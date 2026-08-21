"""逐台水泵建模表 + 泵群逐台频率表。

两个产出，对应两类问题：

- `WX_PUMP_UNIT`（对象 `CHWP01..04` / `CWP01..04`，物模型 `pump_unit.v1`）
  单台：频率 → 功耗。实测 8 台泵的 log-log 斜率全落在 2.96~3.05，
  是这份数据里最干净的物理关系（相似定律 P ∝ f³）。
- `WX_CHWP_UNITFREQ` / `WX_CWP_UNITFREQ`（对象 `CHWPBANK` / `CWPBANK`，
  物模型 `pump_model.v1`）泵群：**逐台频率**（而非群平均）+ 台数 → 总流量。

**为什么要逐台频率**：台间频率确有差异（冷冻侧极差中位 0.51 Hz、p95 3.80、
最大 15.89；冷却侧 1.07 / 5.99 / 34.79），早先用群平均聚合会抹掉这部分信息。

**两侧流量的性质不同，必须分开看**：

- 冷冻侧 `chw_A1.f + chw_A2.f` 是**两个独立支路实测**（逐点相同仅 5.9%）。
- 冷却侧 `cw_A1.f` 两列逐点完全相同，且实测可由
  `(Q_evap + P_chiller)/(cp·ΔT_cw)` 精确还原（相关 1.000000、相对偏差中位
  0.0013%）——它是**冷量反算的虚拟流量计**，不是直接测量。
  用它当标签是正当的（第一定律精确，工程常规做法），但**白名单绝不能含
  `delta_t`**：ΔT 是该算式的分母，放进去就是拿公式的因子预测公式的结果
  （DD-16 循环论证，实测会虚高到 R²=0.764）。
  该标签自带约 4% 的重建噪声底（继承冷冻侧流量计 + 4 个温度探头误差），
  对照实测侧残差 2.66% vs 反算侧 8.59% 可见。

用法::

    python scripts/build_pump_unit_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_data.importer import default_registry, import_parsed
from thermoforge_research.tools import ToolContext

IDX = range(1, 5)

UNIT_COLS: list[tuple[str, str, str, str, str]] = [
    ("power", "kW", "float", "target", "measured"),
    ("frequency", "Hz", "float", "control", "measured"),
    ("bank_run_count", "1", "integer", "state", "measured"),
    ("bank_freq_mean", "Hz", "float", "state", "measured"),
    ("bank_freq_sum", "Hz", "float", "state", "measured"),
    ("chiller_run_count", "1", "integer", "state", "measured"),
]

# 泵群逐台频率表：目标是总流量，特征给到逐台频率
BANK_COLS: list[tuple[str, str, str, str, str]] = [
    ("flow_total", "m3/h", "float", "target", "measured"),
    ("power_total", "kW", "float", "target", "measured"),
    ("run_count", "1", "integer", "control", "measured"),
    ("freq_p1", "Hz", "float", "control", "measured"),
    ("freq_p2", "Hz", "float", "control", "measured"),
    ("freq_p3", "Hz", "float", "control", "measured"),
    ("freq_p4", "Hz", "float", "control", "measured"),
    ("freq_sum", "Hz", "float", "control", "measured"),
    ("freq_mean", "Hz", "float", "control", "measured"),
    ("freq_spread", "Hz", "float", "control", "measured"),
    ("chiller_run_count", "1", "integer", "state", "measured"),
    ("ambient_t", "Cel", "float", "disturbance", "measured"),
]

SIDES = [
    ("chwp", "CHWP", "CHWPBANK", "WX_CHWP_UNITFREQ", "冷冻水泵"),
    ("cwp", "CWP", "CWPBANK", "WX_CWP_UNITFREQ", "冷却水泵"),
]


def _import_store(ctx, ds, axis, columns, registry, lineage, label):
    res = import_parsed(ds, axis, columns, registry=registry)
    if not res.ok:
        for d in res.diagnostics[:10]:
            print("  ", d.code, d.level, getattr(d, "location", ""))
        raise SystemExit(f"{label} 导入失败")
    ref = ctx.vault.store(res, lineage=lineage)
    print(f"{label} {ref}  rows={res.table.num_rows}")
    return ref


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="WX_2025_HVAC@rev_0001")
    ap.add_argument("--vault-root", type=Path, default=Path("vault"))
    ap.add_argument("--research-root", type=Path, default=Path("research"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    args = ap.parse_args(argv)

    ctx = ToolContext(vault_root=args.vault_root, research_root=args.research_root,
                      models_root=args.models_root, actor="human")
    df = ctx.vault.load_data(args.ref).to_pandas()
    ts = pd.to_datetime(df["timestamp"])
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    registry = default_registry()
    for mid in ("pump_unit.v1", "pump_model.v1"):
        if registry.get(mid) is None:
            raise SystemExit(f"物模型未注册: {mid}")

    ch_on = sum((df[f"chiller_0{i}.status_run"] == True).astype(int)  # noqa: E712
                for i in IDX)
    amb_t = num("environment_parameters.ambient_t")

    # ---------------- 逐台泵表（8 个对象共一张表） ----------------
    unit_objects: list[str] = []
    unit_values: dict[str, dict[str, pd.Series]] = {}
    keep_any = pd.Series(False, index=df.index)
    side_state: dict[str, dict] = {}

    for prefix, obj_prefix, _bank, _ds, label in SIDES:
        on = {i: (df[f"{prefix}_0{i}.status_run"] == True).fillna(False)  # noqa: E712
              for i in IDX}
        freq = {i: num(f"{prefix}_0{i}.status_frequency") for i in IDX}
        power = {i: num(f"{prefix}_0{i}.power") for i in IDX}
        count = sum(v.astype(int) for v in on.values())
        fsum = sum(freq[i].where(on[i], 0.0).fillna(0.0) for i in IDX)
        fmean = fsum / count.replace(0, np.nan)
        side_state[prefix] = {"on": on, "freq": freq, "power": power,
                              "count": count, "fsum": fsum, "fmean": fmean}
        for i in IDX:
            obj = f"{obj_prefix}0{i}"
            unit_objects.append(obj)
            run = on[i] & power[i].notna() & (power[i] > 0) & freq[i].notna()
            unit_values[obj] = {
                "power": power[i].where(run),
                "frequency": freq[i].where(run),
                "bank_run_count": count.where(run),
                "bank_freq_mean": fmean.where(run),
                "bank_freq_sum": fsum.where(run),
                "chiller_run_count": ch_on.where(run),
            }
            keep_any |= run.fillna(False)

    axis = sorted(set(ts[keep_any]))
    index = {t: k for k, t in enumerate(axis)}
    print(f"逐台泵表时间轴: {len(axis)} 个时刻 × {len(unit_objects)} 台")
    columns: dict[str, list] = {}
    for obj in unit_objects:
        for code, _u, dtype, _r, _sk in UNIT_COLS:
            col: list = [None] * len(axis)
            for t, v in zip(ts, unit_values[obj][code]):
                if pd.notna(v) and t in index:
                    col[index[t]] = int(v) if dtype == "integer" else float(v)
            columns[f"{obj}.{code}"] = col

    ds = TfdcDataset(
        manifest=TfdcManifest(
            contract="TFDC", contract_version="1.0", dataset_id="WX_PUMP_UNIT",
            dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
            time_resolution="900s", source_system="derived",
            description="逐台水泵建模表（8 台）：频率 → 功耗，相似定律 P∝f³。"
                        "只保留该台在运行且功耗为正的时刻。",
        ),
        objects=[ObjectRecord(object_id=o, object_model_id="pump_unit.v1",
                              object_name=o) for o in unit_objects],
        variables=[
            VariableRecord(
                variable_id=f"{o}.{code}", object_id=o, property_code=code,
                unit=unit, dtype=dtype, role=role, source_kind=sk,
                nullable=True, sample_period="900s")
            for o in unit_objects for code, unit, dtype, role, sk in UNIT_COLS
        ],
    )
    _import_store(ctx, ds, axis, columns, registry, {
        "derived_from": args.ref,
        "scope": "逐台泵（频率→功耗）",
        "note": "8 台泵 log-log 斜率实测 2.96~3.05，符合 P∝f³",
    }, "逐台泵表")

    # ---------------- 泵群逐台频率表（两侧各一张） ----------------
    for prefix, _obj_prefix, bank, dataset_id, label in SIDES:
        st = side_state[prefix]
        on, freq, power = st["on"], st["freq"], st["power"]
        count, fsum, fmean = st["count"], st["fsum"], st["fmean"]
        p_total = sum(power[i].fillna(0.0).where(on[i], 0.0) for i in IDX)
        running = pd.concat([freq[i].where(on[i]) for i in IDX], axis=1)
        spread = running.max(axis=1) - running.min(axis=1)
        if prefix == "chwp":
            # 两个独立支路实测（逐点相同仅 5.9%），相加得总量（A-09）
            flow = num("chw_A1.f").fillna(0.0) + num("chw_A2.f").fillna(0.0)
        else:
            # 两列逐点完全相同；A-02 的闭合检验支持按 ×2 取总量。
            # 该列本身是冷量反算的虚拟流量计，见模块 docstring。
            flow = num("cw_A1.f") * 2.0

        vals = {
            "flow_total": flow, "power_total": p_total, "run_count": count,
            "freq_p1": freq[1].where(on[1]), "freq_p2": freq[2].where(on[2]),
            "freq_p3": freq[3].where(on[3]), "freq_p4": freq[4].where(on[4]),
            "freq_sum": fsum, "freq_mean": fmean, "freq_spread": spread,
            "chiller_run_count": ch_on, "ambient_t": amb_t,
        }
        keep = ((count > 0) & (p_total > 0) & flow.notna()
                & (flow > 0)).fillna(False)
        bank_axis = sorted(set(ts[keep]))
        bank_index = {t: k for k, t in enumerate(bank_axis)}
        print(f"{label}逐台频率表时间轴: {len(bank_axis)} 个时刻")
        bank_columns: dict[str, list] = {}
        for code, _u, dtype, _r, _sk in BANK_COLS:
            col = [None] * len(bank_axis)
            for t, v in zip(ts, vals[code].where(keep)):
                if pd.notna(v) and t in bank_index:
                    col[bank_index[t]] = (int(v) if dtype == "integer"
                                          else float(v))
            bank_columns[f"{bank}.{code}"] = col

        ds = TfdcDataset(
            manifest=TfdcManifest(
                contract="TFDC", contract_version="1.0", dataset_id=dataset_id,
                dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
                time_resolution="900s", source_system="derived",
                description=f"{label}群逐台频率表：各台频率 + 台数 → 总流量/总功耗。"
                            + ("流量为两支路实测之和。" if prefix == "chwp" else
                               "流量为冷量反算的虚拟流量计，白名单严禁含 delta_t。"),
            ),
            objects=[ObjectRecord(object_id=bank,
                                  object_model_id="pump_model.v1",
                                  object_name=label)],
            variables=[
                VariableRecord(
                    variable_id=f"{bank}.{code}", object_id=bank,
                    property_code=code, unit=unit, dtype=dtype, role=role,
                    source_kind=("measured" if prefix == "chwp"
                                 or code != "flow_total" else "estimated"),
                    nullable=True, sample_period="900s")
                for code, unit, dtype, role, _sk in BANK_COLS
            ],
        )
        _import_store(ctx, ds, bank_axis, bank_columns, registry, {
            "derived_from": args.ref, "scope": f"{label}群（逐台频率）",
            "flow_source": ("两支路实测相加" if prefix == "chwp"
                            else "冷量反算：(Q_evap+P_chiller)/(cp·ΔT_cw)，"
                                 "噪声底约 4%"),
        }, f"{label}逐台频率表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
