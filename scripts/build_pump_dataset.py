"""把水泵群建模表作为正式 TFDC 数据集写入 Vault。

产出两个数据集，对象都是「泵群」而非逐台 —— 逐台功耗虽然有，但**流量只有
总管测点**（`chw_A1/A2.f`、`cw_A1/A2.f`），逐台流量得靠泵频加权分摊
（[A-05](docs/data-processing-handbook.md#a-05)），那是个假设不是测量；
泵群口径下总功耗与总流量都是实测，标签和自变量都不依赖分摊假设。

- `WX_CHWP_MODEL` 对象 `CHWPBANK`：冷冻水泵群（chwp_01..04）
- `WX_CWP_MODEL`  对象 `CWPBANK`：冷却水泵群（cwp_01..04）

**与冷却塔风机的关键差别**：风机侧没有任何风量测点，所以风机功耗只能由
频率和台数去猜，实测七个幂律写法全部收敛到 R²≈0.54（信息瓶颈，见
plan/active-plan.md）。水泵侧**流量是实测的**，`P = f(Q, N, f)` 定得住，
不该重蹈覆辙。

口径沿用 handbook：总管流量两路相加（A-02 冷却侧 / A-09 冷冻侧），
温度类重复列只取 A1（A-09），水物性取常数（A-11）。

用法::

    python scripts/build_pump_dataset.py
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

CP = 4.187
IDX = range(1, 5)
MODEL_ID = "pump_model.v1"

# (property_code, unit, dtype, role, source_kind)
PUMP_COLS: list[tuple[str, str, str, str, str]] = [
    ("power_total", "kW", "float", "target", "measured"),
    ("flow_total", "m3/h", "float", "state", "measured"),
    ("run_count", "1", "integer", "control", "measured"),
    ("freq_mean", "Hz", "float", "control", "measured"),
    ("freq_max", "Hz", "float", "control", "measured"),
    ("flow_per_pump", "m3/h", "float", "state", "estimated"),
    ("t_supply", "Cel", "float", "state", "measured"),
    ("t_return", "Cel", "float", "state", "measured"),
    ("delta_t", "K", "float", "state", "estimated"),
    ("chiller_run_count", "1", "integer", "state", "measured"),
    ("ambient_t", "Cel", "float", "disturbance", "measured"),
]

SPECS = [
    ("WX_CHWP_MODEL", "CHWPBANK", "chwp", "chw_A1", "chw_A2", "冷冻水泵群"),
    ("WX_CWP_MODEL", "CWPBANK", "cwp", "cw_A1", "cw_A2", "冷却水泵群"),
]


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
    if registry.get(MODEL_ID) is None:
        raise SystemExit(f"物模型未注册: {MODEL_ID}（应在 contracts/tfom/examples/）")

    ch_on = sum((df[f"chiller_0{i}.status_run"] == True).astype(int)  # noqa: E712
                for i in IDX)
    amb_t = num("environment_parameters.ambient_t")

    for dataset_id, object_id, prefix, hdr_a, hdr_b, label in SPECS:
        power = pd.Series(0.0, index=df.index)
        on_count = pd.Series(0, index=df.index)
        freq_sum = pd.Series(0.0, index=df.index)
        freq_max = pd.Series(0.0, index=df.index)
        for i in IDX:
            base = f"{prefix}_0{i}"
            on = (df[f"{base}.status_run"] == True).fillna(False)   # noqa: E712
            power += num(f"{base}.power").fillna(0.0).where(on, 0.0)
            on_count += on.astype(int)
            f = num(f"{base}.status_frequency").fillna(0.0)
            freq_sum += f.where(on, 0.0)
            freq_max = np.maximum(freq_max, f.where(on, 0.0))
        freq_mean = freq_sum / on_count.replace(0, np.nan)

        flow = num(f"{hdr_a}.f").fillna(0.0) + num(f"{hdr_b}.f").fillna(0.0)
        t_sup, t_ret = num(f"{hdr_a}.t_supply"), num(f"{hdr_a}.t_return")

        vals = {
            "power_total": power,
            "flow_total": flow,
            "run_count": on_count,
            "freq_mean": freq_mean,
            "freq_max": pd.Series(freq_max, index=df.index),
            # 单泵平均流量：泵的工作点落在 Q-H 曲线的哪一段由它决定，
            # 总流量除以台数比总流量本身更接近「每台泵在干什么」
            "flow_per_pump": flow / on_count.replace(0, np.nan),
            "t_supply": t_sup,
            "t_return": t_ret,
            "delta_t": t_ret - t_sup,
            "chiller_run_count": ch_on,
            "ambient_t": amb_t,
        }

        keep = ((on_count > 0) & (power > 0) & flow.notna()
                & (flow > 0)).fillna(False)
        axis = sorted(set(ts[keep]))
        index = {t: k for k, t in enumerate(axis)}
        print(f"{label}建模集时间轴: {len(axis)} 个时刻")

        columns: dict[str, list] = {}
        for code, _u, dtype, _r, _sk in PUMP_COLS:
            col: list = [None] * len(axis)
            series = vals[code].where(keep)
            for t, v in zip(ts, series):
                if pd.notna(v) and t in index:
                    col[index[t]] = int(v) if dtype == "integer" else float(v)
            columns[f"{object_id}.{code}"] = col

        ds = TfdcDataset(
            manifest=TfdcManifest(
                contract="TFDC", contract_version="1.0", dataset_id=dataset_id,
                dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
                time_resolution="900s", source_system="derived",
                description=f"{label}建模表（泵群口径：总功耗与总流量都是实测，"
                            f"不依赖 A-05 的逐台流量分摊假设）。标签 power_total。",
            ),
            objects=[ObjectRecord(object_id=object_id, object_model_id=MODEL_ID,
                                  object_name=label)],
            variables=[
                VariableRecord(
                    variable_id=f"{object_id}.{code}", object_id=object_id,
                    property_code=code, unit=unit, dtype=dtype, role=role,
                    source_kind=sk, nullable=True, sample_period="900s",
                )
                for code, unit, dtype, role, sk in PUMP_COLS
            ],
        )
        res = import_parsed(ds, axis, columns, registry=registry)
        if not res.ok:
            for d in res.diagnostics[:10]:
                print("  ", d.code, d.level, getattr(d, "location", ""))
            raise SystemExit(f"{label}建模集导入失败")
        ref = ctx.vault.store(res, lineage={
            "derived_from": args.ref,
            "scope": f"{label}总管口径",
            "handbook": "docs/data-processing-handbook.md A-02 / A-09 / A-11",
            "assumptions": ["A-02", "A-09", "A-11"],
        })
        print(f"{label}建模集 {ref}  rows={res.table.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
