"""把冷却塔群建模表作为正式 TFDC 数据集写入 Vault。

产出 `WX_CT_MODEL`，对象 `CTBANK`，物模型 `ct_model.v1`。

**为什么是塔群不是逐塔**：本版导出的冷却塔工作表整表为空
（`ct_0i.supply_t` / `return_t` / `instant_flow` 非空计数均为 1，值 0.0，
见 docs/data-survey.md「冷却塔工作表全空」TFDC-306）。逐塔水侧没有任何
数据，只有 12 台风机的电气量是全的，所以水侧只能取总管口径。

口径（沿用 docs/data-processing-handbook.md §4 / A-02 / A-09 / A-11）：

- 塔进水（热）= `cw_A1.t_supply`，塔出水（冷）= `cw_A1.t_return`
  —— 总管命名是**塔视角**：送去塔的是热水，从塔回来的是冷水。
- 冷却水总流量 = `cw_A1.f + cw_A2.f`（A-02，闭合比 1.000）。
- 湿球温度用 Stull(2011) 近似由干球+相对湿度算出，标 `source_kind=estimated`。
  适用域 RH 5~99%、T −20~50 °C，本数据集全部落在域内。
- `heat_reject` 由进出水温差算出，**含标签**，只能进 `fan_power_total`
  目标的白名单（DD-16）。

用法::

    python scripts/build_cooling_tower_dataset.py
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
FAN_IDX = [(i, j) for i in range(1, 5) for j in range(1, 4)]
OBJECT_ID = "CTBANK"
MODEL_ID = "ct_model.v1"

# (property_code, unit, dtype, role, source_kind)
CT_COLS: list[tuple[str, str, str, str, str]] = [
    ("t_ct_out", "Cel", "float", "target", "measured"),
    ("fan_power_total", "kW", "float", "target", "measured"),
    ("t_ct_in", "Cel", "float", "state", "measured"),
    ("f_cw_total", "m3/h", "float", "state", "measured"),
    ("t_wetbulb", "Cel", "float", "disturbance", "estimated"),
    ("ambient_t", "Cel", "float", "disturbance", "measured"),
    ("ambient_h", "%", "float", "disturbance", "measured"),
    ("fan_run_count", "1", "integer", "control", "measured"),
    ("fan_freq_mean", "Hz", "float", "control", "measured"),
    ("fan_freq_max", "Hz", "float", "control", "measured"),
    ("cwp_run_count", "1", "integer", "state", "measured"),
    ("chiller_run_count", "1", "integer", "state", "measured"),
    ("heat_reject", "kW", "float", "state", "estimated"),
    ("approach", "K", "float", "target", "estimated"),
]


def wet_bulb(t_db: pd.Series, rh: pd.Series) -> pd.Series:
    """Stull (2011) 湿球温度近似（J. Appl. Meteor. Climatol. 50, 2267）。

    适用域 RH 5~99%、T −20~50 °C，海平面气压；域外置 NaN 而不是外推 ——
    近似式在域外偏差会放大到数 K，静默外推等于往标签里掺噪声。
    """
    t = pd.to_numeric(t_db, errors="coerce")
    h = pd.to_numeric(rh, errors="coerce")
    tw = (t * np.arctan(0.151977 * np.sqrt(h + 8.313659))
          + np.arctan(t + h)
          - np.arctan(h - 1.676331)
          + 0.00391838 * np.power(h, 1.5) * np.arctan(0.023101 * h)
          - 4.686035)
    return tw.where((h >= 5) & (h <= 99) & (t >= -20) & (t <= 50))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="WX_2025_HVAC@rev_0001")
    ap.add_argument("--vault-root", type=Path, default=Path("vault"))
    ap.add_argument("--research-root", type=Path, default=Path("research"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--season-only", action="store_true",
                    help="只保留塔真正带负荷的时刻（有冷机在运行），产出 "
                         "WX_CT_SEASON。全年口径下逼近度被季节性支配："
                         "冬季湿球低到 −18℃ 而塔出水被限在 11.5℃ 以上，"
                         "approach 与湿球相关 −0.954，单变量线性就能到 "
                         "R²≈0.91 —— 那考的是天气，不是塔")
    args = ap.parse_args(argv)

    ctx = ToolContext(vault_root=args.vault_root, research_root=args.research_root,
                      models_root=args.models_root, actor="human")
    df = ctx.vault.load_data(args.ref).to_pandas()
    ts = pd.to_datetime(df["timestamp"])

    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731

    t_in = num("cw_A1.t_supply")      # 送去塔的水（热）
    t_out = num("cw_A1.t_return")     # 从塔回来的水（冷）
    f_cw = num("cw_A1.f") + num("cw_A2.f")                      # A-02
    amb_t = num("environment_parameters.ambient_t")
    amb_h = num("environment_parameters.ambient_h")

    fan_power = pd.Series(0.0, index=df.index)
    fan_on = pd.Series(0, index=df.index)
    freq_sum = pd.Series(0.0, index=df.index)
    freq_max = pd.Series(0.0, index=df.index)
    for i, j in FAN_IDX:
        base = f"ct_0{i}_f_0{j}"
        on = (df[f"{base}.status_run"] == True).fillna(False)    # noqa: E712
        p = num(f"{base}.power").fillna(0.0)
        f = num(f"{base}.status_frequency").fillna(0.0)
        fan_power += p.where(on, 0.0)
        fan_on += on.astype(int)
        freq_sum += f.where(on, 0.0)
        freq_max = np.maximum(freq_max, f.where(on, 0.0))
    freq_mean = freq_sum / fan_on.replace(0, np.nan)

    cwp_on = sum((df[f"cwp_0{i}.status_run"] == True).astype(int)  # noqa: E712
                 for i in range(1, 5))
    ch_on = sum((df[f"chiller_0{i}.status_run"] == True).astype(int)  # noqa: E712
                for i in range(1, 5))

    q_reject = f_cw / 3.6 * CP * (t_in - t_out)
    t_wb = wet_bulb(amb_t, amb_h)
    # 逼近度 = 出水温 − 湿球温度，冷却塔的行业标准性能指标。
    # 它才是「塔干得好不好」：出水温本身有 98% 的变化能被进水温解释，
    # 拿它当标签等于考一道送分题（handbook §4 / plan 2026-08-18 夜）。
    # 注意白名单：`heat_reject` 与 `t_ct_out` 都必须排除 —— 由
    # t_ct_in、heat_reject、f_cw_total 可反解出 t_ct_out，再减湿球即标签。
    approach = t_out - t_wb

    vals = {
        "t_ct_out": t_out,
        "fan_power_total": fan_power,
        "t_ct_in": t_in,
        "f_cw_total": f_cw,
        "t_wetbulb": t_wb,
        "ambient_t": amb_t,
        "ambient_h": amb_h,
        "fan_run_count": fan_on,
        "fan_freq_mean": freq_mean,
        "fan_freq_max": pd.Series(freq_max, index=df.index),
        "cwp_run_count": cwp_on,
        "chiller_run_count": ch_on,
        "heat_reject": q_reject,
        "approach": approach,
    }

    # 只保留「塔在工作」的时刻：有水在循环 + 至少一台风机在转 + 温度可读。
    # 保留全年时间轴会让大量行全空，rolling_cv 的早期折整折落在空窗里
    # （冷机建模集踩过同一个坑，见 build_modeling_datasets.py 注释）。
    keep = (t_in.notna() & t_out.notna() & f_cw.notna()
            & (f_cw > 0) & (fan_on > 0) & (cwp_on > 0)).fillna(False)
    if args.season_only:
        keep = keep & (ch_on > 0)
    axis = sorted(set(ts[keep]))
    index = {t: k for k, t in enumerate(axis)}
    print(f"冷却塔建模集时间轴: {len(axis)} 个时刻"
          f"（全年 {len(set(ts))} 中有效的部分）")

    columns: dict[str, list] = {}
    for code, _u, dtype, _r, _sk in CT_COLS:
        col: list = [None] * len(axis)
        series = vals[code].where(keep)
        for t, v in zip(ts, series):
            if pd.notna(v) and t in index:
                col[index[t]] = int(v) if dtype == "integer" else float(v)
        columns[f"{OBJECT_ID}.{code}"] = col

    registry = default_registry()
    if registry.get(MODEL_ID) is None:
        raise SystemExit(f"物模型未注册: {MODEL_ID}（应在 contracts/tfom/examples/）")

    ds = TfdcDataset(
        manifest=TfdcManifest(
            contract="TFDC", contract_version="1.0",
            # 制冷季口径与全年口径是两套语义，不该做成同一 dataset 的两个
            # 修订版（修订版表示「同一件事的新版本」）——分成两个数据集，
            # 免得视图/实验引用时把「全年」和「带负荷」搞混。
            dataset_id=("WX_CT_SEASON" if args.season_only else "WX_CT_MODEL"),
            dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
            time_resolution="900s", source_system="derived",
            description=(
                "冷却塔群建模表（塔群总管口径；逐塔工作表整表为空，见 "
                "data-survey TFDC-306）。标签 t_ct_out / fan_power_total / "
                "approach。" + ("仅制冷季（有冷机运行、塔真正带负荷）。"
                                if args.season_only else "全年。")),
        ),
        objects=[ObjectRecord(object_id=OBJECT_ID, object_model_id=MODEL_ID,
                              object_name="冷却塔群（4 塔 12 风机）")],
        variables=[
            VariableRecord(
                variable_id=f"{OBJECT_ID}.{code}", object_id=OBJECT_ID,
                property_code=code, unit=unit, dtype=dtype, role=role,
                source_kind=sk, nullable=True, sample_period="900s",
            )
            for code, unit, dtype, role, sk in CT_COLS
        ],
    )
    res = import_parsed(ds, axis, columns, registry=registry)
    if not res.ok:
        for d in res.diagnostics[:10]:
            print("  ", d.code, d.level, getattr(d, "location", ""))
        raise SystemExit("冷却塔建模集导入失败")
    ref = ctx.vault.store(res, lineage={
        "derived_from": args.ref,
        "scope": ("塔群总管口径，仅制冷季（chiller_run_count>0）"
                  if args.season_only
                  else "塔群总管口径（逐塔水侧无数据，TFDC-306）"),
        "handbook": "docs/data-processing-handbook.md §4 / A-02 / A-09",
        "assumptions": ["A-02", "A-09", "A-11"],
        "note": "t_wetbulb 为 Stull(2011) 近似；heat_reject 含标签，"
                "仅限 fan_power_total 目标的白名单",
    })
    print(f"冷却塔建模集 {ref}  rows={res.table.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
