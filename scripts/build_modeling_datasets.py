"""把 §5 口径处理好的建模表作为正式 TFDC 数据集写入 Vault。

产出两个数据集，供 ThermoForge 主智能体直接建模（不必再依赖外部脚本）：

- `WX_CHILLER_MODEL`  对象 `CH01..CH04`，模型 `chiller_model.v1`
  A 模式（仅冷机）+ C 模式（冷机+板换）逐台一行：功耗标签 + 温度/流量/负荷特征
- `WX_HX_MODEL`       对象 `HXBANK`，模型 `hx_model.v1`
  B 模式（仅板换）+ C 模式每时刻一行：换热量标签 + 仅入口端口特征

**为什么必须含 C 模式**：限制模型精度的头号因素是工况覆盖太窄 ——
A 模式的负荷率几乎全挤在 0.55~0.65，而 C 模式提供 0.22~0.36 的低负荷段；
板换侧 B 模式测的是「需求」（板换承担全部负荷），只有 C 模式能测出「能力」
（实测 ε≈1）。C 模式的 η 比 A 模式系统性低约 10%，故加 `hx_run_count` /
`chiller_run_count` 列作工况标识，使该偏差可归因、可在模型里显式处理。

口径与假设见 docs/data-processing-handbook.md §5 / §7，逐条编号 A-01..A-13。
所有列 `source_kind` 显式标注：由测点直读的为 measured，经公式算出的为 estimated
（后者不得进候选输入白名单的目标侧，DD-16）。

用法::

    python scripts/build_modeling_datasets.py
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
RATED_KW = 9672.0
IDX = range(1, 5)

# (property_code, unit, dtype, role, source_kind)
CHILLER_COLS: list[tuple[str, str, str, str, str]] = [
    ("power", "kW", "float", "target", "measured"),
    ("cooling_load", "kW", "float", "state", "estimated"),
    ("plr", "1", "float", "state", "estimated"),
    ("t_evap_out", "Cel", "float", "state", "measured"),
    ("t_cond_in", "Cel", "float", "state", "measured"),
    ("lift", "K", "float", "state", "estimated"),
    ("t_cw_tower_out", "Cel", "float", "state", "measured"),
    ("t_chw_return", "Cel", "float", "state", "measured"),
    ("f_cw_branch", "m3/h", "float", "state", "estimated"),
    ("f_chw_branch", "m3/h", "float", "state", "estimated"),
    ("cwp_frequency", "Hz", "float", "state", "measured"),
    ("chwp_frequency", "Hz", "float", "state", "measured"),
    ("run_count", "1", "integer", "state", "measured"),
    ("hx_run_count", "1", "integer", "state", "measured"),
    ("ambient_t", "Cel", "float", "state", "measured"),
    ("ambient_h", "%", "float", "state", "measured"),
    ("runtime_accum", "s", "float", "state", "measured"),
]

HX_COLS: list[tuple[str, str, str, str, str]] = [
    ("heat_transfer", "kW", "float", "target", "estimated"),
    ("t_chw_in", "Cel", "float", "state", "measured"),
    ("t_cw_in", "Cel", "float", "state", "measured"),
    ("drive_dt", "K", "float", "state", "estimated"),
    ("f_chw", "m3/h", "float", "state", "measured"),
    ("f_cw", "m3/h", "float", "state", "measured"),
    ("flow_ratio", "1", "float", "state", "estimated"),
    ("run_count", "1", "integer", "state", "measured"),
    ("chiller_run_count", "1", "integer", "state", "measured"),
    ("ambient_t", "Cel", "float", "state", "measured"),
    ("ambient_h", "%", "float", "state", "measured"),
]


def _dataset(dataset_id: str, model_id: str, objects: list[str],
             cols: list[tuple[str, str, str, str, str]],
             description: str) -> TfdcDataset:
    return TfdcDataset(
        manifest=TfdcManifest(
            contract="TFDC", contract_version="1.0", dataset_id=dataset_id,
            dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
            time_resolution="900s", source_system="derived",
            description=description,
        ),
        objects=[ObjectRecord(object_id=o, object_model_id=model_id,
                              object_name=o) for o in objects],
        variables=[
            VariableRecord(
                variable_id=f"{o}.{code}", object_id=o, property_code=code,
                unit=unit, dtype=dtype, role=role, source_kind=sk, nullable=True,
                sample_period="900s",
            )
            for o in objects for code, unit, dtype, role, sk in cols
        ],
    )


def _split_flows(df: pd.DataFrame, prefix: str, total: pd.Series) -> dict:
    """按泵频加权把总流量拆到支路（A-05）。"""
    on = {i: (df[f"{prefix}_0{i}.status_run"] == True) for i in IDX}   # noqa: E712
    w = {i: df[f"{prefix}_0{i}.status_frequency"].where(on[i], 0.0).fillna(0.0)
         for i in IDX}
    s = sum(w.values()).replace(0, np.nan)
    return {i: total * w[i] / s for i in IDX}


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

    ch_on = {i: (df[f"chiller_0{i}.status_run"] == True) for i in IDX}  # noqa: E712
    cwp_on = {i: (df[f"cwp_0{i}.status_run"] == True) for i in IDX}     # noqa: E712
    chwp_on = {i: (df[f"chwp_0{i}.status_run"] == True) for i in IDX}   # noqa: E712
    n_ch = sum(v.astype(int) for v in ch_on.values())
    n_cwp = sum(v.astype(int) for v in cwp_on.values())
    n_chwp = sum(v.astype(int) for v in chwp_on.values())

    F_cw = df["cw_A1.f"] + df["cw_A2.f"]                       # A-02
    F_chw = df["chw_A1.f"].fillna(0) + df["chw_A2.f"].fillna(0)
    T_cold, T_hot = df["cw_A1.t_return"], df["cw_A1.t_supply"]
    T_sup, T_ret = df["chw_A1.t_supply"], df["chw_A1.t_return"]
    Q_chw = F_chw / 3.6 * CP * (T_ret - T_sup)
    P_i = {i: df[f"chiller_0{i}.power"].fillna(0) for i in IDX}
    T_mid = {i: df[f"chiller_0{i}.condenser_return_t"] for i in IDX}
    f_cw = _split_flows(df, "cwp", F_cw)
    f_chw = _split_flows(df, "chwp", F_chw)

    A = (n_chwp > 0) & (n_ch > 0) & (n_cwp == n_ch) & (Q_chw > 0)
    B = (n_chwp > 0) & (n_ch == 0) & (n_cwp > 0) & (Q_chw > 0)
    C = (n_chwp > 0) & (n_ch > 0) & (n_cwp > n_ch) & (Q_chw > 0)
    n_hx_eff = (n_cwp - n_ch).clip(lower=0)

    # C 模式的支路拆分（A-13）：用「有冷机运行」支路测得的中间温度代表所有
    # 支路的板换出口 —— 停机冷机的 condenser_return_t 读数无效（问题 13）。
    run_mid = pd.concat([T_mid[i].where(ch_on[i] & cwp_on[i]) for i in IDX],
                        axis=1)
    T_hx_out = run_mid.mean(axis=1)
    f_run = sum(f_cw[i].where(ch_on[i] & cwp_on[i], 0.0).fillna(0.0) for i in IDX)
    f_hxonly = sum(f_cw[i].where(cwp_on[i] & ~ch_on[i], 0.0).fillna(0.0)
                   for i in IDX)
    T_ch_out = (F_cw * T_hot - f_hxonly * T_hx_out) / f_run.replace(0, np.nan)
    Q_hx_capability = (F_cw / 3.6 * CP * (T_hx_out - T_cold)).clip(lower=0)

    registry = default_registry()
    for mid in ("chiller_model.v1", "hx_model.v1"):
        if registry.get(mid) is None:
            raise SystemExit(f"物模型未注册: {mid}（应在 contracts/tfom/examples/）")
    results = []

    # ---------------- 冷机建模集（A 模式逐台） ----------------
    objects = [f"CH0{i}" for i in IDX]
    # 只保留「至少有一台冷机在该工况运行」的时刻。保留全年时间轴会让数据集
    # 里 ~70% 的行全空 —— 而冷机只在暖季运行，rolling_cv 的早期折会整折落在
    # 冬季空窗里，直接报「训练集清洗后无样本」（实测 EXP-0009/0010）。
    keep = pd.Series(False, index=df.index)
    for i in IDX:
        keep |= ((A | C) & ch_on[i] & cwp_on[i] & (P_i[i] > 0)).fillna(False)
    axis = sorted(set(ts[keep]))
    index = {t: k for k, t in enumerate(axis)}
    print(f"冷机建模集时间轴: {len(axis)} 个时刻"
          f"（全年 {len(set(ts))} 中有效的部分）")
    columns: dict[str, list] = {}
    for i in IDX:
        run = (A | C) & ch_on[i] & cwp_on[i] & (P_i[i] > 0)
        # A 模式：全部支路都有冷机 → 冷机公共出口 = 塔热
        # C 模式：需先解出冷机公共出口（A-13），且入口是板换出口而非塔冷
        q_a = (f_cw[i] / 3.6 * CP * (T_hot - T_mid[i]) - P_i[i])
        q_c = (f_cw[i] / 3.6 * CP * (T_ch_out - T_hx_out) - P_i[i])
        q = q_a.where(A, q_c).where(run).clip(lower=0)
        vals = {
            "power": P_i[i], "cooling_load": q, "plr": q / RATED_KW,
            "t_evap_out": T_sup, "t_cond_in": T_mid[i],
            "lift": T_mid[i] - T_sup,
            "t_cw_tower_out": T_cold, "t_chw_return": T_ret,
            "f_cw_branch": f_cw[i], "f_chw_branch": f_chw[i],
            "cwp_frequency": df[f"cwp_0{i}.status_frequency"],
            "chwp_frequency": df[f"chwp_0{i}.status_frequency"],
            "run_count": n_ch, "hx_run_count": n_hx_eff,
            "ambient_t": df["environment_parameters.ambient_t"],
            "ambient_h": df["environment_parameters.ambient_h"],
            "runtime_accum": df[f"chiller_0{i}.runtime_accum"],
        }
        for code, _u, dtype, _r, _sk in CHILLER_COLS:
            col = [None] * len(axis)
            series = vals[code].where(run)
            for t, v in zip(ts, series):
                if pd.notna(v) and t in index:
                    col[index[t]] = int(v) if dtype == "integer" else float(v)
            columns[f"CH0{i}.{code}"] = col

    ds = _dataset("WX_CHILLER_MODEL", "chiller_model.v1", objects, CHILLER_COLS,
                  "A∪C 模式逐台建模表；hx_run_count=0 为 A 模式。见 handbook §5.3/§7.5")
    res = import_parsed(ds, axis, columns, registry=registry)
    if not res.ok:
        for d in res.diagnostics[:10]:
            print("  ", d.code, d.level, getattr(d, "location", ""))
        raise SystemExit("冷机建模集导入失败")
    ref = ctx.vault.store(res, lineage={
        "derived_from": args.ref, "mode": "A（仅冷机运行）",
        "handbook": "docs/data-processing-handbook.md §5.2/§5.3",
        "assumptions": ["A-02", "A-04", "A-05", "A-11", "A-13"],
    })
    results.append((ref, res.table.num_rows, len(res.dataset.variables)))
    print(f"冷机建模集 {ref}  rows={res.table.num_rows}")

    # ---------------- 板换建模集（B 模式，只用入口值） ----------------
    drive = T_ret - T_cold
    # B 模式：板换承担全部冷量（需求驱动）
    # C 模式：只做预冷，Q 由板换段温升算出（能力窗口，ε≈1）
    Q_hx = Q_chw.where(B, Q_hx_capability)
    vals = {
        "heat_transfer": Q_hx, "t_chw_in": T_ret, "t_cw_in": T_cold,
        "drive_dt": drive, "f_chw": F_chw, "f_cw": F_cw,
        "flow_ratio": F_cw / F_chw.replace(0, np.nan),
        "run_count": n_cwp.where(B, n_hx_eff), "chiller_run_count": n_ch,
        "ambient_t": df["environment_parameters.ambient_t"],
        "ambient_h": df["environment_parameters.ambient_h"],
    }
    hx_keep = ((B | C) & Q_hx.notna()).fillna(False)
    axis = sorted(set(ts[hx_keep]))
    index = {t: k for k, t in enumerate(axis)}
    print(f"板换建模集时间轴: {len(axis)} 个时刻")
    columns = {}
    for code, _u, dtype, _r, _sk in HX_COLS:
        col = [None] * len(axis)
        series = vals[code].where(B | C)
        for t, v in zip(ts, series):
            if pd.notna(v) and t in index:
                col[index[t]] = int(v) if dtype == "integer" else float(v)
        columns[f"HXBANK.{code}"] = col

    ds = _dataset("WX_HX_MODEL", "hx_model.v1", ["HXBANK"], HX_COLS,
                  "B∪C 模式建模表；chiller_run_count=0 为 B 模式（需求驱动），"
                  ">0 为 C 模式（能力窗口）。见 handbook §7.3/§7.5")
    res = import_parsed(ds, axis, columns, registry=registry)
    if not res.ok:
        for d in res.diagnostics[:10]:
            print("  ", d.code, d.level, getattr(d, "location", ""))
        raise SystemExit("板换建模集导入失败")
    ref = ctx.vault.store(res, lineage={
        "derived_from": args.ref, "mode": "B（仅板换运行）",
        "handbook": "docs/data-processing-handbook.md §7.3",
        "assumptions": ["A-02", "A-05", "A-11", "A-13"],
        "note": "B 模式标签=冷冻侧全部冷量（无拆分假设）；C 模式标签=板换段能力（依赖 A-13）",
    })
    results.append((ref, res.table.num_rows, len(res.dataset.variables)))
    print(f"板换建模集 {ref}  rows={res.table.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
