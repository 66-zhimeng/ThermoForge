"""导出处理后的数据集：原始参数 + 中间量 + 冷量/换热量结果。

口径见 docs/data-processing-handbook.md §5。输出两张表：

- `processed_plant.csv`  每个时刻一行：工况、总量、板换换热量、冷机池冷量
- `processed_chiller.csv` 每台冷机每个运行时刻一行：逐台冷量、COP、卡诺效率

用法::

    python scripts/export_processed_dataset.py [--out-dir data_out]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CP = 4.187              # kJ/(kg·K)，A-11
RATED_KW = 9672.0       # 单台额定冷量
ETA_LO, ETA_HI = 0.03, 0.45   # A-06
IDX = range(1, 5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default="WX_2025_HVAC@rev_0001")
    p.add_argument("--vault-root", type=Path, default=Path("vault"))
    p.add_argument("--out-dir", type=Path, default=Path("data_out"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_id, revision = args.ref.split("@")
    df = pd.read_parquet(args.vault_root / "datasets" / dataset_id / revision
                         / "canonical" / "data.parquet")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 原始输入 ----------------
    ts = df["timestamp"]
    F_cw = df["cw_A1.f"] + df["cw_A2.f"]                    # A-02 两列相加
    F_chw = df["chw_A1.f"].fillna(0) + df["chw_A2.f"].fillna(0)
    T_cw_cold = df["cw_A1.t_return"]      # 塔来的冷水（塔视角 return = 冷）
    T_cw_hot = df["cw_A1.t_supply"]       # 去塔的热水
    T_chw_sup, T_chw_ret = df["chw_A1.t_supply"], df["chw_A1.t_return"]

    pump_on = {i: (df[f"cwp_0{i}.status_run"] == True) for i in IDX}    # noqa: E712
    ch_on = {i: (df[f"chiller_0{i}.status_run"] == True) for i in IDX}  # noqa: E712
    chwp_on = {i: (df[f"chwp_0{i}.status_run"] == True) for i in IDX}   # noqa: E712
    T_mid_i = {i: df[f"chiller_0{i}.condenser_return_t"] for i in IDX}  # 支路中间温
    P_i = {i: df[f"chiller_0{i}.power"].fillna(0) for i in IDX}
    freq_i = {i: df[f"cwp_0{i}.status_frequency"] for i in IDX}

    n_ch = sum(v.astype(int) for v in ch_on.values())
    n_cwp = sum(v.astype(int) for v in pump_on.values())
    n_chwp = sum(v.astype(int) for v in chwp_on.values())
    n_hx_eff = (n_cwp - n_ch).clip(lower=0)
    P_total = sum(P_i.values())

    # ---------------- 支路流量：频率加权（A-05） ----------------
    w = {i: freq_i[i].where(pump_on[i], 0.0).fillna(0.0) for i in IDX}
    w_sum = sum(w.values()).replace(0, np.nan)
    f_i = {i: F_cw * w[i] / w_sum for i in IDX}

    # 冷冻侧同理：两侧水泵同支路串联（n_chwp == n_cwp 一致率 99.0%）
    freq_chw_i = {i: df[f"chwp_0{i}.status_frequency"] for i in IDX}
    wc = {i: freq_chw_i[i].where(chwp_on[i], 0.0).fillna(0.0) for i in IDX}
    wc_sum = sum(wc.values()).replace(0, np.nan)
    f_chw_i = {i: F_chw * wc[i] / wc_sum for i in IDX}

    # ---------------- 板换换热量：逐支路（§5.2） ----------------
    dT_hx_i = {i: (T_mid_i[i] - T_cw_cold).where(pump_on[i]).clip(lower=0)
               for i in IDX}
    Q_hx_i = {i: (f_i[i] / 3.6 * CP * dT_hx_i[i]) for i in IDX}
    Q_hx = sum(q.fillna(0.0) for q in Q_hx_i.values())

    # ---------------- 反解冷机公共冷凝出水温（A-12，§5.3） ----------------
    num, den = F_cw * T_cw_hot, pd.Series(0.0, index=df.index)
    for i in IDX:
        hx_only = pump_on[i] & ~ch_on[i]
        num = num - (f_i[i] * T_mid_i[i]).where(hx_only, 0.0).fillna(0.0)
        den = den + f_i[i].where(pump_on[i] & ch_on[i], 0.0).fillna(0.0)
    T_ch_out = num / den.replace(0, np.nan)

    # ---------------- 逐台冷机冷量（§5.3） ----------------
    run_i = {i: pump_on[i] & ch_on[i] for i in IDX}
    Q_rej_i = {i: (f_i[i] / 3.6 * CP * (T_ch_out - T_mid_i[i])).where(run_i[i])
               for i in IDX}
    Q_ch_i = {i: (Q_rej_i[i] - P_i[i]).where(run_i[i]).clip(lower=0) for i in IDX}
    lift_i = {i: (T_mid_i[i] - T_chw_sup).where(run_i[i]) for i in IDX}
    carnot_i = {i: (T_chw_sup + 273.15) / lift_i[i].where(lift_i[i] > 0.5)
                for i in IDX}
    cop_i = {i: Q_ch_i[i] / P_i[i].replace(0, np.nan) for i in IDX}
    eta_i = {i: cop_i[i] / carnot_i[i] for i in IDX}

    Q_total_chw = F_chw / 3.6 * CP * (T_chw_ret - T_chw_sup)
    Q_ch_pool = (F_cw / 3.6 * CP * (T_cw_hot - T_cw_cold)
                 - Q_hx - P_total).clip(lower=0)

    mode = pd.Series("其它", index=df.index)
    mode[(n_chwp > 0) & (n_ch > 0) & (n_hx_eff == 0)] = "A_仅冷机"
    mode[(n_chwp > 0) & (n_ch == 0) & (n_hx_eff > 0)] = "B_仅板换"
    mode[(n_chwp > 0) & (n_ch > 0) & (n_hx_eff > 0)] = "C_冷机加板换"
    mode[n_chwp == 0] = "E_泵停"

    # ---------------- 站级表 ----------------
    plant = pd.DataFrame({
        "timestamp": ts,
        "mode": mode,
        "n_chiller_on": n_ch, "n_cwp_on": n_cwp, "n_chwp_on": n_chwp,
        "n_hx_effective": n_hx_eff,
        "F_cw_m3h": F_cw, "F_chw_m3h": F_chw,
        "T_cw_cold_C": T_cw_cold, "T_cw_hot_C": T_cw_hot,
        "T_chw_supply_C": T_chw_sup, "T_chw_return_C": T_chw_ret,
        "T_chiller_out_solved_C": T_ch_out,
        "P_chiller_total_kW": P_total,
        "Q_chw_total_kW": Q_total_chw,
        "Q_hx_total_kW": Q_hx,
        "Q_chiller_pool_kW": Q_ch_pool,
        "hx_share": (Q_hx / Q_total_chw).replace([np.inf, -np.inf], np.nan),
        "closure_check": ((Q_hx + Q_ch_pool) / Q_total_chw).replace(
            [np.inf, -np.inf], np.nan),
    })
    for i in IDX:
        plant[f"f_cw_branch_{i}_m3h"] = f_i[i]
        plant[f"f_chw_branch_{i}_m3h"] = f_chw_i[i]
        plant[f"T_mid_{i}_C"] = T_mid_i[i]
        plant[f"Q_hx_{i}_kW"] = Q_hx_i[i]
    plant.to_csv(args.out_dir / "processed_plant.csv", index=False,
                 encoding="utf-8-sig", lineterminator="\n")

    # ---------------- 逐台表（长格式） ----------------
    frames = []
    for i in IDX:
        sel = run_i[i] & Q_ch_i[i].notna()
        frames.append(pd.DataFrame({
            "timestamp": ts[sel],
            "chiller_id": f"chiller_0{i}",
            "mode": mode[sel],
            "n_chiller_on": n_ch[sel],
            "cwp_frequency_Hz": freq_i[i][sel],
            "chwp_frequency_Hz": freq_chw_i[i][sel],
            "f_cw_branch_m3h": f_i[i][sel],
            "f_chw_branch_m3h": f_chw_i[i][sel],
            "flow_ratio_chw_cw": (f_chw_i[i] / f_i[i].replace(0, np.nan))[sel],
            "T_cw_cold_C": T_cw_cold[sel],
            "T_mid_C": T_mid_i[i][sel],
            "T_chiller_out_C": T_ch_out[sel],
            "T_chw_supply_C": T_chw_sup[sel],
            "T_chw_return_C": T_chw_ret[sel],
            "lift_K": lift_i[i][sel],
            "P_chiller_kW": P_i[i][sel],
            "Q_hx_branch_kW": Q_hx_i[i][sel],
            "Q_reject_kW": Q_rej_i[i][sel],
            "Q_chiller_kW": Q_ch_i[i][sel],
            "PLR": Q_ch_i[i][sel] / RATED_KW,
            "COP": cop_i[i][sel],
            "eta_carnot": eta_i[i][sel],
            "eta_ok": eta_i[i][sel].between(ETA_LO, ETA_HI),
            "ambient_t_C": df["environment_parameters.ambient_t"][sel],
            "ambient_h_pct": df["environment_parameters.ambient_h"][sel],
        }))
    chiller = pd.concat(frames).sort_values(["timestamp", "chiller_id"])
    chiller.to_csv(args.out_dir / "processed_chiller.csv", index=False,
                   encoding="utf-8-sig", lineterminator="\n")

    print(f"站级表  {args.out_dir / 'processed_plant.csv'}  {plant.shape}")
    print(f"逐台表  {args.out_dir / 'processed_chiller.csv'}  {chiller.shape}")
    print(f"  其中 η 合格: {int(chiller['eta_ok'].sum())} 行")
    print("\n逐台表按冷机汇总（仅 η 合格行）:")
    ok = chiller[chiller["eta_ok"]]
    print(ok.groupby("chiller_id").agg(
        n=("Q_chiller_kW", "size"), Q中位=("Q_chiller_kW", "median"),
        PLR中位=("PLR", "median"), COP中位=("COP", "median"),
        η中位=("eta_carnot", "median")).to_string(
        float_format=lambda v: f"{v:9.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
