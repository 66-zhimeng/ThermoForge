"""复算 docs/data-processing-handbook.md 中的全部数值。

章节号与手册一一对应。任何一项对不上，说明数据版本已变，应先更新手册
再继续建模。

拓扑（A-04）：每条支路 = 冷却水泵 → 板换 → 冷机（串联），支路之间并联。
`chiller_0i.condenser_return_t` 是支路的冷却侧中间温度，据此把冷量拆开。

用法::

    python scripts/analyze_chiller_data.py [--ref WX_2025_HVAC@rev_0001]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CP = 4.187                    # kJ/(kg·K)，A-11
ETA_LO, ETA_HI = 0.03, 0.45   # 卡诺效率实用过滤带，A-06
RATED_KW = 9672.0             # 单台额定冷量
IDX = range(1, 5)
PLR_BINS = [0, .1, .2, .3, .4, .5, .55, .6, .65, .7, 1.2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", default="WX_2025_HVAC@rev_0001")
    p.add_argument("--vault-root", type=Path, default=Path("vault"))
    return p.parse_args(argv)


def load(ref: str, vault_root: Path) -> pd.DataFrame:
    dataset_id, revision = ref.split("@")
    return pd.read_parquet(vault_root / "datasets" / dataset_id / revision
                           / "canonical" / "data.parquet")


def running(df: pd.DataFrame, prefix: str) -> pd.Series:
    return sum((df[f"{prefix}_0{i}.status_run"] == True).astype(int)  # noqa: E712
               for i in IDX)


def equal_rate(df: pd.DataFrame, a: str, b: str) -> float:
    both = df[a].notna() & df[b].notna()
    if not both.any():
        return float("nan")
    return float(((df[a][both] - df[b][both]).abs() < 1e-9).mean())


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load(args.ref, args.vault_root)
    print(f"{args.ref}  shape={df.shape}")

    n_ch, n_cwp, n_chwp = (running(df, "chiller"), running(df, "cwp"),
                           running(df, "chwp"))
    n_hx_eff = (n_cwp - n_ch).clip(lower=0)                      # A-05

    F_chw = df["chw_A1.f"].fillna(0) + df["chw_A2.f"].fillna(0)
    F_cw = df["cw_A1.f"] + df["cw_A2.f"]                         # A-02
    T_cold, T_hot = df["cw_A1.t_return"], df["cw_A1.t_supply"]
    T_evap = df["chw_A1.t_supply"]
    Q_total = (F_chw / 3.6 * CP
               * (df["chw_A1.t_return"] - T_evap))               # §5.1
    P_ch = sum(df[f"chiller_0{i}.power"].fillna(0) for i in IDX)

    # ---- §5.2 按支路在冷却侧拆分（频率加权流量） ----------------------
    pump_on = {i: (df[f"cwp_0{i}.status_run"] == True) for i in IDX}   # noqa: E712
    ch_on = {i: (df[f"chiller_0{i}.status_run"] == True) for i in IDX}  # noqa: E712
    T_cond_i = {i: df[f"chiller_0{i}.condenser_return_t"] for i in IDX}
    P_i = {i: df[f"chiller_0{i}.power"].fillna(0) for i in IDX}

    freq = {i: df[f"cwp_0{i}.status_frequency"].where(pump_on[i], 0.0).fillna(0.0)
            for i in IDX}                                        # A-05 频率加权
    w_sum = sum(freq.values()).replace(0, np.nan)
    f_i = {i: F_cw * freq[i] / w_sum for i in IDX}

    Q_hx = pd.Series(0.0, index=df.index)
    for i in IDX:
        rise = (T_cond_i[i] - T_cold).where(pump_on[i], 0.0).clip(lower=0)
        Q_hx = Q_hx + (f_i[i] * CP * rise / 3.6).fillna(0.0)
    Q_reject = F_cw / 3.6 * CP * (T_hot - T_cold)
    Q_pool = (Q_reject - Q_hx - P_ch).clip(lower=0)              # 减去压缩功

    # ---- §5.3 反解公共冷凝出水温 → 逐台冷机负荷（A-12） ---------------
    num, den = F_cw * T_hot, pd.Series(0.0, index=df.index)
    for i in IDX:
        hx_only = pump_on[i] & ~ch_on[i]        # 泵在跑、冷机停：出口 = 中间温度
        num = num - (f_i[i] * T_cond_i[i]).where(hx_only, 0.0).fillna(0.0)
        den = den + f_i[i].where(pump_on[i] & ch_on[i], 0.0).fillna(0.0)
    T_ch_out = num / den.replace(0, np.nan)

    Q_single, eta_single, plr_single = {}, {}, {}
    for i in IDX:
        on = pump_on[i] & ch_on[i]
        q = ((f_i[i] * CP * (T_ch_out - T_cond_i[i]) / 3.6) - P_i[i]).where(on)
        lift_i = (T_cond_i[i] - T_evap).where(on)
        Q_single[i] = q.clip(lower=0)
        eta_single[i] = ((q / P_i[i].replace(0, np.nan))
                         / ((T_evap + 273.15) / lift_i.where(lift_i > 0.5)))
        plr_single[i] = q / RATED_KW

    T_cond = pd.concat(
        [df[f"chiller_0{i}.condenser_return_t"].where(
            df[f"chiller_0{i}.status_run"] == True) for i in IDX],  # noqa: E712
        axis=1).mean(axis=1)
    lift = T_cond - T_evap
    cop_carnot = (T_evap + 273.15) / lift.where(lift > 0.5)
    cop = Q_pool / P_ch.replace(0, np.nan)
    eta = cop / cop_carnot
    plr = Q_pool / n_ch.replace(0, np.nan) / RATED_KW

    base = Q_total.notna() & (Q_total > 0) & (P_ch > 0)
    A = base & (n_chwp > 0) & (n_ch > 0) & (n_hx_eff == 0)
    C = base & (n_chwp > 0) & (n_ch > 0) & (n_hx_eff > 0)

    # ---------------------------------------------------------------- §2
    section("§2 测点归属")
    for i in IDX:
        run = float((df[f"chiller_0{i}.status_run"] == True).mean())  # noqa: E712
        print(f"  chiller_0{i}: evaporator_supply_t 与总管供水相等 "
              f"{equal_rate(df, f'chiller_0{i}.evaporator_supply_t', 'chw_A1.t_supply'):6.1%}"
              f"   运行占比 {run:6.1%}")
    print("  hx_0i.cw_return_t 是否被填成总管热侧（错列，应停用）")
    for i in IDX:
        print(f"    hx_0{i}: 与 cw_A1.t_supply 相等 "
              f"{equal_rate(df, f'hx_0{i}.cw_return_t', 'cw_A1.t_supply'):6.1%}")
    print("  condenser_return_t 在冷机停机时是否仍有记录（支路口径前提）")
    for i in IDX:
        s = df[f"chiller_0{i}.condenser_return_t"]
        off = df[f"chiller_0{i}.status_run"] != True             # noqa: E712
        print(f"    chiller_0{i}: 停机时非空 {int((s.notna() & off).sum()):>6}")

    # ---------------------------------------------------------------- §3
    section("§3 冷却水侧是推算还是实测")
    err = (T_cold + (Q_total + P_ch) / (F_cw / 3.6 * CP) - T_hot).abs()
    for name, sel in [("A 仅冷机", A), ("C 冷机+板换", C)]:
        e = err[sel & err.notna()]
        print(f"  [{name:<12}] |误差| 中位数 {e.median():.6f} K   P95 {e.quantile(.95):.6f} K")
    print("  → 1e-3 K 量级 = 由热平衡算出来的，不是测出来的")
    print("\n  §3.2 冷却水总流量口径（闭合比应为 1.000）")
    for label, f_tot in [("cw_A1.f 单路", df["cw_A1.f"]), ("cw_A1.f + cw_A2.f", F_cw)]:
        ratio = (f_tot / 3.6 * CP * (T_hot - T_cold))[A] / (Q_total[A] + P_ch[A])
        print(f"    {label:<20} {ratio.median():6.3f}")

    # ---------------------------------------------------------------- §4
    section("§4 工况划分（泵台数口径）")
    print(f"  n_cwp < n_ch 的点: {int((n_cwp - n_ch < 0).sum())}（应为 0）")
    print(f"  n_cwp − n_ch == n_hx 一致率: "
          f"{float((n_hx_eff == running(df, 'hx')).mean()):6.1%}")
    mode = pd.Series("其它", index=df.index)
    mode[(n_chwp > 0) & (n_ch > 0) & (n_hx_eff == 0)] = "A 仅冷机"
    mode[(n_chwp > 0) & (n_ch == 0) & (n_hx_eff > 0)] = "B 仅板换"
    mode[(n_chwp > 0) & (n_ch > 0) & (n_hx_eff > 0)] = "C 冷机+板换"
    mode[n_chwp == 0] = "E 泵停"
    for key in ["A 仅冷机", "B 仅板换", "C 冷机+板换", "E 泵停"]:
        cnt = int((mode == key).sum())
        print(f"  {key:<14} {cnt:>6} 点  {cnt/len(df):6.1%}")

    print("\n  §4.3 表观 COP 不能作判据（按 COP 分组看 η）")
    cop_app = Q_total / P_ch.replace(0, np.nan)
    for label, sel in [("COP>11 ", A & (cop_app > 11)), ("COP<=11", A & (cop_app <= 11))]:
        e = eta[sel].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    [{label}] n={int(sel.sum()):>5}  lift 中位 {lift[sel].median():5.2f} K"
              f"  COP 中位 {cop_app[sel].median():6.2f}  η 中位 {e.median():6.3f}")

    # ---------------------------------------------------------------- §5
    section("§5 冷量拆分（支路口径）")
    print(f"{'工况':<6}{'n':>7}{'Q_total':>10}{'Q_hx':>10}{'Q_chiller':>11}{'板换占比':>10}")
    for name, sel in [("A", A), ("C", C)]:
        print(f"{name:<6}{int(sel.sum()):>7}{Q_total[sel].median():>10.0f}"
              f"{Q_hx[sel].median():>10.0f}{Q_pool[sel].median():>11.0f}"
              f"{(Q_hx/Q_total)[sel].median():>10.1%}")
    print("  → A 模式板换占比应接近 0（板换闲置），这是拆分自洽的直接证据")

    print("\n  §5.4 校验一：Q_hx + Q_chiller 应 = Q_total")
    r_all = (Q_hx + Q_pool) / Q_total
    for name, sel in [("A", A), ("C", C)]:
        r = r_all[sel].dropna()
        print(f"    [{name}] 比值 中位 {r.median():7.4f}"
              f"  IQR [{r.quantile(.25):.4f}, {r.quantile(.75):.4f}]")

    print("\n  §5.4 校验三：同负荷率格内 A 与 C 的 η")
    both = pd.DataFrame({"plr": plr, "eta": eta,
                         "mode": np.where(A, "A", np.where(C, "C", None))})
    both = both[both["mode"].notna() & both["plr"].notna()
                & both["eta"].notna() & np.isfinite(both["eta"])]
    both["bin"] = pd.cut(both["plr"], PLR_BINS)
    piv = both.pivot_table(index="bin", columns="mode", values="eta",
                           aggfunc="median", observed=True)
    cnt = both.pivot_table(index="bin", columns="mode", values="eta",
                           aggfunc="size", observed=True)
    out = piv.join(cnt, rsuffix="_n")
    if {"A", "C"} <= set(piv.columns):
        out["C/A"] = out["C"] / out["A"]
    print(out.to_string(float_format=lambda v: f"{v:8.3f}"))
    print("    → 样本充足的格里 C/A 应接近 1；不要比较不分层的整体中位数")

    print("\n  §5.5 部分负荷特性（A∪C 合并）")
    d = pd.DataFrame({"plr": plr, "eta": eta, "cop": cop, "lift": lift})[A | C]
    d = d[np.isfinite(d["eta"]) & d["plr"].notna()]
    g = d.groupby(pd.cut(d["plr"], PLR_BINS), observed=True)
    print(g.agg(n=("eta", "size"), 负荷率=("plr", "median"), lift=("lift", "median"),
                COP=("cop", "median"), η=("eta", "median")).to_string(
        float_format=lambda v: f"{v:8.3f}"))
    print(f"    η>1 的点: {int((eta[A | C] > 1).sum())}（池级口径，硬门禁）")

    section("§5.3 逐台冷机负荷（反解公共冷凝出水温）")
    tc_mean = pd.concat([T_cond_i[i].where(pump_on[i] & ch_on[i]) for i in IDX],
                        axis=1).mean(axis=1)
    for name, sel in [("A", A), ("C", C)]:
        s = sel & T_ch_out.notna()
        print(f"  [{name}] T_ch_out {T_ch_out[s].median():6.2f}"
              f"  支路中间温 {tc_mean[s].median():6.2f}"
              f"  总管热侧 {T_hot[s].median():6.2f}"
              f"  冷机段温升 {(T_ch_out - tc_mean)[s].median():5.2f} K"
              f"  <0 占比 {((T_ch_out - tc_mean)[s] < 0).mean():6.1%}")
    print("  → C 模式 T_ch_out 应高于总管热侧（板换支路出口更冷，混合后被拉低）")

    print(f"\n{'冷机':<12}{'n':>7}{'Q中位':>10}{'负荷率':>9}{'COP中位':>9}{'η中位':>8}{'η IQR':>18}")
    ok_single = {}
    for i in IDX:
        on = (A | C) & Q_single[i].notna()
        ok_single[i] = on & eta_single[i].between(ETA_LO, ETA_HI)
        e = eta_single[i][on].replace([np.inf, -np.inf], np.nan).dropna()
        c = (Q_single[i] / P_i[i].replace(0, np.nan))[on].dropna()
        print(f"chiller_0{i}  {int(on.sum()):>7}{Q_single[i][on].median():>10.0f}"
              f"{plr_single[i][on].median():>9.3f}{c.median():>9.2f}{e.median():>8.3f}"
              f"   [{e.quantile(.25):.3f}, {e.quantile(.75):.3f}]")

    Q_sum = sum(q.fillna(0.0) for q in Q_single.values())
    print("\n  校验：Σ逐台 / 池级，以及 (Q_hx + Σ逐台) / Q_total")
    for name, sel in [("A", A), ("C", C)]:
        r1 = (Q_sum / Q_pool.replace(0, np.nan))[sel].dropna()
        r2 = ((Q_hx + Q_sum) / Q_total)[sel].dropna()
        print(f"    [{name}] Σ逐台/池级 {r1.median():7.4f}"
              f"   闭合 {r2.median():7.4f}")
    print(f"  η>1 的点合计: {sum(int((eta_single[i] > 1).sum()) for i in IDX)}")

    qmat = pd.concat([Q_single[i] for i in IDX], axis=1)
    multi = (A | C) & (qmat.notna().sum(axis=1) >= 2)
    rel = ((qmat.max(axis=1) - qmat.min(axis=1)) / qmat.mean(axis=1))[multi]
    print(f"\n  台间负荷相对极差（n={int(multi.sum())}）: 中位 {rel.median():6.1%}"
          f"  P75 {rel.quantile(.75):6.1%}  P95 {rel.quantile(.95):6.1%}")
    print("  → 显著大于 0 说明各台状态并不相同，A-03 均摊会抹平真实差异")

    print("\n  逐台训练样本量（η 合格）")
    total_single = 0
    for i in IDX:
        n = int(ok_single[i].sum())
        total_single += n
        print(f"    chiller_0{i}: {n:>6} 行")
    print(f"    合计 {total_single} 行（逐台建模可用样本）")

    # ---------------------------------------------------------------- §7
    section("§7 训练样本构造")
    ok = eta.between(ETA_LO, ETA_HI)
    for label, sel in [("0 全年", pd.Series(True, index=df.index)),
                       ("1 泵开 ∧ 有冷机运行 ∧ 有效", base & (n_chwp > 0) & (n_ch > 0)),
                       ("2a ∧ A 模式", A), ("2b ∧ C 模式", C),
                       (f"3 ∧ η∈[{ETA_LO},{ETA_HI}]", (A | C) & ok)]:
        print(f"  {label:<26} {int(sel.sum()):>6} 点  {sel.mean():6.1%}")
    clean = (A | C) & ok
    print(f"\n  台数分布 A∪C : {n_ch[clean].value_counts().sort_index().to_dict()}")
    print(f"  台数分布 仅 A : {n_ch[A & ok].value_counts().sort_index().to_dict()}")
    print("\n  工况覆盖（A∪C）")
    for label, s in [("lift (K)", lift), ("单机负荷率", plr),
                     ("环境温度 (°C)", df["environment_parameters.ambient_t"])]:
        v = s[clean].dropna()
        print(f"    {label:<14} P5 {v.quantile(.05):6.3f}  P25 {v.quantile(.25):6.3f}"
              f"  中位 {v.median():6.3f}  P75 {v.quantile(.75):6.3f}"
              f"  P95 {v.quantile(.95):6.3f}")
    print(f"    负荷率 < 0.3 的样本: {int((plr[clean] < 0.3).sum())} 点")

    solo = {f"chiller_0{i}": int(
        (A & (n_ch == 1) & (df[f"chiller_0{i}.status_run"] == True)).sum())  # noqa: E712
        for i in IDX}
    print(f"\n  A 模式『全站仅此一台运行』点数: {solo}")
    print("  → 全为 0：单机冷量无法由冷冻侧直接归属；改由冷却侧支路口径解出（§5.3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
