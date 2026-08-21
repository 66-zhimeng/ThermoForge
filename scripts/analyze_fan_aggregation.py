"""塔风机：逐台模型 vs 群模型的口径分解。

回答「为什么群总功耗的 CVRMSE 比逐台小这么多」。表面上是 2.43% vs 6.08%，
但那是**两个不同分母**量出来的数，不能直接比：CVRMSE = RMSE / mean(y)，
群的均值是 12 台之和，单台只有其 1/12 左右。

本脚本做一次判决性分解：把逐台模型的预测**按时刻求和**，在同一张判据面上
与群模型直接比。指标一律走 `research/metrics.py`（全仓唯一实现），不自己算。

    .venv/Scripts/python scripts/analyze_fan_aggregation.py

输出 JSON 到 research/tool_artifacts/analysis/，供账本发现引用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermoforge_research.metrics import cvrmse, nmbe, rmse

REPO = Path(__file__).resolve().parents[1]
UNIT_EXP = "EXP-0170"      # ct_fan_density_corrected_v1，逐台物理式
GROUP_EXP = "EXP-0092"     # gbdt_static，群总功耗黑箱
#: 功率分档：低频段单台只有几 kW，是 CVRMSE 的放大器
POWER_BANDS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 10**9)]


def _metrics_of(experiment_id: str) -> dict:
    path = REPO / "research" / "experiments" / experiment_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _surface_cvrmse(metrics: dict, surface: str) -> float | None:
    if surface == "rolling_cv":
        return ((metrics.get("rolling_cv") or {}).get("metrics") or {}).get("CVRMSE")
    block = ((metrics.get("surfaces") or {}).get(surface) or {}).get("metrics") or {}
    return block.get("CVRMSE")


def analyze() -> dict:
    preds = pd.read_parquet(
        REPO / "research" / "experiments" / UNIT_EXP / "predictions.parquet")
    unit_metrics, group_metrics = _metrics_of(UNIT_EXP), _metrics_of(GROUP_EXP)

    surfaces = {}
    for surface in sorted(preds["surface"].unique()):
        d = preds[preds["surface"] == surface]
        y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        # 同一时刻在跑的台全加 —— 这就是群总功耗的定义
        g = d.groupby("timestamp")[["y_true", "y_pred"]].sum()
        gy, gp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        counts = d.groupby("timestamp").size()

        cv_unit, cv_group = cvrmse(y, p), cvrmse(gy, gp)
        surfaces[surface] = {
            "n_rows": int(len(d)),
            "n_timestamps": int(len(g)),
            "units_per_timestamp": {
                "min": int(counts.min()), "median": float(counts.median()),
                "max": int(counts.max())},
            "unit_scale": {
                "mean_kw": float(y.mean()), "rmse_kw": rmse(y, p),
                "cvrmse": cv_unit, "nmbe": nmbe(y, p)},
            "group_scale": {
                "mean_kw": float(gy.mean()), "rmse_kw": rmse(gy, gp),
                "cvrmse": cv_group, "nmbe": nmbe(gy, gp)},
            "aggregation_gain": (cv_unit / cv_group
                                 if cv_unit and cv_group else None),
            "group_model_cvrmse": _surface_cvrmse(group_metrics, surface),
        }

    # 加总为什么没买到 sqrt(N)：逐台误差相关性
    wide = preds[preds["surface"] == "A"].pivot_table(
        index="timestamp", columns="object_id", values=["y_true", "y_pred"])
    err = (wide["y_pred"] - wide["y_true"]).dropna(axis=0, how="any")
    corr = err.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    n_units = int(preds["object_id"].nunique())

    # CVRMSE 随功率档位怎么变（分母效应的直接证据）
    d = preds[preds["surface"] == "A"]
    bands = []
    for lo, hi in POWER_BANDS:
        sel = d[(d["y_true"] >= lo) & (d["y_true"] < hi)]
        if len(sel) < 10:
            continue
        y, p = sel["y_true"].to_numpy(), sel["y_pred"].to_numpy()
        bands.append({"band_kw": f"{lo}-{hi}" if hi < 10**9 else f">{lo}",
                      "n": int(len(sel)), "mean_kw": float(y.mean()),
                      "rmse_kw": rmse(y, p), "cvrmse": cvrmse(y, p)})

    return {
        "question": "群总功耗的 CVRMSE 为何远小于逐台，是模型更好还是口径不同",
        "unit_experiment": UNIT_EXP,
        "group_experiment": GROUP_EXP,
        "n_units": n_units,
        "headline_note": (
            f"{UNIT_EXP} 滚动面 {_surface_cvrmse(unit_metrics, 'rolling_cv'):.4f} 与 "
            f"{GROUP_EXP} 滚动面 {_surface_cvrmse(group_metrics, 'rolling_cv'):.4f} "
            "分母不同（单台 vs 12 台之和），不可直接比较"),
        "surfaces": surfaces,
        "error_correlation": {
            "mean": float(off_diagonal.mean()),
            "median": float(np.median(off_diagonal)),
            "independent_gain_would_be": float(np.sqrt(n_units)),
            "note": ("误差若独立，加总后相对误差应降 sqrt(N)；实测远低于此，"
                     "说明逐台误差同向——与季节性密度偏差是全站共有量一致"),
        },
        "power_bands": bands,
        "caveat": (
            f"{GROUP_EXP} 与 {UNIT_EXP} 建在不同视图上，跨实验的面对面比较仅供"
            f"参考；**{UNIT_EXP} 内部的逐台→加总分解是精确的**，它才是结论依据。"),
    }


def main() -> int:
    result = analyze()
    out_dir = REPO / "research" / "tool_artifacts" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ct_fan_unit_vs_group.json"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
    tmp.replace(out)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
