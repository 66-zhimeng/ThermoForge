"""切分子集的分布画像：train/validate/test 各段的自变量、目标分布对比。

口径与 `thermoforge_data.profile`（整份数据集的画像）保持一致：分位数点位
固定 min/p01/p25/p50/p75/p99/max（implementation-notes §11），只是对象从
「整个数据版本」换成「一次实验切出来的子集」——用来回答"训练集见过的
工况范围，验证/测试集是不是超出去了"。

本模块只算数字，不下结论：是否存在需要关注的工况断层、要不要因此调整
实验，留给读这份制品的人或 Agent 判断。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# 分位数点位固定，与 thermoforge_data.profile.QUANTILE_POINTS 同调
QUANTILE_POINTS = (0.01, 0.25, 0.5, 0.75, 0.99)


def _quantile(sorted_vals: np.ndarray, q: float) -> float:
    if not len(sorted_vals):
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _column_distribution(values: np.ndarray) -> dict[str, Any]:
    """单列的分位数画像：count/missing/min/max/mean/std + 五个分位点。

    传入空数组（子集为空,或该列全为缺失）时只给 count/missing,不编造
    min/max —— 下游按 `"min" in entry` 判断有没有可用的分布。
    """
    n = len(values)
    finite = values[~np.isnan(values)]
    if not len(finite):
        return {"count": 0, "missing": int(n)}
    sorted_vals = np.sort(finite)
    return {
        "count": int(len(finite)),
        "missing": int(n - len(finite)),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "quantiles": {
            f"p{int(q * 100):02d}": _quantile(sorted_vals, q)
            for q in QUANTILE_POINTS
        },
    }


def build_split_profile(
    subsets: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    *,
    reference: str = "train",
) -> dict[str, Any]:
    """各子集逐列分布 + 相对 `reference` 子集的越界样本占比。

    `out_of_<reference>_range_fraction`：该子集里落在 `reference` 子集
    [min, max] 之外的样本比例。数字本身不代表"模型会失准"——train 覆盖窄
    也可能只是数据采集期短，但这是发现"验证/测试集比训练集见过更宽/更
    窄工况范围"的最直接信号。
    """
    profiles: dict[str, dict[str, Any]] = {}
    for name, sub in subsets.items():
        profiles[name] = {
            "n_samples": int(len(sub)),
            "variables": {
                col: _column_distribution(sub[col].to_numpy(dtype=float))
                for col in columns if col in sub.columns
            },
        }

    ref_variables = profiles.get(reference, {}).get("variables", {})
    for name, doc in profiles.items():
        if name == reference:
            continue
        sub = subsets.get(name)
        if sub is None:
            continue
        for col, entry in doc["variables"].items():
            ref_entry = ref_variables.get(col)
            if not ref_entry or "min" not in ref_entry or col not in sub.columns:
                continue
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if not len(vals):
                continue
            lo, hi = ref_entry["min"], ref_entry["max"]
            entry[f"out_of_{reference}_range_fraction"] = float(
                np.mean((vals < lo) | (vals > hi))
            )
    return {"reference": reference, "subsets": profiles}
