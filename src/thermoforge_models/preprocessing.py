"""模型模板共用的预处理（implementation-notes §8.1）。

- **特征顺序显式存储为列表**，不依赖 dict/DataFrame 列顺序（§8.1）。
- scaler 只在训练集上 fit，并随模型一起序列化（§4.4 泄漏来源表）。
- 缺失值用训练集均值填充（填充值属于 scaler 状态，随模型序列化）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureScaler:
    """标准化器：显式特征顺序 + 训练集均值/标准差。"""

    feature_order: tuple[str, ...]
    means: dict[str, float]
    stds: dict[str, float]  # 训练集标准差，0 时记为 1.0（常数特征）

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_order: Sequence[str]) -> "FeatureScaler":
        if not feature_order:
            raise ValueError("feature_order 不能为空")
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for name in feature_order:
            if name not in df.columns:
                raise ValueError(f"训练数据缺少特征列: {name!r}")
            col = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=np.float64)
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                raise ValueError(f"特征列全为缺失: {name!r}")
            mean = float(np.mean(valid))
            std = float(np.std(valid))  # 总体标准差（ddof=0），确定性
            means[name] = mean
            stds[name] = std if std > 0.0 else 1.0
        return cls(tuple(feature_order), means, stds)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """按显式特征顺序输出 (n, k) 矩阵；NaN 用训练集均值填充。"""
        n = len(df)
        out = np.empty((n, len(self.feature_order)), dtype=np.float64)
        for j, name in enumerate(self.feature_order):
            if name not in df.columns:
                raise ValueError(f"输入缺少特征列: {name!r}")
            col = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=np.float64)
            col = np.where(np.isnan(col), self.means[name], col)
            out[:, j] = (col - self.means[name]) / self.stds[name]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_order": list(self.feature_order),
            "means": dict(self.means),
            "stds": dict(self.stds),
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "FeatureScaler":
        return cls(
            tuple(str(f) for f in doc["feature_order"]),
            {str(k): float(v) for k, v in doc["means"].items()},
            {str(k): float(v) for k, v in doc["stds"].items()},
        )
