"""线性基线模型（research-loop.md §3 策略 1：可解释的朴素基线）。

sklearn LinearRegression / Ridge。交付格式为**系数 JSON**，
不用 pickle（implementation-notes §8.1：跨版本不可加载 + 任意代码执行）。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from .preprocessing import FeatureScaler

MODEL_FORMAT = "thermoforge.linear_baseline.v1"


class LinearBaseline:
    """线性基线：scaler（训练集 fit）+ 线性回归。

    用法::

        model = LinearBaseline(method="ridge", alpha=1.0)
        model.fit(train_df, y_train, feature_order=["f", "t_supply"])
        y_hat = model.predict(test_df)
        model.save(dir); loaded = LinearBaseline.load(dir)
    """

    def __init__(self, method: str = "linear", alpha: float = 1.0):
        if method not in ("linear", "ridge"):
            raise ValueError(f"未知 method: {method!r}（允许 linear/ridge）")
        self.method = method
        self.alpha = float(alpha)
        self.scaler: FeatureScaler | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0

    @property
    def feature_order(self) -> list[str]:
        if self.scaler is None:
            raise RuntimeError("模型尚未 fit")
        return list(self.scaler.feature_order)

    def fit(
        self,
        df: pd.DataFrame,
        y: Sequence[float],
        feature_order: Sequence[str] | None = None,
    ) -> "LinearBaseline":
        order = list(feature_order) if feature_order else sorted(df.columns)
        self.scaler = FeatureScaler.fit(df, order)
        X = self.scaler.transform(df)
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) != len(df):
            raise ValueError("y 与 df 行数不一致")
        est = (
            Ridge(alpha=self.alpha)
            if self.method == "ridge"
            else LinearRegression()
        )
        est.fit(X, y_arr)
        self._coef = np.asarray(est.coef_, dtype=np.float64)
        self._intercept = float(est.intercept_)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self._coef is None:
            raise RuntimeError("模型尚未 fit")
        X = self.scaler.transform(df)
        return X @ self._coef + self._intercept

    # ---------------------------------------------------------------- 序列化

    def to_dict(self) -> dict[str, Any]:
        if self.scaler is None or self._coef is None:
            raise RuntimeError("模型尚未 fit")
        return {
            "format": MODEL_FORMAT,
            "method": self.method,
            "alpha": self.alpha,
            "coefficients": {
                name: float(c)
                for name, c in zip(self.scaler.feature_order, self._coef)
            },
            "intercept": self._intercept,
            "scaler": self.scaler.to_dict(),
            "standardized": True,  # 系数作用于标准化特征
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "LinearBaseline":
        if doc.get("format") != MODEL_FORMAT:
            raise ValueError(f"未知模型格式: {doc.get('format')!r}")
        model = cls(method=str(doc["method"]), alpha=float(doc["alpha"]))
        model.scaler = FeatureScaler.from_dict(doc["scaler"])
        coefs = doc["coefficients"]
        model._coef = np.array(
            [float(coefs[name]) for name in model.scaler.feature_order],
            dtype=np.float64,
        )
        model._intercept = float(doc["intercept"])
        return model

    def save(self, directory: str | Path) -> Path:
        """写 `model.json`（临时文件 + os.replace，§10.1）。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.json"
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            fp.write("\n")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "LinearBaseline":
        with open(Path(directory) / "model.json", encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp))
