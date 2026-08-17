"""残差混合模型（research-loop.md §4 混合建模方式 1、DD-07）。

`Y = Y_physics + XGBoost(残差)`：物理主干预测主体，梯度提升拟合残差。

- XGBoost 用**原生 .json 格式**保存（implementation-notes §8.1）。
- 固定 `seed` 与 `nthread`（§7.2：固定线程数是复现的必要条件）。
- 支持 `monotone_constraints`（§6.3：优先训练时强制约束而非事后检查）。
- DD-07 强制配套：`physics_only_report()` 给出物理主干单独指标输入，
  以及残差项量级占比，防止残差项吸收物理主干的系统性错误。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from .identification import (
    GN_FORMAT,
    NTU_FORMAT,
    EffectivenessNTU,
    GordonNgChiller,
)
from .physics import (
    MODEL_FORMAT as PHYSICS_FORMAT,
    MODEL_FORMAT_V2 as PHYSICS_FORMAT_V2,
    ChillerPhysicsModel,
    load_physics_model,
)
from .preprocessing import FeatureScaler

MODEL_FORMAT = "thermoforge.residual_hybrid.v1"

#: 能量平衡族（v1/v2）主干格式：参数真源是 params.yaml，load 走
#: load_physics_model；系统辨识族（GN/NTU）的 save 只写 model.json，
#: 会被 hybrid 的 meta 覆盖，须单独落 base_model.json（save/load 见下）。
_PHYSICS_FAMILY_FORMATS = (PHYSICS_FORMAT, PHYSICS_FORMAT_V2)

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "exact",  # 确定性优先：避免近似算法的分裂点不确定性
}


class ResidualHybrid:
    """残差混合：物理/辨识主干 + XGBoost(残差)。

    主干可以是能量平衡族（ChillerPhysicsModel / ChillerPhysicsV2）或
    系统辨识族（GordonNgChiller / EffectivenessNTU）——两者都提供
    fit/predict，序列化路径不同（save/load 注释）。
    """

    def __init__(
        self,
        physics: ChillerPhysicsModel | GordonNgChiller | EffectivenessNTU,
        *,
        seed: int,
        nthread: int = 1,
        xgb_params: Mapping[str, Any] | None = None,
        monotone_constraints: Mapping[str, int] | None = None,
    ):
        self.physics = physics
        self.seed = int(seed)
        self.nthread = int(nthread)
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **dict(xgb_params or {})}
        self.monotone_constraints = {
            str(k): int(v) for k, v in (monotone_constraints or {}).items()
        }
        self.scaler: FeatureScaler | None = None
        self._booster: xgb.Booster | None = None

    @property
    def feature_order(self) -> list[str]:
        if self.scaler is None:
            raise RuntimeError("模型尚未 fit")
        return list(self.scaler.feature_order)

    def _monotone_tuple(self) -> tuple[int, ...]:
        order = self.feature_order
        unknown = set(self.monotone_constraints) - set(order)
        if unknown:
            raise ValueError(f"monotone_constraints 引用了未知特征: {sorted(unknown)}")
        return tuple(self.monotone_constraints.get(name, 0) for name in order)

    def fit(
        self,
        df: pd.DataFrame,
        y: Sequence[float],
        feature_order: Sequence[str] | None = None,
    ) -> "ResidualHybrid":
        y_arr = np.asarray(y, dtype=np.float64)
        self.physics.fit(df, y_arr)
        residual = y_arr - self.physics.predict(df)

        order = list(feature_order) if feature_order else sorted(df.columns)
        self.scaler = FeatureScaler.fit(df, order)
        X = self.scaler.transform(df)

        params = dict(self.xgb_params)
        n_estimators = int(params.pop("n_estimators"))
        params.pop("monotone_constraints", None)  # 统一由构造参数提供
        dtrain = xgb.DMatrix(X, label=residual, feature_names=self.feature_order)
        self._booster = xgb.train(
            params={
                "objective": "reg:squarederror",
                "seed": self.seed,
                "nthread": self.nthread,
                "monotone_constraints": self._monotone_tuple(),
                **params,
            },
            dtrain=dtrain,
            num_boost_round=n_estimators,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self._booster is None:
            raise RuntimeError("模型尚未 fit")
        X = self.scaler.transform(df)
        dm = xgb.DMatrix(X, feature_names=self.feature_order)
        return self.physics.predict(df) + self._booster.predict(dm)

    def physics_only(self, df: pd.DataFrame) -> np.ndarray:
        """物理主干单独的预测（DD-07 配套：必须同时报告）。"""
        return self.physics.predict(df)

    def residual_share(self, df: pd.DataFrame) -> float:
        """残差项量级占比：RMS(残差) / RMS(总预测)（DD-07 配套）。"""
        if self.scaler is None or self._booster is None:
            raise RuntimeError("模型尚未 fit")
        X = self.scaler.transform(df)
        dm = xgb.DMatrix(X, feature_names=self.feature_order)
        res = self._booster.predict(dm)
        total = self.predict(df)
        denom = float(np.sqrt(np.mean(total**2)))
        return float(np.sqrt(np.mean(res**2)) / denom) if denom > 0 else 0.0

    # ---------------------------------------------------------------- 序列化

    def save(self, directory: str | Path) -> Path:
        """写 `booster.json`（XGBoost 原生）+ 主干参数 + `model.json`。

        主干落盘按族分派：能量平衡族（v1/v2）写 `params.yaml`（其
        model.json 会被下面的 hybrid meta 覆盖，真源是 YAML）；系统辨识族
        （GN/NTU）的 save 只写 model.json，必须单独落 `base_model.json`，
        否则主干参数被覆盖丢失、load 无从重建。
        """
        if self.scaler is None or self._booster is None:
            raise RuntimeError("模型尚未 fit")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        booster_path = directory / "booster.json"
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        os.close(fd)
        self._booster.save_model(tmp)  # 原生 JSON 格式（§8.1）
        os.replace(tmp, booster_path)

        if isinstance(self.physics, ChillerPhysicsModel):
            base_format = self.physics.params_dict()["format"]
            self.physics.save(directory)  # params.yaml（YAML 明文参数）
        else:
            base_doc = self.physics.to_dict()
            base_format = base_doc["format"]
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_",
                                       suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
                json.dump(base_doc, fp, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False)
                fp.write("\n")
            os.replace(tmp, directory / "base_model.json")

        meta = {
            "format": MODEL_FORMAT,
            "base_format": base_format,
            "seed": self.seed,
            "nthread": self.nthread,
            "xgb_params": dict(self.xgb_params),
            "monotone_constraints": dict(self.monotone_constraints),
            "scaler": self.scaler.to_dict(),
        }
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(meta, fp, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
            fp.write("\n")
        os.replace(tmp, directory / "model.json")
        return booster_path

    @classmethod
    def load(cls, directory: str | Path) -> "ResidualHybrid":
        directory = Path(directory)
        with open(directory / "model.json", encoding="utf-8") as fp:
            meta = json.load(fp)
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"未知模型格式: {meta.get('format')!r}")
        base_format = meta.get("base_format")
        if base_format is None or base_format in _PHYSICS_FAMILY_FORMATS:
            # 旧包没有 base_format 字段；能量平衡族的真源是 params.yaml
            base = load_physics_model(directory)
        else:
            base_path = directory / "base_model.json"
            if not base_path.exists():
                raise ValueError(f"混合模型缺少主干参数文件: {base_path}")
            with open(base_path, encoding="utf-8") as fp:
                base_doc = json.load(fp)
            if base_format == GN_FORMAT:
                base = GordonNgChiller.from_dict(base_doc)
            elif base_format == NTU_FORMAT:
                base = EffectivenessNTU.from_dict(base_doc)
            else:
                raise ValueError(f"未知主干模型格式: {base_format!r}")
        model = cls(
            physics=base,
            seed=int(meta["seed"]),
            nthread=int(meta["nthread"]),
            xgb_params=meta["xgb_params"],
            monotone_constraints=meta["monotone_constraints"],
        )
        model.scaler = FeatureScaler.from_dict(meta["scaler"])
        booster = xgb.Booster()
        booster.load_model(str(directory / "booster.json"))
        model._booster = booster
        return model
