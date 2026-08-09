"""物理模型模板（research-loop.md §4 物理模型路线）。

冷水机功率的半经验物理模型：

```text
Q   = m·Cp·ΔT            # 制冷量（kW），m 由体积流量与密度换算
PLR = Q / rated_capacity
COP = c0 + c1·T_chws + c2·T_cws + c3·PLR
P   = Q / COP
```

COP 参数由历史数据辨识（最小二乘，仅在 Q>0 且 P>0 的样本上），
参数范围、单位、方程版本明文记录，交付格式为 **YAML 明文参数**
（implementation-notes §8.1：可读、可审计、可人工复核）。

密度/比热取定值（implementation-notes §14 待决策 #4：定值还是温度相关
尚未定稿），所用取值写入参数文件，作为物理模型版本的一部分（§6.1）。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

EQUATION_VERSION = "cooling_balance_v1"
MODEL_FORMAT = "thermoforge.chiller_physics.v1"

# 定值物性（[草案]，implementation-notes §6.1）：温度相关模型待标定后替换
DEFAULT_RHO_KG_PER_M3 = 998.0  # 水密度，约 25 °C
DEFAULT_CP_KJ_PER_KG_K = 4.186  # 水比热

# COP 回归系数的合法范围（最小二乘结果裁剪到此范围并记录）
COP_COEF_BOUNDS = {
    "c0": (0.0, 20.0),       # 截距
    "c1": (-1.0, 1.0),       # 冷冻水供水温度（Cel）
    "c2": (-1.0, 1.0),       # 冷却水供水温度（Cel）
    "c3": (0.0, 10.0),       # 部分负荷率 PLR（1）
}

# 默认输入列名（property_code）；可通过 inputs 参数重映射
DEFAULT_INPUTS = {
    "chw_flow": "chw_flow",                # m3/h
    "chw_supply_temp": "chw_supply_temp",  # Cel
    "chw_return_temp": "chw_return_temp",  # Cel
    "cw_supply_temp": "cw_supply_temp",    # Cel
}


def _write_text_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_",
                               suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(text)
    os.replace(tmp, path)


class ChillerPhysicsModel:
    """Q = m·Cp·ΔT、P = Q/COP 物理模型，COP 参数可辨识。"""

    def __init__(
        self,
        rated_capacity_kw: float,
        rated_power_kw: float | None = None,
        *,
        inputs: Mapping[str, str] | None = None,
        rho_kg_per_m3: float = DEFAULT_RHO_KG_PER_M3,
        cp_kj_per_kg_k: float = DEFAULT_CP_KJ_PER_KG_K,
    ):
        if rated_capacity_kw <= 0:
            raise ValueError("rated_capacity_kw 必须为正")
        self.rated_capacity_kw = float(rated_capacity_kw)
        self.rated_power_kw = float(rated_power_kw) if rated_power_kw else None
        self.inputs = dict(DEFAULT_INPUTS if inputs is None else inputs)
        self.rho = float(rho_kg_per_m3)
        self.cp = float(cp_kj_per_kg_k)
        self.cop_coefs: dict[str, float] | None = None  # c0..c3
        self.identification: dict[str, Any] = {}  # 参数辨识记录

    # ---------------------------------------------------------------- 方程

    def cooling_capacity(self, df: pd.DataFrame) -> np.ndarray:
        """Q = m·Cp·ΔT（kW）。体积流量 m3/h → 质量流量 kg/s 经密度换算。"""
        flow = pd.to_numeric(df[self.inputs["chw_flow"]]).to_numpy(np.float64)
        t_s = pd.to_numeric(df[self.inputs["chw_supply_temp"]]).to_numpy(np.float64)
        t_r = pd.to_numeric(df[self.inputs["chw_return_temp"]]).to_numpy(np.float64)
        mass_flow = flow * self.rho / 3600.0  # kg/s
        return mass_flow * self.cp * (t_r - t_s)  # kJ/s = kW

    def cop(self, df: pd.DataFrame, q_kw: np.ndarray) -> np.ndarray:
        """COP = c0 + c1·T_chws + c2·T_cws + c3·PLR。"""
        if self.cop_coefs is None:
            raise RuntimeError("模型尚未 fit")
        t_s = pd.to_numeric(df[self.inputs["chw_supply_temp"]]).to_numpy(np.float64)
        t_cw = pd.to_numeric(df[self.inputs["cw_supply_temp"]]).to_numpy(np.float64)
        plr = q_kw / self.rated_capacity_kw
        c = self.cop_coefs
        return c["c0"] + c["c1"] * t_s + c["c2"] * t_cw + c["c3"] * plr

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P = Q / COP。"""
        q = self.cooling_capacity(df)
        cop = self.cop(df, q)
        return q / cop

    # ---------------------------------------------------------------- 辨识

    def fit(self, df: pd.DataFrame, y: Sequence[float]) -> "ChillerPhysicsModel":
        """最小二乘辨识 COP 系数（仅 Q>0 且 P>0 样本），结果裁剪到合法范围。"""
        q = self.cooling_capacity(df)
        p = np.asarray(y, dtype=np.float64)
        if len(p) != len(q):
            raise ValueError("y 与 df 行数不一致")
        mask = (q > 0) & (p > 0) & np.isfinite(q) & np.isfinite(p)
        n_used = int(mask.sum())
        if n_used < 4:
            raise ValueError(f"可用于 COP 辨识的样本不足: {n_used} < 4")
        cop_obs = q[mask] / p[mask]
        t_s = pd.to_numeric(df[self.inputs["chw_supply_temp"]]).to_numpy(np.float64)[mask]
        t_cw = pd.to_numeric(df[self.inputs["cw_supply_temp"]]).to_numpy(np.float64)[mask]
        plr = q[mask] / self.rated_capacity_kw
        A = np.column_stack([np.ones(n_used), t_s, t_cw, plr])
        sol, _, _, _ = np.linalg.lstsq(A, cop_obs, rcond=None)
        names = ("c0", "c1", "c2", "c3")
        coefs: dict[str, float] = {}
        clipped: list[str] = []
        for name, value in zip(names, sol):
            lo, hi = COP_COEF_BOUNDS[name]
            v = float(value)
            if not (lo <= v <= hi):
                clipped.append(name)
                v = min(max(v, lo), hi)
            coefs[name] = v
        self.cop_coefs = coefs

        cop_pred = A @ np.array([coefs[n] for n in names])
        ss_res = float(np.sum((cop_obs - cop_pred) ** 2))
        ss_tot = float(np.sum((cop_obs - np.mean(cop_obs)) ** 2))
        self.identification = {
            "method": "least_squares",
            "n_samples": n_used,
            "n_dropped_nonpositive": int((~mask).sum()),
            "cop_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
            "cop_observed_range": [float(np.min(cop_obs)), float(np.max(cop_obs))],
            "clipped_coefficients": clipped,
        }
        return self

    # ---------------------------------------------------------------- 序列化

    def params_dict(self) -> dict[str, Any]:
        """参数 YAML 文档：范围、单位、方程版本明文记录。"""
        if self.cop_coefs is None:
            raise RuntimeError("模型尚未 fit")
        return {
            "format": MODEL_FORMAT,
            "equation_version": EQUATION_VERSION,
            "equations": [
                "Q = rho * flow / 3600 * Cp * (chw_return_temp - chw_supply_temp)",
                "PLR = Q / rated_capacity",
                "COP = c0 + c1*chw_supply_temp + c2*cw_supply_temp + c3*PLR",
                "P = Q / COP",
            ],
            "parameters": {
                "rated_capacity_kw": {"value": self.rated_capacity_kw, "unit": "kW"},
                "rated_power_kw": (
                    {"value": self.rated_power_kw, "unit": "kW"}
                    if self.rated_power_kw is not None else None
                ),
                "rho": {"value": self.rho, "unit": "kg/m3",
                        "note": "定值（§6.1 [草案]），温度相关模型待标定"},
                "cp": {"value": self.cp, "unit": "kJ/(kg.K)",
                       "note": "定值（§6.1 [草案]）"},
                "cop_coefficients": {
                    name: {
                        "value": self.cop_coefs[name],
                        "bounds": list(COP_COEF_BOUNDS[name]),
                        "unit": "1",
                    }
                    for name in ("c0", "c1", "c2", "c3")
                },
            },
            "inputs": dict(self.inputs),
            "identification": self.identification,
        }

    def save(self, directory: str | Path) -> Path:
        """写 `params.yaml`（YAML 明文参数）+ `model.json`（输入映射）。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        params_path = directory / "params.yaml"
        text = yaml.safe_dump(
            self.params_dict(), allow_unicode=True, sort_keys=True
        )
        _write_text_atomic(params_path, text)
        _write_text_atomic(
            directory / "model.json",
            json.dumps({"format": MODEL_FORMAT, "inputs": self.inputs,
                        "rated_capacity_kw": self.rated_capacity_kw,
                        "rated_power_kw": self.rated_power_kw,
                        "rho": self.rho, "cp": self.cp},
                       ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        return params_path

    @classmethod
    def load(cls, directory: str | Path) -> "ChillerPhysicsModel":
        directory = Path(directory)
        with open(directory / "params.yaml", encoding="utf-8") as fp:
            doc = yaml.safe_load(fp)
        if doc.get("format") != MODEL_FORMAT:
            raise ValueError(f"未知模型格式: {doc.get('format')!r}")
        params = doc["parameters"]
        model = cls(
            rated_capacity_kw=float(params["rated_capacity_kw"]["value"]),
            rated_power_kw=(
                float(params["rated_power_kw"]["value"])
                if params.get("rated_power_kw") else None
            ),
            inputs=doc.get("inputs"),
            rho_kg_per_m3=float(params["rho"]["value"]),
            cp_kj_per_kg_k=float(params["cp"]["value"]),
        )
        model.cop_coefs = {
            name: float(params["cop_coefficients"][name]["value"])
            for name in ("c0", "c1", "c2", "c3")
        }
        model.identification = dict(doc.get("identification") or {})
        return model
