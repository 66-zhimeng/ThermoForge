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

    @property
    def condenser_col(self) -> str:
        """冷凝侧温度列（Carnot 检查用，I-40）。v1 以冷却水供水温度近似。"""
        return self.inputs["cw_supply_temp"]


# ---------------------------------------------------------------- v2：DOE-2 三曲线

EQUATION_VERSION_V2 = "cooling_balance_v2"
MODEL_FORMAT_V2 = "thermoforge.chiller_physics.v2"

# v2 额外输入：冷却水回水温度（冷凝侧代理候选）与运行台数（可选）
DEFAULT_INPUTS_V2 = {
    **DEFAULT_INPUTS,
    "cw_return_temp": "cw_return_temp",  # Cel
    "run_count": "run_count",            # 1
}

# 双二次/二次曲线系数的明文物理范围（沿用 v1 做法：裁剪并记录）。
# 注意：v2 的曲线作用在**归一化温度** u=(t−center)/scale 上（center/scale
# 为训练集统计量，随参数 YAML 落盘）——原始温度下双二次设计矩阵在窄温域
# 严重病态（条件数 ~1e5，辨识系数爆炸性互消），归一化后条件数 ~4。
_BIQUAD_BOUNDS = {
    "const": (0.0, 3.0),
    "t1": (-2.0, 2.0),      # 归一化 T_chws 一次项
    "t1_sq": (-1.0, 1.0),
    "t2": (-2.0, 2.0),      # 归一化冷凝侧温度一次项
    "t2_sq": (-1.0, 1.0),
    "t1_t2": (-1.0, 1.0),
}
CURVE_BOUNDS_V2 = {
    "capft": dict(_BIQUAD_BOUNDS),
    "eirft": dict(_BIQUAD_BOUNDS),
    "eirfplr": {"const": (-0.5, 2.0), "plr": (-3.0, 3.0), "plr_sq": (-2.0, 2.0)},
}

# 预测时的曲线取值保护范围（防外推病态），明文记录在参数 YAML
CURVE_GUARDS_V2 = {
    "capft": (0.3, 2.0),
    "eirft": (0.2, 3.0),
    "eirfplr": (0.0, 1.5),
    "plr": (0.0, 1.5),
}

_BIQUAD_TERMS = ("const", "t1", "t1_sq", "t2", "t2_sq", "t1_t2")
_EIRFPLR_TERMS = ("const", "plr", "plr_sq")

# 辨识迭代次数固定（不定早停）：保证复现路径唯一（§7 可复现性）
_ALS_OUTER = 3
_ALS_INNER = 20


def _biquad_design(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """双二次设计矩阵：[1, t1, t1², t2, t2², t1·t2]。"""
    return np.column_stack([
        np.ones_like(t1), t1, t1 * t1, t2, t2 * t2, t1 * t2,
    ])


class ChillerPhysicsV2(ChillerPhysicsModel):
    """DOE-2 三曲线离心机模型（cooling_balance_v2）。

    ```text
    Q       = m·Cp·ΔT（与 v1 相同）
    CAPFT   = f(T_chws, T_cond)            # 可用容量比，双二次
    PLR     = Q / (Q_rated · units · CAPFT)
    EIRFT   = g(T_chws, T_cond)            # 能效比温度修正，双二次
    EIRFPLR = c0 + c1·PLR + c2·PLR²        # 部分负荷修正
    P       = P_rated · units · PLR · EIRFT · EIRFPLR
    ```

    - `T_cond`：冷凝侧温度代理，拟合时在 cw_supply_temp / cw_return_temp
      两个候选中按**辨识残差**选择（I-40 修正），选择依据写入参数 YAML。
    - `units`：inputs 含 `run_count` 时按运行台数缩放（rated_* 为**单台**
      额定）；否则 units ≡ 1（rated_* 为系统级）。
    - 辨识：交替最小二乘（双二次可线性化），EIRFPLR 归一 Σc=1
      （PLR=1 处为 1，DOE-2 额定工况归一约定），CAPFT 归一训练集
      中位工况为 1；迭代次数固定（_ALS_OUTER × _ALS_INNER），无随机源。
    """

    def __init__(self, rated_capacity_kw: float,
                 rated_power_kw: float | None = None, *,
                 inputs: Mapping[str, str] | None = None,
                 rho_kg_per_m3: float = DEFAULT_RHO_KG_PER_M3,
                 cp_kj_per_kg_k: float = DEFAULT_CP_KJ_PER_KG_K):
        super().__init__(rated_capacity_kw, rated_power_kw,
                         inputs=inputs or DEFAULT_INPUTS_V2,
                         rho_kg_per_m3=rho_kg_per_m3, cp_kj_per_kg_k=cp_kj_per_kg_k)
        self.capft_coefs: dict[str, float] | None = None
        self.eirft_coefs: dict[str, float] | None = None
        self.eirfplr_coefs: dict[str, float] | None = None
        self.condenser_proxy: str | None = None  # 逻辑名（cw_supply_temp / cw_return_temp）
        # 曲线输入归一化（训练集统计量，随参数落盘）：u = (t − center)/scale
        self.normalization: dict[str, dict[str, float]] | None = None

    # ---------------------------------------------------------------- 曲线

    @property
    def condenser_col(self) -> str:
        """冷凝侧温度列（I-40 修正：用辨识选出的代理，不再固定供水）。"""
        if self.condenser_proxy:
            return self.inputs[self.condenser_proxy]
        return self.inputs["cw_supply_temp"]

    @property
    def per_unit_rated(self) -> bool:
        """rated_* 为单台额定（按 run_count 缩放）时为 True。"""
        return bool(self.inputs.get("run_count"))

    def _temps(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """归一化后的（T_chws, T_cond）；未 fit 时返回原始值（仅供辨识）。"""
        t1 = pd.to_numeric(df[self.inputs["chw_supply_temp"]]).to_numpy(np.float64)
        t2 = pd.to_numeric(df[self.condenser_col]).to_numpy(np.float64)
        if self.normalization is None:
            return t1, t2
        n1, n2 = self.normalization["t_chws"], self.normalization["t_cond"]
        return (t1 - n1["center"]) / n1["scale"], (t2 - n2["center"]) / n2["scale"]

    def _units(self, df: pd.DataFrame) -> np.ndarray:
        col = self.inputs.get("run_count")
        if col and col in df.columns:
            units = pd.to_numeric(df[col]).to_numpy(np.float64)
            return np.maximum(units, 1.0)
        return np.ones(len(df))

    def _curve(self, coefs: dict[str, float], terms: tuple[str, ...],
               design: np.ndarray) -> np.ndarray:
        weights = np.array([coefs[t] for t in terms])
        return design @ weights

    def capft(self, df: pd.DataFrame) -> np.ndarray:
        t1, t2 = self._temps(df)
        raw = self._curve(self.capft_coefs, _BIQUAD_TERMS,
                          _biquad_design(t1, t2))
        lo, hi = CURVE_GUARDS_V2["capft"]
        return np.clip(raw, lo, hi)

    def eirft(self, df: pd.DataFrame) -> np.ndarray:
        t1, t2 = self._temps(df)
        raw = self._curve(self.eirft_coefs, _BIQUAD_TERMS,
                          _biquad_design(t1, t2))
        lo, hi = CURVE_GUARDS_V2["eirft"]
        return np.clip(raw, lo, hi)

    def eirfplr(self, plr: np.ndarray) -> np.ndarray:
        c = self.eirfplr_coefs
        raw = c["const"] + c["plr"] * plr + c["plr_sq"] * plr * plr
        lo, hi = CURVE_GUARDS_V2["eirfplr"]
        return np.clip(raw, lo, hi)

    def plr(self, df: pd.DataFrame, q_kw: np.ndarray) -> np.ndarray:
        denom = self.rated_capacity_kw * self._units(df) * self.capft(df)
        lo, hi = CURVE_GUARDS_V2["plr"]
        return np.clip(q_kw / np.maximum(denom, 1e-9), lo, hi)

    def cop(self, df: pd.DataFrame, q_kw: np.ndarray) -> np.ndarray:
        """有效 COP = Q/P = Q_rated·CAPFT / (P_rated·EIRFT·EIRFPLR)。"""
        if self.capft_coefs is None:
            raise RuntimeError("模型尚未 fit")
        plr = self.plr(df, q_kw)
        num = self.rated_capacity_kw * self.capft(df)
        den = (self.rated_power_kw or 1.0) * self.eirft(df) * self.eirfplr(plr)
        return num / np.maximum(den, 1e-9)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P = P_rated · units · PLR · EIRFT · EIRFPLR。"""
        if self.capft_coefs is None:
            raise RuntimeError("模型尚未 fit")
        if self.rated_power_kw is None:
            raise RuntimeError("v2 模型需要 rated_power_kw")
        q = self.cooling_capacity(df)
        plr = self.plr(df, q)
        return (self.rated_power_kw * self._units(df)
                * plr * self.eirft(df) * self.eirfplr(plr))

    # ---------------------------------------------------------------- 辨识

    def _identify(self, q: np.ndarray, p: np.ndarray, t1_raw: np.ndarray,
                  t2_raw: np.ndarray, units: np.ndarray) -> tuple[float, dict]:
        """对给定冷凝侧温度列做交替最小二乘，返回 (相对 RMSE, 归一化参数)。

        曲线输入先按训练集均值/标准差归一化（原始温度下双二次设计矩阵
        在窄温域病态，归一化是辨识可数值求解的前提）。
        """
        norm = {
            "t_chws": {"center": float(np.mean(t1_raw)),
                       "scale": float(np.std(t1_raw)) or 1.0},
            "t_cond": {"center": float(np.mean(t2_raw)),
                       "scale": float(np.std(t2_raw)) or 1.0},
        }
        t1 = (t1_raw - norm["t_chws"]["center"]) / norm["t_chws"]["scale"]
        t2 = (t2_raw - norm["t_cond"]["center"]) / norm["t_cond"]["scale"]
        phi = _biquad_design(t1, t2)
        y = p / (self.rated_power_kw * units)  # 归一功率
        qr = self.rated_capacity_kw * units

        a = np.zeros(6)
        a[0] = 1.0  # CAPFT ≡ 1 起步
        c = np.array([0.1, 0.9, 0.0])  # EIRFPLR 起步（Σ=1）
        for _ in range(_ALS_OUTER):
            capft = np.clip(phi @ a, *CURVE_GUARDS_V2["capft"])
            plr = np.clip(q / np.maximum(qr * capft, 1e-9), 1e-6, None)
            b = np.zeros(6)
            b[0] = 0.3
            for _ in range(_ALS_INNER):
                h = np.clip(c[0] + c[1] * plr + c[2] * plr * plr, 1e-3, None)
                # 固定 c 解 b：y/(plr·h) = φ·b
                b, _, _, _ = np.linalg.lstsq(phi, y / (plr * h), rcond=None)
                eirft = np.clip(phi @ b, 1e-3, None)
                # 固定 b 解 c：y/eirft = c0·plr + c1·plr² + c2·plr³
                design_c = np.column_stack([plr, plr * plr, plr ** 3])
                c, _, _, _ = np.linalg.lstsq(design_c, y / eirft, rcond=None)
                # DOE-2 归一：EIRFPLR(1)=1，尺度并入 EIRFT
                s = float(c.sum())
                if abs(s) > 1e-9:
                    c = c / s
                    b = b * s
            # CAPFT 更新：R = y/(eirft·h) = q/(qr·capft) → capft = q/(qr·R)
            h = np.clip(c[0] + c[1] * plr + c[2] * plr * plr, 1e-3, None)
            eirft = np.clip(phi @ b, 1e-3, None)
            r = np.clip(y / (eirft * h), 1e-6, None)
            capft_target = np.clip(q / np.maximum(qr * r, 1e-9),
                                   *CURVE_GUARDS_V2["capft"])
            a, _, _, _ = np.linalg.lstsq(phi, capft_target, rcond=None)
            # CAPFT 归一：训练集中位工况 = 1（消尺度漂移，形状仍被辨识）
            median_capft = float(np.median(np.clip(
                phi @ a, *CURVE_GUARDS_V2["capft"])))
            if median_capft > 1e-9:
                a = a / median_capft

        self.capft_coefs = dict(zip(_BIQUAD_TERMS, (float(v) for v in a)))
        self.eirft_coefs = dict(zip(_BIQUAD_TERMS, (float(v) for v in b)))
        self.eirfplr_coefs = dict(zip(_EIRFPLR_TERMS, (float(v) for v in c)))
        pred = self._predict_arrays(q, t1, t2, units)
        rel_rmse = float(np.sqrt(np.mean((pred - p) ** 2)) / np.mean(p))
        return rel_rmse, norm

    def _predict_arrays(self, q, t1, t2, units) -> np.ndarray:
        phi = _biquad_design(t1, t2)
        capft = np.clip(phi @ np.array([self.capft_coefs[t] for t in _BIQUAD_TERMS]),
                        *CURVE_GUARDS_V2["capft"])
        plr = np.clip(q / np.maximum(self.rated_capacity_kw * units * capft, 1e-9),
                      *CURVE_GUARDS_V2["plr"])
        eirft = np.clip(phi @ np.array([self.eirft_coefs[t] for t in _BIQUAD_TERMS]),
                        *CURVE_GUARDS_V2["eirft"])
        c = self.eirfplr_coefs
        h = np.clip(c["const"] + c["plr"] * plr + c["plr_sq"] * plr * plr,
                    *CURVE_GUARDS_V2["eirfplr"])
        return self.rated_power_kw * units * plr * eirft * h

    def fit(self, df: pd.DataFrame, y: Sequence[float]) -> "ChillerPhysicsV2":
        """交替最小二乘辨识三曲线；冷凝侧代理按辨识残差选择（I-40）。"""
        if self.rated_power_kw is None:
            raise ValueError("v2 模型必须提供 rated_power_kw")
        q = self.cooling_capacity(df)
        p = np.asarray(y, dtype=np.float64)
        if len(p) != len(q):
            raise ValueError("y 与 df 行数不一致")
        units_all = self._units(df)
        t1_all = pd.to_numeric(
            df[self.inputs["chw_supply_temp"]]).to_numpy(np.float64)
        mask = ((q > 0) & (p > 0) & np.isfinite(q) & np.isfinite(p)
                & np.isfinite(t1_all) & (units_all > 0))
        candidates = [k for k in ("cw_supply_temp", "cw_return_temp")
                      if self.inputs.get(k) in df.columns]
        if not candidates:
            raise ValueError("v2 需要至少一个冷凝侧温度候选列")
        for name in candidates:
            col = pd.to_numeric(df[self.inputs[name]]).to_numpy(np.float64)
            mask &= np.isfinite(col)
        n_used = int(mask.sum())
        if n_used < 12:
            raise ValueError(f"可用于曲线辨识的样本不足: {n_used} < 12")

        scores: dict[str, float] = {}
        best: dict[str, tuple] = {}
        for name in candidates:
            t2 = pd.to_numeric(df[self.inputs[name]]).to_numpy(np.float64)
            rel_rmse, norm = self._identify(
                q[mask], p[mask], t1_all[mask], t2[mask], units_all[mask])
            scores[name] = rel_rmse
            best[name] = (dict(self.capft_coefs), dict(self.eirft_coefs),
                          dict(self.eirfplr_coefs), norm)
        self.condenser_proxy = min(scores, key=scores.get)
        (self.capft_coefs, self.eirft_coefs, self.eirfplr_coefs,
         self.normalization) = best[self.condenser_proxy]

        # 系数裁剪到明文物理范围并记录（沿用 v1 做法）
        clipped: list[str] = []
        for curve, terms in (("capft", _BIQUAD_TERMS), ("eirft", _BIQUAD_TERMS),
                             ("eirfplr", _EIRFPLR_TERMS)):
            coefs = getattr(self, f"{curve}_coefs")
            for term in terms:
                lo, hi = CURVE_BOUNDS_V2[curve][term]
                v = coefs[term]
                if not (lo <= v <= hi):
                    clipped.append(f"{curve}.{term}")
                    coefs[term] = min(max(v, lo), hi)
        # 裁剪后重申 EIRFPLR 归一（Σc=1）
        s = sum(self.eirfplr_coefs[t] for t in _EIRFPLR_TERMS)
        if abs(s) > 1e-9 and s != 1.0:
            for t in _EIRFPLR_TERMS:
                self.eirfplr_coefs[t] /= s

        self.identification = {
            "method": "alternating_least_squares",
            "outer_iterations": _ALS_OUTER,
            "inner_iterations": _ALS_INNER,
            "n_samples": n_used,
            "n_dropped_nonpositive": int((~mask).sum()),
            "condenser_proxy": {
                "selected": self.condenser_proxy,
                "column": self.condenser_col,
                "rel_rmse_by_candidate": scores,
                "criterion": "train 相对 RMSE（越小冷凝侧解释力越强）",
            },
            "power_rel_rmse": scores[self.condenser_proxy],
            "clipped_coefficients": clipped,
        }
        return self

    # ---------------------------------------------------------------- 序列化

    def params_dict(self) -> dict[str, Any]:
        if self.capft_coefs is None:
            raise RuntimeError("模型尚未 fit")
        per_unit = bool(self.inputs.get("run_count"))
        rated_note = ("单台额定（按 run_count 缩放）" if per_unit
                      else "系统级额定")
        return {
            "format": MODEL_FORMAT_V2,
            "equation_version": EQUATION_VERSION_V2,
            "equations": [
                "Q = rho * flow / 3600 * Cp * (chw_return_temp - chw_supply_temp)",
                "u = (T - center) / scale  # 曲线输入归一化，见 input_normalization",
                "CAPFT = biquad(u_chws, u_cond)",
                "PLR = Q / (Q_rated * units * CAPFT)",
                "EIRFT = biquad(u_chws, u_cond)",
                "EIRFPLR = c0 + c1*PLR + c2*PLR^2",
                "P = P_rated * units * PLR * EIRFT * EIRFPLR",
            ],
            "parameters": {
                "rated_capacity_kw": {"value": self.rated_capacity_kw,
                                      "unit": "kW", "note": rated_note},
                "rated_power_kw": {"value": self.rated_power_kw,
                                   "unit": "kW", "note": rated_note},
                "rho": {"value": self.rho, "unit": "kg/m3",
                        "note": "定值（§6.1 [草案]），温度相关模型待标定"},
                "cp": {"value": self.cp, "unit": "kJ/(kg.K)",
                       "note": "定值（§6.1 [草案]）"},
                "input_normalization": {
                    k: {"center": v["center"], "scale": v["scale"],
                        "unit": "Cel",
                        "note": "训练集均值/标准差，曲线作用在归一化温度上"}
                    for k, v in (self.normalization or {}).items()
                },
                "curves": {
                    curve: {
                        term: {
                            "value": getattr(self, f"{curve}_coefs")[term],
                            "bounds": list(CURVE_BOUNDS_V2[curve][term]),
                            "unit": "1",
                        }
                        for term in terms
                    }
                    for curve, terms in (
                        ("capft", _BIQUAD_TERMS), ("eirft", _BIQUAD_TERMS),
                        ("eirfplr", _EIRFPLR_TERMS))
                },
                "curve_guards": {k: list(v) for k, v in CURVE_GUARDS_V2.items()},
            },
            "inputs": dict(self.inputs),
            "condenser_proxy": self.condenser_proxy,
            "identification": self.identification,
        }

    def save(self, directory: str | Path) -> Path:
        """写 `params.yaml`（YAML 明文参数）+ `model.json`。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        params_path = directory / "params.yaml"
        _write_text_atomic(params_path, yaml.safe_dump(
            self.params_dict(), allow_unicode=True, sort_keys=True))
        _write_text_atomic(
            directory / "model.json",
            json.dumps({"format": MODEL_FORMAT_V2, "inputs": self.inputs,
                        "rated_capacity_kw": self.rated_capacity_kw,
                        "rated_power_kw": self.rated_power_kw,
                        "rho": self.rho, "cp": self.cp,
                        "condenser_proxy": self.condenser_proxy},
                       ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        return params_path

    @classmethod
    def load(cls, directory: str | Path) -> "ChillerPhysicsV2":
        directory = Path(directory)
        with open(directory / "params.yaml", encoding="utf-8") as fp:
            doc = yaml.safe_load(fp)
        if doc.get("format") != MODEL_FORMAT_V2:
            raise ValueError(f"未知模型格式: {doc.get('format')!r}")
        params = doc["parameters"]
        model = cls(
            rated_capacity_kw=float(params["rated_capacity_kw"]["value"]),
            rated_power_kw=float(params["rated_power_kw"]["value"]),
            inputs=doc.get("inputs"),
            rho_kg_per_m3=float(params["rho"]["value"]),
            cp_kj_per_kg_k=float(params["cp"]["value"]),
        )
        curves = params["curves"]
        model.capft_coefs = {t: float(curves["capft"][t]["value"])
                             for t in _BIQUAD_TERMS}
        model.eirft_coefs = {t: float(curves["eirft"][t]["value"])
                             for t in _BIQUAD_TERMS}
        model.eirfplr_coefs = {t: float(curves["eirfplr"][t]["value"])
                               for t in _EIRFPLR_TERMS}
        model.condenser_proxy = doc.get("condenser_proxy")
        norm = params.get("input_normalization") or {}
        model.normalization = {
            str(k): {"center": float(v["center"]), "scale": float(v["scale"])}
            for k, v in norm.items()
        } or None
        model.identification = dict(doc.get("identification") or {})
        return model


def load_physics_model(directory: str | Path) -> ChillerPhysicsModel:
    """按 params.yaml 的 format 字段分派加载 v1 / v2（v1 保留兼容）。"""
    with open(Path(directory) / "params.yaml", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    fmt = doc.get("format")
    if fmt == MODEL_FORMAT:
        return ChillerPhysicsModel.load(directory)
    if fmt == MODEL_FORMAT_V2:
        return ChillerPhysicsV2.load(directory)
    raise ValueError(f"未知物理模型格式: {fmt!r}")
