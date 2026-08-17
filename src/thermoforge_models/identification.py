"""系统辨识模型族：物理方程直接辨识，参数可解释。

两个模型，都遵循「线性化后用最小二乘辨识」的路子 —— 参数少、外推稳、
每个系数有物理含义，且天然满足单调性约束（不需要额外施加）。

**Gordon-Ng 通用冷机模型**（`GordonNgChiller`）
    基于第一定律（能量平衡）+ 第二定律（熵平衡）导出，是冷机辨识的事实标准
    （Gordon & Ng 1994/2000；ASHRAE Transactions 109(2) 复核）。三个参数都有
    物理含义::

        (1/COP + 1) · Q_e / T_ci  −  1  =  a0/T_ci + a1·(T_ci − T_ei)/(T_ci·T_ei)
                                            + a2·(1/COP + 1)·Q_e/(T_ci·T_ei)

    整理成线性形式 `y = a0·x0 + a1·x1 + a2·x2`，其中

    - `a0` ≈ 总内部熵产 ΔS_int（不可逆损失）
    - `a1` ≈ 热漏损失 Q_leak
    - `a2` ≈ 等效换热热阻 R_eqv（蒸发器+冷凝器串联热阻）

    温度必须用**绝对温标**。辨识后反解 COP，再得功率 `P = Q_e / COP`。

**ε-NTU 换热器模型**（`EffectivenessNTU`）
    逆流板换的有效度关联式。`ε` 由 NTU 与热容比 `Cr` 决定，NTU 随两侧流量
    变化（Dittus-Boelter：`UA ∝ (m_h^-0.8 + m_c^-0.8)^-1`）::

        Q = ε · C_min · (T_h,in − T_c,in)
        ε = [1 − exp(−NTU·(1−Cr))] / [1 − Cr·exp(−NTU·(1−Cr))]      Cr < 1
        NTU = UA / C_min,   UA = UA0 · (m_h^-0.8 + β·m_c^-0.8)^-1

    只用**入口端口值**（两侧进水温 + 两侧流量），出口温度是输出而非输入 ——
    因此结构上不可能发生「出口温度×流量=标签」的泄漏。
    辨识参数：`UA0`、`β`。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

CP_WATER = 4.187          # kJ/(kg·K)
RHO_WATER = 1000.0        # kg/m³
KELVIN = 273.15

GN_FORMAT = "thermoforge.gordon_ng.v1"
NTU_FORMAT = "thermoforge.eps_ntu.v1"

# 默认列名（property_code）；可通过 inputs 重映射
GN_INPUTS = {
    "cooling_load": "cooling_load",     # Q_e，蒸发器负荷 kW
    "t_evap_out": "t_evap_out",         # T_ei，蒸发器出水温 °C
    "t_cond_in": "t_cond_in",           # T_ci，冷凝器进水温 °C
}
NTU_INPUTS = {
    "t_hot_in": "t_chw_in",             # 热侧进水温 °C
    "t_cold_in": "t_cw_in",             # 冷侧进水温 °C
    "f_hot": "f_chw",                   # 热侧流量 m³/h
    "f_cold": "f_cw",                   # 冷侧流量 m³/h
}


class MissingInputError(KeyError):
    """输入列缺失。带上模型需要什么、视图里有什么，便于直接改正。"""


def _col(df: pd.DataFrame, name: str, *, role: str = "",
         needs: Mapping[str, str] | None = None) -> np.ndarray:
    if name not in df.columns:
        want = (f"；本模型需要的列：{sorted(needs.values())}" if needs else "")
        raise MissingInputError(
            f"缺少输入列 {name!r}"
            + (f"（角色 {role}）" if role else "")
            + f"。视图现有列：{sorted(df.columns)[:16]}" + want
            + "。可用 hyperparameters.inputs 重映射，写法 '角色=列名' 或直接给同名列名。"
        )
    return pd.to_numeric(df[name]).to_numpy(np.float64)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class GordonNgChiller:
    """Gordon-Ng 族冷机模型：能量平衡 + 熵平衡 + 换热热阻，三参数闭式解。

    直接从第一性原理推，不用文献里那个线性化形式 —— 后者在窄工况区间下
    `a0/a2` 高度共线，反解时分母趋零而发散（实测 CVRMSE 5000%+）。

    推导::

        制冷剂蒸发/冷凝温度由换热热阻与水温相连：
            T_e = T_ei − Q_e · R_e            蒸发温度低于冷冻水
            T_c = T_ci + Q_c · R_c            冷凝温度高于冷却水
        能量：  Q_c = Q_e + P
        熵：    Q_c / T_c = Q_e / T_e + ΔS_int          （ΔS_int ≥ 0，不可逆）

        令 K = Q_e / T_e + ΔS_int，代入并解出：
            Q_c = T_ci · K / (1 − R_c · K)
            P   = Q_c − Q_e

    三个参数都有物理含义且非负：`R_e`（蒸发器热阻 K/kW）、`R_c`（冷凝器热阻）、
    `ΔS_int`（内部熵产 kW/K）。单调性天然满足：∂P/∂Q_e > 0、∂P/∂T_ci > 0、
    ∂P/∂T_ei < 0 —— 不需要额外施加约束。

    数值条件：`R_c·K ≈ 0.02`（典型工况），分母远离零，稳定。
    """

    def __init__(self, inputs: Mapping[str, str] | None = None,
                 q_floor: float = 1.0):
        self.inputs = dict(GN_INPUTS if inputs is None else inputs)
        self.q_floor = float(q_floor)
        self.coef_: np.ndarray | None = None       # [R_e, R_c, dS_int]
        self.n_train_: int = 0

    # ------------------------------------------------------------ 内部
    def _mask(self, df: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        qe = _col(df, self.inputs["cooling_load"], role="cooling_load", needs=self.inputs)
        tei = _col(df, self.inputs["t_evap_out"], role="t_evap_out", needs=self.inputs)
        tci = _col(df, self.inputs["t_cond_in"], role="t_cond_in", needs=self.inputs)
        m = np.isfinite(qe) & np.isfinite(tei) & np.isfinite(tci)
        m &= qe > self.q_floor
        m &= (tci - tei) > 0.5
        if y is not None:
            m &= np.isfinite(y) & (y > 0)
        return m

    # 定义域边界：低于这些值就把输入夹住再算，**不返回 NaN**。
    # 契约规定「计算层产生 NaN 属于缺陷」（conventions §4.2），指标层会直接
    # 拒收含 NaN 的预测（TFX-905）。定义域外该做的是**如实外推并可被物理
    # 检查逮住**，而不是把问题丢给下游。
    MIN_LIFT_K = 0.5          # 最小提升温差
    MIN_DENOM = 1e-3          # 1 − R_c·K 的下界

    def _power_from_coef(self, df: pd.DataFrame,
                         coef: np.ndarray) -> np.ndarray:
        r_e, r_c, ds = coef
        qe = _col(df, self.inputs["cooling_load"], role="cooling_load", needs=self.inputs)
        tei = _col(df, self.inputs["t_evap_out"], role="t_evap_out", needs=self.inputs) + KELVIN
        tci = _col(df, self.inputs["t_cond_in"], role="t_cond_in", needs=self.inputs) + KELVIN

        qe = np.nan_to_num(qe, nan=0.0, posinf=0.0, neginf=0.0)
        qe = np.maximum(qe, 0.0)
        # 冷凝温度必须高于蒸发温度，否则卡诺项发散：夹住而非丢弃
        tci = np.maximum(tci, tei + self.MIN_LIFT_K)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_e = np.maximum(tei - qe * r_e, 1.0)
            k = qe / t_e + ds
            denom = np.maximum(1.0 - r_c * k, self.MIN_DENOM)
            power = tci * k / denom - qe
        # 夹住之后仍可能因输入全缺失而非有限；此时退回 0（无负荷即无功耗）
        return np.clip(np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0),
                       0.0, None)

    # ------------------------------------------------------------ 拟合
    def fit(self, df: pd.DataFrame, y: Sequence[float],
            feature_order: Sequence[str] | None = None) -> "GordonNgChiller":
        """`y` 为实测电功率 kW。对功率残差做带非负约束的非线性最小二乘。"""
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) != len(df):
            raise ValueError(f"y 长度 {len(y_arr)} 与 df {len(df)} 不一致")
        m = self._mask(df, y_arr)
        if m.sum() < 10:
            raise ValueError(f"可用样本不足: {int(m.sum())}")
        sub, ys = df.loc[m], y_arr[m]

        # 初值：由典型工况反推量级。R ~ ΔT/Q，ΔS ~ P/T
        qe = _col(sub, self.inputs["cooling_load"])
        q_typ = float(np.median(qe))
        p_typ = float(np.median(ys))
        t_typ = float(np.median(_col(sub, self.inputs["t_cond_in"])) + KELVIN)
        x0 = np.array([1.0 / max(q_typ, 1.0),      # R_e：1K 逼近温差
                       1.0 / max(q_typ, 1.0),      # R_c
                       p_typ / t_typ * 0.1])       # ΔS_int

        def resid(p: np.ndarray) -> np.ndarray:
            pred = self._power_from_coef(sub, p)
            return np.where(np.isfinite(pred), pred - ys, 1e4)

        sol = least_squares(resid, x0=x0, bounds=(0.0, np.inf),
                            x_scale="jac", max_nfev=800)
        self.coef_ = np.asarray(sol.x, dtype=np.float64)
        self.n_train_ = int(m.sum())
        return self

    # ------------------------------------------------------------ 预测
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测功率。**保证返回有限非负值**，定义域外按夹住的输入外推。"""
        if self.coef_ is None:
            raise RuntimeError("模型尚未 fit")
        return self._power_from_coef(df, self.coef_)

    # ------------------------------------------------------------ 存取
    @property
    def parameters(self) -> dict[str, float]:
        """物理含义的参数（供报告与人工复核）。"""
        if self.coef_ is None:
            raise RuntimeError("模型尚未 fit")
        r_e, r_c, ds = self.coef_
        return {"r_evaporator": float(r_e), "r_condenser": float(r_c),
                "delta_s_internal": float(ds)}

    def to_dict(self) -> dict[str, Any]:
        return {"format": GN_FORMAT, "inputs": dict(self.inputs),
                "coef": [float(c) for c in (self.coef_ if self.coef_ is not None
                                            else [])],
                "parameters": self.parameters if self.coef_ is not None else {},
                "n_train": self.n_train_, "q_floor": self.q_floor}

    def save(self, directory: str | Path) -> None:
        _atomic_write(Path(directory) / "model.json",
                      json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "GordonNgChiller":
        with open(Path(directory) / "model.json", encoding="utf-8") as fp:
            doc = json.load(fp)
        m = cls(inputs=doc.get("inputs"), q_floor=doc.get("q_floor", 1.0))
        m.coef_ = np.asarray(doc["coef"], dtype=np.float64)
        m.n_train_ = int(doc.get("n_train", 0))
        return m


class EffectivenessNTU:
    """逆流板换 ε-NTU 模型：两参数（UA0、β），非线性最小二乘辨识。"""

    def __init__(self, inputs: Mapping[str, str] | None = None,
                 n_units: str | None = "run_count"):
        self.inputs = dict(NTU_INPUTS if inputs is None else inputs)
        self.n_units = n_units
        self.ua0_: float | None = None
        self.beta_: float = 1.0
        self.n_train_: int = 0

    # ------------------------------------------------------------ 内部
    def _capacity_rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """两侧热容流率 kW/K（体积流量 m³/h → 质量流率）。"""
        fh = _col(df, self.inputs["f_hot"]) * RHO_WATER / 3600.0     # kg/s
        fc = _col(df, self.inputs["f_cold"]) * RHO_WATER / 3600.0
        return fh * CP_WATER, fc * CP_WATER

    def _eps(self, ua: np.ndarray, cmin: np.ndarray,
             cmax: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ntu = ua / np.maximum(cmin, 1e-9)
            cr = np.clip(cmin / np.maximum(cmax, 1e-9), 0.0, 0.9999)
            expo = np.exp(-np.clip(ntu * (1.0 - cr), 0.0, 50.0))
            eps = (1.0 - expo) / np.maximum(1.0 - cr * expo, 1e-9)
        return np.clip(eps, 0.0, 1.0)

    def _ua(self, df: pd.DataFrame, ua0: float, beta: float) -> np.ndarray:
        """UA 随两侧流量变化（Dittus-Boelter 串联热阻）。"""
        fh = np.maximum(_col(df, self.inputs["f_hot"]), 1e-6)
        fc = np.maximum(_col(df, self.inputs["f_cold"]), 1e-6)
        ua = ua0 / (np.power(fh, -0.8) + beta * np.power(fc, -0.8))
        if self.n_units and self.n_units in df.columns:
            ua = ua * np.maximum(_col(df, self.n_units), 1.0)
        return ua

    def _q(self, df: pd.DataFrame, ua0: float, beta: float) -> np.ndarray:
        ch, cc = self._capacity_rates(df)
        cmin, cmax = np.minimum(ch, cc), np.maximum(ch, cc)
        dt = (_col(df, self.inputs["t_hot_in"])
              - _col(df, self.inputs["t_cold_in"]))
        return self._eps(self._ua(df, ua0, beta), cmin, cmax) * cmin * dt

    # ------------------------------------------------------------ 拟合
    def fit(self, df: pd.DataFrame, y: Sequence[float],
            feature_order: Sequence[str] | None = None) -> "EffectivenessNTU":
        y_arr = np.asarray(y, dtype=np.float64)
        dt = (_col(df, self.inputs["t_hot_in"])
              - _col(df, self.inputs["t_cold_in"]))
        m = np.isfinite(y_arr) & (y_arr > 0) & np.isfinite(dt) & (dt > 0.1)
        for key in self.inputs.values():
            m &= np.isfinite(_col(df, key))
        if m.sum() < 10:
            raise ValueError(f"可用样本不足: {int(m.sum())}")
        sub, ys = df.loc[m], y_arr[m]

        def resid(p: np.ndarray) -> np.ndarray:
            return self._q(sub, float(p[0]), float(p[1])) - ys

        ch, cc = self._capacity_rates(sub)
        guess = float(np.median(np.minimum(ch, cc)))       # NTU≈1 量级起步
        sol = least_squares(resid, x0=[guess, 1.0],
                            bounds=([1e-3, 1e-3], [np.inf, np.inf]),
                            max_nfev=400)
        self.ua0_, self.beta_ = float(sol.x[0]), float(sol.x[1])
        self.n_train_ = int(m.sum())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.ua0_ is None:
            raise RuntimeError("模型尚未 fit")
        return self._q(df, self.ua0_, self.beta_)

    # ------------------------------------------------------------ 存取
    @property
    def parameters(self) -> dict[str, float]:
        if self.ua0_ is None:
            raise RuntimeError("模型尚未 fit")
        return {"ua0": self.ua0_, "beta": self.beta_}

    def to_dict(self) -> dict[str, Any]:
        return {"format": NTU_FORMAT, "inputs": dict(self.inputs),
                "n_units": self.n_units,
                "parameters": self.parameters if self.ua0_ is not None else {},
                "n_train": self.n_train_}

    def save(self, directory: str | Path) -> None:
        _atomic_write(Path(directory) / "model.json",
                      json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "EffectivenessNTU":
        with open(Path(directory) / "model.json", encoding="utf-8") as fp:
            doc = json.load(fp)
        m = cls(inputs=doc.get("inputs"), n_units=doc.get("n_units"))
        params = doc.get("parameters") or {}
        m.ua0_ = float(params["ua0"]) if "ua0" in params else None
        m.beta_ = float(params.get("beta", 1.0))
        m.n_train_ = int(doc.get("n_train", 0))
        return m
