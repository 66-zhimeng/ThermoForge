"""物理验证（implementation-notes.md §6、research-loop.md §7 Physics 维度）。

- **可证伪硬约束**（§6.2，不依赖标定）：
  `COP > 0`、`COP < COP_Carnot`（开尔文）、
  `input_power ≤ rated × 上限系数`、制冷量为正时 `input_power > 0`。
- **单调性**（§6.3）：在训练好的模型上做受控扰动扫描——固定其他输入
  于典型值，单独扫描目标输入，检查输出是否单调。
- **违规率口径**（§6.4）：每条约束单独报告 + 总体口径（任一违规即计入），
  验收阈值绑定总体口径。
- 违反硬约束的模型不得进入发布候选（research-loop §7）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

CEL_TO_K = 273.15
DEFAULT_UPPER_POWER_FACTOR = 1.1  # input_power ≤ rated × 上限系数 [草案]


@dataclass(frozen=True)
class ConstraintResult:
    """单条约束的检查结果。"""

    name: str
    applicable: int  # 该约束可判定的样本数
    violations: int
    rate: float  # violations / applicable（无适用样本时为 None 的替代：-1 不用，用 None）

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "violations": self.violations,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class PhysicsReport:
    """物理验证报告：每条单独口径 + 总体口径（§6.4）。"""

    n_samples: int
    hard_constraints: dict[str, ConstraintResult]
    monotonicity: dict[str, ConstraintResult] = field(default_factory=dict)
    overall_violations: int = 0  # 任一硬约束违规的样本数（总体口径）
    overall_rate: float = 0.0

    def is_publish_candidate(self, max_violation_rate: float) -> bool:
        """发布门禁绑定总体口径（§6.4）；违反硬约束不得进入发布候选。"""
        return self.overall_rate <= max_violation_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "hard_constraints": {k: v.to_dict() for k, v in self.hard_constraints.items()},
            "monotonicity": {k: v.to_dict() for k, v in self.monotonicity.items()},
            "overall_violations": self.overall_violations,
            "overall_rate": self.overall_rate,
        }


def check_hard_constraints(
    df: pd.DataFrame,
    *,
    power_col: str = "input_power",
    cooling_col: str | None = None,
    chw_supply_col: str | None = None,  # 近似蒸发温度（Cel）
    cw_return_col: str | None = None,  # 近似冷凝温度（Cel）
    rated_power: float | None = None,
    upper_power_factor: float = DEFAULT_UPPER_POWER_FACTOR,
) -> PhysicsReport:
    """可证伪硬约束检查（§6.2）。

    每条约束只在其所需列存在且数值有限、物理上有意义的样本上判定
    （如 COP 类约束要求 Q > 0）；总体口径按「任一适用约束违规即计入」。
    """
    n = len(df)
    power = _col(df, power_col)
    cooling = _col(df, cooling_col) if cooling_col else None

    masks: dict[str, np.ndarray] = {}  # 约束名 → 违规布尔掩码
    applicable: dict[str, int] = {}

    def _register(name: str, applicable_mask: np.ndarray, violation: np.ndarray) -> None:
        applicable[name] = int(applicable_mask.sum())
        masks[name] = violation & applicable_mask

    # COP > 0（要求 Q > 0 时 P > 0）
    if cooling is not None:
        cop_base = np.isfinite(power) & np.isfinite(cooling) & (cooling > 0)
        _register("cop_positive", cop_base, power <= 0)
        # 制冷量为正时 input_power > 0（与 cop_positive 同条件，单列以便对照 §6.2）
        _register("positive_cooling_positive_power", cop_base, power <= 0)

    # COP < COP_Carnot = T_evap / (T_cond − T_evap)（开尔文）
    if cooling is not None and chw_supply_col and cw_return_col:
        t_evap = _col(df, chw_supply_col) + CEL_TO_K
        t_cond = _col(df, cw_return_col) + CEL_TO_K
        base = (
            np.isfinite(power) & np.isfinite(cooling) & np.isfinite(t_evap)
            & np.isfinite(t_cond) & (cooling > 0) & (power > 0)
            & (t_cond > t_evap)
        )
        cop = np.where(base, cooling / np.maximum(power, 1e-12), 0.0)
        cop_carnot = np.where(base, t_evap / np.maximum(t_cond - t_evap, 1e-12), 0.0)
        _register("cop_below_carnot", base, cop >= cop_carnot)

    # input_power ≤ rated × 上限系数（含非负）
    if rated_power is not None:
        base = np.isfinite(power)
        violation = (power < 0) | (power > rated_power * upper_power_factor)
        _register("power_within_rated", base, violation)

    overall = np.zeros(n, dtype=bool)
    results: dict[str, ConstraintResult] = {}
    for name, mask in masks.items():
        overall |= mask
        app = applicable[name]
        results[name] = ConstraintResult(
            name=name,
            applicable=app,
            violations=int(mask.sum()),
            rate=(int(mask.sum()) / app) if app else 0.0,
        )
    return PhysicsReport(
        n_samples=n,
        hard_constraints=results,
        overall_violations=int(overall.sum()),
        overall_rate=float(overall.mean()) if n else 0.0,
    )


def check_monotonicity(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    base_row: Mapping[str, Any],
    feature: str,
    grid: Sequence[float],
    *,
    direction: int = 1,
    rel_tol: float = 1e-9,
) -> ConstraintResult:
    """训练后模型的受控扰动单调性扫描（§6.3）。

    固定其他输入于 `base_row`，单独把 `feature` 扫过 `grid`，
    检查输出沿 `direction`（+1 递增 / −1 递减）是否单调。
    违规率 = 违规相邻对数 / 相邻对总数。
    """
    if len(grid) < 2:
        raise ValueError("grid 至少需要两个点")
    if direction not in (1, -1):
        raise ValueError("direction 必须是 +1 或 -1")
    rows = []
    for value in grid:
        row = dict(base_row)
        row[feature] = value
        rows.append(row)
    y_hat = np.asarray(predict_fn(pd.DataFrame(rows)), dtype=np.float64)
    diffs = direction * np.diff(y_hat)
    tol = rel_tol * max(float(np.max(np.abs(y_hat))), 1.0)
    violations = int(np.sum(diffs < -tol))
    pairs = len(grid) - 1
    return ConstraintResult(
        name=f"monotonic:{feature}:{'+' if direction > 0 else '-'}",
        applicable=pairs,
        violations=violations,
        rate=violations / pairs,
    )


def combine_reports(report: PhysicsReport,
                    monotonicity: Mapping[str, ConstraintResult]) -> PhysicsReport:
    """把单调性结果并入报告（单调性违规计入总体口径，§6.4）。"""
    mono = dict(monotonicity)
    mono_violations = sum(r.violations for r in mono.values())
    overall_violations = report.overall_violations + mono_violations
    n = report.n_samples
    return PhysicsReport(
        n_samples=n,
        hard_constraints=report.hard_constraints,
        monotonicity=mono,
        overall_violations=overall_violations,
        overall_rate=(overall_violations / n) if n else 0.0,
    )


def _col(df: pd.DataFrame, name: str | None) -> np.ndarray:
    if name is None or name not in df.columns:
        raise ValueError(f"缺少物理检查所需列: {name!r}")
    return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)
