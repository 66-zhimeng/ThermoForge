"""指标模块（implementation-notes.md §5、research-loop.md §7、DD-14）。

RMSE / MAE / MAPE / CVRMSE / NMBE / R2 的**单一实现**，比率制（非百分比）。
所有实验的指标必须由本模块计算，禁止各建模脚本自行实现（§5.3）。

口径约定：

- 默认 **micro**（全样本合并），同时附带 per-object 明细（§5.3）。
- MAPE 剔除 `|y| < y_floor` 的样本并报告 `mape_valid_fraction`；
  `y_floor` 默认取 `0.05 * max(|y|)`（额定功率 5% 的代理，[草案]，§5.1）。
  有效样本比例 < `min_valid_fraction`（默认 0.8）报 **TFX-905**。
- CVRMSE / NMBE 在 `mean(y) ≈ 0` 时未定义，R2 在 `var(y) ≈ 0` 时未定义；
  三者一律记为 None 并给出说明，不产生一个看似正常的数字。
- 输入中的 NaN 属于缺陷（conventions §4.2），直接报错而不是静默剔除。

R2 是 data-survey.md 通篇用来表达结论的指标（§F1 的同源判定、§5 的
模型对比都以 R² 陈述），但实现层此前缺席，导致报告里的 R² 无法被实验
产物复现校验——补齐见 issues.md I-54。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .errors import ResearchError

METRIC_NAMES = ("RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2")

DEFAULT_MIN_VALID_FRACTION = 0.8  # §5.1 [草案：80%]
DEFAULT_Y_FLOOR_RATIO = 0.05  # §5.1 [草案：额定功率的 5%]
_MEAN_EPS = 1e-12  # mean(y) ≈ 0 的判定阈值（相对 max|y|）
_VAR_EPS = 1e-24  # var(y) ≈ 0 的判定阈值（相对 max(y)²）


def _as_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维序列")
    if np.isnan(arr).any():
        raise ResearchError(
            "TFX-905", f"{name} 含 NaN：计算层产生 NaN 属于缺陷（conventions §4.2）"
        )
    if np.isinf(arr).any():
        raise ResearchError("TFX-905", f"{name} 含 inf（TFDC-602）")
    return arr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray, y_pred: np.ndarray, y_floor: float
) -> tuple[float | None, float]:
    """MAPE 与有效样本比例。`|y| < y_floor` 的样本剔除（§5.1）。"""
    mask = np.abs(y_true) >= y_floor
    valid_fraction = float(np.mean(mask)) if len(mask) else 0.0
    if not mask.any():
        return None, valid_fraction
    value = float(
        np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    )
    return value, valid_fraction


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """CVRMSE = RMSE / mean(y)；mean(y) ≈ 0 时未定义（§5 表）。"""
    mean_y = float(np.mean(y_true))
    scale = float(np.max(np.abs(y_true))) if len(y_true) else 0.0
    if abs(mean_y) <= _MEAN_EPS * max(scale, 1.0):
        return None
    return rmse(y_true, y_pred) / mean_y


def nmbe(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """NMBE = sum(y-ŷ) / (n · mean(y))（§5.2，DD-14 必报偏差指标）。"""
    mean_y = float(np.mean(y_true))
    scale = float(np.max(np.abs(y_true))) if len(y_true) else 0.0
    if abs(mean_y) <= _MEAN_EPS * max(scale, 1.0):
        return None
    return float(np.sum(y_true - y_pred) / (len(y_true) * mean_y))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R² = 1 − SS_res / SS_tot；`var(y) ≈ 0` 时未定义。

    取值可以为负（模型比常数均值预测还差），这是有信息量的结果而不是
    错误——data-survey §5 里冷冻水侧单独建模的 R² = −8.47 正是靠这一点
    读出来的，因此不做任何截断。
    """
    mean_y = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    scale = float(np.max(np.abs(y_true))) if len(y_true) else 0.0
    if ss_tot <= _VAR_EPS * len(y_true) * max(scale * scale, 1.0):
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class MetricReport:
    """一次评估的完整指标报告（micro + per-object 明细）。"""

    n_samples: int
    y_floor: float
    metrics: dict[str, float | None]  # micro 口径
    mape_valid_fraction: float | None
    undefined: dict[str, str] = field(default_factory=dict)  # 指标 → 未定义原因
    per_object: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "y_floor": self.y_floor,
            "metrics": self.metrics,
            "mape_valid_fraction": self.mape_valid_fraction,
            "undefined": self.undefined,
            "per_object": self.per_object,
        }


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric_names: Sequence[str] = METRIC_NAMES,
    *,
    y_floor: float | None = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    object_ids: Sequence[str] | None = None,
) -> MetricReport:
    """在一组预测结果上计算指标（唯一入口，§5.3）。

    - `y_floor=None` 时取 `0.05 * max(|y|)`（[草案] 默认值）。
    - MAPE 有效样本比例 < `min_valid_fraction` 报 TFX-905（§5.1）。
    - `object_ids` 提供时附 per-object 明细（micro 为主口径）。
    """
    y = _as_array(y_true, "y_true")
    yhat = _as_array(y_pred, "y_pred")
    if len(y) != len(yhat):
        raise ValueError("y_true 与 y_pred 长度不一致")
    if len(y) == 0:
        raise ResearchError("TFX-905", "指标计算需要至少一个样本")
    for name in metric_names:
        if name not in METRIC_NAMES:
            raise ValueError(f"未登记的指标: {name!r}（允许 {METRIC_NAMES}）")

    floor = float(y_floor) if y_floor is not None else DEFAULT_Y_FLOOR_RATIO * float(
        np.max(np.abs(y))
    )
    values: dict[str, float | None] = {}
    undefined: dict[str, str] = {}
    valid_fraction: float | None = None

    for name in metric_names:
        if name == "RMSE":
            values[name] = rmse(y, yhat)
        elif name == "MAE":
            values[name] = mae(y, yhat)
        elif name == "MAPE":
            value, valid_fraction = mape(y, yhat, floor)
            if valid_fraction < min_valid_fraction:
                raise ResearchError(
                    "TFX-905",
                    f"MAPE 有效样本比例 {valid_fraction:.3f} < {min_valid_fraction}"
                    f"（|y| >= y_floor={floor} 的样本不足，§5.1）",
                )
            values[name] = value
            if value is None:
                undefined[name] = "无 |y| >= y_floor 的样本"
        elif name == "CVRMSE":
            values[name] = cvrmse(y, yhat)
            if values[name] is None:
                undefined[name] = "mean(y) ≈ 0，CVRMSE 未定义（§5）"
        elif name == "NMBE":
            values[name] = nmbe(y, yhat)
            if values[name] is None:
                undefined[name] = "mean(y) ≈ 0，NMBE 未定义（§5）"
        elif name == "R2":
            values[name] = r2(y, yhat)
            if values[name] is None:
                undefined[name] = "var(y) ≈ 0，R² 未定义（目标为常数）"

    per_object: dict[str, dict[str, Any]] = {}
    if object_ids is not None:
        objs = np.asarray(object_ids)
        if len(objs) != len(y):
            raise ValueError("object_ids 长度与 y_true 不一致")
        for obj in sorted(set(objs.tolist())):
            mask = objs == obj
            sub = compute_metrics(
                y[mask].tolist(), yhat[mask].tolist(), metric_names,
                y_floor=floor, min_valid_fraction=min_valid_fraction,
            )
            per_object[str(obj)] = sub.to_dict()

    return MetricReport(
        n_samples=len(y),
        y_floor=floor,
        metrics=values,
        mape_valid_fraction=valid_fraction,
        undefined=undefined,
        per_object=per_object,
    )


def aggregate_fold_metrics(reports: Sequence[MetricReport]) -> dict[str, Any]:
    """跨 fold 指标聚合（滚动原点 CV）：mean / std / min / max。

    - 逐 fold 报告必须来自 `compute_metrics`（同一实现，micro 默认、
      y_floor 规则一致、NMBE 必报），本函数不做任何重算。
    - std 为总体标准差（ddof=0），纯算术、无随机源，结果确定。
    - 某 fold 未定义（None）的指标按 fold 跳过，并记录 `n_undefined`。
    """
    if not reports:
        raise ValueError("reports 不能为空")
    names: list[str] = []
    for report in reports:
        for name in report.metrics:
            if name not in names:
                names.append(name)
    per_metric: dict[str, Any] = {}
    for name in sorted(names):
        values = [r.metrics[name] for r in reports
                  if r.metrics.get(name) is not None]
        n_undefined = len(reports) - len(values)
        if values:
            arr = np.asarray(values, dtype=np.float64)
            per_metric[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n_defined": len(values),
                "n_undefined": n_undefined,
            }
        else:
            per_metric[name] = {
                "mean": None, "std": None, "min": None, "max": None,
                "n_defined": 0, "n_undefined": n_undefined,
            }
    return {
        "n_folds": len(reports),
        "total_eval_samples": sum(r.n_samples for r in reports),
        "per_metric": per_metric,
    }
