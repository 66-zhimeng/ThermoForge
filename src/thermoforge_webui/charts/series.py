"""绘图数据准备：抽稀、残差、切分区间、指标对比矩阵。

这一层是纯函数 + 纯数据结构，不 import 任何绘图库，因此可以直接被
pytest 测——图好不好看要人看，但「残差算对没有、区间切在哪」必须能自动
验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..services.experiments import ExperimentDetail, ExperimentSummary

# 预测时序图的点数上限。测试面一般一两千点，训练面上万，浏览器会卡。
PREDICTION_MAX_POINTS = 3000
DEFAULT_HIST_BINS = 40


@dataclass(frozen=True)
class PredictionSeries:
    """一个面（可再按对象过滤）的预测点。"""

    frame: pd.DataFrame  # timestamp / y_true / y_pred / residual
    surface: str
    object_ids: list[str]
    total_points: int
    stride: int

    @property
    def downsampled(self) -> bool:
        return self.stride > 1

    @property
    def empty(self) -> bool:
        return self.frame.empty

    @property
    def caption(self) -> str:
        text = f"{self.surface} 面 · {self.total_points} 个点"
        if self.downsampled:
            text += f"（图上每 {self.stride} 点取 1，共 {len(self.frame)} 点）"
        if len(self.object_ids) > 1:
            text += f" · {len(self.object_ids)} 个对象合并"
        return text


def prepare_predictions(
    predictions: pd.DataFrame | None,
    surface: str,
    *,
    object_id: str | None = None,
    max_points: int = PREDICTION_MAX_POINTS,
) -> PredictionSeries:
    """筛面 → 筛对象 → 按时间排序 → 抽稀 → 算残差。

    抽稀等间隔取点，不做平均：平均会把预测跟不上的尖峰抹平，而那正是
    要在图上看见的东西。
    """
    empty = pd.DataFrame(columns=["timestamp", "y_true", "y_pred", "residual"])
    if predictions is None or predictions.empty:
        return PredictionSeries(empty, surface, [], 0, 1)
    frame = predictions[predictions["surface"] == surface]
    if object_id:
        frame = frame[frame["object_id"] == object_id]
    if frame.empty:
        return PredictionSeries(empty, surface, [], 0, 1)

    objects = sorted(str(o) for o in frame["object_id"].unique())
    frame = frame.sort_values("timestamp")
    total = len(frame)
    stride = max(1, -(-total // max_points))
    if stride > 1:
        frame = frame.iloc[::stride]
    frame = frame[["timestamp", "y_true", "y_pred"]].copy()
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    return PredictionSeries(frame.reset_index(drop=True), surface, objects,
                            total, stride)


@dataclass(frozen=True)
class ResidualStats:
    """残差分布：直方图 + 关键分位。用全量残差算，不受时序图抽稀影响。"""

    values: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    mean: float
    std: float
    p05: float
    p95: float

    @property
    def caption(self) -> str:
        return (f"均值 {self.mean:,.2f} · 标准差 {self.std:,.2f} · "
                f"90% 区间 [{self.p05:,.2f}, {self.p95:,.2f}]")


def prepare_residuals(predictions: pd.DataFrame | None, surface: str,
                      *, object_id: str | None = None,
                      bins: int = DEFAULT_HIST_BINS) -> ResidualStats | None:
    """残差统计。注意用**全量**点，不用抽稀后的——分布必须是真的。"""
    if predictions is None or predictions.empty:
        return None
    frame = predictions[predictions["surface"] == surface]
    if object_id:
        frame = frame[frame["object_id"] == object_id]
    if frame.empty:
        return None
    values = (frame["y_true"] - frame["y_pred"]).to_numpy(dtype=float)
    counts, edges = np.histogram(values, bins=bins, range=_hist_range(values))
    return ResidualStats(
        values=values, counts=counts, edges=edges,
        mean=float(np.mean(values)), std=float(np.std(values)),
        p05=float(np.percentile(values, 5)),
        p95=float(np.percentile(values, 95)),
    )


def _hist_range(values: np.ndarray) -> tuple[float, float]:
    """直方图范围。

    残差恒定时（模型输出与实测差一个常数偏移）`np.histogram` 会因为
    「区间宽度为 0」直接抛错。给它一个围绕该值的小区间，让图能画出来
    ——一根柱子恰恰说明「误差是纯偏置」，是有信息的画面。

    判据用**相对宽度**而不是 `high == low`：浮点减法会让本该恒定的残差
    差出 1e-14 量级的噪声，宽度虽非零但仍分不出 40 个不同的边界。
    """
    low = float(np.min(values))
    high = float(np.max(values))
    scale = max(abs(low), abs(high), 1.0)
    if high - low > 1e-9 * scale:
        return low, high
    margin = 0.01 * scale
    return low - margin, low + margin


@dataclass(frozen=True)
class ScatterFit:
    """实测-预测散点 + 45° 参考线的端点。"""

    y_true: np.ndarray
    y_pred: np.ndarray
    low: float
    high: float


def prepare_scatter(series: PredictionSeries) -> ScatterFit | None:
    if series.empty:
        return None
    y_true = series.frame["y_true"].to_numpy(dtype=float)
    y_pred = series.frame["y_pred"].to_numpy(dtype=float)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    margin = (high - low) * 0.02 or 1.0
    return ScatterFit(y_true, y_pred, low - margin, high + margin)


@dataclass(frozen=True)
class SplitSegment:
    name: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    count: int | None


@dataclass(frozen=True)
class SplitTimeline:
    """时间切分示意：三段 + 中间的 purge/embargo 间隙。"""

    segments: list[SplitSegment]
    purge_seconds: float
    embargo_seconds: float
    dropped: int
    resolution_seconds: float | None
    boundaries: list[pd.Timestamp] = field(default_factory=list)

    @property
    def caption(self) -> str:
        gap = self.purge_seconds + self.embargo_seconds
        text = (f"purge {self.purge_seconds:.0f}s + embargo "
                f"{self.embargo_seconds:.0f}s = 边界间隙 {gap / 60:.0f} 分钟")
        if self.dropped:
            text += f" · 丢弃 {self.dropped} 个跨界样本"
        return text


_SEGMENT_LABELS = {"train": "训练", "validate": "验证", "test": "测试"}


def prepare_split(split: dict[str, Any]) -> SplitTimeline | None:
    """从 split.json 还原时间轴。没有 range 就画不出来，返回 None。"""
    if not split:
        return None
    counts = split.get("counts") or {}
    segments: list[SplitSegment] = []
    for name in ("train", "validate", "test"):
        span = split.get(f"{name}_range")
        if not span or len(span) != 2 or not span[0] or not span[1]:
            continue
        segments.append(SplitSegment(
            name=name, label=_SEGMENT_LABELS[name],
            start=pd.Timestamp(span[0]), end=pd.Timestamp(span[1]),
            count=counts.get(name),
        ))
    if not segments:
        return None
    boundaries = [pd.Timestamp(split[key]) for key in ("b1", "b2")
                  if split.get(key)]
    return SplitTimeline(
        segments=segments,
        purge_seconds=float(split.get("purge_seconds") or 0.0),
        embargo_seconds=float(split.get("embargo_seconds") or 0.0),
        dropped=int(counts.get("purged_or_embargoed") or 0),
        resolution_seconds=split.get("resolution_seconds"),
        boundaries=boundaries,
    )


@dataclass(frozen=True)
class MetricBars:
    """多实验单指标对比。`lower_is_better` 决定排序方向和高亮谁。"""

    metric: str
    labels: list[str]
    values: list[float]
    subtitles: list[str]
    lower_is_better: bool
    best_index: int | None


# 越小越好的指标；R² 反过来，越大越好。NMBE 看绝对值，单独处理。
LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE", "CVRMSE"}


def prepare_metric_bars(summaries: Sequence[ExperimentSummary],
                        metric: str) -> MetricBars | None:
    """按指标排好序的对比条。缺该指标的实验直接不进图（不画 0）。"""
    rows = [(item, item.metric(metric)) for item in summaries]
    rows = [(item, value) for item, value in rows if value is not None]
    if not rows:
        return None
    lower_better = metric in LOWER_IS_BETTER
    if metric == "NMBE":  # 偏差看绝对值，越接近 0 越好
        rows.sort(key=lambda pair: abs(float(pair[1])))
        lower_better = True
    else:
        rows.sort(key=lambda pair: float(pair[1]), reverse=not lower_better)
    return MetricBars(
        metric=metric,
        labels=[item.experiment_id for item, _ in rows],
        values=[float(value) for _, value in rows],
        subtitles=[item.model_label for item, _ in rows],
        lower_is_better=lower_better,
        best_index=0 if rows else None,
    )


@dataclass(frozen=True)
class ProgressLine:
    """指标随实验推进的演进（按开始时间排序）。看 AI 有没有真的在变好。"""

    metric: str
    labels: list[str]
    values: list[float]
    best_so_far: list[float]


def prepare_progress(summaries: Sequence[ExperimentSummary],
                     metric: str = "CVRMSE") -> ProgressLine | None:
    rows = [(item, item.metric(metric)) for item in summaries]
    rows = [(item, float(value)) for item, value in rows if value is not None]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda pair: str(pair[0].started_at or ""))
    values = [value for _, value in rows]
    lower_better = metric in LOWER_IS_BETTER
    running: list[float] = []
    for value in values:
        if not running:
            running.append(value)
        elif lower_better:
            running.append(min(running[-1], value))
        else:
            running.append(max(running[-1], value))
    return ProgressLine(metric=metric,
                        labels=[item.experiment_id for item, _ in rows],
                        values=values, best_so_far=running)


@dataclass(frozen=True)
class MissingRates:
    variables: list[str]
    rates: list[float]
    threshold: float = 0.1


def prepare_missing_rates(profile: dict[str, Any],
                          *, top: int = 25) -> MissingRates | None:
    """缺失率排行（只画最高的若干条，宽表变量多了图会挤成一团）。"""
    rows = [(str(v.get("variable_id")), float(v.get("missing_rate") or 0.0))
            for v in profile.get("variables") or []]
    if not rows:
        return None
    rows.sort(key=lambda pair: pair[1], reverse=True)
    rows = rows[:top]
    return MissingRates([name for name, _ in rows], [rate for _, rate in rows])


def objects_in(predictions: pd.DataFrame | None, surface: str) -> list[str]:
    if predictions is None or predictions.empty:
        return []
    frame = predictions[predictions["surface"] == surface]
    return sorted(str(o) for o in frame["object_id"].unique())


def detail_series(detail: ExperimentDetail, predictions: pd.DataFrame | None,
                  surface: str, object_id: str | None = None
                  ) -> tuple[PredictionSeries, ResidualStats | None]:
    """页面常用的一组：时序 + 残差一起取，避免两处重复筛选逻辑。"""
    series = prepare_predictions(predictions, surface, object_id=object_id)
    residuals = prepare_residuals(predictions, surface, object_id=object_id)
    return series, residuals
