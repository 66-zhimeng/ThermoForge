"""Plotly 渲染器：界面上看的图，以及自包含 HTML 报告里内嵌的图。

只做画法，取数与统计在 `series.py`。所有函数返回 `go.Figure`。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import PALETTE, SURFACE_COLORS
from .series import (
    CoefBars,
    MetricBars,
    MissingRates,
    PredictionSeries,
    ProgressLine,
    ResidualStats,
    ScatterFit,
    SplitTimeline,
    UsageBars,
)

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=24, t=48, b=48),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def _finish(fig: go.Figure, title: str, x_label: str, y_label: str,
            height: int = 380) -> go.Figure:
    fig.update_layout(title=title, height=height, **_LAYOUT)
    fig.update_xaxes(title_text=x_label, gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text=y_label, gridcolor=PALETTE["grid"])
    return fig


def predictions_figure(series: PredictionSeries, *, y_label: str = "目标值",
                       title: str = "预测 vs 实测") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.frame["timestamp"], y=series.frame["y_true"],
        name="实测", mode="lines",
        line=dict(color=PALETTE["actual"], width=1.6)))
    fig.add_trace(go.Scatter(
        x=series.frame["timestamp"], y=series.frame["y_pred"],
        name="预测", mode="lines",
        line=dict(color=PALETTE["predicted"], width=1.4, dash="dot")))
    return _finish(fig, title, "时间", y_label, height=420)


def residual_series_figure(series: PredictionSeries,
                           y_label: str = "残差") -> go.Figure:
    """残差随时间：看误差是不是集中在某个时段（比只看分布信息量大）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.frame["timestamp"], y=series.frame["residual"],
        name="实测 − 预测", mode="lines",
        line=dict(color=PALETTE["residual"], width=1.1)))
    fig.add_hline(y=0, line=dict(color=PALETTE["reference"], width=1))
    return _finish(fig, "残差随时间", "时间", y_label, height=280)


def scatter_figure(scatter: ScatterFit, *, unit: str = "") -> go.Figure:
    suffix = f"（{unit}）" if unit else ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scatter.y_true, y=scatter.y_pred, mode="markers", name="样本",
        marker=dict(color=PALETTE["actual"], size=4, opacity=0.45)))
    fig.add_trace(go.Scatter(
        x=[scatter.low, scatter.high], y=[scatter.low, scatter.high],
        mode="lines", name="理想（y=x）",
        line=dict(color=PALETTE["reference"], width=1.4, dash="dash")))
    fig = _finish(fig, "实测 vs 预测", f"实测{suffix}", f"预测{suffix}",
                  height=420)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def residual_hist_figure(stats: ResidualStats,
                         unit: str = "") -> go.Figure:
    centers = (stats.edges[:-1] + stats.edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=stats.counts, name="样本数",
                         marker=dict(color=PALETTE["residual"]),
                         width=float(stats.edges[1] - stats.edges[0])))
    fig.add_vline(x=0, line=dict(color=PALETTE["reference"], width=1.4,
                                 dash="dash"))
    fig.add_vline(x=stats.mean, line=dict(color=PALETTE["bad"], width=1.2))
    suffix = f"（{unit}）" if unit else ""
    fig = _finish(fig, "残差分布", f"实测 − 预测{suffix}", "样本数", height=320)
    fig.update_layout(hovermode="closest", bargap=0.02)
    return fig


def split_timeline_figure(timeline: SplitTimeline) -> go.Figure:
    """时间切分横条图：一眼看出三段的位置、长度，以及边界间隙。"""
    fig = go.Figure()
    for segment in timeline.segments:
        # plotly 日期轴的内部表示是「epoch 毫秒数」：x 直接给 Timedelta 会把
        # 整条轴渲染成时长（P255DT12H… 这种），必须换成毫秒再声明 type=date
        base_ms = segment.start.value / 1e6  # Timestamp.value 是纳秒
        width_ms = (segment.end - segment.start).total_seconds() * 1000.0
        fig.add_trace(go.Bar(
            x=[width_ms], y=["切分"], base=[base_ms],
            orientation="h", name=f"{segment.label}（{segment.count or 0}）",
            marker=dict(color=SURFACE_COLORS.get(
                segment.name, PALETTE["train"])),
            hovertemplate=(f"{segment.label}<br>"
                           f"{segment.start:%Y-%m-%d %H:%M} → "
                           f"{segment.end:%Y-%m-%d %H:%M}"
                           f"<br>{segment.count or 0} 个样本<extra></extra>")))
    for boundary in timeline.boundaries:
        fig.add_vline(x=boundary.value / 1e6,
                      line=dict(color=PALETTE["gap"], width=2))
    fig.update_layout(barmode="stack", title="时间切分与边界间隙",
                      height=220, showlegend=True, **{
                          k: v for k, v in _LAYOUT.items()
                          if k not in ("hovermode",)})
    fig.update_xaxes(title_text="时间", gridcolor=PALETTE["grid"], type="date")
    fig.update_yaxes(title_text="", showticklabels=False)
    return fig


def metric_bars_figure(bars: MetricBars) -> go.Figure:
    colors = [PALETTE["good"] if i == bars.best_index else PALETTE["actual"]
              for i in range(len(bars.values))]
    fig = go.Figure(go.Bar(
        x=bars.values, y=bars.labels, orientation="h",
        marker=dict(color=colors), text=[f"{v:.4g}" for v in bars.values],
        textposition="outside",
        customdata=bars.subtitles,
        hovertemplate="%{y}<br>%{customdata}<br>"
                      f"{bars.metric}=%{{x:.5g}}<extra></extra>"))
    direction = "越小越好" if bars.lower_is_better else "越大越好"
    fig = _finish(fig, f"{bars.metric} 对比（{direction}）", bars.metric, "",
                  height=max(240, 40 * len(bars.values) + 120))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def progress_figure(progress: ProgressLine) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=progress.labels, y=progress.values, mode="lines+markers",
        name="本次实验", line=dict(color=PALETTE["actual"], width=1.6)))
    fig.add_trace(go.Scatter(
        x=progress.labels, y=progress.best_so_far, mode="lines",
        name="历史最优", line=dict(color=PALETTE["good"], width=2,
                                   dash="dash")))
    return _finish(fig, f"{progress.metric} 随实验推进", "实验（按时间）",
                   progress.metric, height=320)


def missing_rates_figure(rates: MissingRates) -> go.Figure:
    colors = [PALETTE["bad"] if rate >= rates.threshold else PALETTE["actual"]
              for rate in rates.rates]
    fig = go.Figure(go.Bar(
        x=[rate * 100 for rate in rates.rates], y=rates.variables,
        orientation="h", marker=dict(color=colors),
        text=[f"{rate * 100:.2f}%" for rate in rates.rates],
        textposition="outside"))
    fig = _finish(fig, "缺失率排行", "缺失率（%）", "",
                  height=max(260, 22 * len(rates.variables) + 120))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def raw_series_figure(frame: pd.DataFrame, variable_ids: list[str],
                      *, title: str = "原始时序") -> go.Figure:
    """多变量原始时序。量纲不同的变量放一起会互相压扁，所以分配到左右轴：
    第一条走左轴，其余同量级的跟随，明显不同量级的走右轴。"""
    fig = go.Figure()
    palette = ["#2f6feb", "#e8710a", "#0b7a48", "#6f42c1", "#b3261e",
               "#0891b2", "#a16207", "#be185d"]
    scales = _axis_assignment(frame, variable_ids)
    for index, variable in enumerate(variable_ids):
        if variable not in frame.columns:
            continue
        fig.add_trace(go.Scatter(
            x=frame["timestamp"], y=frame[variable], name=variable,
            mode="lines", line=dict(width=1.2,
                                    color=palette[index % len(palette)]),
            yaxis="y2" if scales.get(variable) == "right" else "y"))
    fig.update_layout(title=title, height=420,
                      yaxis2=dict(overlaying="y", side="right",
                                  showgrid=False, title="右轴"),
                      **_LAYOUT)
    fig.update_xaxes(title_text="时间", gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="左轴", gridcolor=PALETTE["grid"])
    return fig


def _axis_assignment(frame: pd.DataFrame,
                     variable_ids: list[str]) -> dict[str, str]:
    """量级差 100 倍以上的变量甩到右轴，否则小量纲的曲线会被压成一条线。"""
    magnitudes: dict[str, float] = {}
    for variable in variable_ids:
        if variable not in frame.columns:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce").abs()
        scale = float(values.quantile(0.9)) if values.notna().any() else 0.0
        magnitudes[variable] = scale if np.isfinite(scale) else 0.0
    if not magnitudes:
        return {}
    reference = max(magnitudes.values()) or 1.0
    return {name: ("right" if scale > 0 and reference / scale >= 100 else "left")
            for name, scale in magnitudes.items()}


def usage_figure(bars: UsageBars, *,
                 title: str = "每次模型调用的 token 用量",
                 height: int = 340) -> go.Figure:
    """输入/输出 tokens 堆叠柱 + 累计金额折线（右轴，配了单价才画）。"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bars.labels, y=bars.prompt_tokens,
                         name="输入 tokens", marker_color=PALETTE["actual"]))
    fig.add_trace(go.Bar(x=bars.labels, y=bars.completion_tokens,
                         name="输出 tokens", marker_color=PALETTE["predicted"]))
    if bars.cumulative_cost is not None:
        fig.add_trace(go.Scatter(
            x=bars.labels, y=bars.cumulative_cost,
            name="累计金额（元）", mode="lines+markers", yaxis="y2",
            line=dict(color=PALETTE["good"], width=1.8)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                      title_text="元", showgrid=False))
    fig.update_layout(barmode="stack")
    return _finish(fig, title, "模型调用", "tokens", height=height)


def coef_bars_figure(bars: CoefBars, *, title: str = "系数") -> go.Figure:
    """横向条形图：正绿负红，第一个标签在顶部（阅读顺序即特征顺序）。"""
    colors = [PALETTE["good"] if v >= 0 else PALETTE["bad"]
              for v in bars.values]
    fig = go.Figure(go.Bar(x=bars.values, y=bars.labels, orientation="h",
                           marker_color=colors))
    fig = _finish(fig, title, bars.unit or "取值", "",
                  height=max(240, 26 * len(bars.labels) + 140))
    fig.update_yaxes(autorange="reversed")
    return fig


def structure_flow_figure(nodes: Sequence[tuple[float, float, str]],
                          edges: Sequence[tuple[int, int]], *,
                          title: str = "模型结构",
                          height: int = 260) -> go.Figure:
    """组合结构框图：nodes 为 (x, y, 文本)，edges 为 (起, 止) 下标对。

    用 annotation 画框和箭头而不是 scatter——纯说明图，不需要坐标轴。
    """
    fig = go.Figure()
    for a, b in edges:
        fig.add_annotation(x=nodes[b][0], y=nodes[b][1],
                           ax=nodes[a][0], ay=nodes[a][1],
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.5,
                           arrowcolor=PALETTE["reference"], arrowwidth=1.6)
    for x, y, label in nodes:
        fig.add_annotation(x=x, y=y, text=label.replace("\n", "<br>"),
                           showarrow=False, font=dict(size=13),
                           bgcolor="#ffffff", bordercolor=PALETTE["actual"],
                           borderwidth=1.4, borderpad=8)
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    fig.update_xaxes(visible=False, range=[min(xs) - 1.0, max(xs) + 1.0])
    fig.update_yaxes(visible=False, range=[min(ys) - 0.9, max(ys) + 0.9])
    fig.update_layout(title=title, height=height, **_LAYOUT)
    return fig
