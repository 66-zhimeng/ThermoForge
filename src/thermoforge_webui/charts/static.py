"""Matplotlib 渲染器：Markdown 与 PDF 报告用的静态 PNG。

与 `interactive.py` 同名函数一一对应，输入同一份 `series.py` 的数据结构，
输出 PNG 字节。中文字体在导入时解析一次：Windows 有 微软雅黑/黑体，
Linux 常见 Noto Sans CJK，都找不到就退回默认字体并在日志里说清楚——
让图出得来但中文变方块，好过整个导出失败。
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前设定

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from . import PALETTE, SURFACE_COLORS  # noqa: E402
from .series import (  # noqa: E402
    MetricBars,
    MissingRates,
    PredictionSeries,
    ProgressLine,
    ResidualStats,
    ScatterFit,
    SplitTimeline,
)

logger = logging.getLogger(__name__)

_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei",
                   "SimSun")
DPI = 130


def _configure_fonts() -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    found = [name for name in _CJK_CANDIDATES if name in available]
    if found:
        plt.rcParams["font.sans-serif"] = [*found, "DejaVu Sans"]
    else:
        logger.warning("未找到中文字体，导出的静态图里中文会显示为方块；"
                       "安装任一字体即可：%s", "、".join(_CJK_CANDIDATES[:3]))
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = PALETTE["grid"]
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return found[0] if found else None


CJK_FONT = _configure_fonts()


def _render(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def predictions_png(series: PredictionSeries, *, y_label: str = "目标值",
                    title: str = "预测 vs 实测") -> bytes:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(series.frame["timestamp"], series.frame["y_true"],
            color=PALETTE["actual"], linewidth=1.1, label="实测")
    ax.plot(series.frame["timestamp"], series.frame["y_pred"],
            color=PALETTE["predicted"], linewidth=1.0, linestyle=":",
            label="预测")
    ax.set_title(title)
    ax.set_xlabel("时间")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right", frameon=False)
    fig.autofmt_xdate()
    return _render(fig)


def scatter_png(scatter: ScatterFit, *, unit: str = "") -> bytes:
    suffix = f"（{unit}）" if unit else ""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(scatter.y_true, scatter.y_pred, s=6, alpha=0.35,
               color=PALETTE["actual"], edgecolors="none")
    ax.plot([scatter.low, scatter.high], [scatter.low, scatter.high],
            color=PALETTE["reference"], linestyle="--", linewidth=1.2,
            label="理想（y=x）")
    ax.set_title("实测 vs 预测")
    ax.set_xlabel(f"实测{suffix}")
    ax.set_ylabel(f"预测{suffix}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False)
    return _render(fig)


def residual_hist_png(stats: ResidualStats, unit: str = "") -> bytes:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    centers = (stats.edges[:-1] + stats.edges[1:]) / 2
    width = float(stats.edges[1] - stats.edges[0])
    ax.bar(centers, stats.counts, width=width, color=PALETTE["residual"])
    ax.axvline(0, color=PALETTE["reference"], linestyle="--", linewidth=1.2)
    ax.axvline(stats.mean, color=PALETTE["bad"], linewidth=1.1)
    suffix = f"（{unit}）" if unit else ""
    ax.set_title("残差分布")
    ax.set_xlabel(f"实测 − 预测{suffix}")
    ax.set_ylabel("样本数")
    return _render(fig)


def split_timeline_png(timeline: SplitTimeline) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 1.9))
    for segment in timeline.segments:
        start = segment.start.to_pydatetime()
        end = segment.end.to_pydatetime()
        ax.barh(0, end - start, left=start, height=0.5,
                color=SURFACE_COLORS.get(segment.name, PALETTE["train"]),
                label=f"{segment.label}（{segment.count or 0}）")
    for boundary in timeline.boundaries:
        ax.axvline(boundary.to_pydatetime(), color=PALETTE["gap"],
                   linewidth=2)
    ax.set_yticks([])
    ax.set_title("时间切分与边界间隙")
    ax.set_xlabel("时间")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncols=3,
              frameon=False)
    fig.autofmt_xdate()
    return _render(fig)


def metric_bars_png(bars: MetricBars) -> bytes:
    height = max(2.2, 0.42 * len(bars.values) + 1.0)
    fig, ax = plt.subplots(figsize=(7.4, height))
    colors = [PALETTE["good"] if i == bars.best_index else PALETTE["actual"]
              for i in range(len(bars.values))]
    positions = range(len(bars.values))
    ax.barh(list(positions), bars.values, color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"{label}\n{sub}" for label, sub
                        in zip(bars.labels, bars.subtitles)], fontsize=8)
    ax.invert_yaxis()
    direction = "越小越好" if bars.lower_is_better else "越大越好"
    ax.set_title(f"{bars.metric} 对比（{direction}）")
    ax.set_xlabel(bars.metric)
    for index, value in enumerate(bars.values):
        ax.text(value, index, f" {value:.4g}", va="center", fontsize=8)
    return _render(fig)


def progress_png(progress: ProgressLine) -> bytes:
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(progress.labels, progress.values, marker="o", markersize=4,
            color=PALETTE["actual"], linewidth=1.4, label="本次实验")
    ax.plot(progress.labels, progress.best_so_far, linestyle="--",
            color=PALETTE["good"], linewidth=1.8, label="历史最优")
    ax.set_title(f"{progress.metric} 随实验推进")
    ax.set_xlabel("实验（按时间）")
    ax.set_ylabel(progress.metric)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.legend(frameon=False)
    return _render(fig)


def missing_rates_png(rates: MissingRates) -> bytes:
    height = max(2.4, 0.26 * len(rates.variables) + 1.0)
    fig, ax = plt.subplots(figsize=(7.4, height))
    colors = [PALETTE["bad"] if rate >= rates.threshold else PALETTE["actual"]
              for rate in rates.rates]
    positions = range(len(rates.variables))
    ax.barh(list(positions), [rate * 100 for rate in rates.rates],
            color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(rates.variables, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("缺失率排行")
    ax.set_xlabel("缺失率（%）")
    return _render(fig)
