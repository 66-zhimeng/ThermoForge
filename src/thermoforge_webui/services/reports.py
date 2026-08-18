"""报告数据组装：把一次研究的结果整理成与输出格式无关的文档模型。

三种导出（HTML / Markdown / PDF）共用这一份文档模型，各自只负责排版。
图用**惰性渲染**：文档里存的是「怎么画」而不是画好的图，HTML 拿 plotly
交互版，Markdown 和 PDF 拿 matplotlib 的 PNG，同一份数据两种画法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from ..charts import interactive, series, static
from . import experiments as exp_service
from . import inspect_model
from .experiments import STATUS_LABELS, SURFACE_LABELS

# 与「数据质量」页同一套中文口径；报告里不出现 PASS/FAIL
VERDICT_LABELS = {"PASS": "通过", "FAIL": "不通过", "UNKNOWN": "未体检"}


@dataclass
class Figure:
    """一张图：两种渲染方式各自惰性求值。"""

    key: str
    title: str
    caption: str
    _plotly: Callable[[], Any]
    _png: Callable[[], bytes]

    def plotly(self):
        return self._plotly()

    def png(self) -> bytes:
        return self._png()


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None
    table_caption: str = ""
    figures: list[Figure] = field(default_factory=list)
    key_values: dict[str, str] = field(default_factory=dict)
    # 主表之外的附加表（如模型参数明细）：（表题, 数据）
    extra_tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


@dataclass
class ReportDocument:
    title: str
    subtitle: str
    generated_at: str
    sections: list[Section] = field(default_factory=list)
    footer: str = ""


METRIC_COLUMNS = ("CVRMSE", "R2", "NMBE", "MAPE", "RMSE", "MAE")


def build_report(experiment_ids: list[str], *, title: str,
                 subtitle: str = "", include_charts: bool = True,
                 quality_report: Any = None,
                 quality_ref: str = "") -> ReportDocument:
    """组装报告。实验按选择顺序出现，最优模型单独点名。"""
    now = datetime.now(timezone.utc).astimezone()
    document = ReportDocument(
        title=title,
        subtitle=subtitle,
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        footer=("由 ThermoForge 控制台生成。指标由 research/metrics.py 单一实现"
                "计算，比率制（非百分比）；R² 缺失的历史实验由预测点现算补齐。"),
    )
    details = [d for d in (exp_service.load_detail(eid) for eid in experiment_ids)
               if d is not None]
    if not details:
        document.sections.append(Section(
            title="没有可用的实验",
            paragraphs=["选中的实验读不到报告文件。"]))
        return document

    summaries = exp_service.enrich_with_r2(
        [item for item in exp_service.list_experiments()
         if item.experiment_id in set(experiment_ids)])

    document.sections.append(_overview_section(details, summaries))
    if quality_report is not None:
        document.sections.append(_quality_section(quality_report, quality_ref))
    document.sections.append(_comparison_section(summaries, include_charts))
    for detail in details:
        document.sections.append(_experiment_section(detail, include_charts))
    document.sections.append(_method_section(details))
    return document


def _overview_section(details: list[exp_service.ExperimentDetail],
                      summaries: list[exp_service.ExperimentSummary]) -> Section:
    goals = sorted({d.report.get("goal_id") for d in details
                    if d.report.get("goal_id")})
    views = sorted({d.report.get("dataset_view") for d in details
                    if d.report.get("dataset_view")})
    targets = sorted({d.target for d in details if d.target})
    best = _best(summaries)
    section = Section(title="概览", key_values={
        "研究目标": "、".join(str(g) for g in goals) or "—",
        "数据视图": "、".join(str(v) for v in views) or "—",
        "建模目标": "、".join(targets) or "—",
        "实验数量": str(len(details)),
        "最优模型": (f"{best.experiment_id}（{best.model_label}）"
                     if best else "—"),
    })
    if best:
        section.paragraphs.append(
            f"在参与对比的 {len(summaries)} 次实验中，**{best.experiment_id}**"
            f"（{best.model_label}）在 {SURFACE_LABELS.get(best.surface or '', best.surface or '')}"
            f"上取得最优 CVRMSE {_fmt(best.metric('CVRMSE'), ratio=True)}，"
            f"R² {_fmt(best.metric('R2'))}，NMBE {_fmt(best.metric('NMBE'), ratio=True)}。")
    return section


def _best(summaries: list[exp_service.ExperimentSummary]):
    scored = [s for s in summaries if s.metric("CVRMSE") is not None]
    return min(scored, key=lambda s: float(s.metric("CVRMSE"))) if scored else None


def _quality_section(report: Any, ref: str) -> Section:
    section = Section(
        title="数据质量",
        key_values={"数据集": ref or "—",
                    "门禁判定": VERDICT_LABELS.get(
                        str(getattr(report, "verdict", "")), "—"),
                    "阻断项": str(len(getattr(report, "blockers", []))),
                    "警告项": str(len(getattr(report, "warnings", [])))},
        paragraphs=[getattr(report, "headline", "")])
    rows = []
    for finding in getattr(report, "findings", []):
        rows.append({
            "级别": finding.severity_label,
            "检查": finding.title,
            "结论": finding.detail.splitlines()[0][:120],
            "处方": (finding.prescriptions[0].text[:120]
                     if finding.prescriptions else "—"),
        })
    if rows:
        section.table = pd.DataFrame(rows)
        section.table_caption = "语义门禁与统计画像的逐项结论"
    return section


def _comparison_section(summaries: list[exp_service.ExperimentSummary],
                        include_charts: bool) -> Section:
    section = Section(title="模型对比")
    section.table = exp_service.comparison_frame(summaries)
    section.table_caption = ("各实验主测试面指标。CVRMSE/NMBE/MAPE 为比率制，"
                             "R² 可为负（负值表示比常数均值预测更差）。")
    if include_charts:
        bars = series.prepare_metric_bars(summaries, "CVRMSE")
        if bars:
            section.figures.append(Figure(
                key="compare_cvrmse", title="CVRMSE 对比",
                caption="越小越好；绿色为当前最优。",
                _plotly=lambda b=bars: interactive.metric_bars_figure(b),
                _png=lambda b=bars: static.metric_bars_png(b)))
        progress = series.prepare_progress(summaries, "CVRMSE")
        if progress:
            section.figures.append(Figure(
                key="progress_cvrmse", title="CVRMSE 随实验推进",
                caption="虚线是「到此为止的最优」，走平意味着新实验不再带来信息增益。",
                _plotly=lambda p=progress: interactive.progress_figure(p),
                _png=lambda p=progress: static.progress_png(p)))
    return section


def _experiment_section(detail: exp_service.ExperimentDetail,
                        include_charts: bool) -> Section:
    report = detail.report
    section = Section(
        title=f"实验 {detail.experiment_id}",
        key_values={
            "模型": detail.model_label,
            "状态": STATUS_LABELS.get(detail.status, detail.status),
            "假设": str(report.get("hypothesis_id") or "—"),
            "数据视图": str(report.get("dataset_view") or "—"),
            "随机种子": str(report.get("random_seed") or "—"),
            "耗时": f"{report.get('duration_seconds', 0):.1f}s",
        })
    if detail.status != "completed":
        section.paragraphs.append(
            f"实验未完成：`{report.get('error_code') or '—'}` "
            f"{report.get('failure_reason') or ''}")
        return section

    rows = []
    for name, surface in sorted(detail.surfaces.items()):
        metrics = surface.get("metrics") or {}
        rows.append({
            "面": SURFACE_LABELS.get(name, name),
            "样本": surface.get("n_samples"),
            **{key: metrics.get(key) for key in METRIC_COLUMNS},
        })
    if rows:
        section.table = pd.DataFrame(rows)
        section.table_caption = "各评估面指标"
    if detail.r2_backfilled:
        section.paragraphs.append(
            "> 这次实验跑在 R² 进入指标登记表之前，表中 R² 由 "
            "`predictions.parquet` 用同一实现现算补齐，未改写历史产物。")

    physics = detail.physics or {}
    if physics.get("overall_rate") is not None:
        section.paragraphs.append(
            f"物理一致性检查：违规率 {physics['overall_rate'] * 100:.2f}%"
            f"（{physics.get('overall_violations', 0)} / "
            f"{physics.get('n_samples', 0)} 样本）。")

    _attach_model_structure(section, detail, include_charts)
    if include_charts:
        section.figures.extend(_experiment_figures(detail))
    return section


def _attach_model_structure(section: Section,
                            detail: exp_service.ExperimentDetail,
                            include_charts: bool) -> None:
    """把「模型内部是什么」写进实验小节：公式（等宽文本）、参数表、结构图。

    与页面「模型结构」区共用 inspect_model 的解析结果；失败实验没有
    model/ 目录时安静跳过。公式用等宽纯文本而不是 LaTeX——Markdown/PDF
    管线没有 LaTeX 渲染，且与 params.yaml 的明文同口径。
    """
    doc = inspect_model.load_model_doc(detail.directory / "model")
    if doc is None or doc.kind == "unknown":
        return
    section.paragraphs.append(f"建模方式：{doc.summary}。")
    if doc.equations:
        lines = list(doc.equations)
        if doc.substituted:
            lines.extend(["", "代入辨识参数：", *doc.substituted])
        section.paragraphs.append("```text\n" + "\n".join(lines) + "\n```")
    if doc.params:
        section.extra_tables.append(
            ("模型参数（辨识取值与合法范围）", _params_frame(doc)))
    if include_charts:
        section.figures.extend(_structure_figures(detail.experiment_id, doc))


def _params_frame(doc: inspect_model.ModelDoc) -> pd.DataFrame:
    rows = [{"参数": p.name,
             "取值": p.value,
             "单位": p.unit or "—",
             "合法范围": (f"[{p.bounds[0]:.4g}, {p.bounds[1]:.4g}]"
                        if p.bounds else "—"),
             "备注": p.note or "—"}
            for p in doc.params]
    return pd.DataFrame(rows)


def _structure_figures(experiment_id: str,
                       doc: inspect_model.ModelDoc) -> list[Figure]:
    figures: list[Figure] = []
    if doc.kind == "hybrid":
        nodes, edges = inspect_model.hybrid_flow_spec(doc)
        figures.append(Figure(
            key=f"{experiment_id}_flow", title="组合结构",
            caption="物理主干给出可解析的 baseline，XGBoost 只修正残差。",
            _plotly=lambda n=nodes, e=edges: interactive.structure_flow_figure(n, e),
            _png=lambda n=nodes, e=edges: static.structure_flow_png(n, e)))
    importance = (doc.xgb or {}).get("importance") or {}
    if doc.kind in ("hybrid", "lab") and importance:
        bars = series.prepare_coef_bars(importance, unit="%")
        title = "残差特征重要性" if doc.kind == "hybrid" else "特征重要性"
        if bars:
            figures.append(Figure(
                key=f"{experiment_id}_importance", title=title,
                caption="XGBoost gain 占比，前 12 位。",
                _plotly=lambda b=bars, t=title: interactive.coef_bars_figure(
                    b, title=t),
                _png=lambda b=bars, t=title: static.coef_bars_png(
                    b, title=t)))
    if doc.kind == "linear" and doc.coefficients:
        bars = series.prepare_coef_bars(doc.coefficients, unit="每标准差")
        if bars:
            figures.append(Figure(
                key=f"{experiment_id}_coef", title="标准化特征系数",
                caption="作用在标准化特征上，|w| 可直接比大小。",
                _plotly=lambda b=bars: interactive.coef_bars_figure(
                    b, title="标准化特征系数"),
                _png=lambda b=bars: static.coef_bars_png(
                    b, title="标准化特征系数")))
    return figures


def _experiment_figures(detail: exp_service.ExperimentDetail) -> list[Figure]:
    figures: list[Figure] = []
    predictions = exp_service.load_predictions(detail.experiment_id)
    surfaces = exp_service.surface_options(detail)
    surface = next((s for s in ("A", "C", "validate") if s in surfaces),
                   surfaces[0] if surfaces else None)
    if surface and predictions is not None:
        prediction_series = series.prepare_predictions(predictions, surface)
        residuals = series.prepare_residuals(predictions, surface)
        label = SURFACE_LABELS.get(surface, surface)
        if not prediction_series.empty:
            figures.append(Figure(
                key=f"{detail.experiment_id}_timeseries",
                title=f"预测 vs 实测（{label}）",
                caption=prediction_series.caption,
                _plotly=lambda s=prediction_series: interactive.predictions_figure(
                    s, y_label=detail.target),
                _png=lambda s=prediction_series: static.predictions_png(
                    s, y_label=detail.target)))
            scatter = series.prepare_scatter(prediction_series)
            if scatter:
                figures.append(Figure(
                    key=f"{detail.experiment_id}_scatter",
                    title=f"实测 vs 预测散点（{label}）",
                    caption="点越贴近虚线越好；系统性偏离一侧说明有偏差。",
                    _plotly=lambda s=scatter: interactive.scatter_figure(s),
                    _png=lambda s=scatter: static.scatter_png(s)))
        if residuals:
            figures.append(Figure(
                key=f"{detail.experiment_id}_residual",
                title=f"残差分布（{label}）",
                caption=residuals.caption,
                _plotly=lambda s=residuals: interactive.residual_hist_figure(s),
                _png=lambda s=residuals: static.residual_hist_png(s)))
    timeline = series.prepare_split(detail.split)
    if timeline:
        figures.append(Figure(
            key=f"{detail.experiment_id}_split",
            title="时间切分",
            caption=timeline.caption,
            _plotly=lambda t=timeline: interactive.split_timeline_figure(t),
            _png=lambda t=timeline: static.split_timeline_png(t)))
    return figures


def _method_section(details: list[exp_service.ExperimentDetail]) -> Section:
    locks = sorted({str(d.report.get("environment_lock") or "")[:16]
                    for d in details})
    codes = sorted({str(d.report.get("code_version") or "")[:12]
                    for d in details})
    return Section(
        title="方法与可复现性",
        key_values={"环境锁": "、".join(l for l in locks if l) or "—",
                    "代码版本": "、".join(c for c in codes if c) or "—"},
        paragraphs=[
            "**切分**：按时间而非行数切分，边界向下取整到采样周期整数倍；"
            "训练尾部 purge、验证头部 embargo，防止相邻样本把未来信息漏进训练集。",
            "**执行**：每次实验在独立子进程里跑，线程数环境变量在 numpy/sklearn "
            "导入之前设定，随机种子写进产物。同一台机器 + 同一个环境锁，重跑必须"
            "得到逐位相同的指标。",
            "**指标**：RMSE / MAE / MAPE / CVRMSE / NMBE / R² 全部由 "
            "`research/metrics.py` 单一实现计算。MAPE 剔除 |y| < y_floor 的样本"
            "并报告有效比例；CVRMSE/NMBE 在 mean(y)≈0、R² 在 var(y)≈0 时记为"
            "未定义，而不是给一个看似正常的数字。",
        ])


def _fmt(value: float | None, *, ratio: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if ratio else f"{value:.4f}"
