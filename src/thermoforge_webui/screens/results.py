"""实验结果页：单实验详情 + 多实验对比。

这一页是「跑完看不到结果」的正面回答：指标卡、预测/实测时序、散点、
残差分布、切分时间轴、物理检查、per-object 拆解、原始预测点下载，
都不需要再问 Agent。
"""

from __future__ import annotations

import streamlit as st

from .. import cache
from ..charts import interactive, series
from ..services import experiments as exp_service
from ..services.experiments import SURFACE_LABELS
from .structure import render_model_structure
from ..ui import (
    copilot_banner,
    empty_state,
    fmt_metric,
    fmt_seconds,
    fmt_time,
    metric_row,
    status_badge,
)

COMPARE_METRICS = ("CVRMSE", "R2", "NMBE", "MAPE", "RMSE", "MAE")


def results_page() -> None:
    st.title("实验结果")
    copilot_banner("results")
    summaries = cache.experiment_list()
    if not summaries:
        empty_state("还没有实验结果",
                    "去「AI 研究」页跑一轮，回来这里就能看到图和指标。",
                    icon="📈")
        return
    tab_single, tab_compare = st.tabs(["单实验详情", "多实验对比"])
    with tab_single:
        _single(summaries)
    with tab_compare:
        _compare(summaries)


# ---------------------------------------------------------------- 单实验


def _single(summaries: list[exp_service.ExperimentSummary]) -> None:
    labels = {item.experiment_id:
              f"{item.experiment_id} · {item.model_label} · "
              f"{status_badge(item.status)}" for item in summaries}
    experiment_id = st.selectbox("实验", options=list(labels),
                                 format_func=lambda k: labels[k],
                                 key="results_exp")
    detail = cache.experiment_detail(str(experiment_id))
    if detail is None:
        st.error("读不到这个实验的报告文件。")
        return

    _header(detail)
    if detail.status != "completed":
        _failure(detail)
        return

    _structure_block(detail)
    surfaces = exp_service.surface_options(detail)
    if not surfaces:
        st.warning("实验完成了但没有任何评估面，无法出图。")
        return
    default = next((s for s in ("A", "C", "validate") if s in surfaces), surfaces[0])
    columns = st.columns([2, 2, 3])
    surface = columns[0].selectbox(
        "评估面", surfaces, index=surfaces.index(default),
        format_func=lambda s: SURFACE_LABELS.get(s, s), key="results_surface")

    predictions = cache.predictions(detail.experiment_id)
    objects = series.objects_in(predictions, str(surface))
    object_id = None
    if len(objects) > 1:
        picked = columns[1].selectbox("对象", ["（全部合并）", *objects],
                                      key="results_object")
        object_id = None if picked == "（全部合并）" else str(picked)

    surface_report = detail.surfaces.get(str(surface)) or {}
    _metrics_block(detail, surface_report, object_id)
    _charts(detail, predictions, str(surface), object_id)
    _split_block(detail)
    _physics_block(detail)
    _reproducibility(detail)
    _raw_points(detail, predictions, str(surface))


def _header(detail: exp_service.ExperimentDetail) -> None:
    report = detail.report
    st.markdown(f"### {detail.experiment_id}　{status_badge(detail.status)}")
    columns = st.columns(5)
    columns[0].caption(f"**模型**\n\n{detail.model_label}")
    columns[1].caption(f"**目标**\n\n`{detail.target or '—'}`")
    columns[2].caption(f"**数据视图**\n\n`{report.get('dataset_view') or '—'}`")
    columns[3].caption(f"**假设**\n\n`{report.get('hypothesis_id') or '—'}`")
    columns[4].caption(f"**耗时**\n\n{fmt_seconds(report.get('duration_seconds'))}"
                       f"　{fmt_time(report.get('started_at'))}")


def _failure(detail: exp_service.ExperimentDetail) -> None:
    report = detail.report
    st.error(f"**{report.get('error_code') or '未知错误'}**　"
             f"{report.get('failure_reason') or report.get('error') or ''}")
    stderr = detail.directory / "stderr.log"
    if stderr.is_file():
        text = stderr.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            with st.expander("子进程 stderr"):
                st.code(text[-8000:], language="text")


def _structure_block(detail: exp_service.ExperimentDetail) -> None:
    """模型结构：建模方式、方程与辨识参数——「这个模型到底是什么」。"""
    if not (detail.directory / "model").is_dir():
        return
    st.markdown("#### 模型结构")
    render_model_structure(detail.directory / "model",
                           key_prefix=f"results_{detail.experiment_id}")


def _metrics_block(detail: exp_service.ExperimentDetail,
                   surface_report: dict, object_id: str | None) -> None:
    source = surface_report
    if object_id:
        source = (surface_report.get("per_object") or {}).get(object_id, {})
    metrics = dict(source.get("metrics") or {})
    metric_row(metrics)

    notes: list[str] = []
    n_samples = source.get("n_samples")
    if n_samples:
        notes.append(f"样本 {n_samples:,}")
    fraction = source.get("mape_valid_fraction")
    if fraction is not None:
        notes.append(f"MAPE 有效样本 {fraction * 100:.1f}%"
                     f"（y_floor={source.get('y_floor')}）")
    if detail.r2_backfilled:
        notes.append("R² 由预测点现算补齐（该实验跑在 I-54 之前，"
                     "产物里没有记录 R²）")
    undefined = source.get("undefined") or {}
    for name, reason in undefined.items():
        notes.append(f"{name} 未定义：{reason}")
    if notes:
        st.caption("　·　".join(notes))


def _charts(detail: exp_service.ExperimentDetail, predictions,
            surface: str, object_id: str | None) -> None:
    if predictions is None:
        st.info("这个实验没有留下 predictions.parquet，画不了图。")
        return
    prediction_series, residuals = series.detail_series(
        detail, predictions, surface, object_id)
    if prediction_series.empty:
        st.info("这个面（或这个对象）没有预测点。")
        return

    unit = _target_unit(detail)
    st.plotly_chart(
        interactive.predictions_figure(
            prediction_series, y_label=f"{detail.target}{f'（{unit}）' if unit else ''}"),
        width="stretch")
    st.caption(prediction_series.caption)

    left, right = st.columns(2)
    scatter = series.prepare_scatter(prediction_series)
    if scatter:
        left.plotly_chart(interactive.scatter_figure(scatter, unit=unit),
                          width="stretch")
    if residuals:
        right.plotly_chart(interactive.residual_hist_figure(residuals, unit),
                           width="stretch")
        right.caption(residuals.caption + "　（用全量点统计，不受上图抽稀影响）")
    st.plotly_chart(interactive.residual_series_figure(prediction_series),
                    width="stretch")


def _target_unit(detail: exp_service.ExperimentDetail) -> str:
    """目标的单位：从 spec 里的视图定义拿不到就留空，不猜。"""
    view = (detail.spec.get("view_definition") or {})
    return str(view.get("target_unit") or "")


def _split_block(detail: exp_service.ExperimentDetail) -> None:
    timeline = series.prepare_split(detail.split)
    if timeline is None:
        return
    st.markdown("#### 时间切分")
    st.plotly_chart(interactive.split_timeline_figure(timeline),
                    width="stretch")
    st.caption(timeline.caption + "　·　切分按时间而非行数，边界向下取整到"
               "采样周期整数倍；训练尾部 purge、验证头部 embargo，"
               "是为了防止相邻样本把未来信息漏进训练集。")


def _physics_block(detail: exp_service.ExperimentDetail) -> None:
    physics = detail.physics
    if not physics:
        return
    st.markdown("#### 物理一致性")
    rate = physics.get("overall_rate")
    columns = st.columns(4)
    columns[0].metric("总违规率",
                      f"{rate * 100:.2f}%" if rate is not None else "—")
    columns[1].metric("违规样本", physics.get("overall_violations", 0))
    columns[2].metric("检查样本", physics.get("n_samples", 0))
    columns[3].metric("冷凝温度代理列", physics.get("condenser_col") or "—")
    constraints = physics.get("hard_constraints") or {}
    monotonicity = physics.get("monotonicity") or {}
    if constraints or monotonicity:
        with st.expander("逐项检查明细"):
            if constraints:
                st.markdown("**硬约束**")
                st.json(constraints)
            if monotonicity:
                st.markdown("**单调性**")
                st.json(monotonicity)
    elif rate == 0:
        st.caption("没有登记具体的物理检查项——总违规率 0 只说明「没有触发"
                   "任何已启用的检查」，不等于「物理上一定对」。")
    if physics.get("note"):
        st.caption(physics["note"])


def _reproducibility(detail: exp_service.ExperimentDetail) -> None:
    report = detail.report
    with st.expander("可复现性信息"):
        st.caption(
            "同一台机器 + 同一个环境锁，重跑必须得到逐位相同的指标。"
            "线程数环境变量在 numpy/sklearn 导入之前就设好了，随机种子"
            "写进产物；环境锁对不上会直接报 TFX-901。")
        columns = st.columns(2)
        columns[0].code(
            f"随机种子　　{report.get('random_seed')}\n"
            f"环境锁　　　{str(report.get('environment_lock'))[:32]}…\n"
            f"代码版本　　{str(report.get('code_version'))[:12]}",
            language="text")
        environment = detail.environment
        if environment:
            columns[1].json(environment, expanded=False)


def _raw_points(detail: exp_service.ExperimentDetail, predictions,
                surface: str) -> None:
    if predictions is None:
        return
    subset = predictions[predictions["surface"] == surface]
    with st.expander(f"原始预测点（{len(subset):,} 行）"):
        st.dataframe(subset.head(500), width="stretch",
                     hide_index=True)
        st.download_button(
            "下载该面全部预测点 CSV",
            subset.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{detail.experiment_id}_{surface}_predictions.csv",
            mime="text/csv")


# ---------------------------------------------------------------- 对比


def _compare(summaries: list[exp_service.ExperimentSummary]) -> None:
    completed = [item for item in summaries if item.status == "completed"]
    if not completed:
        st.info("还没有跑完的实验可以对比。")
        return

    goals = sorted({item.goal_id for item in completed if item.goal_id})
    scope = st.selectbox("范围", ["（全部实验）", *goals], key="compare_goal")
    pool = (completed if scope == "（全部实验）"
            else [item for item in completed if item.goal_id == scope])
    picked = st.multiselect(
        "参与对比的实验", options=[item.experiment_id for item in pool],
        default=[item.experiment_id for item in pool][:8],
        key="compare_ids", placeholder="请选择…")
    chosen = [item for item in pool if item.experiment_id in picked]
    if not chosen:
        st.caption("选至少一个实验。")
        return

    with st.spinner("补齐 R²…"):
        chosen = exp_service.enrich_with_r2(chosen)

    metric = st.radio("排序指标", COMPARE_METRICS, horizontal=True,
                      key="compare_metric")
    bars = series.prepare_metric_bars(chosen, str(metric))
    if bars:
        st.plotly_chart(interactive.metric_bars_figure(bars),
                        width="stretch")
        best = bars.labels[bars.best_index or 0]
        st.caption(f"当前指标下最优：**{best}**　"
                   f"（{fmt_metric(bars.metric, bars.values[0])}）。"
                   "注意不同实验若用了不同的评估面，指标不可直接比——"
                   "「面」这一列要一致才有意义。")
    else:
        st.info(f"选中的实验都没有 {metric} 这个指标。")

    frame = exp_service.comparison_frame(chosen)
    st.dataframe(
        frame, width="stretch", hide_index=True,
        column_config={
            "CVRMSE": st.column_config.NumberColumn(format="%.4f"),
            "R2": st.column_config.NumberColumn(format="%.4f"),
            "NMBE": st.column_config.NumberColumn(format="%.4f"),
            "MAPE": st.column_config.NumberColumn(format="%.4f"),
            "物理违规率": st.column_config.NumberColumn(format="%.4f"),
        })
    st.download_button("下载对比表 CSV",
                       frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name="experiment_comparison.csv", mime="text/csv")
