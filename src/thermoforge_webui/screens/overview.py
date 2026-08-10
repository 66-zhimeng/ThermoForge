"""总览页：一屏看清「现在到哪了」。

回答四个问题：研究目标进展如何、最近实验跑得怎么样、有没有东西卡着等
人处理、指标有没有在变好。
"""

from __future__ import annotations

import streamlit as st

from .. import cache
from ..charts import interactive, series
from ..services import experiments as exp_service
from ..ui import copilot_banner, empty_state, fmt_metric, fmt_seconds, fmt_time, status_badge

# Ledger 里 goal 初始状态存的是小写 "active"，流转后是大写状态名，
# 两种都要能查到，所以键一律小写、查表前先 lower()
GOAL_STATUS_LABELS = {
    "draft": "草稿",
    "active": "进行中",
    "data_profiling": "数据画像",
    "modelability_assessment": "可建模性门禁",
    "baseline_modeling": "基线建模",
    "hypothesis_generation": "生成假设",
    "experiment_design": "实验设计",
    "experiment_running": "实验运行中",
    "result_analysis": "结果分析",
    "model_review": "模型评审",
    "publish": "已发布",
    "stopped": "已停止",
}

# goal 终态：停止原因面板里显示的那一列
TERMINAL_LABELS = {"PUBLISH": "已发布", "STOPPED": "已停止"}

STOP_REASON_LABELS = {
    "acceptance_met": "达到验收标准",
    "budget_exhausted": "预算耗尽",
    "no_information_gain": "连续无信息增益",
    "insufficient_data_coverage": "数据覆盖不足",
    "missing_required_variables": "缺少必需变量",
    "human_confirmation_required": "需要人工确认",
    "modelability_failed": "可建模性门禁未过",
}


def overview_page() -> None:
    st.title("总览")
    copilot_banner("overview")
    status = cache.status_panel(recent=8)
    _headline(status)
    _pending_actions(status)

    left, right = st.columns([3, 2], gap="large")
    with left:
        _recent_experiments()
    with right:
        _goals(status)
        _stop_reasons(status)


def _headline(status: dict) -> None:
    summaries = cache.experiment_list()
    completed = [item for item in summaries if item.status == "completed"]
    failed = [item for item in summaries if item.status == "failed"]
    columns = st.columns(4)
    columns[0].metric("研究目标", len(status.get("goals") or []))
    columns[1].metric("实验", len(summaries),
                      help=f"完成 {len(completed)} · 失败 {len(failed)}")
    columns[2].metric("生产模型",
                      sum(1 for m in status.get("production_models") or []
                          if m.get("production")))
    best = _best(completed)
    columns[3].metric("最优 CVRMSE",
                      fmt_metric("CVRMSE", best.metric("CVRMSE")) if best else "—",
                      help=f"来自 {best.experiment_id}" if best else "还没有完成的实验")


def _best(summaries: list[exp_service.ExperimentSummary]):
    scored = [item for item in summaries if item.metric("CVRMSE") is not None]
    return min(scored, key=lambda i: float(i.metric("CVRMSE"))) if scored else None


def _pending_actions(status: dict) -> None:
    """卡着等人处理的事优先顶到最上面——它们是整个循环的实际瓶颈。"""
    pending = (status.get("preprocess") or {}).get("pending_rules") or 0
    if pending:
        st.warning(
            f"有 **{pending}** 条预处理规则等待人工审批，规则集不批准就无法"
            "产生新的数据修订版。去「数据质量」页处理。", icon="⏳")


def _recent_experiments() -> None:
    st.subheader("最近实验")
    summaries = cache.experiment_list(limit=8)
    if not summaries:
        empty_state("还没有实验",
                    "去「AI 研究」页让 Agent 跑一轮，或在那里手动建一个实验。")
        return
    for item in summaries:
        with st.container(border=True):
            head, metrics = st.columns([2, 3])
            with head:
                st.markdown(f"**{item.experiment_id}**　{status_badge(item.status)}")
                st.caption(f"{item.model_label}")
                st.caption(f"{fmt_time(item.started_at)} · "
                           f"{fmt_seconds(item.duration_seconds)} · "
                           f"{item.goal_id or '—'}")
            with metrics:
                if item.status == "completed":
                    cells = st.columns(4)
                    for cell, name in zip(cells, ("CVRMSE", "R2", "NMBE", "MAPE")):
                        cell.metric(name, fmt_metric(name, item.metric(name)))
                else:
                    st.error(f"{item.error_code or '—'}　"
                             f"{item.failure_reason or '（无失败原因）'}")

    progress = series.prepare_progress(cache.experiment_list(), "CVRMSE")
    if progress:
        st.plotly_chart(interactive.progress_figure(progress),
                        width="stretch")
        st.caption("虚线是「到此为止的最优」。它长时间走平，说明再试下去"
                   "拿不到新信息——编排器的 no_information_gain 停止条件"
                   "看的就是这个。")


def _goals(status: dict) -> None:
    st.subheader("研究目标")
    goals = status.get("goals") or []
    if not goals:
        empty_state("还没有研究目标", "去「AI 研究」页新建一个。", icon="🎯")
        return
    for goal in goals:
        experiments = goal.get("experiments") or {}
        with st.container(border=True):
            st.markdown(f"**{goal['goal_id']}**　"
                        f"`{GOAL_STATUS_LABELS.get(str(goal['status']).lower(), goal['status'])}`")
            st.caption(goal.get("name") or "—")
            st.caption(f"实验 {experiments.get('total', 0)} · "
                       f"假设 {(goal.get('hypotheses') or {}).get('total', 0)} · "
                       f"模型 {(goal.get('models') or {}).get('total', 0)}")
            st.caption(f"最后活动 {fmt_time(goal.get('last_activity'))}")


def _stop_reasons(status: dict) -> None:
    reasons = status.get("stop_reasons") or {}
    if not reasons:
        return
    st.subheader("目标停止原因")
    for key, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        state, _, reason = key.partition(":")
        label = STOP_REASON_LABELS.get(reason, reason or "—")
        st.caption(f"{TERMINAL_LABELS.get(state, state)}　{label} × {count}")
