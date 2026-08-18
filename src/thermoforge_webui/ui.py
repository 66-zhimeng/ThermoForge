"""页面共用的小组件：指标卡、状态徽章、信封结果展示、数字格式化。

指标格式化集中在这里有个实际原因：CVRMSE/MAPE/NMBE 在 `metrics.py` 里
是**比率**不是百分比，界面上要显示成百分比就必须乘 100，而这个换算散落
在各页面里迟早会漏一处，出现「0.12」和「12%」并排的画面。
"""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

# 比率制指标：显示时乘 100 加 %（conventions：存储一律比率，不存百分数）
RATIO_METRICS = {"CVRMSE", "NMBE", "MAPE"}

METRIC_HELP = {
    "CVRMSE": "变异系数均方根误差 = RMSE / mean(y)。主指标（ASHRAE 口径），越小越好。",
    "R2": "决定系数 = 1 − SS_res/SS_tot。可以为负，负值表示比直接用均值预测还差。",
    "NMBE": "归一化平均偏差。正=系统性低估，负=系统性高估；看绝对值大小。",
    "MAPE": "平均绝对百分比误差，已剔除 |y| < y_floor 的样本。",
    "RMSE": "均方根误差，与目标同量纲。",
    "MAE": "平均绝对误差，与目标同量纲。",
}

STATUS_BADGES = {
    "completed": ("✅", "已完成"),
    "failed": ("❌", "失败"),
    "running": ("⏳", "运行中"),
    "planned": ("📋", "已计划"),
    "PUBLISH": ("🚀", "已发布"),
    "STOPPED": ("⏹", "已停止"),
}


def fmt_metric(name: str, value: float | None) -> str:
    if value is None:
        return "—"
    if name in RATIO_METRICS:
        return f"{value * 100:.2f}%"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f}s"
    return f"{value / 60:.1f}min"


def fmt_time(value: str | None) -> str:
    """时间戳只留到分钟——秒和微秒在界面上是噪声。"""
    if not value:
        return "—"
    return str(value).replace("T", " ")[:16]


def fmt_tokens(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def fmt_cost(value: float | None) -> str:
    """金额（元）。None = 未配置单价，此时界面只显示 token 不算钱。"""
    if value is None:
        return "—"
    return f"¥{value:.4f}"


def usage_caption(usage: Mapping[str, Any] | None,
                  cost: float | None) -> str:
    """一行用量摘要：输入/输出 tokens + 金额（有单价时）。"""
    usage = usage or {}
    parts = [f"输入 {fmt_tokens(usage.get('prompt_tokens'))}",
             f"输出 {fmt_tokens(usage.get('completion_tokens'))} tokens"]
    if cost is not None:
        parts.append(f"约 {fmt_cost(cost)}")
    return " · ".join(parts)


def status_badge(status: str | None) -> str:
    icon, label = STATUS_BADGES.get(str(status), ("•", str(status or "未知")))
    return f"{icon} {label}"


def status_text(status: str | None) -> str:
    """只要中文文字、不要图标的场合（表格单元格、导出的报告）。"""
    return STATUS_BADGES.get(str(status), ("", str(status or "未知")))[1]


def metric_row(metrics: dict[str, float | None],
               names: tuple[str, ...] = ("CVRMSE", "R2", "NMBE", "MAPE",
                                         "RMSE", "MAE")) -> None:
    """一排指标卡。没有值的指标也占位显示 —— 让人看见「这个没算」。"""
    columns = st.columns(len(names))
    for column, name in zip(columns, names):
        column.metric(name, fmt_metric(name, metrics.get(name)),
                      help=METRIC_HELP.get(name))


def envelope_result(envelope: dict[str, Any], *,
                    success: str = "完成") -> bool:
    """统一展示工具信封。

    工具层的约定是**已知错误返回 ok=False 而不是抛异常**，所以这里必须
    显式看 `ok`——只看有没有异常会把失败当成功（CLI 那边同理：exit 0
    不代表工具成功）。
    """
    if envelope.get("ok"):
        identifier = envelope.get("id")
        st.success(f"{success}{f'：{identifier}' if identifier else ''}")
    else:
        st.error(_envelope_error(envelope))
    diagnostics = envelope.get("diagnostics") or []
    if diagnostics:
        with st.expander(f"诊断 {len(diagnostics)} 条", expanded=not envelope.get("ok")):
            for item in diagnostics:
                level = str(item.get("level") or "INFO").upper()
                icon = {"ERROR": "🔴", "WARN": "🟡"}.get(level, "🔵")
                st.write(f"{icon} `{item.get('code', '—')}` {item.get('message', '')}")
    if envelope.get("summary"):
        with st.expander("完整信封"):
            st.json(envelope)
    return bool(envelope.get("ok"))


def _envelope_error(envelope: dict[str, Any]) -> str:
    diagnostics = envelope.get("diagnostics") or []
    errors = [d for d in diagnostics
              if str(d.get("level", "")).upper() == "ERROR"]
    if errors:
        first = errors[0]
        return f"失败（{first.get('code', '—')}）：{first.get('message', '')}"
    summary = envelope.get("summary")
    return f"失败：{summary if summary else '工具返回 ok=false'}"


def caption_list(items: list[str]) -> None:
    for item in items:
        st.caption(item)


def empty_state(title: str, hint: str, icon: str = "📭") -> None:
    """空状态要说清楚「下一步该干什么」，而不是只说「暂无数据」。"""
    st.info(f"{icon} **{title}**\n\n{hint}")


def copilot_banner(page_key: str) -> None:
    """副驾把你带到这一页时，在顶部显示它的结论。

    结论跟着跳转走、显示在**目标页**而不是留在对话框里，是因为它讲的就是
    这一页上的图——两边分开看，等于让人自己在脑子里做对照。
    """
    banner = st.session_state.get("copilot_banner")
    if not banner or banner.get("page") != page_key:
        return
    note = str(banner.get("note") or "").strip()
    if not note:
        return
    with st.container(border=True):
        left, right = st.columns([9, 1])
        left.markdown(f"🤖 **AI 助手**　{note}")
        if right.button("知道了", key=f"dismiss_banner_{page_key}"):
            st.session_state.pop("copilot_banner", None)
            st.rerun()
