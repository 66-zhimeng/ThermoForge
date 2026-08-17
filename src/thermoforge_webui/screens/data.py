"""数据页：数据集浏览、变量表、画像、原始时序预览。

这一页只回答「数据长什么样」，不做判断——「这份数据能不能拿来建模」在
「数据质量」页，两者刻意分开：前者是事实，后者是带门禁语义的结论。
"""

from __future__ import annotations

import streamlit as st

from .. import cache
from ..charts import interactive, series
from ..services import catalog
from ..ui import copilot_banner, empty_state, fmt_time
from ..envelope import entries

# 默认预览的变量条数：一次画太多曲线看不清，也拖慢浏览器
DEFAULT_PREVIEW_VARIABLES = 3


def data_page() -> None:
    st.title("数据")
    copilot_banner("data")
    revisions = cache.revisions()
    if not revisions:
        empty_state(
            "Vault 里还没有数据集",
            "用 `tf dataset import <工作簿.xlsx>` 导入，或跑一次 "
            "`examples/chiller_power/run_demo.py` 生成演示数据。", icon="🗂")
        return

    ref = _picker(revisions)
    schema = cache.schema(ref)
    profile = cache.profile(ref)
    _summary_cards(ref, revisions, profile)

    tab_vars, tab_profile, tab_series, tab_lineage = st.tabs(
        ["变量", "画像", "时序预览", "血缘"])
    with tab_vars:
        _variables(schema)
    with tab_profile:
        _profile(profile)
    with tab_series:
        _series(ref, schema)
    with tab_lineage:
        _lineage(ref)


def _picker(revisions: list[catalog.RevisionRow]) -> str:
    labels = {row.ref: f"{row.dataset_id} · {row.revision}"
              f"（{row.rows or 0:,} 行）" for row in revisions}
    ref = st.selectbox("数据集修订版", options=list(labels),
                       format_func=lambda r: labels[r],
                       key="data_ref")
    return str(ref)


def _summary_cards(ref: str, revisions: list[catalog.RevisionRow],
                   profile: dict) -> None:
    row = next((r for r in revisions if r.ref == ref), None)
    columns = st.columns(5)
    columns[0].metric("记录数", f"{profile.get('record_count', 0):,}")
    columns[1].metric("变量数",
                      (profile.get("coverage") or {}).get("variable_count", 0))
    columns[2].metric("对象数",
                      (profile.get("coverage") or {}).get("object_count", 0))
    columns[3].metric("时间断档", profile.get("gap_count", 0),
                      help="相邻记录间隔超过采样周期的次数")
    columns[4].metric("越界值", profile.get("range_violations", 0),
                      help="超出变量声明 min/max 的取值个数")
    if row:
        st.caption(
            f"时间范围 {fmt_time(row.time_range[0])} → "
            f"{fmt_time(row.time_range[1])}　·　"
            f"内容指纹 `{row.content_sha256[:16]}…`　·　"
            f"入库 {fmt_time(row.created_at)}")
    intervals = profile.get("interval_stats") or {}
    if intervals.get("off_resolution_fraction"):
        st.warning(
            f"有 {intervals['off_resolution_fraction'] * 100:.2f}% 的记录不落在"
            f"采样网格上（中位间隔 {intervals.get('median_s')}s）。时间轴不齐会"
            "影响切分边界的对齐，去「数据质量」页看处置建议。", icon="⏱")


def _variables(schema: dict) -> None:
    frame = catalog.variables_frame(schema)
    if frame.empty:
        st.info("这个修订版没有变量定义。")
        return
    derived = int((frame["来源"] == "派生").sum())
    st.caption(
        f"共 {len(frame)} 个变量，其中**派生 {derived} 个**。派生变量不能进"
        "候选输入白名单（DD-16）——用公式推出来的量去预测它的原料，"
        "得到的是循环论证，指标会漂亮得不真实。")
    st.dataframe(frame, width="stretch", hide_index=True,
                 column_config={
                     "来源": st.column_config.TextColumn(
                         "来源", help="measured=实测；derived=表内公式推导"),
                 })


def _profile(profile: dict) -> None:
    frame = catalog.profile_frame(profile)
    if frame.empty:
        st.info("没有画像数据。")
        return
    rates = series.prepare_missing_rates(profile)
    if rates:
        st.plotly_chart(interactive.missing_rates_figure(rates),
                        width="stretch")
        st.caption("红色是缺失率 ≥ 10% 的变量。分位点是固定的 7 个"
                   "（min/p01/p25/p50/p75/p99/max），由工具信封约定。")
    st.dataframe(
        frame, width="stretch", hide_index=True,
        column_config={
            "缺失率": st.column_config.ProgressColumn(
                "缺失率", format="%.2f%%", min_value=0.0, max_value=1.0),
        })


def _series(ref: str, schema: dict) -> None:
    variables = [str(v.get("variable_id"))
                 for v in entries(schema.get("variables"))]
    if not variables:
        st.info("没有可画的变量。")
        return
    chosen = st.multiselect(
        "选择变量", options=variables,
        default=variables[:DEFAULT_PREVIEW_VARIABLES], key="data_series_vars", placeholder="请选择…")
    if not chosen:
        st.caption("选至少一个变量。")
        return
    preview = cache.series_preview(ref, tuple(chosen))
    if preview.frame.empty:
        st.info("这个时间窗里没有数据。")
        return
    st.plotly_chart(
        interactive.raw_series_figure(preview.frame, chosen),
        width="stretch")
    note = f"{preview.total_rows:,} 行"
    if preview.downsampled:
        note += (f"，图上每 {preview.stride} 点取 1（等间隔抽稀，不做平均——"
                 "平均会把毛刺抹掉，而毛刺正是要看的）")
    st.caption(note)
    with st.expander("原始点（前 200 行）"):
        st.dataframe(preview.frame.head(200), width="stretch",
                     hide_index=True)


def _lineage(ref: str) -> None:
    """血缘：这份数据从哪来、被谁用过。"""
    st.markdown("#### 这个修订版被谁用了")
    view_docs = [doc for doc in cache.views()
                 if (doc.get("definition") or {}).get("dataset") == ref]
    experiments = cache.experiment_list()
    view_ids = {doc.get("id") for doc in view_docs}
    used_by = [item for item in experiments if item.dataset_view in view_ids]

    if not view_docs:
        st.info("还没有基于这个修订版登记的 Dataset View。")
    for doc in view_docs:
        definition = doc.get("definition") or {}
        with st.container(border=True):
            st.markdown(f"**{doc.get('id')}**　目标 `{definition.get('target')}`")
            st.caption(f"特征 {len(definition.get('features') or [])} 个"
                       f" · 对象 {'、'.join(definition.get('objects') or []) or '—'}"
                       f" · 过滤 {definition.get('filter') or '无'}")
            children = [item for item in used_by if item.dataset_view == doc.get("id")]
            if children:
                st.caption("→ 实验：" + "、".join(i.experiment_id for i in children))
    st.caption(
        "链路是：原始工作簿 → 数据修订版 → Dataset View → 实验 → 模型包。"
        "每一环都带指纹，view_hash 里含数据修订版 ID，所以数据一变，"
        "旧的视图缓存不会被错误复用。")
