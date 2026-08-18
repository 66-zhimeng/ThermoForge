"""报告导出页：挑内容 → 预览 → 下载 HTML / Markdown / PDF。

同一份文档模型出三种格式，各有各的场合：HTML 发给别人看（自包含、图能
交互），Markdown 进 git（可 diff 的版本化记录），PDF 用来打印和归档。
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from .. import cache
from ..export import render_html, render_markdown_bundle, render_pdf
from ..services.reports import build_report
from ..ui import copilot_banner, empty_state

PREVIEW_HEIGHT = 640


def report_page() -> None:
    st.title("报告导出")
    copilot_banner("report")
    summaries = cache.experiment_list()
    if not summaries:
        empty_state("还没有实验可以写进报告",
                    "先去「AI 研究」页跑一轮。", icon="📄")
        return

    completed = [item for item in summaries if item.status == "completed"]
    options = [item.experiment_id for item in (completed or summaries)]
    labels = {item.experiment_id: f"{item.experiment_id} · {item.model_label}"
              for item in (completed or summaries)}

    columns = st.columns([3, 2])
    picked = columns[0].multiselect(
        "写进报告的实验", options=options, default=options[:5],
        format_func=lambda k: labels.get(k, k),
        help="第一个选中的会出现在最前面；最优模型由 CVRMSE 自动判定。", placeholder="请选择…")
    title = columns[1].text_input("报告标题", value="ThermoForge 建模实验报告")
    subtitle = columns[1].text_input(
        "副标题", value=f"生成于 {datetime.now():%Y-%m-%d}")
    include_charts = columns[1].checkbox("包含图表", value=True)
    include_quality = columns[1].checkbox(
        "附带数据质量体检结论", value=True,
        help="用「数据质量」页最近一次体检的结果。没跑过体检就不会出现这一节。")

    if not picked:
        st.info("至少选一个实验。")
        return

    quality_result = (st.session_state.get("quality_result")
                      if include_quality else None)
    quality_report = None
    quality_ref = ""
    if quality_result and (quality_result.get("envelope") or {}).get("ok"):
        from ..services import quality as quality_service

        quality_report = quality_service.build_report(
            quality_result["envelope"].get("summary") or {},
            quality_result.get("profile"))
        quality_ref = str(quality_result.get("ref") or "")
    elif include_quality:
        st.caption("还没有体检结果——去「数据质量」页跑一次，报告里就会多一节。")

    with st.spinner("组装报告…"):
        document = build_report(
            list(picked), title=title.strip() or "ThermoForge 实验报告",
            subtitle=subtitle.strip(), include_charts=include_charts,
            quality_report=quality_report, quality_ref=quality_ref)

    _downloads(document)
    _preview(document)


def _downloads(document) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.markdown("#### 下载")
    columns = st.columns(3)

    with columns[0]:
        st.caption("**HTML**　自包含单文件，双击就能看，图可交互。")
        if st.button("生成 HTML", width="stretch", type="primary"):
            st.session_state["report_html"] = render_html(document)
        if st.session_state.get("report_html"):
            st.download_button(
                "下载 .html", st.session_state["report_html"].encode("utf-8"),
                file_name=f"thermoforge_report_{stamp}.html",
                mime="text/html", width="stretch")

    with columns[1]:
        st.caption("**Markdown**　正文 + PNG 图，zip 打包，可进 git 版本化。")
        if st.button("生成 Markdown", width="stretch"):
            with st.spinner("渲染静态图…"):
                st.session_state["report_md"] = render_markdown_bundle(document)
        if st.session_state.get("report_md"):
            st.download_button(
                "下载 .zip", st.session_state["report_md"],
                file_name=f"thermoforge_report_{stamp}.zip",
                mime="application/zip", width="stretch")

    with columns[2]:
        st.caption("**PDF**　固定排版，中文用内置字体，不依赖浏览器。")
        if st.button("生成 PDF", width="stretch"):
            with st.spinner("排版中…"):
                st.session_state["report_pdf"] = render_pdf(document)
        if st.session_state.get("report_pdf"):
            st.download_button(
                "下载 .pdf", st.session_state["report_pdf"],
                file_name=f"thermoforge_report_{stamp}.pdf",
                mime="application/pdf", width="stretch")


def _preview(document) -> None:
    st.markdown("#### 预览")
    st.caption(f"{len(document.sections)} 节　·　"
               f"{sum(len(s.figures) for s in document.sections)} 张图")
    for section in document.sections:
        with st.expander(section.title,
                         expanded=section.title in ("概览", "模型对比")):
            if section.key_values:
                columns = st.columns(min(4, len(section.key_values)))
                for index, (key, value) in enumerate(section.key_values.items()):
                    columns[index % len(columns)].metric(key, value)
            for paragraph in section.paragraphs:
                st.markdown(paragraph)
            if section.table is not None and not section.table.empty:
                st.dataframe(section.table, width="stretch",
                             hide_index=True)
                if section.table_caption:
                    st.caption(section.table_caption)
            for caption, frame in section.extra_tables:
                if frame is None or frame.empty:
                    continue
                st.dataframe(frame, width="stretch", hide_index=True)
                if caption:
                    st.caption(caption)
            for figure in section.figures:
                st.plotly_chart(figure.plotly(), width="stretch",
                                key=f"preview_{figure.key}")
                if figure.caption:
                    st.caption(figure.caption)
