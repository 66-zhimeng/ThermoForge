"""Streamlit 入口：导航、全局侧栏、副驾快捷提问。

用 `streamlit run` 跑本文件（`tools/webui.py` 会代劳）。页面本身都在
`screens/` 下，一个模块一个页面函数，方便单独读和单独改。

目录名刻意不叫 `pages/`：Streamlit 会把入口脚本旁边的 `pages/` 当成自动
多页发现目录，和这里显式的 `st.navigation` 撞车。

本文件用**绝对导入**：`streamlit run` 是把脚本 exec 进一个没有 package
上下文的模块里，相对导入会直接 ImportError。
"""

from __future__ import annotations

import streamlit as st

from thermoforge_webui import APP_TITLE, navigation
from thermoforge_webui.config import read_config
from thermoforge_webui.context import roots_summary
from thermoforge_webui.screens import (
    copilot_page,
    data_page,
    models_page,
    overview_page,
    quality_page,
    report_page,
    research_page,
    research_v2_page,
    results_page,
    settings_page,
)
from thermoforge_webui.screens.copilot import PENDING_QUESTION_KEY

st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

# 这里的 key 必须与 navigation.PAGE_CATALOG 对齐——副驾按 key 决定往哪跳，
# 对不上就跳不过去（apply_and_switch 会返回 False 并提示）。
_SPECS = (
    ("copilot", copilot_page, "AI 助手", "🤖"),
    ("overview", overview_page, "总览", "🏠"),
    ("data", data_page, "数据", "🗂"),
    ("quality", quality_page, "数据质量", "🩺"),
    ("research", research_page, "AI 研究", "🔬"),
    ("research_v2", research_v2_page, "V2 研究运行", "🧪"),
    ("results", results_page, "实验结果", "📈"),
    ("models", models_page, "模型", "📦"),
    ("report", report_page, "报告导出", "📄"),
    ("settings", settings_page, "设置", "⚙️"),
)

PAGES = []
for _key, _screen, _title, _icon in _SPECS:
    _page = st.Page(_screen, title=_title, icon=_icon, url_path=_key,
                    default=_key == "copilot")
    navigation.register(_key, _page)
    PAGES.append(_page)


def _quick_ask(configured: bool) -> None:
    """侧栏快捷提问：任何页面上都能直接问，问完自动切到「AI 助手」。"""
    with st.form("sidebar_ask", clear_on_submit=True, border=False):
        question = st.text_input(
            "问 AI", placeholder="哪次实验最好？为什么？",
            label_visibility="collapsed", disabled=not configured)
        submitted = st.form_submit_button("🤖 问一下", width="stretch",
                                          disabled=not configured)
    if submitted and question.strip():
        st.session_state[PENDING_QUESTION_KEY] = question.strip()
        st.switch_page(navigation.page_object("copilot"))


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### 🔬 {APP_TITLE}")
        config = read_config()
        _quick_ask(config.configured)
        if config.configured:
            source = "环境变量" if config.from_env else "配置文件"
            st.caption(f"模型 `{config.model}` · 密钥来自{source}")
        else:
            st.info("网页 AI 助手尚未配置 API；V2 研究使用本机 Codex 登录。", icon="ℹ️")
        with st.expander("工件目录"):
            for name, path in roots_summary().items():
                st.caption(f"**{name}**　`{path}`")
        st.caption("只监听 127.0.0.1，同网段的人打不开这个界面。")


def main() -> None:
    _sidebar()
    st.navigation(PAGES).run()


main()
