"""页面注册表与跳转。

副驾要能「把你带到对应页面并选好对象」，就得在运行时拿到 `st.Page` 对象
——`st.switch_page` 对函数式页面只认对象，不认路径。但页面对象是在
`app.py` 里构造的，副驾模块又被 `app.py` 间接导入，直接互相 import 会成环。

所以这里只放一个**空注册表**：`app.py` 构造完页面后回填，副驾只依赖本
模块。没有循环，也不需要谁去 import 谁的实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st


@dataclass(frozen=True)
class PageInfo:
    """一个页面的元信息。`selection_keys` 是副驾能预选的 widget key。"""

    key: str
    title: str
    purpose: str
    selection_keys: tuple[str, ...] = ()


# 副驾看到的页面清单。`purpose` 会进系统提示词，决定它往哪跳，
# 所以写的是「什么问题该来这一页」，不是「这一页有什么控件」。
PAGE_CATALOG: tuple[PageInfo, ...] = (
    PageInfo("overview", "总览",
             "整体进展：有几个研究目标、最近实验跑得怎么样、有没有卡着等人处理的事"),
    PageInfo("data", "数据",
             "某个数据集里有什么变量、缺失多少、原始曲线长什么样",
             ("data_ref", "data_series_vars")),
    PageInfo("quality", "数据质量",
             "这份数据能不能拿来建模、有哪些阻断项、该怎么处理",
             ("quality_ref", "quality_target")),
    PageInfo("research", "AI 研究",
             "开始一轮自主研究，或新建研究目标与数据视图"),
    PageInfo("research_v2", "V2 研究运行",
             "启动和管理主 Codex 与五个独立候选的后台研究，查看轨迹、来源与比较报告",
             ("v2_run_id",)),
    PageInfo("results", "实验结果",
             "某次实验跑得怎么样、预测准不准、误差在哪、几次实验哪个更好",
             ("results_exp", "results_surface", "compare_ids", "compare_goal")),
    PageInfo("models", "模型",
             "已发布的模型包、版本状态、发布门禁过没过"),
    PageInfo("report", "报告导出",
             "把结果导成能发给别人的 HTML / Markdown / PDF 报告"),
    PageInfo("settings", "设置",
             "模型接口配置、密钥、网络连不上时的分层排查"),
)

PAGE_KEYS = tuple(info.key for info in PAGE_CATALOG)
PAGE_BY_KEY = {info.key: info for info in PAGE_CATALOG}

# app.py 构造完 st.Page 后回填
_REGISTRY: dict[str, Any] = {}

# 副驾的分析结论跟着跳转一起带到目标页顶部
BANNER_KEY = "copilot_banner"


def register(key: str, page: Any) -> None:
    _REGISTRY[key] = page


def page_object(key: str) -> Any | None:
    return _REGISTRY.get(key)


def catalog_for_prompt() -> str:
    """给副驾看的页面清单（进系统提示词）。"""
    return "\n".join(f"- {info.key}（{info.title}）：{info.purpose}"
                     for info in PAGE_CATALOG)


def apply_and_switch(intent: dict[str, Any],
                     validator: Callable[[str, Any], bool] | None = None) -> bool:
    """套用预选 → 设置横幅 → 跳页。返回是否真的跳了。

    预选值会写进 widget 的 session_state key。**必须先验证取值仍然有效**：
    给 selectbox 塞一个不在 options 里的值，Streamlit 会直接抛异常，
    而副驾完全可能引用一个刚被删掉的实验。
    """
    key = str(intent.get("page") or "")
    page = _REGISTRY.get(key)
    if page is None:
        return False
    info = PAGE_BY_KEY.get(key)
    allowed = set(info.selection_keys) if info else set()
    for name, value in (intent.get("selections") or {}).items():
        if name not in allowed:
            continue
        if validator is not None and not validator(name, value):
            continue
        st.session_state[name] = value
    st.session_state[BANNER_KEY] = {
        "page": key,
        "note": str(intent.get("note") or ""),
    }
    st.switch_page(page)
    return True
