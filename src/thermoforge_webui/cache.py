"""Streamlit 缓存包装。

Streamlit 每次交互都重跑整个脚本，读 parquet / 建 DuckDB 连接的代价会
被反复付。这里集中包一层缓存，并且**只缓存只读查询**——任何会写工件的
调用都不进这里，写完统一 `invalidate()`。

TTL 而不是永久缓存：实验可能由后台线程或另一个终端产生，界面得能自己
发现新工件，不必让人手动刷新。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .context import tool_context
from .services import catalog, experiments

_TTL = 20  # 秒


@st.cache_data(ttl=_TTL, show_spinner=False)
def status_panel(recent: int = 8) -> dict[str, Any]:
    from thermoforge_cli.status import build_status

    return build_status(tool_context(), recent=recent)


@st.cache_data(ttl=_TTL, show_spinner=False)
def experiment_list(limit: int | None = None
                    ) -> list[experiments.ExperimentSummary]:
    return experiments.list_experiments(limit=limit)


@st.cache_data(ttl=_TTL, show_spinner=False)
def experiment_detail(experiment_id: str
                      ) -> experiments.ExperimentDetail | None:
    return experiments.load_detail(experiment_id)


@st.cache_data(ttl=_TTL, show_spinner=False)
def predictions(experiment_id: str) -> pd.DataFrame | None:
    return experiments.load_predictions(experiment_id)


@st.cache_data(ttl=_TTL, show_spinner=False)
def revisions() -> list[catalog.RevisionRow]:
    return catalog.list_revisions(tool_context())


@st.cache_data(ttl=_TTL, show_spinner=False)
def schema(ref: str) -> dict[str, Any]:
    return catalog.schema_of(tool_context(), ref)


@st.cache_data(ttl=_TTL, show_spinner="正在统计数据画像…")
def profile(ref: str) -> dict[str, Any]:
    return catalog.profile_of(tool_context(), ref)


@st.cache_data(ttl=_TTL, show_spinner="正在读取原始数据…")
def series_preview(ref: str, variable_ids: tuple[str, ...],
                   start: str | None = None,
                   end: str | None = None) -> catalog.SeriesPreview:
    return catalog.series_preview(tool_context(), ref, list(variable_ids),
                                  start=start, end=end)


@st.cache_data(ttl=_TTL, show_spinner=False)
def goals() -> list[dict[str, Any]]:
    ctx = tool_context()
    if not ctx.research_root.exists():
        return []
    return ctx.ledger.list_goals()


@st.cache_data(ttl=_TTL, show_spinner=False)
def views() -> list[dict[str, Any]]:
    """已登记的 Dataset View（实验计划必须引用一个已存在的 View）。"""
    import yaml

    ctx = tool_context()
    directory = ctx.research_root / "views"
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("VIEW-*.yaml")):
        try:
            with open(path, encoding="utf-8") as fp:
                doc = yaml.safe_load(fp)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out.append(doc)
    return out


def invalidate() -> None:
    """写过工件之后调用：让下一次读取重新落盘取数。"""
    st.cache_data.clear()
