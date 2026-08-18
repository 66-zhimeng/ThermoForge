"""模型结构渲染：公式 + 参数 + 结构图。实验结果 / 模型 / 研究 三页共用。

解析在 `services/inspect_model.py`（纯函数，可测），这里只负责画。
报告导出不走这里——`services/reports.py` 用同一份 ModelDoc 组装
纯文本公式与双轨图。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..charts import interactive, series
from ..services import inspect_model
from ..services.inspect_model import ModelDoc, ParamRow, load_model_doc


def render_model_structure(directory: str | Path,
                           *, key_prefix: str = "structure") -> None:
    """渲染一个模型工件目录的结构。没有工件就安静不出现（调用方不用判断）。"""
    doc = load_model_doc(directory)
    if doc is None:
        return
    if doc.kind == "unknown":
        st.caption(f"未识别的模型格式 `{doc.format}`，只能展示原始清单：")
        st.json(doc.raw, expanded=False)
        return
    st.markdown(f"**{doc.summary}**　`{doc.format}`")
    _equations(doc)
    if doc.kind == "hybrid":
        _hybrid_body(doc, key_prefix)
    elif doc.kind == "linear":
        _linear_body(doc, key_prefix)
    elif doc.kind == "lab":
        _lab_body(doc, key_prefix)
    if doc.params:
        _params_table(doc.params)
    if doc.inputs and any(k != v for k, v in doc.inputs.items()):
        st.caption("输入映射：" + "　".join(f"{k} → `{v}`"
                                          for k, v in doc.inputs.items()))
    if doc.notes:
        st.caption("　·　".join(doc.notes))


# ---------------------------------------------------------------- 各路线


def _equations(doc: ModelDoc) -> None:
    if doc.latex:
        for tex in doc.latex:
            st.latex(tex)
    elif doc.equations:
        st.code("\n".join(doc.equations), language="text")
    if doc.substituted:
        st.markdown("**代入辨识参数**")
        st.code("\n".join(doc.substituted), language="text")


def _linear_body(doc: ModelDoc, key_prefix: str) -> None:
    bars = series.prepare_coef_bars(doc.coefficients, unit="目标单位 / 每标准差")
    if bars:
        st.plotly_chart(
            interactive.coef_bars_figure(bars, title="标准化系数（岭回归）"),
            width="stretch", key=f"{key_prefix}_coef")
    scaler = doc.scaler or {}
    order = scaler.get("feature_order") or []
    if order:
        with st.expander("标准化参数（训练集均值 / 标准差）"):
            means, stds = scaler.get("means") or {}, scaler.get("stds") or {}
            st.dataframe(pd.DataFrame([{
                "特征": name,
                "均值": means.get(name),
                "标准差": stds.get(name),
            } for name in order]), hide_index=True, width="stretch")


def _hybrid_body(doc: ModelDoc, key_prefix: str) -> None:
    nodes, edges = inspect_model.hybrid_flow_spec(doc)
    st.plotly_chart(interactive.structure_flow_figure(nodes, edges),
                    width="stretch", key=f"{key_prefix}_flow")
    _xgb_spec(doc)
    importance = (doc.xgb or {}).get("importance") or {}
    bars = series.prepare_coef_bars(importance, unit="%")
    if bars:
        st.plotly_chart(
            interactive.coef_bars_figure(
                bars, title="残差项特征重要性（gain 占比）"),
            width="stretch", key=f"{key_prefix}_imp")


def _lab_body(doc: ModelDoc, key_prefix: str) -> None:
    _xgb_spec(doc)
    importance = (doc.xgb or {}).get("importance") or {}
    bars = series.prepare_coef_bars(importance, unit="%")
    if bars:
        st.plotly_chart(
            interactive.coef_bars_figure(bars, title="特征重要性（gain 占比）"),
            width="stretch", key=f"{key_prefix}_imp")
    if doc.lab_source:
        with st.expander("模块源码（模型实验室提交时过结构校验的原始代码）"):
            st.code(doc.lab_source, language="python")


def _xgb_spec(doc: ModelDoc) -> None:
    xgb = doc.xgb or {}
    params = xgb.get("params") or {}
    bits = [f"{xgb.get('n_trees') or '?'} 棵树",
            f"深度 ≤ {params.get('max_depth', '?')}",
            f"学习率 {params.get('learning_rate', '?')}"]
    if xgb.get("seed") is not None:
        bits.append(f"种子 {xgb['seed']}")
    st.caption("XGBoost 残差项：" + " · ".join(bits))
    mono = xgb.get("monotone_constraints") or {}
    if mono:
        st.caption("单调约束：" + "、".join(
            f"`{k}` {'递增' if v > 0 else '递减'}" for k, v in mono.items()))


def _params_table(params: tuple[ParamRow, ...]) -> None:
    def _bounds(row: ParamRow) -> str:
        if not row.bounds:
            return "—"
        return f"[{row.bounds[0]:.4g}, {row.bounds[1]:.4g}]"

    st.dataframe(pd.DataFrame([{
        "参数": p.name,
        "取值": f"{p.value:.6g}" if p.value is not None else "—",
        "单位": p.unit or "—",
        "合法范围": _bounds(p),
        "备注": p.note,
    } for p in params]), hide_index=True, width="stretch")
