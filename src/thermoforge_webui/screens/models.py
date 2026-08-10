"""模型页：注册表、版本状态机、发布门禁结果。

模型包不是 `model.pkl`——它带签名、约束、指标和数据/研究血缘，还有
`checksums.json` 与 `golden.parquet`（冷加载冒烟会在全新解释器里重放
这组预测）。这一页就是把这些东西摊开给人看。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from ..context import MODELS_ROOT
from ..ui import copilot_banner, empty_state, fmt_time

# 版本状态机（model-package §5）：只有回滚是反向边
STATE_LABELS = {
    "candidate": "候选",
    "validated": "已验证",
    "approved": "已批准",
    "production": "生产",
    "deprecated": "已弃用",
    "retired": "已退役",
}
STATE_COLORS = {
    "production": "🟢", "approved": "🔵", "validated": "🔵",
    "candidate": "⚪", "deprecated": "🟡", "retired": "⚫",
}

# 发布门禁项（registry.py 里的 gate name → 中文）
GATE_LABELS = {
    "integrity": "完整性校验",
    "signature_tfom": "签名与对象模型兼容",
    "acceptance": "验收指标达标",
    "latency": "推理延迟",
    "smoke": "冷加载冒烟",
    "rollback_recorded": "回滚留痕",
    "failed": "门禁执行失败",
}


def models_page() -> None:
    st.title("模型")
    copilot_banner("models")
    registries = _load_registries()
    if not registries:
        empty_state("还没有发布的模型包",
                    "实验达标后在「实验结果」里选中它，用 `tf model publish` "
                    "发布；或让 AI 研究循环在达到验收标准时自动发布。",
                    icon="📦")
        return

    names = [item["model_id"] for item in registries]
    model_id = st.selectbox("模型", names, key="models_pick")
    registry = next(item for item in registries if item["model_id"] == model_id)
    _overview(registry)
    _versions(registry)


def _load_registries() -> list[dict[str, Any]]:
    if not MODELS_ROOT.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(MODELS_ROOT.glob("*/registry.json")):
        try:
            with open(path, encoding="utf-8") as fp:
                doc = json.load(fp)
        except (OSError, ValueError):
            continue
        doc.setdefault("model_id", path.parent.name)
        doc["_dir"] = str(path.parent)
        out.append(doc)
    return out


def _overview(registry: dict[str, Any]) -> None:
    versions = registry.get("versions") or {}
    production = registry.get("production")
    columns = st.columns(4)
    columns[0].metric("当前生产版本", production or "—")
    columns[1].metric("版本总数", len(versions))
    columns[2].metric("上线历史", len(registry.get("production_history") or []))
    columns[3].metric("目录", Path(registry["_dir"]).name)
    if production and production in versions:
        _gates(versions[production], "当前生产版本的门禁结果")


def _gates(version: dict[str, Any], title: str) -> None:
    gates = version.get("last_gate_results") or []
    if not gates:
        return
    st.markdown(f"#### {title}")
    for gate in gates:
        icon = "✅" if gate.get("ok") else "❌"
        name = str(gate.get("name") or "")
        st.markdown(f"{icon} **{GATE_LABELS.get(name, name)}**　"
                    f"{gate.get('detail', '')}")


def _versions(registry: dict[str, Any]) -> None:
    st.markdown("#### 版本")
    versions = registry.get("versions") or {}
    production = registry.get("production")
    for name in sorted(versions, reverse=True):
        version = versions[name]
        history = version.get("history") or []
        state = str(history[-1].get("to")) if history else "unknown"
        icon = STATE_COLORS.get(state, "•")
        label = STATE_LABELS.get(state, state)
        badge = "　**← 生产中**" if name == production else ""
        with st.expander(f"{icon} `{name}`　{label}{badge}"):
            st.caption(f"内容指纹 `{str(version.get('content_id'))[:24]}…`")
            _gates(version, "发布门禁")
            st.markdown("**状态流转**")
            for step in history:
                st.caption(
                    f"{fmt_time(step.get('at'))}　"
                    f"`{step.get('from') or '—'}` → `{step.get('to')}`　"
                    f"{step.get('reason', '')}　（{step.get('actor')}）")
            st.caption("状态机只有回滚是反向边；跨级跳转会被 `can_transition` "
                       "拒绝，发布留痕绕不过去。")
