"""数据质量页：诊断 → 处方 → 一键提案 → 人工审批。

三块内容：

1. **语义门禁**（modelability）：派生链、同源相关、设备多样性、物理合理性、
   工况覆盖。FAIL 会直接挡住建模，所以放最上面。
2. **统计画像**：时间轴、断档、缺失率、越界值。
3. **预处理闭环**：能由规则库解决的生成 `status=proposed` 草案，
   审批必须 `actor=human`——Agent 拿不到这个身份，这是 I-49 的留痕语义。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .. import cache
from ..context import human_context, tool_context
from ..services import catalog, quality
from ..ui import copilot_banner, empty_state, envelope_result

# 门禁判定：报告里存的是 PASS/FAIL，界面显示中文（存的值不动）
VERDICT_LABELS = {"PASS": "通过", "FAIL": "不通过", "UNKNOWN": "未体检"}

RULE_STATUS_LABELS = {"approved": "已批准", "proposed": "待批准",
                      "deprecated": "已废弃"}


def quality_page() -> None:
    st.title("数据质量")
    copilot_banner("quality")
    revisions = cache.revisions()
    if not revisions:
        empty_state("Vault 里还没有数据集", "先去「数据」页导入数据。", icon="🩺")
        return

    tab_check, tab_rules = st.tabs(["体检", "预处理规则"])
    with tab_check:
        _checkup(revisions)
    with tab_rules:
        _rulesets()


# ---------------------------------------------------------------- 体检


def _checkup(revisions) -> None:
    labels = {row.ref: f"{row.dataset_id} · {row.revision}" for row in revisions}
    ref = str(st.selectbox("数据集修订版", options=list(labels),
                           format_func=lambda r: labels[r], key="quality_ref"))
    schema = cache.schema(ref)
    property_codes = catalog.property_codes(schema)
    if not property_codes:
        st.warning("这个修订版没有变量定义，没法做语义体检。")
        return

    st.caption("语义门禁需要知道「要预测什么、用什么预测」——同一份数据"
               "换个目标，结论可能完全不同。")
    columns = st.columns([2, 5])
    target = columns[0].selectbox("目标", property_codes, key="quality_target")
    default_inputs = [code for code in property_codes if code != target]
    inputs = columns[1].multiselect("候选输入", options=default_inputs,
                                    default=default_inputs,
                                    key="quality_inputs", placeholder="请选择…")

    if st.button("开始体检", type="primary"):
        st.session_state["quality_result"] = _run_checkup(ref, str(target),
                                                          list(inputs))
    result = st.session_state.get("quality_result")
    if not result or result.get("ref") != ref:
        st.info("选好目标与候选输入，点「开始体检」。")
        return
    _render_report(result, ref)


def _run_checkup(ref: str, target: str, inputs: list[str]) -> dict[str, Any]:
    from thermoforge_research.tools import tf_dataset_modelability

    ctx = tool_context()
    envelope = tf_dataset_modelability(ctx, ref, target=target,
                                       candidate_inputs=inputs)
    profile = cache.profile(ref)
    return {"ref": ref, "target": target, "inputs": inputs,
            "envelope": envelope, "profile": profile}


def _render_report(result: dict[str, Any], ref: str) -> None:
    envelope = result["envelope"]
    if not envelope.get("ok"):
        envelope_result(envelope)
        return
    modelability = envelope.get("summary") or {}
    report = quality.build_report(modelability, result.get("profile"))

    if report.passed:
        st.success(report.headline, icon="✅")
    else:
        st.error(report.headline, icon="🛑")

    columns = st.columns(3)
    columns[0].metric("阻断", len(report.blockers))
    columns[1].metric("警告", len(report.warnings))
    columns[2].metric("门禁判定", VERDICT_LABELS.get(report.verdict,
                                                     report.verdict))

    for finding in report.findings:
        _finding_card(finding, ref)


def _finding_card(finding: quality.Finding, ref: str) -> None:
    expanded = finding.severity == "blocker"
    with st.expander(f"{finding.icon} {finding.severity_label}　{finding.title}",
                     expanded=expanded):
        st.markdown(finding.detail)
        for index, prescription in enumerate(finding.prescriptions):
            st.markdown(f"**处方**：{prescription.text}")
            if prescription.actionable:
                _proposal_form(finding, prescription, index, ref)
            if prescription.cli_hint:
                st.code(prescription.cli_hint, language="bash")
        if finding.evidence:
            st.caption("证据")
            st.json(finding.evidence, expanded=False)


def _proposal_form(finding: quality.Finding, prescription: quality.Prescription,
                   index: int, ref: str) -> None:
    """一键生成 status=proposed 规则草案。

    规则**只能选参数，不能写代码**：`rule_type` 必须已在 `RULE_LIBRARY`
    注册，参数由人在这里填。这是「提案 + 审批，绝不生成代码」的落点。
    """
    key = f"proposal_{finding.key}_{index}"
    with st.form(key):
        st.caption(f"生成预处理规则草案（类型 `{prescription.rule_type}`）。"
                   "草案是 proposed 状态，未经人工审批不能产生数据修订版。")
        columns = st.columns(3)
        ruleset_id = columns[0].text_input(
            "规则集 ID", value=f"{ref.split('@')[0]}_cleanup",
            key=f"{key}_rs")
        rule_id = columns[1].text_input(
            "规则 ID", value=f"{prescription.rule_type}_1", key=f"{key}_rid")
        sheet = columns[2].text_input(
            "目标工作表", value="", placeholder="原始工作簿里的表名",
            key=f"{key}_sheet")
        rationale = st.text_input(
            "理由（写进留痕）", value=finding.detail.splitlines()[0][:120],
            key=f"{key}_why")
        if st.form_submit_button("生成草案"):
            if not sheet.strip():
                st.error("目标工作表不能为空——规则作用在原始工作簿的某张表上。")
                return
            _submit_proposal(ruleset_id.strip(), rule_id.strip(), sheet.strip(),
                             prescription, rationale.strip())


def _submit_proposal(ruleset_id: str, rule_id: str, sheet: str,
                     prescription: quality.Prescription,
                     rationale: str) -> None:
    from thermoforge_research.tools import tf_preprocess_propose

    ruleset = {
        "ruleset_id": ruleset_id,
        "rules": [{
            "rule_id": rule_id,
            "rule_type": prescription.rule_type,
            "sheet": sheet,
            "params": dict(prescription.rule_params),
            "status": "proposed",
            "proposer": "webui",
            "rationale": rationale or None,
        }],
    }
    envelope = tf_preprocess_propose(tool_context(), ruleset)
    if envelope_result(envelope, success="草案已提交"):
        cache.invalidate()
        st.info("去下面的「预处理规则」页签审批。审批要求 actor=human，"
                "这一步没法由 Agent 代劳。")


# ---------------------------------------------------------------- 规则集


def _rulesets() -> None:
    from thermoforge_research.tools import tf_preprocess_list

    envelope = tf_preprocess_list(tool_context())
    if not envelope.get("ok"):
        envelope_result(envelope)
        return
    rulesets = (envelope.get("summary") or {}).get("rulesets") or []
    if not rulesets:
        empty_state("还没有预处理规则集",
                    "在「体检」页签里，对能由规则库解决的问题点「生成草案」。",
                    icon="🧾")
        return

    st.caption(
        "预处理是**提案 + 审批**，不是生成代码：Agent 只能为已注册的确定性"
        "变换提参数。proposed 状态的规则集不能产生数据修订版，审批必须由人"
        "以 actor=human 执行。")

    for ruleset in rulesets:
        rules = ruleset.get("rules") or []
        pending = [r for r in rules if r.get("status") == "proposed"]
        title = (f"{ruleset.get('ruleset_id')} · v{ruleset.get('version')}"
                 f"　（{len(rules)} 条规则"
                 f"{f'，{len(pending)} 条待批' if pending else ''}）")
        with st.expander(title, expanded=bool(pending)):
            for rule in rules:
                icon = {"approved": "✅", "proposed": "⏳",
                        "deprecated": "🚫"}.get(str(rule.get("status")), "•")
                status = str(rule.get("status"))
                st.markdown(f"{icon} `{rule.get('rule_id')}`　类型 "
                            f"`{rule.get('rule_type')}`　表 "
                            f"`{rule.get('sheet')}`　状态 "
                            f"{RULE_STATUS_LABELS.get(status, status)}")
            if pending:
                _approval(ruleset, pending)


def _approval(ruleset: dict[str, Any], pending: list[dict[str, Any]]) -> None:
    from thermoforge_research.tools import tf_preprocess_approve

    ids = [str(r.get("rule_id")) for r in pending]
    with st.form(f"approve_{ruleset.get('ruleset_id')}_{ruleset.get('version')}"):
        picked = st.multiselect("批准哪些规则", options=ids, default=ids, placeholder="请选择…")
        note = st.text_input("审批备注", placeholder="写清为什么批准，会留痕")
        if st.form_submit_button("以 actor=human 批准", type="primary"):
            envelope = tf_preprocess_approve(
                human_context(), str(ruleset.get("ruleset_id")),
                version=int(ruleset.get("version")),
                rule_ids=list(picked), note=note or None)
            if envelope_result(envelope, success="已批准"):
                cache.invalidate()
                st.rerun()
