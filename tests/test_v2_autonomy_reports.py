"""Frozen proposals, feedback and reuse remain distinct in external research reports."""

from copy import deepcopy
import json
import re

import pytest

from thermoforge_v2.report_flow import build_flow
from thermoforge_v2.reports import build_report, render_html, render_markdown


@pytest.fixture()
def snapshot():
    def job(identifier, track, idea, proposal, metric, **extra):
        return {"id": identifier, "track_id": track, "idea_id": idea, "proposal_id": proposal,
                "status": "completed", "executed": True, "experiment_fingerprint": "model-a",
                "result": {"experiment_id": "EXP-" + identifier, "feedback_surface": "validate",
                           "protocol_fingerprint": "p1", "metrics": {"CVRMSE": metric}}, **extra}

    def proposal(identifier, track, idea, version, **extra):
        return {"id": identifier, "track_id": track, "idea_id": idea, "version": version,
                "status": "committed", "purpose": "explore" if version == 1 else "refine",
                "model": {"estimator": "ridge", "hyperparameters": {"alpha": 1}},
                "protocol_fingerprint": "p1", "experiment_fingerprint": "model-a",
                "parent_job_ids": [], **extra}

    return {
        "run": {"run_id": "RUN-A", "research_stage": "sharing", "protocol": {"fingerprint": "p1"}},
        "tracks": [{"track_id": "candidate-1", "role": "candidate"},
                   {"track_id": "candidate-2", "role": "candidate"}],
        "ideas": [{"id": "I1", "track_id": "candidate-1", "origin": "conjecture", "statement": "基线"},
                  {"id": "I2", "track_id": "candidate-1", "origin": "history", "statement": "收缩系数"},
                  {"id": "I3", "track_id": "candidate-2", "origin": "conjecture", "statement": "另一首轮"}],
        "proposals": [proposal("P1", "candidate-1", "I1", 1),
                      proposal("P2", "candidate-1", "I2", 2, parent_job_ids=["J1"]),
                      proposal("P3", "candidate-2", "I3", 1)],
        "jobs": [job("J1", "candidate-1", "I1", "P1", 0.1),
                 job("J2", "candidate-1", "I2", "P2", 0.09),
                 job("J3", "candidate-2", "I3", "P3", 0.1, duplicate_of_job_id="J1")],
        "findings": [{"id": "F1", "track_id": "candidate-1", "job_ids": ["J1", "J2"],
                      "statement": "误差降低但窗口有限"}],
        "stops": [{"id": "STOP1", "track_id": "candidate-1", "reason": "余下预算不足以完成消融",
                   "evidence_ids": ["F1"], "job_ids": ["J1", "J2"]}],
    }


def nodes(flow):
    return {node["id"]: node for node in flow["nodes"]}


def edges(flow):
    return {(edge["from"], edge["to"], edge["relation"]) for edge in flow["edges"]}


def test_frozen_proposal_revision_and_stop_keep_explicit_evidence(snapshot):
    before = deepcopy(snapshot)
    report = build_report(snapshot)
    flow = report["flow"]
    assert snapshot == before
    assert {("I1", "P1", "idea"), ("P1", "J1", "proposal"),
            ("J1", "P2", "parent_experiment"), ("P2", "J2", "proposal"),
            ("F1", "STOP1", "evidence"), ("J2", "STOP1", "experiment")} <= edges(flow)
    assert "首轮独立冻结" in nodes(flow)["P1"]["label"]
    assert "修订提案" in nodes(flow)["P2"]["label"]
    assert nodes(flow)["P2"]["detail"]["model"] == snapshot["proposals"][1]["model"]
    assert report["summary"]["proposals"] == 3
    assert report["summary"]["stops"] == 1
    assert not flow["missing_references"]
    assert not report["lineage"]["missing_references"]
    markdown = render_markdown(report)
    for text in ("首轮独立冻结提案 P1", "修订提案 P2", "停止研究决定 STOP1",
                 "余下预算不足以完成消融", "不自动表示目标已经达到", "证据共享"):
        assert text in markdown


def test_equal_configuration_does_not_invent_inspiration_or_reuse(snapshot):
    flow = build_flow(snapshot)
    assert ("J1", "J3", "duplicate_configuration") in edges(flow)
    assert ("J1", "J3", "result_reuse") not in edges(flow)
    assert nodes(flow)["J3"]["detail"]["execution_label"] == "重复配置，分别执行"
    assert not any(edge[0] == "J1" and edge[1] in {"I3", "P3"} for edge in edges(flow))


def test_reused_result_keeps_score_but_is_not_an_independent_measurement(snapshot):
    snapshot["jobs"][2].update(executed=False, reused_from_job_id="J1")
    report = build_report(snapshot)
    flow = report["flow"]
    assert ("J1", "J3", "result_reuse") in edges(flow)
    assert nodes(flow)["J3"]["metrics"] == {"CVRMSE": 0.1}
    assert nodes(flow)["J3"]["detail"]["execution_label"] == "复用既有结果（非独立复现）"
    compared = next(row for row in report["comparability"]["groups"][0]["rows"] if row["job_id"] == "J3")
    assert compared["reused_from_job_id"] == "J1"
    assert "非独立复现" in compared["execution"]
    markdown = render_markdown(report)
    assert "明确实际执行 2 个，结果复用 1 个" in markdown
    assert "不构成新增实测或独立复现" in markdown


def test_scoped_report_references_peer_reuse_without_copying_peer_research(snapshot):
    snapshot["jobs"][2].update(executed=False, reused_from_job_id="J1")
    report = build_report(snapshot, track_id="candidate-2")
    flow = report["flow"]
    assert nodes(flow)["J1"]["status"] == "outside_scope"
    assert "P1" not in nodes(flow) and "STOP1" not in nodes(flow)
    assert [p["id"] for p in report["proposals"]] == ["P3"]
    assert report["stops"] == []
    assert "收缩系数" not in json.dumps(report, ensure_ascii=False)


def test_shared_message_cites_evidence_without_inventing_candidate_inspiration(snapshot):
    snapshot["messages"] = [{"id": "M1", "track_id": "main", "from": "main", "to": "candidate-2",
                             "text": "可评估另一候选的收缩结果", "evidence_ids": ["J2"],
                             "research_stage": "sharing"}]
    flow = build_flow(snapshot)
    assert ("J2", "M1", "evidence") in edges(flow)
    assert ("M1", "track:candidate-2", "message_to") in edges(flow)
    assert nodes(flow)["M1"]["detail"]["research_stage"] == "sharing"
    assert not any(edge[0] == "M1" and edge[1] in {"I3", "P3", "J3"} for edge in edges(flow))


def test_uncommitted_proposal_is_not_presented_as_frozen(snapshot):
    snapshot["proposals"][0]["status"] = "draft"
    report = build_report(snapshot)
    assert "待冻结提案" in nodes(report["flow"])["P1"]["label"]
    assert "首轮独立冻结提案 P1" not in render_markdown(report)


def test_pending_request_never_becomes_a_measurement(snapshot):
    snapshot["jobs"][0].update(status="pending", executed=False)
    report = build_report(snapshot)
    assert nodes(report["flow"])["J1"]["metrics"] == {}
    assert nodes(report["flow"])["J1"]["detail"]["execution_label"] == "未执行"
    assert "J1" in {row["job_id"] for row in report["comparability"]["excluded"]}


def test_new_records_do_not_export_private_thoughts_or_unbounded_payloads(snapshot):
    for collection in ("proposals", "stops"):
        snapshot[collection][0]["chain_of_thought"] = "private-trace"
        snapshot[collection][0]["evidence"] = {"hidden_holdout": 12345}
    serialized = json.dumps(build_flow(snapshot))
    assert "private-trace" not in serialized
    assert "hidden_holdout" not in serialized


def test_flow_payload_safely_preserves_stop_and_proposal_text(snapshot):
    malicious = '</script><img src=x onerror="alert(1)">'
    snapshot["stops"][0]["reason"] = malicious
    snapshot["proposals"][0]["model"]["name"] = malicious
    html = render_html(build_report(snapshot))
    payload = re.search(r'<script type="application/json" id="tf-flow-data">(.*?)</script>',
                        html, re.DOTALL).group(1)
    assert "<" not in payload
    decoded = json.loads(payload)
    assert nodes(decoded)["STOP1"]["detail"]["reason"] == malicious
    assert '<img src=x onerror=' not in html
    assert "复用的验证指标" in html


def test_legacy_snapshot_does_not_claim_autonomous_proposal_protocol(snapshot):
    snapshot.pop("proposals")
    snapshot.pop("stops")
    snapshot["run"].pop("research_stage")
    for job in snapshot["jobs"]:
        job.pop("proposal_id")
        job.pop("executed")
        job.pop("duplicate_of_job_id", None)
    report = build_report(snapshot)
    assert report["proposals"] == [] and report["stops"] == []
    assert "自主研究阶段与实验执行" not in render_markdown(report)
    assert nodes(report["flow"])["J1"]["detail"]["execution_label"] == "执行方式未记录"
    assert not report["flow"]["missing_references"]
