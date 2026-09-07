"""The flow is a view of evidence, not a fabricated multi-agent conversation."""

from copy import deepcopy
import json

import pytest

from thermoforge_v2.report_flow import build_flow


@pytest.fixture()
def snapshot():
    return {
        "run": {"run_id": "RUN-1", "protocol": {"fingerprint": "p1"},
                "config": {"strategy": "independent", "experiment_workers": 1}},
        "tracks": [{"track_id": "main", "role": "main", "status": "completed"},
                   {"track_id": "candidate-1", "role": "candidate", "status": "completed"},
                   {"track_id": "candidate-2", "role": "candidate", "status": "running"}],
        "messages": [{"id": "M-1", "from": "main", "to": "all", "track_id": "main",
                      "initial_task": True, "text": "先做线性基线，再根据结果检验岭回归。", "evidence_ids": []}],
        "sources": [{"id": "S-1", "track_id": "candidate-1", "kind": "history", "title": "自身首轮实验",
                     "verification": "agent_supplied", "read_scope": "history", "history_ids": ["J-1"]}],
        "ideas": [{"id": "I-1", "track_id": "candidate-1", "origin": "conjecture", "statement": "线性基线",
                   "reason": "共同任务指定", "prediction": "误差低于阈值", "falsification": "误差超过阈值"},
                  {"id": "I-2", "track_id": "candidate-1", "origin": "history", "statement": "固定强度岭回归",
                   "reason": "检验正则化是否仍有改善", "source_ids": ["S-1"],
                   "parent_idea_ids": ["I-1"], "parent_job_ids": ["J-1"]},
                  {"id": "I-3", "track_id": "candidate-2", "origin": "conjecture", "statement": "其他独立候选"}],
        "jobs": [{"id": "J-1", "track_id": "candidate-1", "idea_id": "I-1", "status": "completed",
                  "request": {"model": {"estimator": "linear"}},
                  "result": {"experiment_id": "EXP-1", "feedback_surface": "validate", "protocol_fingerprint": "p1",
                             "metrics": {"CVRMSE": 0.1, "RMSE": 5}, "n_samples": 174}},
                 {"id": "J-2", "track_id": "candidate-1", "idea_id": "I-2", "status": "completed",
                  "request": {"model": {"estimator": "ridge", "hyperparameters": {"alpha": 1}}},
                  "result": {"experiment_id": "EXP-2", "feedback_surface": "validate", "protocol_fingerprint": "p1",
                             "metrics": {"CVRMSE": 0.09, "RMSE": 4.5}, "n_samples": 174}},
                 {"id": "J-3", "track_id": "candidate-2", "idea_id": "I-3", "status": "running"}],
        "findings": [{"id": "F-1", "track_id": "candidate-1", "idea_ids": ["I-1", "I-2"], "job_ids": ["J-1", "J-2"],
                      "statement": "验证误差下降", "limitations": "单次固定验证", "failure_category": "none"}],
        "decisions": [{"id": "D-1", "track_id": "main", "action": "retain", "reason": "保留验证优选",
                       "evidence_ids": ["F-1"]}],
        "reports": [{"id": "R-1", "track_id": "candidate-1", "kind": "track", "title": "候选报告",
                     "idea_ids": ["I-1", "I-2"], "job_ids": ["J-1", "J-2"], "finding_ids": ["F-1"],
                     "summary": "岭回归验证较优"},
                    {"id": "R-2", "track_id": "main", "kind": "team", "report_stage": "final",
                     "title": "综合报告", "evidence_ids": ["R-1", "J-2"]}],
        "final_evaluation": {"id": "FINAL-1", "status": "evaluated", "job_id": "J-2",
                             "track_id_selected": "candidate-1", "feedback_to_agents": False,
                             "surfaces": {"A": {"metrics": {"CVRMSE": 0.12}, "n_samples": 180}}},
    }


def _nodes(flow):
    return {node["id"]: node for node in flow["nodes"]}


def _links(flow):
    return {(edge["from"], edge["to"], edge["relation"]) for edge in flow["edges"]}


def test_preserves_explicit_history_experiments_and_report_citations(snapshot):
    before = deepcopy(snapshot)
    flow = build_flow(snapshot)
    assert snapshot == before
    assert {("J-1", "S-1", "history"), ("S-1", "I-2", "source"),
            ("I-1", "I-2", "parent_idea"), ("J-1", "I-2", "parent_experiment"),
            ("I-2", "J-2", "idea"), ("J-2", "F-1", "experiment"),
            ("F-1", "R-1", "finding"), ("R-1", "R-2", "evidence")} <= _links(flow)
    assert _nodes(flow)["I-2"]["detail"]["reason"] == "检验正则化是否仍有改善"
    assert _nodes(flow)["J-2"]["metrics"] == {"CVRMSE": 0.09, "RMSE": 4.5}
    assert _nodes(flow)["J-2"]["detail"]["comparable"] is True
    assert not flow["missing_references"]
    assert any("训练任务按队列串行" in note for note in flow["notes"])


def test_parallel_lanes_do_not_invent_candidate_debate_or_task_causality(snapshot):
    flow = build_flow(snapshot)
    assert {("track:main", "M-1", "message_sent"), ("M-1", "track:candidate-1", "message_to"),
            ("M-1", "track:candidate-2", "message_to")} <= _links(flow)
    assert not any(edge["from"] == "M-1" and edge["to"].startswith("I-") for edge in flow["edges"])
    candidate_two = {node["id"] for node in flow["nodes"] if node["track_id"] == "candidate-2"}
    candidate_one = {node["id"] for node in flow["nodes"] if node["track_id"] == "candidate-1"}
    assert not any(edge["from"] in candidate_one and edge["to"] in candidate_two for edge in flow["edges"])
    snapshot["messages"] = []
    assert not any(edge["relation"].startswith("message") for edge in build_flow(snapshot)["edges"])


def test_targeted_message_only_reaches_named_candidate(snapshot):
    snapshot["messages"][0]["to"] = "candidate-2"
    links = _links(build_flow(snapshot))
    assert ("M-1", "track:candidate-2", "message_to") in links
    assert ("M-1", "track:candidate-1", "message_to") not in links
    assert "M-1" not in _nodes(build_flow(snapshot, "candidate-1"))


def test_holdout_only_links_frozen_job_and_has_no_feedback(snapshot):
    flow = build_flow(snapshot)
    final = _nodes(flow)["FINAL-1"]
    assert final["kind"] == "final_evaluation" and final["track_id"] is None
    assert final["detail"]["surfaces"]["A"]["metrics"]["CVRMSE"] == 0.12
    assert final["detail"]["feedback_to_agents"] is False
    assert [(e["from"], e["to"]) for e in flow["edges"] if e["to"] == "FINAL-1"] == [("J-2", "FINAL-1")]
    assert not any(e["from"] == "FINAL-1" for e in flow["edges"])
    assert "FINAL-1" not in _nodes(build_flow(snapshot, "candidate-2"))
    assert "0.12" not in json.dumps(build_flow(snapshot, "candidate-2"))


def test_track_view_preserves_citations_without_other_research_contents(snapshot):
    snapshot["ideas"][1]["parent_idea_ids"].append("I-3")
    flow = build_flow(snapshot, "candidate-1")
    nodes = _nodes(flow)
    assert nodes["I-3"]["kind"] == "missing" and nodes["I-3"]["status"] == "outside_scope"
    assert "J-3" not in nodes and "R-2" not in nodes and "D-1" not in nodes
    assert nodes["track:main"]["detail"]["context_only"] is True
    assert "其他独立候选" not in json.dumps(flow, ensure_ascii=False)
    assert nodes["S-1"]["detail"]["history_ids"] == ["J-1"]
    with pytest.raises(ValueError, match="没有这条研究轨迹"):
        build_flow(snapshot, "candidate-missing")


def test_cited_shared_source_is_kept_but_unreferenced_foreign_source_is_hidden(snapshot):
    snapshot["sources"][0]["track_id"] = "main"
    snapshot["sources"].append({"id": "S-2", "track_id": "candidate-2", "title": "不相关来源"})
    nodes = _nodes(build_flow(snapshot, "candidate-1"))
    assert nodes["S-1"]["detail"]["verification"] == "agent_supplied"
    assert "S-2" not in nodes


def test_missing_evidence_is_a_visible_placeholder_not_silently_dropped(snapshot):
    snapshot["sources"] = []
    snapshot["ideas"][1]["source_ids"] = ["S-MISSING", "S-MISSING"]
    flow = build_flow(snapshot)
    assert _nodes(flow)["S-MISSING"]["status"] == "missing"
    assert {"record_id": "I-2", "reference_id": "S-MISSING", "reason": "missing"} in flow["missing_references"]
    assert sum(e["from"] == "S-MISSING" and e["to"] == "I-2" for e in flow["edges"]) == 1


@pytest.mark.parametrize("status,result,expected", [
    ("running", None, {}),
    ("failed", {"feedback_surface": "validate", "metrics": {"CVRMSE": 0.0}, "error": "training crashed"}, {}),
    ("completed", {"feedback_surface": "A", "metrics": {"CVRMSE": 0.33}}, {}),
    ("completed", {"metrics": {"CVRMSE": 0.33}}, {}),
    ("completed", {"feedback_surface": "validate", "metrics": {"CVRMSE": float("nan"), "R2": float("inf")}}, {}),
    ("completed", {"feedback_surface": "validate", "metrics": {"CVRMSE": 0.1, "R2": True}}, {"CVRMSE": 0.1}),
])
def test_failed_missing_and_wrong_surface_metrics_cannot_become_validation_scores(snapshot, status, result, expected):
    snapshot["jobs"][0].update(status=status, result=result)
    flow = build_flow(snapshot)
    node = _nodes(flow)["J-1"]
    assert node["metrics"] == expected
    assert node["detail"]["comparable"] is False
    json.dumps(flow, allow_nan=False)
    if status == "failed":
        assert node["detail"]["result"]["error"] == "training crashed"


def test_records_are_whitelisted_instead_of_exporting_internal_reasoning(snapshot):
    for name in ("sources", "ideas", "jobs", "findings", "reports", "messages"):
        snapshot[name][0]["chain_of_thought"] = "private-unpublished-trace"
    snapshot["sources"][0]["content"] = "unbounded-source-content"
    snapshot["messages"][0]["evidence"] = [{"secret_result": "foreign-embedded-evidence"}]
    snapshot["jobs"][0]["result"]["artifact_path"] = "hidden-holdout-artifact"
    snapshot["events"] = [{"payload": {"reasoning": "private-event-content"}}]
    serialized = json.dumps(build_flow(snapshot))
    for private in ("private-unpublished-trace", "unbounded-source-content", "foreign-embedded-evidence",
                    "hidden-holdout-artifact", "private-event-content"):
        assert private not in serialized


@pytest.mark.parametrize("actual_model", [
    {"spec": {"estimator": "ridge", "hyperparameters": {"alpha": 2}}},
    {"estimator": "ridge", "hyperparameters": {"alpha": 2}},
])
def test_job_label_and_details_prefer_actual_model_to_request(snapshot, actual_model):
    snapshot["jobs"][0]["request"]["model"] = {"estimator": "linear", "hyperparameters": {}}
    snapshot["jobs"][0]["result"]["model"] = actual_model
    snapshot["jobs"][0]["result"]["unrelated_private_field"] = "must-not-be-exported"
    node = _nodes(build_flow(snapshot))["J-1"]
    assert node["label"] == "EXP-1 · ridge"
    assert node["detail"]["result"]["model"] == actual_model
    assert node["detail"]["request"]["model"]["estimator"] == "linear"
    assert "unrelated_private_field" not in node["detail"]["result"]
    node["detail"]["result"]["model"].clear()
    assert snapshot["jobs"][0]["result"]["model"] == actual_model


def test_empty_and_partial_snapshots_are_readable_and_json_serializable():
    empty = build_flow({})
    assert empty["nodes"] == [] and empty["edges"] == [] and empty["lanes"] == []
    partial = build_flow({"run": {"run_id": "R"},
                          "tracks": [{"track_id": "candidate-1"}],
                          "jobs": [{"id": "J", "track_id": "candidate-1", "status": "queued"}],
                          "final_evaluation": {"status": "pending", "reason": "等待冻结候选"}})
    assert _nodes(partial)["J"]["metrics"] == {}
    assert _nodes(partial)["final_evaluation:R"]["status"] == "pending"
    assert not any(e["to"] == "final_evaluation:R" for e in partial["edges"])
    json.dumps(partial, allow_nan=False)
