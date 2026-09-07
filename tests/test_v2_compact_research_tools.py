"""Incremental evidence views preserve identity boundaries and genuine diagnostics."""

from __future__ import annotations

import base64
import json

import pytest

from test_v2_research_tools import setup, idea, experiment
from thermoforge_v2.evidence_projection import compact_record, safe_feedback
from thermoforge_v2.feedback import research_diagnostics
from thermoforge_v2.research_tools import ResearchTools


def history(tools, **args):
    result = tools.call("research_history", args)
    assert result["ok"], result
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 32 * 1024
    return result["summary"]


def test_history_pages_all_records_and_reuses_empty_cursor_for_new_records(setup):
    _, store, run_id, tools = setup
    ids = [store.add_record(run_id, "ideas", {"statement": f"idea {i}"},
                           track_id="candidate_1")["id"] for i in range(57)]
    received, cursor = [], None
    for _ in range(8):
        page = history(tools, kind="ideas", limit=11, **({"cursor": cursor} if cursor else {}))
        received += [item["id"] for item in page["records"]]
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
    assert received == ids
    empty = history(tools, kind="ideas", cursor=cursor)
    assert empty["records"] == [] and empty["next_cursor"] == cursor
    new = idea(tools)
    assert [r["id"] for r in history(tools, kind="ideas", cursor=cursor)["records"]] == [new]


def test_history_cursor_scope_kind_and_anchor_are_checked(setup):
    ctx, store, run_id, tools = setup
    idea(tools)
    cursor = history(tools, kind="ideas")["next_cursor"]
    foreign = ResearchTools(store, ctx, run_id, "candidate_2")
    assert not foreign.call("research_history", {"kind": "ideas", "cursor": cursor})["ok"]
    assert not tools.call("research_history", {"kind": "jobs", "cursor": cursor})["ok"]
    doc = json.loads(base64.urlsafe_b64decode(cursor))
    doc["after"] = "missing-id"
    forged = base64.urlsafe_b64encode(json.dumps(doc).encode()).decode()
    assert not tools.call("research_history", {"kind": "ideas", "cursor": forged})["ok"]
    for value in ("invalid", [], "" * 0, "x" * 2001):
        assert not tools.call("research_history", {"cursor": value})["ok"]


def test_full_id_lookup_uses_same_sharing_boundary_as_evidence_references(setup):
    ctx, store, run_id, tools = setup
    other = ResearchTools(store, ctx, run_id, "candidate_2")
    foreign = idea(other)
    args = {"kind": "ideas", "record_ids": [foreign], "compact": False}
    assert not tools.call("research_history", args)["ok"]
    store.add_record(run_id, "messages", {"from": "main", "to": "candidate_1", "text": "share",
                     "evidence_ids": [foreign]}, track_id="main")
    store.update_track(run_id, "candidate_1", {"turns": 2})
    assert not tools.call("research_history", args)["ok"]  # independent remains private
    store.update_run(run_id, {"config": store.get_run(run_id)["config"] | {"strategy": "top_k"}})
    assert history(tools, **args)["records"][0]["id"] == foreign
    # Sharing one ID never implicitly grants access to its other referenced IDs.
    private = idea(other)
    assert not tools.call("research_history", args | {"record_ids": [foreign, private]})["ok"]


def test_projection_omits_source_bodies_code_and_heldout_sentinels(setup):
    _, store, run_id, tools = setup
    source = store.add_record(run_id, "sources", {"title": "Paper", "content": "PRIVATE_SOURCE_BODY" * 500,
                    "read_scope": "fulltext", "verification": "retrieved", "source": "MODEL_SOURCE_SENTINEL"},
                    track_id="candidate_1")
    for compact in (True, False):
        result = history(tools, kind="sources", record_ids=[source["id"]], compact=compact)
        assert "PRIVATE_SOURCE_BODY" not in json.dumps(result)
        assert "MODEL_SOURCE_SENTINEL" not in json.dumps(result)
        assert result["records"][0]["content_chars"] > 0
    bad = tools.call("research_history", {"kind": "sources", "record_ids": [source["id"]],
                                          "compact": False, "fields": ["content"]})
    assert not bad["ok"]
    job = store.add_record(run_id, "jobs", {"status": "completed", "artifacts": "HOLDOUT_ARTIFACT",
        "request": {"model": {"category": "lab", "source": "MODEL_SOURCE_SENTINEL"}},
        "result": {"feedback_surface": "validate", "metrics": {"CVRMSE": .2, "A": "HOLDOUT_METRIC"},
                   "surfaces": {"A": "HOLDOUT_SURFACE"}, "physics_overall_rate": "HOLDOUT_PHYSICS"}},
        track_id="candidate_1")
    for compact in (True, False):
        response = history(tools, record_ids=[job["id"]], compact=compact)
        text = json.dumps(response)
        assert "HOLDOUT_" not in text and "MODEL_SOURCE_SENTINEL" not in text
        assert response["records"][0]["result"]["metrics"] == {"CVRMSE": .2}
    assert "HOLDOUT_" not in json.dumps(tools._job_response(job))


def test_large_record_pages_do_not_advance_over_undelivered_records(setup):
    _, store, run_id, tools = setup
    ids = [store.add_record(run_id, "ideas", {key: "论" * 12000 for key in
            ("statement", "reason", "prediction", "falsification")}, track_id="candidate_1")["id"]
           for _ in range(12)]
    received, cursor = [], None
    for _ in range(12):
        page = history(tools, kind="ideas", limit=50, **({"cursor": cursor} if cursor else {}))
        received.extend(r["id"] for r in page["records"])
        cursor = page["next_cursor"]
        assert page["records"]
        if not page["has_more"]:
            break
    assert received == ids


def test_full_record_projection_and_long_text_can_be_reconstructed_by_id(setup):
    _, store, run_id, tools = setup
    body = "可追溯正文" * 17000
    record = store.add_record(run_id, "reports", {"title": "report", "summary": "summary", "body": body},
                              track_id="candidate_1")
    compact = history(tools, kind="reports")["records"][0]
    assert "body" not in compact and compact["truncated"]
    parts, offset = [], 0
    for _ in range(20):
        page = history(tools, kind="reports", record_ids=[record["id"]], compact=False,
                       fields=["body"], offset=offset)["records"][0]
        parts.append(page["body"])
        offset = page["text_page"]["next_offset"]
        if not page["text_page"]["has_more"]:
            break
    assert "".join(parts) == body
    assert history(tools, kind="reports", record_ids=[record["id"]], compact=False,
                   fields=["title"])["records"] == [{"id": record["id"], "title": "report"}]
    for args in ({"offset": 0}, {"compact": False, "offset": 0}, {"fields": ["artifacts"]}):
        assert not tools.call("research_history", {"kind": "reports"} | args)["ok"]


def test_by_id_batch_reports_unreturned_ids_instead_of_skipping(setup):
    _, store, run_id, tools = setup
    records = [store.add_record(run_id, "ideas", {"statement": "论" * 6000}, track_id="candidate_1")
               for _ in range(4)]
    ids = [r["id"] for r in records]
    result = history(tools, kind="ideas", record_ids=ids, compact=False)
    assert [r["id"] for r in result["records"]] + result["remaining_record_ids"] == ids
    assert result["has_more"]


def test_compact_projection_is_bounded_even_with_many_large_hyperparameters():
    model = {"category": "lab", "hyperparameters": {f"parameter{i}": "论" * 12000 for i in range(80)}}
    record = {"id": "JOB-safe", "status": "completed", "request": {"model": model},
              "result": {"feedback_surface": "validate", "model": {"spec": model}}}
    result = compact_record("jobs", record)
    assert result["id"] == "JOB-safe" and result["truncated"]
    assert len(json.dumps(result, ensure_ascii=False).encode()) < 14 * 1024
    assert "evidence" not in compact_record("messages", {"id": "message", "text": "hi",
                                                           "evidence": [record], "evidence_ids": ["JOB-safe"]})


def test_diagnostics_do_not_infer_training_or_search_physics_from_holdout():
    feedback = research_diagnostics({"surfaces": {
        "validate": {"n_samples": 9, "metrics": {"CVRMSE": .2, "NMBE": .03}},
        "A": {"n_samples": 300, "metrics": {"CVRMSE": "HELD_OUT"}},
        "B": {"metrics": {"NMBE": "HELD_OUT"}}}, "physics_overall_rate": "HELD_OUT"})
    assert "HELD_OUT" not in json.dumps(feedback)
    assert feedback["train"]["status"] == "unavailable"
    assert feedback["generalization_gap"]["status"] == "unavailable"
    assert feedback["physics"]["status"] == "unavailable"
    assert feedback["bias"]["nmbe"] == .03
    assert feedback["bias"]["direction"] == "normalized_positive"
    assert "目标均值为正" in feedback["bias"]["convention"]


def test_diagnostics_use_only_valid_available_training_and_validation_values():
    feedback = research_diagnostics({"surfaces": {
        "train": {"n_samples": 30, "metrics": {"CVRMSE": .1, "NMBE": -.02, "R2": .9}},
        "validate": {"n_samples": 9, "metrics": {"CVRMSE": .2, "NMBE": -.03, "R2": .8,
                                                 "RMSE": float("nan"), "MAE": True}}}})
    assert feedback["generalization_gap"]["validate_minus_train"]["CVRMSE"] == pytest.approx(.1)
    assert feedback["generalization_gap"]["validate_minus_train"]["R2"] == pytest.approx(-.1)
    assert feedback["bias"]["direction"] == "normalized_negative"
    assert "RMSE" not in feedback["validate"]["metrics"] and "MAE" not in feedback["validate"]["metrics"]


def test_real_experiment_feedback_includes_available_bias_and_explicit_missing_train(setup):
    _, _, _, tools = setup
    response = experiment(tools, idea(tools))
    assert response["ok"], response
    feedback = response["summary"]["result"]
    assert feedback["diagnostics"]["bias"]["nmbe"] == feedback["metrics"]["NMBE"]
    assert feedback["diagnostics"]["train"]["status"] == "unavailable"
    assert feedback["diagnostics"]["physics"]["status"] == "unavailable"


def test_constraint_projection_only_preserves_verified_validation_evidence():
    verified = {"evaluated_on": "validate", "verified": True, "value": .02}
    result = safe_feedback({"feedback_surface": "validate", "constraint_evidence": {
        "physics_violation_rate": verified | {"raw": "PRIVATE_ARTIFACT"},
        "inference_latency_ms": {"evaluated_on": "test", "verified": True, "value": 7001},
        "extrapolation": {"evaluated_on": "validate", "verified": True, "value": "HOLDOUT"}}})
    assert result["constraint_evidence"] == {"physics_violation_rate": verified}
    assert "PRIVATE_ARTIFACT" not in json.dumps(result) and "HOLDOUT" not in json.dumps(result)
    for patch in ({"value": float("nan")}, {"value": True}, {"value": -1}, {"verified": False}, {"evaluated_on": "A"}):
        assert "constraint_evidence" not in safe_feedback({"constraint_evidence": {
            "physics_violation_rate": verified | patch}})
