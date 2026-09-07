"""提前为报告留出 token；停止新研究，不丢失已预留工作和结题证据。"""

import math

import pytest

from thermoforge_v2.contracts import V2Error
from thermoforge_v2.store import RunStore


MODEL = {"category": "data", "estimator": "ridge", "hyperparameters": {"alpha": 1}}


def setup_run(tmp_path, *, mode="autonomous", budget=10000, candidates=1):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001",
                            "research_mode": mode, "token_budget": budget,
                            "candidates": candidates}, {"fingerprint": "frozen"}, "closure-test")
    rid = run["run_id"]
    store.update_run(rid, {"status": "running"})
    store.create_track(rid, "main", "main")
    for index in range(1, candidates + 1):
        store.create_track(rid, f"candidate-{index}", "candidate")
    return store, rid


def idea(store, rid, tid="candidate-1", **fields):
    return store.add_record(rid, "ideas", {
        "statement": "研究假说", "origin": "conjecture", "reason": "检验模型的实际表现",
        "prediction": "验证指标改善", "falsification": "同协议下无改善",
        "protocol_fingerprint": "frozen", **fields}, track_id=tid)


def proposal(store, rid, key="first"):
    item = idea(store, rid)
    return store.commit_proposal(rid, "candidate-1", item["id"], MODEL,
                                  {"purpose": "explore"}, key)


def request(plan):
    return {"model": plan["model"], "proposal_id": plan["id"],
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "experiment_fingerprint": plan["experiment_fingerprint"]}


def reserve(store, rid, plan, key="first-job"):
    return store.reserve_job(rid, "candidate-1", plan["idea_id"], request(plan), key)


def completed_evidence(store, rid, job):
    store.settle_job(job["id"], "completed", {"feedback_surface": "validate", "metrics": {"CVRMSE": .1}})
    finding = store.add_record(rid, "findings", {
        "job_ids": [job["id"]], "statement": "验证结果", "interpretation": "记录实际反馈"}, track_id="candidate-1")
    report = store.add_record(rid, "reports", {
        "kind": "track", "idea_ids": [job["idea_id"]], "job_ids": [job["id"]],
        "finding_ids": [finding["id"]], "summary": "按预算收尾，未声称达到目标"}, track_id="candidate-1")
    return finding, report


def test_no_usage_uses_fraction_only_and_closure_event_is_durable(tmp_path):
    store, rid = setup_run(tmp_path, budget=1000)
    store.update_run(rid, {"tokens_used": 749})
    state = store.autonomy_state(rid)
    assert state["report_token_reserve"] == 250
    assert state["tokens_remaining"] == 251 and state["closing_for_tokens"] is False
    store.update_run(rid, {"tokens_used": 750})
    assert store.autonomy_state(rid)["closing_for_tokens"] is True
    closure = store.get_run(rid)["research_closure"]
    assert closure["reason"] == "token_report_reserve"
    assert closure["usage_reserve_tokens"] == 0
    assert "不保证足够" in closure["estimate_note"]
    reopened = RunStore(store.root)
    assert reopened.autonomy_state(rid)["closing_for_tokens"] is True
    assert reopened.get_run(rid)["research_closure"] == closure
    assert sum(event["kind"] == "research.closure_started" for event in reopened.events(rid)["events"]) == 1


def test_usage_last_reserves_multiple_calls_and_main_context_without_double_counting_cache(tmp_path):
    store, rid = setup_run(tmp_path, budget=100000, candidates=2)
    store.update_track(rid, "candidate-1", {"usage": {"last": {
        "inputTokens": 5500, "outputTokens": 500, "totalTokens": 6000, "cachedInputTokens": 4000}}})
    store.update_track(rid, "candidate-2", {"usage": {"last": {
        "inputTokens": 4000, "outputTokens": 1000, "cachedInputTokens": 3000}}})
    store.update_track(rid, "main", {"usage": {"last": {"totalTokens": 1000}}})
    reserve_tokens = math.ceil((6000 + 5000 + 6000) * 4 * 1.2)
    store.update_run(rid, {"tokens_used": 100000 - reserve_tokens})
    state = store.autonomy_state(rid)
    assert state["closing_for_tokens"] and state["report_token_reserve"] == reserve_tokens
    closure = store.get_run(rid)["research_closure"]
    assert closure["observed_last_call_tokens_by_track"]["main"] == 1000
    assert closure["estimated_call_tokens_by_track"]["main"] == 6000
    assert closure["estimated_call_tokens_by_track"]["candidate-2"] == 5000


def test_rejected_new_job_persists_closure_without_reserving_an_experiment(tmp_path):
    store, rid = setup_run(tmp_path)
    plan = proposal(store, rid)
    store.update_run(rid, {"tokens_used": 7500})
    with pytest.raises(V2Error) as denied:
        reserve(store, rid, plan)
    assert denied.value.code == "TFV2-REPORT-RESERVE"
    assert store.get_run(rid)["research_closure"]["reason"] == "token_report_reserve"
    assert store.get_run(rid)["experiments_reserved"] == 0
    assert store.records(rid, "jobs") == []
    assert any(e["kind"] == "research.closure_started" for e in store.events(rid)["events"])


def test_rejected_new_proposal_preserves_closure_but_original_retry_succeeds(tmp_path):
    store, rid = setup_run(tmp_path)
    plan = proposal(store, rid)
    later = idea(store, rid)
    store.update_run(rid, {"tokens_used": 7500})
    with pytest.raises(V2Error) as denied:
        store.commit_proposal(rid, "candidate-1", later["id"], MODEL, {"purpose": "explore"}, "new")
    assert denied.value.code == "TFV2-REPORT-RESERVE"
    assert store.get_run(rid)["research_closure"]["reason"] == "token_report_reserve"
    repeated = store.commit_proposal(rid, "candidate-1", plan["idea_id"], MODEL,
                                     {"purpose": "explore"}, "first")
    assert repeated["id"] == plan["id"] and not repeated["fresh"]
    assert len(store.records(rid, "proposals")) == 1


def test_reserved_work_retry_and_report_stop_remain_available_during_closure(tmp_path):
    store, rid = setup_run(tmp_path)
    plan = proposal(store, rid)
    job = reserve(store, rid, plan)
    store.update_run(rid, {"tokens_used": 7500})
    assert store.autonomy_state(rid)["closing_for_tokens"]
    repeated = reserve(store, rid, plan)
    alias = reserve(store, rid, plan, key="same-proposal-new-key")
    assert repeated["id"] == alias["id"] == job["id"]
    assert not repeated["fresh"] and not alias["fresh"]
    assert store.start_job(job["id"])["started"] is True
    finding, report = completed_evidence(store, rid, job)
    stopped = store.request_research_stop(rid, "candidate-1", "为报告保留预算后结题",
                                           [finding["id"], report["id"]])
    assert stopped["job_ids"] == [job["id"]]
    assert store.get_run(rid)["experiments_reserved"] == store.get_run(rid)["experiments_settled"] == 1


def test_same_budget_resume_stays_closed_and_explicit_increase_can_release(tmp_path):
    store, rid = setup_run(tmp_path)
    store.update_run(rid, {"tokens_used": 7500})
    assert store.autonomy_state(rid)["closing_for_tokens"]
    store.control(rid, "pause")
    store.update_run(rid, {"status": "paused"})
    store.control(rid, "update", changes={"token_budget": 10000, "guidance": "先整理报告"})
    store.control(rid, "resume")
    assert RunStore(store.root).autonomy_state(rid)["closing_for_tokens"]
    store.control(rid, "update", changes={"token_budget": 20000})
    state = store.autonomy_state(rid)
    assert state["closing_for_tokens"] is False and state["tokens_remaining"] == 12500
    assert "research_closure" not in store.get_run(rid)
    assert any(e["kind"] == "research.closure_released" for e in store.events(rid)["events"])


def test_small_increase_retains_closure_when_report_estimate_still_needs_it(tmp_path):
    store, rid = setup_run(tmp_path)
    store.update_track(rid, "candidate-1", {"usage": {"last": {"totalTokens": 1000}}})
    store.update_run(rid, {"tokens_used": 7500})
    assert store.autonomy_state(rid)["closing_for_tokens"]
    store.control(rid, "update", changes={"token_budget": 11000})
    assert store.autonomy_state(rid)["closing_for_tokens"]
    assert store.get_run(rid)["research_closure"]["token_budget"] == 11000


def test_budget_increase_never_removes_existing_research_stops(tmp_path):
    store, rid = setup_run(tmp_path)
    plan = proposal(store, rid)
    job = reserve(store, rid, plan)
    store.start_job(job["id"])
    finding, report = completed_evidence(store, rid, job)
    store.update_run(rid, {"tokens_used": 7500})
    store.autonomy_state(rid)
    stop = store.request_research_stop(rid, "candidate-1", "依据已完成实验停止", [report["id"], finding["id"]])
    store.control(rid, "update", changes={"token_budget": 20000})
    assert not store.autonomy_state(rid)["closing_for_tokens"]
    assert store.records(rid, "stops")[0]["id"] == stop["id"]


def test_acceptance_mode_does_not_use_report_reserve_or_block_new_jobs(tmp_path):
    store, rid = setup_run(tmp_path, mode="acceptance")
    store.update_run(rid, {"tokens_used": 9900})
    store.update_track(rid, "candidate-1", {"usage": {"last": {"totalTokens": 100000}}})
    state = store.autonomy_state(rid)
    assert not state["closing_for_tokens"] and state["report_token_reserve"] == 0
    assert "research_closure" not in store.get_run(rid)
    job = store.reserve_job(rid, "candidate-1", "legacy-idea", {"model": "linear"}, "legacy")
    assert job["fresh"] is True
    assert store.start_job(job["id"])["started"] is True
