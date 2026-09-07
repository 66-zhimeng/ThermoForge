"""Autonomous research gates survive retries, concurrency and store reopening."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from thermoforge_v2.contracts import V2Error
from thermoforge_v2.store import RunStore


def setup_run(tmp_path, *, candidates=2, **config):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001",
                            "research_mode": "autonomous", "candidates": candidates,
                            "max_experiments_per_track": 8, **config},
                           {"fingerprint": "frozen-data-features-seed-split-env"}, "autonomy")
    rid = run["run_id"]
    store.begin_run(rid)
    store.create_track(rid, "main", "main", status="running")
    for index in range(1, candidates + 1):
        store.create_track(rid, f"candidate-{index}", "candidate", status="running")
    return store, rid


def idea(store, rid, track, *, parents=()):
    return store.add_record(rid, "ideas", {
        "statement": "A falsifiable regularization hypothesis", "reason": "Own prior evidence",
        "origin": "history" if parents else "conjecture", "parent_job_ids": list(parents),
        "protocol_fingerprint": store.get_run(rid)["protocol"]["fingerprint"]}, track_id=track)


def proposal(store, rid, track, alpha=1.0, *, parents=(), purpose="explore"):
    hypothesis = idea(store, rid, track, parents=parents)
    return store.commit_proposal(rid, track, hypothesis["id"],
                                 {"category": "data", "estimator": "ridge",
                                  "hyperparameters": {"alpha": alpha}},
                                 {"purpose": purpose, "expected_cost": "one training run"},
                                 "proposal-" + hypothesis["id"])


def request(p):
    return {"proposal_id": p["id"], "model": p["model"],
            "protocol_fingerprint": p["protocol_fingerprint"],
            "experiment_fingerprint": p["experiment_fingerprint"],
            "lab_content_hash": p["lab_content_hash"]}


def reserve(store, rid, p, key=None):
    return store.reserve_job(rid, p["track_id"], p["idea_id"], request(p), key or p["id"])


def complete(store, rid, p, score=.1):
    job = reserve(store, rid, p)
    started = store.start_job(job["id"])
    assert started["started"] is True
    return store.settle_job(job["id"], "completed", {
        "feedback_surface": "validate", "protocol_fingerprint": p["protocol_fingerprint"],
        "metrics": {"CVRMSE": score}, "experiment_id": "EXP-" + job["id"]})


def finding(store, rid, job):
    return store.add_record(rid, "findings", {
        "statement": "Observed validation outcome", "interpretation": "Revise the hypothesis",
        "job_ids": [job["id"]]}, track_id=job["track_id"])


def report(store, rid, track, jobs):
    return store.add_record(rid, "reports", {"kind": "track", "body": "Evidence and limitations",
                            "job_ids": [j["id"] for j in jobs],
                            "idea_ids": [j["idea_id"] for j in jobs],
                            "finding_ids": [f["id"] for f in store.records(rid, "findings", track)]}, track_id=track)


def test_all_first_proposals_must_be_frozen_before_any_training_reservation(tmp_path):
    store, rid = setup_run(tmp_path)
    first = proposal(store, rid, "candidate-1")
    with pytest.raises(V2Error) as error:
        reserve(store, rid, first)
    assert error.value.code == "TFV2-PHASE"
    assert store.get_run(rid)["experiments_reserved"] == 0
    assert store.records(rid, "jobs") == []
    assert store.autonomy_state(rid)["stage"] == "independent_proposals"

    second = proposal(store, rid, "candidate-2", alpha=2.0)
    assert store.autonomy_state(rid)["stage"] == "independent_experiments"
    assert store.start_job(reserve(store, rid, first)["id"])["started"] is True
    assert store.start_job(reserve(store, rid, second)["id"])["started"] is True


def test_reopened_store_preserves_proposals_and_research_stage(tmp_path):
    store, rid = setup_run(tmp_path, strategy="top_k")
    first = proposal(store, rid, "candidate-1")
    reopened = RunStore(store.root)
    assert reopened.autonomy_state(rid)["stage"] == "independent_proposals"
    assert reopened.records(rid, "proposals")[0]["id"] == first["id"]
    second = proposal(reopened, rid, "candidate-2", alpha=2.0)
    assert RunStore(store.root).autonomy_state(rid)["stage"] == "independent_experiments"
    complete(reopened, rid, first)
    complete(reopened, rid, second)
    assert RunStore(store.root).autonomy_state(rid)["stage"] == "sharing"
    assert RunStore(store.root).get_run(rid)["research_stage"] == "sharing"


def test_sharing_depends_on_all_first_feedback_not_chat_turn_counts(tmp_path):
    store, rid = setup_run(tmp_path, strategy="adaptive")
    first = proposal(store, rid, "candidate-1")
    second = proposal(store, rid, "candidate-2", alpha=2.0)
    for track in ("main", "candidate-1", "candidate-2"):
        store.update_track(rid, track, {"turns": 200})
    assert store.autonomy_state(rid)["sharing_ready"] is False
    complete(store, rid, first)
    second_job = reserve(store, rid, second)
    store.start_job(second_job["id"])
    assert store.autonomy_state(rid)["sharing_ready"] is False
    store.settle_job(second_job["id"], "failed", {"error": "Fit failed: inspect model implementation"})
    assert store.autonomy_state(rid)["sharing_ready"] is True


def test_independent_duplicate_results_train_twice_and_only_external_snapshot_links_them(tmp_path):
    store, rid = setup_run(tmp_path)
    first = proposal(store, rid, "candidate-1")
    second = proposal(store, rid, "candidate-2")
    assert first["experiment_fingerprint"] == second["experiment_fingerprint"]
    first_job = complete(store, rid, first)
    second_job = complete(store, rid, second)
    assert first_job["executed"] is True and second_job["executed"] is True
    assert store.get_run(rid)["experiments_reserved"] == 2
    assert store.get_run(rid)["experiments_settled"] == 2
    assert store.autonomy_state(rid)["sharing_ready"] is False
    assert all("duplicate_of_job_id" not in j for j in store.records(rid, "jobs"))
    visible = store.snapshot(rid)["jobs"]
    assert "duplicate_of_job_id" not in visible[0]
    assert visible[1]["duplicate_of_job_id"] == first_job["id"]
    assert "duplicate_of_job_id" not in store.records(rid, "jobs", "candidate-2")[0]


def test_reuse_option_does_not_skip_first_independent_experiments(tmp_path):
    store, rid = setup_run(tmp_path, strategy="top_k", reuse_experiments=True)
    first = proposal(store, rid, "candidate-1")
    second = proposal(store, rid, "candidate-2")
    complete(store, rid, first)
    assert store.autonomy_state(rid)["sharing_ready"] is False
    second_job = complete(store, rid, second)
    assert second_job["executed"] is True
    assert "reused_from_job_id" not in second_job


@pytest.mark.parametrize("reuse,alpha,purpose,should_reuse", [
    (True, 1.0, "refine", True),
    (False, 1.0, "refine", False),
    (True, 3.0, "refine", False),
    (True, 1.0, "replicate", False),
])
def test_shared_reuse_requires_opt_in_exact_identity_and_non_replication(
        tmp_path, reuse, alpha, purpose, should_reuse):
    store, rid = setup_run(tmp_path, strategy="top_k", reuse_experiments=reuse)
    first = proposal(store, rid, "candidate-1", alpha=1.0)
    second = proposal(store, rid, "candidate-2", alpha=2.0)
    first_job = complete(store, rid, first, score=.08)
    second_job = complete(store, rid, second, score=.12)
    assert store.autonomy_state(rid)["sharing_ready"] is True
    finding(store, rid, second_job)
    revised = proposal(store, rid, "candidate-2", alpha=alpha,
                       parents=[second_job["id"]], purpose=purpose)
    reserved = reserve(store, rid, revised)
    started = store.start_job(reserved["id"])
    assert started["started"] is (not should_reuse)
    assert started["executed"] is (not should_reuse)
    if should_reuse:
        assert started["reused"] is True
        assert started["status"] == "completed"
        assert started["reused_from_job_id"] == first_job["id"]
        assert started["result"] == first_job["result"]
        assert store.get_run(rid)["experiments_settled"] == 3
        assert store.start_job(reserved["id"])["started"] is False
        assert store.get_run(rid)["experiments_settled"] == 3
    else:
        assert "reused_from_job_id" not in started
        assert store.get_run(rid)["experiments_settled"] == 2


def test_explicit_stop_requires_own_evidence_settled_jobs_and_fresh_complete_report(tmp_path):
    store, rid = setup_run(tmp_path)
    first = proposal(store, rid, "candidate-1")
    second = proposal(store, rid, "candidate-2")
    first_job = reserve(store, rid, first)
    store.start_job(first_job["id"])
    with pytest.raises(V2Error, match="尚未结束"):
        store.request_research_stop(rid, "candidate-1", "Cannot improve further", [first_job["id"]])
    stale_report = report(store, rid, "candidate-1", [first_job])
    settled = store.settle_job(first_job["id"], "completed", {"metrics": {"CVRMSE": .1}})
    with pytest.raises(V2Error, match="轨迹报告"):
        store.request_research_stop(rid, "candidate-1", "Cannot improve further", [settled["id"]])
    with pytest.raises(V2Error, match="本轨迹"):
        store.request_research_stop(rid, "candidate-1", "Cannot improve further", [second["id"]])
    with pytest.raises(V2Error, match="具体研究理由"):
        store.request_research_stop(rid, "candidate-1", "  ", [settled["id"]])
    report(store, rid, "candidate-1", [settled])
    with pytest.raises(V2Error, match="发现分析"):
        store.request_research_stop(rid, "candidate-1", "Cannot improve further", [settled["id"]])
    finding(store, rid, settled)
    current_report = report(store, rid, "candidate-1", [settled])
    stop = store.request_research_stop(rid, "candidate-1", " Further work needs new data ", [settled["id"]])
    assert stop["reason"] == "Further work needs new data"
    assert current_report["id"] in stop["evidence_ids"]
    assert stale_report["id"] not in stop["evidence_ids"]
    assert stop["job_ids"] == [settled["id"]]
    assert store.request_research_stop(rid, "candidate-1", "Same decision retried", [settled["id"]])["id"] == stop["id"]
    with pytest.raises(V2Error, match="停止"):
        proposal(store, rid, "candidate-1", alpha=3.0, parents=[settled["id"]])


def test_valid_no_experiment_stop_records_blocker_and_releases_proposal_barrier(tmp_path):
    store, rid = setup_run(tmp_path)
    first = proposal(store, rid, "candidate-1")
    blocker = idea(store, rid, "candidate-2")
    report(store, rid, "candidate-2", [])
    stop = store.request_research_stop(rid, "candidate-2", "Available data cannot test this hypothesis", [blocker["id"]])
    assert stop["job_ids"] == []
    assert store.autonomy_state(rid)["proposal_barrier_open"] is True
    assert store.start_job(reserve(store, rid, first)["id"])["started"] is True


def test_revision_requires_latest_real_feedback_and_analysis_before_new_proposal(tmp_path):
    store, rid = setup_run(tmp_path, candidates=0)
    first = proposal(store, rid, "main")
    finished = complete(store, rid, first)
    with pytest.raises(V2Error, match="最近一次真实实验反馈"):
        proposal(store, rid, "main", alpha=2.0)
    with pytest.raises(V2Error, match="发现与结果分析"):
        proposal(store, rid, "main", alpha=2.0, parents=[finished["id"]])
    analysis = finding(store, rid, finished)
    revised = proposal(store, rid, "main", alpha=2.0, parents=[finished["id"]])
    assert revised["version"] == 2
    assert revised["parent_job_ids"] == [finished["id"]]
    assert revised["finding_ids"] == [analysis["id"]]


def test_proposal_retry_after_new_findings_returns_original_frozen_snapshot(tmp_path):
    store, rid = setup_run(tmp_path, candidates=0)
    first = proposal(store, rid, "main")
    finished = complete(store, rid, first)
    analysis = finding(store, rid, finished)
    retry = store.commit_proposal(rid, "main", first["idea_id"], first["model"],
                                  {"purpose": "explore", "expected_cost": "one training run",
                                   "finding_ids": [analysis["id"]]}, first["idempotency_key"])
    assert retry["id"] == first["id"]
    assert retry["fresh"] is False
    assert retry["finding_ids"] == first["finding_ids"] == []
    assert len(store.records(rid, "proposals")) == 1


def test_repeat_of_own_exact_experiment_requires_explicit_replication_purpose(tmp_path):
    store, rid = setup_run(tmp_path, candidates=0)
    first = proposal(store, rid, "main")
    finished = complete(store, rid, first)
    finding(store, rid, finished)
    with pytest.raises(V2Error) as error:
        proposal(store, rid, "main", parents=[finished["id"]], purpose="refine")
    assert error.value.code == "TFV2-DUPLICATE"
    repeated = proposal(store, rid, "main", parents=[finished["id"]], purpose="replicate")
    assert repeated["experiment_fingerprint"] == first["experiment_fingerprint"]
    assert repeated["purpose"] == "replicate"
    assert complete(store, rid, repeated)["executed"] is True


def test_same_proposal_reserved_concurrently_with_new_keys_consumes_one_job(tmp_path):
    store, rid = setup_run(tmp_path, candidates=0, max_experiments=1)
    first = proposal(store, rid, "main")
    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(lambda index: reserve(store, rid, first, f"retry-{index}"), range(8)))
    assert len({j["id"] for j in reservations}) == 1
    assert sum(j["fresh"] for j in reservations) == 1
    assert store.get_run(rid)["experiments_reserved"] == 1
    assert len(store.records(rid, "jobs")) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        starts = list(pool.map(lambda job: store.start_job(job["id"]), reservations))
    assert sum(j["started"] for j in starts) == 1


@pytest.mark.parametrize("field,value", [
    ("model", {"category": "data", "estimator": "linear"}),
    ("experiment_fingerprint", "forged-identity"),
    ("protocol_fingerprint", "different-split-or-seed"),
    ("proposal_id", "missing-proposal"),
    ("lab_content_hash", "a" * 64),
])
def test_experiment_request_cannot_change_frozen_proposal(tmp_path, field, value):
    store, rid = setup_run(tmp_path, candidates=0)
    first = proposal(store, rid, "main")
    with pytest.raises(V2Error) as error:
        store.reserve_job(rid, "main", first["idea_id"], request(first) | {field: value}, "changed")
    assert error.value.code == "TFV2-PROPOSAL"
    assert store.get_run(rid)["experiments_reserved"] == 0
