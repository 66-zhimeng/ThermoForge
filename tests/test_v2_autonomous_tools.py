"""自主工具通过真实存储/实验内核验证方案屏障、反馈谱系和有据停止。"""

from __future__ import annotations

import json

import pytest

from phase2_helpers import FEATURES, TARGET
from phase34_helpers import make_ctx
from thermoforge_research.tools import tf_goal_create
from thermoforge_v2.research_tools import ResearchTools, prepare_protocol
from thermoforge_v2.store import RunStore


MODEL = {"category": "data", "estimator": "ridge", "hyperparameters": {"alpha": 1.0}}


@pytest.fixture()
def autonomy(tmp_path):
    ctx, ref = make_ctx(tmp_path, n_steps=600)
    goal = tf_goal_create(ctx, {
        "name": "自主研究工具验收", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES), "acceptance": {"cvrmse_max": .5},
    })
    assert goal["ok"], goal
    config = {"goal_id": goal["id"], "dataset_ref": ref, "candidates": 2,
              "research_mode": "autonomous", "max_experiments": 8,
              "max_experiments_per_track": 4, "seed": 79,
              "purge_seconds": 0, "embargo_seconds": 2700}
    store = RunStore(tmp_path / "v2")
    run = store.create_run(config, prepare_protocol(ctx, config), "autonomous-tools")
    run_id = run["run_id"]
    store.update_run(run_id, {"status": "running"})
    for track_id in ("candidate-1", "candidate-2"):
        store.create_track(run_id, track_id, "candidate",
                           research_root=str(tmp_path / "tracks" / track_id),
                           workspace=str(tmp_path / "workspaces" / track_id))
    return store, run_id, [ResearchTools(store, ctx, run_id, track_id)
                           for track_id in ("candidate-1", "candidate-2")]


def new_idea(tools, **updates):
    args = {"statement": "自主选择正则化模型作为起点", "reason": "用可比较实验检验收缩假说",
            "prediction": "验证误差下降", "falsification": "相同协议验证误差不降则否定",
            "origin": "conjecture"} | updates
    response = tools.call("research_idea_create", args)
    assert response["ok"], response
    return response["id"]


def proposal(tools, idea_id=None, model=None, key="proposal-1", purpose="explore"):
    return tools.call("research_proposal_commit", {
        "idea_id": idea_id or new_idea(tools), "model": model or MODEL,
        "purpose": purpose, "expected_cost": "一项本地训练实验",
        "idempotency_key": key})


def run_proposal(tools, frozen, key="experiment-1", model=None):
    return tools.call("research_experiment_run", {
        "idea_id": frozen["summary"]["idea_id"], "model": model or MODEL,
        "proposal_id": frozen["id"], "idempotency_key": key})


def ready(autonomy):
    store, run_id, tracks = autonomy
    frozen = [proposal(track) for track in tracks]
    assert all(item["ok"] for item in frozen), frozen
    return store, run_id, tracks, frozen


def finding(tools, job_id, idea_id):
    response = tools.call("research_finding_create", {
        "statement": "记录首轮验证结果", "interpretation": "据实判断下一轮是否扩大正则强度",
        "limitations": "一次验证无统计显著性证据", "job_ids": [job_id], "idea_ids": [idea_id]})
    assert response["ok"], response
    return response["id"]


def test_autonomous_experiment_requires_committed_plan_and_all_first_proposals(autonomy):
    store, run_id, (first, second) = autonomy
    idea_id = new_idea(first)
    missing = first.call("research_experiment_run", {
        "idea_id": idea_id, "model": MODEL, "idempotency_key": "missing"})
    assert not missing["ok"] and "proposal_id" in missing["error"]
    frozen = proposal(first, idea_id)
    assert frozen["ok"], frozen
    assert not run_proposal(first, frozen)["ok"]
    assert store.get_run(run_id)["experiments_reserved"] == 0
    stage = first.call("research_protocol", {})["summary"]["research_stage"]
    assert not stage["proposal_barrier_open"]
    assert proposal(second)["ok"]
    finished = run_proposal(first, frozen)
    assert finished["ok"] and finished["summary"]["status"] == "completed", finished
    assert finished["summary"]["result"]["metrics"]["CVRMSE"] is not None
    assert '"A"' not in json.dumps(finished)
    repeated = run_proposal(first, frozen, key="same-proposal-new-request")
    assert repeated["ok"] and repeated["id"] == finished["id"]
    assert repeated["summary"]["duplicate"]
    assert store.get_run(run_id)["experiments_reserved"] == 1


def test_committed_plan_is_idempotent_and_model_cannot_drift(autonomy):
    store, run_id, tracks, frozen = ready(autonomy)
    first, second = tracks
    again = proposal(first, frozen[0]["summary"]["idea_id"])
    assert again["ok"] and again["id"] == frozen[0]["id"], again
    assert len(store.records(run_id, "proposals", track_id=first.track_id)) == 1
    changed = MODEL | {"hyperparameters": {"alpha": 50.0}}
    conflict = proposal(first, frozen[0]["summary"]["idea_id"], model=changed)
    assert not conflict["ok"]
    assert not run_proposal(first, frozen[0], model=changed)["ok"]
    assert not run_proposal(second, frozen[0])["ok"]
    assert store.get_run(run_id)["experiments_reserved"] == 0


def test_revision_requires_actual_own_feedback_and_analyzing_finding(autonomy):
    store, run_id, tracks, frozen = ready(autonomy)
    first = tracks[0]
    finished = run_proposal(first, frozen[0])
    assert finished["ok"], finished
    changed = MODEL | {"hyperparameters": {"alpha": 2.0}}
    ungrounded = proposal(first, model=changed, key="revision", purpose="refine")
    assert not ungrounded["ok"]
    next_idea = new_idea(first, origin="history", parent_job_ids=[finished["id"]],
                         parent_idea_ids=[frozen[0]["summary"]["idea_id"]],
                         reason="根据首轮真实误差测试不同收缩强度")
    unanalyzed = proposal(first, next_idea, changed, "revision", "refine")
    assert not unanalyzed["ok"]
    analysis = finding(first, finished["id"], frozen[0]["summary"]["idea_id"])
    committed = proposal(first, next_idea, changed, "revision", "refine")
    assert committed["ok"], committed
    assert committed["summary"]["version"] == 2
    assert analysis in committed["summary"]["finding_ids"]
    assert finished["id"] in committed["summary"]["parent_job_ids"]
    second = run_proposal(first, committed, "experiment-2", changed)
    assert second["summary"]["status"] == "completed", second
    assert store.get_run(run_id)["experiments_reserved"] == 2


def test_repeating_exact_plan_requires_explicit_replication_purpose(autonomy):
    _, _, tracks, frozen = ready(autonomy)
    first = tracks[0]
    finished = run_proposal(first, frozen[0])
    assert finished["ok"], finished
    finding(first, finished["id"], frozen[0]["summary"]["idea_id"])
    repeat_idea = new_idea(first, origin="history", parent_job_ids=[finished["id"]])
    hidden = proposal(first, repeat_idea, key="replication", purpose="refine")
    assert not hidden["ok"]
    explicit = proposal(first, repeat_idea, key="replication", purpose="replicate")
    assert explicit["ok"], explicit
    result = run_proposal(first, explicit, "replication-job")
    assert result["ok"] and result["summary"]["status"] == "completed", result


def test_reading_history_does_not_reveal_other_candidates(autonomy):
    _, _, tracks, frozen = ready(autonomy)
    first, second = tracks
    history = first.call("research_history", {"kind": "proposals"})
    assert history["ok"]
    assert [item["id"] for item in history["summary"]["records"]] == [frozen[0]["id"]]
    state = first.call("research_protocol", {})["summary"]
    assert frozen[1]["id"] not in json.dumps(state)
    assert second.track_id not in json.dumps(state)


def test_turn_count_cannot_unlock_unfinished_autonomous_sharing(autonomy):
    store, run_id, (first, second) = autonomy
    config = store.get_run(run_id)["config"] | {"strategy": "top_k"}
    store.update_run(run_id, {"config": config})
    source = second.call("research_source_register", {
        "title": "未共享的研究资料", "kind": "paper", "url": "https://example.org/study",
        "read_scope": "abstract", "content": "资料摘要"})
    store.add_record(run_id, "messages", {"from": "main", "to": first.track_id,
        "text": "提前广播不应放行", "evidence_ids": [source["id"]]}, track_id="main")
    store.update_track(run_id, first.track_id, {"turns": 99})
    denied = first.call("research_source_excerpt", {"source_id": source["id"]})
    assert not denied["ok"]


def test_coordinator_cannot_read_independent_candidate_evidence_until_final_report(autonomy):
    store, run_id, (first, _) = autonomy
    track_root = store.root / "tracks" / "main"
    store.create_track(run_id, "main", "main", research_root=str(track_root / "research"),
                       workspace=str(track_root / "workspace"))
    main = ResearchTools(store, first.base_ctx, run_id, "main")
    common = main.call("research_source_register", {
        "title": "共同任务资料", "kind": "paper", "url": "https://example.org/common-study",
        "read_scope": "abstract", "content": "事先确定的共同初始证据"})
    assert common["ok"], common
    store.add_record(run_id, "messages", {"from": "main", "to": "all", "initial_task": True,
        "text": "共同资料", "evidence_ids": [common["id"]]}, track_id="main")
    assert first.call("research_source_excerpt", {"source_id": common["id"]})["ok"]
    source = first.call("research_source_register", {
        "title": "独立候选资料", "kind": "paper", "url": "https://example.org/private-study",
        "read_scope": "abstract", "content": "尚未开放共享的独立研究资料"})
    assert source["ok"], source
    assert not main.call("research_source_excerpt", {"source_id": source["id"]})["ok"]
    store.update_run(run_id, {"final_report_phase": True})
    opened = main.call("research_source_excerpt", {"source_id": source["id"]})
    assert opened["ok"] and opened["summary"]["content"] == "尚未开放共享的独立研究资料"


def test_restored_candidate_still_completes_own_first_experiment_before_sharing(autonomy):
    store, run_id, (first, second) = autonomy
    store.update_run(run_id, {"config": store.get_run(run_id)["config"] | {"strategy": "top_k"}})
    store.update_track(run_id, second.track_id, {"status": "failed"})
    frozen = proposal(first)
    assert frozen["ok"], frozen
    finished = run_proposal(first, frozen)
    assert finished["ok"] and finished["summary"]["status"] == "completed", finished
    assert store.autonomy_state(run_id)["sharing_ready"]
    source = first.call("research_source_register", {
        "title": "首轮候选资料", "kind": "paper", "url": "https://example.org/first-study",
        "read_scope": "abstract", "content": "真实研究资料摘要"})
    assert source["ok"], source
    store.add_record(run_id, "messages", {"from": "main", "to": second.track_id,
        "text": "首轮后分享", "evidence_ids": [source["id"]]}, track_id="main")
    store.update_track(run_id, second.track_id, {"status": "running", "turns": 99})
    assert not second.call("research_source_excerpt", {"source_id": source["id"]})["ok"]
    restored = proposal(second)
    assert restored["ok"], restored
    assert not second.call("research_source_excerpt", {"source_id": source["id"]})["ok"]
    completed = run_proposal(second, restored)
    assert completed["ok"] and completed["summary"]["status"] == "completed", completed
    assert second.call("research_source_excerpt", {"source_id": source["id"]})["ok"]


def test_stop_requires_complete_report_and_real_own_evidence(autonomy):
    _, _, tracks, frozen = ready(autonomy)
    first = tracks[0]
    finished = run_proposal(first, frozen[0])
    assert finished["ok"], finished
    analysis = finding(first, finished["id"], frozen[0]["summary"]["idea_id"])
    assert not first.call("research_stop", {"reason": "结果充分", "evidence_ids": [analysis]})["ok"]
    incomplete = first.call("research_report_submit", {
        "title": "未覆盖发现的阶段报告", "summary": "仅列出作业编号",
        "body": "尚未整理所有反馈分析。", "limitations": "缺少分析依据",
        "idea_ids": [frozen[0]["summary"]["idea_id"]], "job_ids": [finished["id"]],
        "finding_ids": []})
    assert incomplete["ok"], incomplete
    assert not first.call("research_stop", {
        "reason": "结果充分", "evidence_ids": [incomplete["id"]]})["ok"]
    report = first.call("research_report_submit", {
        "title": "当前研究终结报告", "summary": "现有证据不足以支持扩大搜索",
        "body": "报告记录实际验证反馈与可复现模型。", "limitations": "单次结果不证明泛化改善",
        "idea_ids": [frozen[0]["summary"]["idea_id"]], "job_ids": [finished["id"]],
        "finding_ids": [analysis]})
    assert report["ok"], report
    forged = first.call("research_stop", {"reason": "结果充分", "evidence_ids": ["FINDING-forged"]})
    assert not forged["ok"]
    stopped = first.call("research_stop", {
        "reason": "已完成有界假说检验，当前证据不足以支持继续消耗预算",
        "evidence_ids": [analysis, report["id"]]})
    assert stopped["ok"], stopped
    history = first.call("research_history", {"kind": "stops"})
    assert history["ok"] and len(history["summary"]["records"]) == 1


def test_lab_plan_binds_immutable_source_hash(autonomy):
    _, _, (first, second) = autonomy

    class LabFixture:
        content_hash = "a" * 64

        def require_runnable(self, name, version):
            assert (name, version) == ("custom", 1)

        def get(self, name, version):
            return {"content_hash": self.content_hash}

    first.ctx._lab_store = LabFixture()
    custom = {"category": "lab", "hyperparameters": {"lab": "custom@v1"}}
    frozen = proposal(first, model=custom)
    assert frozen["ok"], frozen
    assert frozen["summary"]["lab_content_hash"] == "a" * 64
    assert proposal(second)["ok"]
    first.ctx._lab_store.content_hash = "b" * 64
    changed = run_proposal(first, frozen, model=custom)
    assert not changed["ok"]
    assert "提案" in changed["error"]


def test_persisted_run_without_mode_keeps_legacy_acceptance_requests(autonomy):
    store, run_id, (first, _) = autonomy
    config = dict(store.get_run(run_id)["config"])
    config.pop("research_mode")
    store.update_run(run_id, {"config": config})
    result = first.call("research_experiment_run", {
        "idea_id": new_idea(first), "model": MODEL, "idempotency_key": "legacy"})
    assert result["ok"] and result["summary"]["status"] == "completed", result
    assert set(store.records(run_id, "jobs")[0]["request"]) == {"model", "protocol_fingerprint"}
    assert first.call("research_protocol", {})["summary"]["research_mode"] == "acceptance"
