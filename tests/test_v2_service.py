"""独立服务应用层：真实协议和制品、假调度引擎，不启动 Codex。"""

from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from phase2_helpers import FEATURES, TARGET
from phase34_helpers import make_ctx
from thermoforge_research.tools import tf_goal_create
from thermoforge_v2.contracts import V2Error
from thermoforge_v2.service import ResearchService


class FakeEngine:
    def __init__(self, store, ctx):
        self.store, self.ctx = store, ctx
        self.launched = []
        self.recovered = False
        self.closed = False
        self.changed = threading.Event()

    async def recover(self):
        self.recovered = True

    def launch(self, run_id):
        self.launched.append(run_id)
        run = self.store.get_run(run_id)
        for track_id in ["main"] + [f"candidate_{n + 1}" for n in range(run["config"]["candidates"])]:
            self.store.create_track(run_id, track_id, "main" if track_id == "main" else "candidate",
                                    research_root=str(self.store.root / "runs" / run_id / track_id / "research"),
                                    workspace=str(self.store.root / "runs" / run_id / track_id / "workspace"))
        self.store.update_run(run_id, {"status": "running"})
        self.changed.set()

    async def close(self):
        self.closed = True


@pytest.fixture()
def service_config(tmp_path, monkeypatch):
    ctx, ref = make_ctx(tmp_path, n_steps=600)
    goal = tf_goal_create(ctx, {"name": "服务集成测试", "object_model": "chiller.v1",
        "purpose": "optimization", "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": .5}})
    assert goal["ok"]
    monkeypatch.setattr("thermoforge_v2.codex.resolve_codex_command", lambda: ["codex-test-only"])
    service = ResearchService(ctx, engine_factory=FakeEngine)
    service.start()
    try:
        yield service, {"goal_id": goal["id"], "dataset_ref": ref, "research_mode": "acceptance", "candidates": 5}
    finally:
        service.close()
    assert service.engine.closed


def start(service, config):
    service.engine.changed.clear()
    result = service.dispatch("start", {"config": config, "idempotency_key": "service-run"})
    assert service.engine.changed.wait(5)
    return result["run_id"]


def completed_job(service, rid, *, score=.1, holdout=.2, key="winner"):
    store = service.store
    fingerprint = store.get_run(rid)["protocol"]["fingerprint"]
    idea = store.add_record(rid, "ideas", {"statement": "模型假设", "origin": "conjecture",
        "reason": "先建立基线", "prediction": "验证误差低", "falsification": "误差不改善",
        "source_ids": [], "protocol_fingerprint": fingerprint}, track_id="candidate_1")
    job = store.reserve_job(rid, "candidate_1", idea["id"], {"model": {}}, key)
    started = store.start_job(job["id"])
    assert started["started"]
    exp_id = "EXP-0001" if key == "winner" else "EXP-0002"
    directory = Path(store.get_track(rid, "candidate_1")["research_root"]) / "experiments" / exp_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps({
        "experiment_id": exp_id, "status": "completed",
        "metrics": {"surfaces": {"validate": {"metrics": {"CVRMSE": score}, "n_samples": 100},
                                    "A": {"metrics": {"CVRMSE": holdout}, "n_samples": 90}}},
        "physics": {"overall_rate": .01}}), encoding="utf-8", newline="\n")
    store.update_record(rid, "jobs", job["id"], {"experiment_id": exp_id})
    return store.settle_job(job["id"], "completed", {"experiment_id": exp_id,
        "protocol_fingerprint": fingerprint, "feedback_surface": "validate",
        "metrics": {"CVRMSE": score}, "n_samples": 100})


def test_prepare_reports_missing_input_backend_and_real_frozen_protocol(service_config):
    service, config = service_config
    empty = service.dispatch("prepare", {"config": {}})
    assert set(empty["missing"]) == {"goal_id", "dataset_ref"}
    assert not empty["ready"]
    result = service.dispatch("prepare", {"config": config})
    assert result["ready"], result
    assert result["protocol"]["feedback_surface"] == "validate"
    assert result["protocol"]["modelability"]["evaluated_on"] == "train"
    assert result["defaults"]["candidates"] == 5
    assert service.engine.launched == []


def test_start_status_control_and_idempotency_share_one_background_run(service_config):
    service, config = service_config
    rid = start(service, config)
    again = service.start_run(config, "service-run")
    assert again["run_id"] == rid
    assert service.engine.launched == [rid]
    snapshot = service.dispatch("status", {"run_id": rid})
    assert snapshot["run"]["status"] == "running"
    assert len(snapshot["tracks"]) == 6
    paused = service.dispatch("control", {"run_id": rid, "action": "pause",
        "expected_version": snapshot["run"]["version"]})
    assert paused["status"] == "pausing"
    with pytest.raises(V2Error):
        service.control(rid, "update", snapshot["run"]["version"], {"guidance": "旧版本不得覆盖"})
    service.store.update_run(rid, {"status": "paused"})  # 假引擎模拟检查点已静止。
    service.engine.changed.clear()
    resumed = service.control(rid, "resume")
    assert resumed["status"] == "queued"
    assert service.engine.changed.wait(5)
    assert service.engine.launched == [rid, rid]
    assert service.dispatch("events", {"run_id": rid})["events"]


def test_idempotent_retry_does_not_refreeze_changed_environment(service_config, monkeypatch):
    service, config = service_config
    rid = start(service, config)
    def invalid_now(*args):
        raise ValueError("目标或环境在提交后改变")
    monkeypatch.setattr("thermoforge_v2.service.prepare_protocol", invalid_now)
    assert service.start_run(config, "service-run")["run_id"] == rid
    with pytest.raises(V2Error, match="不同"):
        service.start_run(config | {"seed": 99}, "service-run")


def test_report_reads_true_holdout_shape_only_after_selection_and_preserves_it(service_config):
    service, config = service_config
    rid = start(service, config)
    best = completed_job(service, rid, score=.1, holdout=.3)
    completed_job(service, rid, score=.2, holdout=.01, key="validation-worse")
    interim = service.get_report(rid)
    assert not service.store.records(rid, "finalizations")
    assert not interim.get("final_evaluation")
    service.store.update_run(rid, {"status": "completed"})
    report = service.get_report(rid)
    frozen = service.store.records(rid, "finalizations")
    assert len(frozen) == 1
    assert frozen[0]["job_id"] == best["id"]  # 不能按留出反过来选另一个模型。
    assert frozen[0]["surfaces"]["A"]["metrics"]["CVRMSE"] == .3
    assert frozen[0]["feedback_to_agents"] is False
    assert report["final_evaluation"]["job_id"] == best["id"]
    assert all(Path(a["path"]).is_file() for a in report["artifacts"])
    service.get_report(rid)
    assert len(service.store.records(rid, "finalizations")) == 1
    assert all("A" not in (j.get("result") or {}).get("metrics", {})
               for j in service.store.records(rid, "jobs"))


def test_waiting_for_input_does_not_finalize_resumable_research(service_config):
    service, config = service_config
    rid = start(service, config)
    completed_job(service, rid)
    service.store.update_run(rid, {"status": "needs_input"})
    service.get_report(rid)
    assert service.store.records(rid, "finalizations") == []


def test_prepare_missing_codex_returns_unavailable_without_model_requests(service_config, monkeypatch):
    service, config = service_config

    def absent():
        raise FileNotFoundError("未安装 Codex")

    monkeypatch.setattr("thermoforge_v2.codex.resolve_codex_command", absent)
    result = service.prepare(config)
    assert result["available"] is False and result["ready"] is False
    assert any("Codex" in e for e in result["errors"])
    assert service.engine.launched == []


def test_reused_winner_reads_original_artifact_without_pretending_local_execution(service_config):
    service, config = service_config
    rid = start(service, config)
    source = completed_job(service, rid, score=.1, holdout=.3)
    original_id = source["id"]
    service.store.update_record(rid, "jobs", original_id, {"experiment_fingerprint": "identical", "executed": True})
    service.store.create_track(rid, "candidate-reused", "candidate", research_root=str(service.store.root / "no-artifacts"))
    service.store.add_record(rid, "jobs", {"idea_id": source["idea_id"], "status": "completed",
        "result": source["result"], "reused_from_job_id": original_id, "executed": False,
        "experiment_fingerprint": "identical"}, track_id="candidate-reused", record_id="JOB-000")
    service.store.update_run(rid, {"status": "completed"})
    final = service.get_report(rid)["final_evaluation"]
    assert final["job_id"] == "JOB-000"
    assert final["artifact_job_id"] == original_id
    assert final["artifact_track_id"] == source["track_id"]
    assert final["surfaces"]["A"]["metrics"]["CVRMSE"] == .3


def test_report_and_dispatch_reject_nonexistent_track_or_operation(service_config):
    service, config = service_config
    rid = start(service, config)
    with pytest.raises((ValueError, V2Error)):
        service.get_report(rid, "../../foreign")
    with pytest.raises(V2Error):
        service.dispatch("unknown", {})
