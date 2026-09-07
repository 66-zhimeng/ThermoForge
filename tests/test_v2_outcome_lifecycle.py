"""Evidence survives budget/control failure, and a met target can conclude early."""
import asyncio
import json

import pytest

from thermoforge_v2.engine import ResearchEngine
from thermoforge_v2.store import RunStore
from test_v2_autonomy_engine import AutonomousScenario, AutonomousTools


def test_cancelling_paused_run_updates_checkpoint_without_an_active_agent(tmp_path):
    store, rid, _ = engine_run(tmp_path, UnanalysedScenario())
    store.control(rid, "pause")
    path = store.root / "runs" / rid / "checkpoints" / "latest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "paused"
    store.control(rid, "cancel")
    brief = json.loads(path.read_text(encoding="utf-8"))
    assert brief["status"] == "cancelled" and brief["reason"] == "control.cancel"
    assert brief["counts"]["agent_reports"] == 0


@pytest.mark.parametrize("before,after", [("pausing", "paused"), ("cancelling", "cancelled")])
def test_service_recovery_checkpoints_final_control_state(tmp_path, before, after):
    store, rid, engine = engine_run(tmp_path, UnanalysedScenario())
    store.update_run(rid, {"status": before})
    asyncio.run(engine.recover())
    path = store.root / "runs" / rid / "checkpoints" / "latest.json"
    brief = json.loads(path.read_text(encoding="utf-8"))
    assert brief["status"] == after and brief["reason"] == "service.recovered"
    assert not engine.tasks


def engine_run(tmp_path, scenario):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001",
        "candidates": 0, "max_turns": 6, "max_experiments_per_track": 4},
        {"fingerprint": "frozen", "goal_definition": {"acceptance": {
            "evaluated_on": "validate", "cvrmse_max": 0.25}}}, "target")
    return store, run["run_id"], ResearchEngine(store, None, session_factory=scenario.session,
                                                tools_factory=scenario.tools)


class ReviewedTools(AutonomousTools):
    def execute(self, proposal):
        job = super().execute(proposal)
        self.store.add_record(self.rid, "findings", {"statement": "Observed measured error",
            "interpretation": "The frozen validation threshold was met", "limitations": "No external evaluation",
            "protocol_fingerprint": "frozen", "job_ids": [job["id"]]}, track_id=self.tid)
        self.report()
        return job


class ReviewedScenario(AutonomousScenario):
    def tools(self, store, ctx, run_id, track_id, **kwargs):
        return ReviewedTools(self, store, run_id, track_id)


def test_target_concludes_after_evidence_review_without_using_remaining_experiments(tmp_path):
    async def run():
        scenario = ReviewedScenario()
        store, rid, engine = engine_run(tmp_path, scenario)
        await asyncio.wait_for(engine.launch(rid), 12)
        assert store.get_run(rid)["status"] == "completed"
        assert len(store.records(rid, "jobs")) == 1
        assert len(store.records(rid, "stops")) == 1
        assert "无需机械用完实验额度" in scenario.prompts["main"][-1]
        brief = json.loads((store.root / "runs" / rid / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
        assert brief["objective"]["threshold_status"] == "pass"
        assert brief["objective"]["review_status"] == "pass"
        assert brief["tracks"][0]["missing"]["agent_report"] is False
        assert brief["reason"] == "run.exit"
    asyncio.run(run())


class UnanalysedTools(AutonomousTools):
    def execute(self, proposal):
        job = self.store.reserve_job(self.rid, self.tid, proposal["idea_id"], self.request(proposal),
                                     "job-" + proposal["id"])
        self.store.start_job(job["id"])
        return self.store.settle_job(job["id"], "completed", {"experiment_id": "EXP-0001",
            "protocol_fingerprint": "frozen", "feedback_surface": "validate", "metrics": {"CVRMSE": 0.2}})


class UnanalysedScenario(AutonomousScenario):
    def tools(self, store, ctx, run_id, track_id, **kwargs):
        return UnanalysedTools(self, store, run_id, track_id)


def test_cancel_after_measurement_exports_facts_without_fabricating_analysis_or_stop(tmp_path):
    async def run():
        scenario = UnanalysedScenario()
        scenario.control_after_first_job = "cancel"
        store, rid, engine = engine_run(tmp_path, scenario)
        await asyncio.wait_for(engine.launch(rid), 12)
        assert store.get_run(rid)["status"] == "cancelled"
        assert store.records(rid, "reports") == store.records(rid, "findings") == store.records(rid, "stops") == []
        folder = store.root / "runs" / rid / "checkpoints"
        brief = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
        assert brief["counts"]["successful_jobs"] == 1
        assert brief["is_agent_report"] is False
        assert brief["objective"]["review_status"] == "unknown"
        missing = brief["tracks"][0]["missing"]
        assert missing["agent_report"] and missing["stop_decision"] and len(missing["finding_job_ids"]) == 1
        assert (folder / "team-flow.html").is_file()
    asyncio.run(run())


def test_export_failure_keeps_settled_evidence_and_can_regenerate(tmp_path, monkeypatch):
    from thermoforge_v2 import checkpoints
    store, rid, _ = engine_run(tmp_path, UnanalysedScenario())
    store.begin_run(rid)
    store.create_track(rid, "main", "main", turns=1)
    tools = UnanalysedTools(UnanalysedScenario(), store, rid, "main")
    proposal = tools.commit_next()
    original = checkpoints.export_checkpoint
    def disk_failure(*args):
        raise OSError("disk temporarily unavailable")
    monkeypatch.setattr(checkpoints, "export_checkpoint", disk_failure)
    job = tools.execute(proposal)
    assert job["status"] == "completed"
    assert store.get_run(rid)["experiments_settled"] == 1
    assert any(e["kind"] == "checkpoint.failed" for e in store.events(rid, limit=1000)["events"])
    monkeypatch.setattr(checkpoints, "export_checkpoint", original)
    assert store.save_checkpoint(rid, "rebuild")["ok"]
    assert len(store.records(rid, "jobs")) == 1
    assert store.records(rid, "reports") == []
