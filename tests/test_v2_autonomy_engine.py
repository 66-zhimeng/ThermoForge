"""Autonomous lifecycle integration with real SQLite and deterministic fake sessions."""

import asyncio
from dataclasses import replace
import json
import time

import pytest

from thermoforge_v2.contracts import V2Error
from test_v2_engine import FakeSession, FakeTools, Scenario, make_engine


class AutonomousScenario(Scenario):
    def __init__(self):
        super().__init__()
        self.prompts = {}
        self.delay_first_track = None
        self.control_after_first_job = None

    def session(self, config, specs, handler, event_handler):
        session = AutonomousSession(self, config, handler, event_handler)
        self.instances.append(session)
        return session

    def tools(self, store, ctx, run_id, track_id, **kwargs):
        return AutonomousTools(self, store, run_id, track_id)


class AutonomousSession(FakeSession):
    async def turn(self, prompt):
        self.scenario.prompts.setdefault(self.tid, []).append(prompt)
        return await super().turn(prompt)


class AutonomousTools(FakeTools):
    def report(self):
        jobs = self.store.records(self.rid, "jobs", self.tid)
        return self.store.add_record(self.rid, "reports", {
            "kind": "track", "summary": "Current evidence and limits", "body": "Review saved validation feedback",
            "job_ids": [job["id"] for job in jobs],
            "idea_ids": [idea["id"] for idea in self.store.records(self.rid, "ideas", self.tid)],
            "finding_ids": [finding["id"] for finding in self.store.records(self.rid, "findings", self.tid)],
            "limitations": "Fake execution checks lifecycle, not model quality"}, track_id=self.tid)

    def commit_next(self):
        jobs = self.store.records(self.rid, "jobs", self.tid)
        proposals = self.store.records(self.rid, "proposals", self.tid)
        number = len(proposals) + 1
        idea = self.store.add_record(self.rid, "ideas", {
            "statement": f"Frozen independent model {number}",
            "origin": "history" if jobs else "conjecture",
            "reason": "Revise from measured validation feedback" if jobs else "Initial hypothesis",
            "parent_job_ids": [jobs[-1]["id"]] if jobs else [],
            "protocol_fingerprint": "frozen"}, track_id=self.tid)
        return self.store.commit_proposal(self.rid, self.tid, idea["id"],
            {"category": "data", "estimator": "ridge", "hyperparameters": {"alpha": number}},
            {"purpose": "refine" if jobs else "explore", "expected_cost": "one experiment"},
            f"proposal-{self.tid}-{number}")

    @staticmethod
    def request(proposal):
        return {"proposal_id": proposal["id"], "model": proposal["model"],
                "protocol_fingerprint": proposal["protocol_fingerprint"],
                "experiment_fingerprint": proposal["experiment_fingerprint"],
                "lab_content_hash": proposal.get("lab_content_hash")}

    def execute(self, proposal):
        job = self.store.reserve_job(self.rid, self.tid, proposal["idea_id"], self.request(proposal),
                                     "job-" + proposal["id"])
        started = self.store.start_job(job["id"])
        assert started["started"] is True
        number = len(self.store.records(self.rid, "jobs", self.tid))
        settled = self.store.settle_job(job["id"], "completed", {
            "protocol_fingerprint": "frozen", "feedback_surface": "validate",
            "metrics": {"CVRMSE": 0.2 / number}, "experiment_id": "EXP-" + job["id"]})
        self.store.add_record(self.rid, "findings", {
            "statement": "Observed validation error", "interpretation": "Test the next coefficient",
            "job_ids": [job["id"]], "limitations": "Deterministic fake measurement"}, track_id=self.tid)
        # A complete report exists after EVERY job. It must not stop exploration by itself.
        self.report()
        return settled

    def call(self, name, args):
        prompt = self.scenario.prompts[self.tid][-1]
        jobs = self.store.records(self.rid, "jobs", self.tid)
        proposals = self.store.records(self.rid, "proposals", self.tid)
        if "结题阶段" in prompt:
            report = self.report()
            stop = self.store.request_research_stop(self.rid, self.tid, "Turn budget reserved for closure",
                                                    [report["id"]])
            return {"ok": True, "id": stop["id"]}
        if name == "fake_report":
            return {"ok": True, "id": self.report()["id"]}
        if not proposals:
            if self.scenario.delay_first_track == self.tid:
                time.sleep(0.05)
            return {"ok": True, "id": self.commit_next()["id"]}
        proposal = proposals[-1]
        if any(job.get("proposal_id") == proposal["id"] for job in jobs):
            proposal = self.commit_next()
        job = self.execute(proposal)
        if not jobs and self.scenario.control_after_first_job:
            self.store.control(self.rid, self.scenario.control_after_first_job)
        return {"ok": True, "id": job["id"]}


def test_five_candidates_freeze_before_experiments_and_continue_after_two_reports(tmp_path):
    async def run():
        scenario = AutonomousScenario()
        scenario.delay_first_track = "candidate-5"
        store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                        candidates=5, max_turns=6, max_experiments=15,
                                        max_experiments_per_track=3)
        await asyncio.wait_for(engine.launch(rid), 15)
        assert store.get_run(rid)["status"] == "completed", store.get_run(rid)
        assert len(scenario.instances) == 6
        assert len({session.thread_id for session in scenario.instances}) == 6
        first = [p for p in store.records(rid, "proposals") if p["version"] == 1]
        jobs = store.records(rid, "jobs")
        assert len(first) == 5 and len(jobs) == 15
        assert max(p["created_at"] for p in first) < min(j["created_at"] for j in jobs)
        for index in range(1, 6):
            tid = f"candidate-{index}"
            own_jobs = store.records(rid, "jobs", tid)
            proposals = store.records(rid, "proposals", tid)
            reports = store.records(rid, "reports", tid)
            assert len(own_jobs) == 3 and len(reports) == 3
            assert reports[1]["created_at"] < own_jobs[2]["created_at"]
            assert proposals[1]["parent_job_ids"] == [own_jobs[0]["id"]]
            assert proposals[2]["parent_job_ids"] == [own_jobs[1]["id"]]
            assert len({job["research_turn"] for job in own_jobs}) == 3
            assert len(store.records(rid, "stops", tid)) == 1
            assert store.get_track(rid, tid)["status"] == "completed"
        assert engine._current_team_report(rid) is not None
        assert not engine.sessions and all(session.closed for session in scenario.instances)
    asyncio.run(run())


def test_three_turn_single_main_executes_then_uses_last_turn_for_evidence_stop(tmp_path):
    async def run():
        scenario = AutonomousScenario()
        store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                        candidates=0, max_turns=3, max_experiments_per_track=3)
        await asyncio.wait_for(engine.launch(rid), 10)
        assert store.get_run(rid)["status"] == "completed", store.get_run(rid)
        assert len(store.records(rid, "jobs", "main")) == 1
        assert len(store.records(rid, "proposals", "main")) == 1
        assert len(store.records(rid, "stops", "main")) == 1
        assert "结题阶段" in scenario.prompts["main"][-1]
        assert store.get_track(rid, "main")["turns"] == 3
        assert engine._current_track_report(rid, "main") is not None
    asyncio.run(run())


@pytest.mark.parametrize("action, final", [("pause", "paused"), ("cancel", "cancelled")])
@pytest.mark.parametrize("control_when", ["after_report", "stop_transaction"])
def test_control_after_quota_exhaustion_and_report_keeps_requested_state(tmp_path, monkeypatch,
                                                                       action, final, control_when):
    async def run():
        scenario = AutonomousScenario()
        scenario.control_after_first_job = action if control_when == "after_report" else None
        store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                        candidates=0, max_turns=3, max_experiments=1,
                                        max_experiments_per_track=1)
        stop_attempts = []
        if control_when == "stop_transaction":
            original = store.request_research_stop

            def race_stop(*args, **kwargs):
                stop_attempts.append(True)
                store.control(rid, action)
                return original(*args, **kwargs)

            monkeypatch.setattr(store, "request_research_stop", race_stop)
        await asyncio.wait_for(engine.launch(rid), 10)
        assert len(store.records(rid, "jobs")) == 1
        assert engine._current_track_report(rid, "main") is not None
        assert store.get_run(rid)["status"] == final, store.get_run(rid)
        assert store.get_track(rid, "main")["status"] == final
        assert store.get_run(rid)["error"] is None
        assert store.records(rid, "stops") == []
        if control_when == "stop_transaction":
            assert stop_attempts == [True]
    asyncio.run(run())


def test_main_team_view_is_metadata_only_before_gate_and_unlocked_for_final_report(tmp_path):
    scenario = AutonomousScenario()
    store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous", candidates=2)
    store.begin_run(rid)
    for tid, role in (("main", "main"), ("candidate-1", "candidate"), ("candidate-2", "candidate")):
        store.create_track(rid, tid, role)
    proposal = AutonomousTools(scenario, store, rid, "candidate-1").commit_next()
    hidden = engine._mediator(rid, "main", "research_team", {})
    assert hidden["ok"] is True and hidden["research_stage"] == "independent_proposals"
    assert hidden["tracks"]
    for key in ("ideas", "jobs", "proposals", "findings", "sources", "reports", "stops", "decisions"):
        assert hidden[key] == []
    assert proposal["id"] not in json.dumps(hidden)
    store.update_run(rid, {"final_report_phase": True})
    visible = engine._mediator(rid, "main", "research_team", {})
    assert [p["id"] for p in visible["proposals"]] == [proposal["id"]]
    assert [i["id"] for i in visible["ideas"]] == [proposal["idea_id"]]


def test_second_job_in_same_research_turn_is_rejected_without_spending_budget(tmp_path):
    scenario = AutonomousScenario()
    store, rid, _ = make_engine(tmp_path, scenario, research_mode="autonomous", candidates=0)
    store.begin_run(rid)
    store.create_track(rid, "main", "main", turns=1)
    tools = AutonomousTools(scenario, store, rid, "main")
    first = tools.commit_next()
    tools.execute(first)
    second = tools.commit_next()
    with pytest.raises(V2Error) as rejected:
        tools.execute(second)
    assert rejected.value.code == "TFV2-PHASE"
    assert store.get_run(rid)["experiments_reserved"] == 1
    assert len(store.records(rid, "jobs")) == 1
    store.update_track(rid, "main", {"turns": 2})
    assert tools.execute(second)["status"] == "completed"


def test_resumed_late_candidate_cannot_read_shared_messages_before_own_first_experiment(tmp_path):
    scenario = AutonomousScenario()
    store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                    candidates=2, strategy="top_k")
    store.begin_run(rid)
    for tid, role in (("main", "main"), ("candidate-1", "candidate"), ("candidate-2", "candidate")):
        store.create_track(rid, tid, role, turns=1)
    tools = AutonomousTools(scenario, store, rid, "candidate-1")
    proposal = tools.commit_next()
    store.update_track(rid, "candidate-2", {"status": "failed"})
    assert store.autonomy_state(rid)["proposal_barrier_open"]
    job = tools.execute(proposal)
    assert store.autonomy_state(rid)["sharing_ready"]
    store.update_track(rid, "main", {"turns": 2})
    sent = engine._mediator(rid, "main", "research_send_message", {
        "to": "candidate-2", "text": "A peer result is ready", "evidence_ids": [job["id"]]})
    assert sent["ok"]
    # A new process on resume sets ready, while the global sharing stage remains durable.
    store.update_track(rid, "candidate-2", {"status": "ready"})
    inbox = engine._mediator(rid, "candidate-2", "research_messages", {})
    assert inbox["messages"] == []
    late_tools = AutonomousTools(scenario, store, rid, "candidate-2")
    late_proposal = late_tools.commit_next()
    assert engine._mediator(rid, "candidate-2", "research_messages", {})["messages"] == []
    late_tools.execute(late_proposal)
    assert [m["id"] for m in engine._mediator(rid, "candidate-2", "research_messages", {})["messages"]] == [sent["id"]]


def test_resume_clears_crashed_coordination_when_main_only_has_final_turn_left(tmp_path):
    async def run():
        scenario = AutonomousScenario()
        store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                        candidates=1, strategy="top_k", max_turns=6,
                                        max_experiments_per_track=3)
        store.begin_run(rid)
        for tid, role, turns in (("main", "main", 5), ("candidate-1", "candidate", 2)):
            track_root = store.root / "runs" / rid / "tracks" / tid
            (track_root / "workspace").mkdir(parents=True)
            store.create_track(rid, tid, role, turns=turns,
                               workspace=str(track_root / "workspace"),
                               research_root=str(track_root / "research"))
        helper = AutonomousTools(scenario, store, rid, "candidate-1")
        helper.execute(helper.commit_next())
        store.add_record(rid, "messages", {
            "from": "main", "to": "all", "text": "Common research objective",
            "initial_task": True, "evidence_ids": []}, track_id="main")
        assert store.autonomy_state(rid)["sharing_ready"]
        # The process died before the coordinator's finally block could clear this flag.
        store.update_run(rid, {"status": "queued", "coordination_in_progress": True})
        await asyncio.wait_for(engine.launch(rid), 8)
        assert store.get_run(rid)["status"] == "completed", store.get_run(rid)
        assert store.get_run(rid)["coordination_in_progress"] is False
        assert len(store.records(rid, "jobs", "candidate-1")) == 3
        assert store.get_track(rid, "main")["turns"] == 6
        assert engine._current_team_report(rid) is not None
        assert not any("候选尚在运行" in prompt for prompt in scenario.prompts["main"])
    asyncio.run(run())


@pytest.mark.parametrize("consume_hard_limit", [False, True], ids=["report-reserve", "hard-cap"])
def test_token_closure_preserves_reports_when_possible_and_honors_hard_cap(tmp_path, consume_hard_limit):
    class BudgetScenario(AutonomousScenario):
        def __init__(self):
            super().__init__()
            self.emit_usage = False
            self.injected = False
            self.store = None
            self.rid = None

        def session(self, config, specs, handler, event_handler):
            instance = BudgetSession(self, config, handler, event_handler)
            self.instances.append(instance)
            return instance

    class BudgetSession(AutonomousSession):
        total = 0

        async def turn(self, prompt):
            result = await super().turn(prompt)
            previous = self.total
            self.total += 100
            scenario = self.scenario
            if (self.tid == "candidate-1" and not scenario.injected
                    and scenario.store.records(scenario.rid, "jobs", self.tid)):
                state = scenario.store.autonomy_state(scenario.rid)
                budget = scenario.store.get_run(scenario.rid)["config"]["token_budget"]
                # The first completed experiment's Codex turn consumes the remaining
                # exploration allowance. Report turns themselves remain under the cap.
                other_tokens = sum((track.get("usage") or {}).get("total", {}).get("totalTokens", 0)
                                   for track in scenario.store.list_tracks(scenario.rid)
                                   if track["track_id"] != self.tid)
                target = budget if consume_hard_limit else budget - state["report_token_reserve"] + 1
                self.total = target - other_tokens
                scenario.injected = True
            usage = {"total": {"totalTokens": self.total},
                     "last": {"totalTokens": self.total - previous}}
            return replace(result, usage=usage)

    async def run():
        scenario = BudgetScenario()
        store, rid, engine = make_engine(tmp_path, scenario, research_mode="autonomous",
                                        candidates=1, strategy="top_k", max_turns=8,
                                        max_experiments=4, max_experiments_per_track=4,
                                        token_budget=200000)
        scenario.store, scenario.rid = store, rid
        await asyncio.wait_for(engine.launch(rid), 10)
        assert scenario.injected is True
        run_record = store.get_run(rid)
        main = store.get_track(rid, "main")
        assert main["pid"] is None
        assert not engine.sessions and all(instance.closed for instance in scenario.instances)
        if consume_hard_limit:
            assert run_record["status"] == "budget_exhausted", run_record
            assert run_record["tokens_used"] == run_record["config"]["token_budget"]
            assert main["status"] == "budget_exhausted" and main["phase"] == "stopped"
            assert len(scenario.prompts["main"]) == 1
            assert len(store.records(rid, "jobs", "candidate-1")) == 1
            assert engine._current_team_report(rid) is None
            return
        assert run_record["status"] == "completed", run_record
        assert main["status"] == "completed"
        assert run_record["research_closure"]
        assert run_record["tokens_used"] < run_record["config"]["token_budget"]
        assert len(store.records(rid, "jobs", "candidate-1")) == 1
        assert len(store.records(rid, "proposals", "candidate-1")) == 1
        assert len(store.records(rid, "stops", "candidate-1")) == 1
        assert "结题阶段" in scenario.prompts["candidate-1"][-1]
        assert store.get_track(rid, "candidate-1")["turns"] < run_record["config"]["max_turns"]
        assert run_record["experiments_reserved"] < run_record["config"]["max_experiments"]
        assert engine._current_track_report(rid, "candidate-1") is not None
        assert engine._current_team_report(rid) is not None
        assert not any("候选尚在运行" in prompt for prompt in scenario.prompts["main"])
    asyncio.run(run())
