"""真实 RunStore 与可控制 Codex 后端验证 V2 调度，不调用模型。"""
import asyncio
import threading
import time

import pytest

from thermoforge_v2.codex import CodexError, TurnResult
from thermoforge_v2.engine import ResearchEngine
from thermoforge_v2.store import RunStore


class Scenario:
    def __init__(self):
        self.instances = []
        self.starts = []
        self.thread_turns = {}
        self.active = set()
        self.peak = 0
        self.main_overlap = asyncio.Event()
        self.require_main_overlap = False
        self.second_turn_delay = 0
        self.waiting = False
        self.entered = asyncio.Event()
        self.emit_usage = True
        self.crash_once = False
        self.crashed = False
        self.start_failure = None
        self.block_tool = False
        self.tool_started = threading.Event()
        self.tool_release = threading.Event()
        self.resolved_model = "test-codex-model"
        self.resolved_effort = "high"
        self.send_refine = False
        self.refine_sent = False
        self.refine_received = False
        self.coordination_started = asyncio.Event()
        self.pause_coordination = False
        self.fail_coordination = False
        self.hold_review_for_refine = False
        self.refine_executed = asyncio.Event()
        self.stage_report_only = False
        self.report_turns = 0
        self.final_team_report = True
        self.stage_team_report = False

    def session(self, config, specs, handler, event_handler):
        instance = FakeSession(self, config, handler, event_handler)
        self.instances.append(instance)
        return instance

    def tools(self, store, ctx, run_id, track_id, **kwargs):
        return FakeTools(self, store, run_id, track_id)


class FakeTools:
    def __init__(self, scenario, store, run_id, track_id):
        self.scenario, self.store, self.rid, self.tid = scenario, store, run_id, track_id

    def tool_specs(self):
        return [{"name": name, "description": "确定性测试实验或报告",
                 "inputSchema": {"type": "object", "properties": {}}}
                for name in ("fake_experiment", "fake_report")]

    def report(self):
        jobs = self.store.records(self.rid, "jobs", track_id=self.tid)
        return self.store.add_record(self.rid, "reports", {
            "kind": "track", "job_ids": [j["id"] for j in jobs],
            "evidence_ids": [j["idea_id"] for j in jobs]}, track_id=self.tid)

    def call(self, name, args):
        if name == "fake_report":
            self.scenario.report_turns += 1
            return {"ok": True, "id": self.report()["id"]}
        if self.scenario.block_tool:
            self.scenario.tool_started.set()
            assert self.scenario.tool_release.wait(10)
        number = len(self.store.records(self.rid, "jobs", track_id=self.tid)) + 1
        idea = self.store.add_record(self.rid, "ideas", {"title": f"假设{number}"}, track_id=self.tid)
        job = self.store.add_record(self.rid, "jobs", {
            "idea_id": idea["id"], "status": "completed", "finished_at": time.time(), "result": {
                "metrics": {"CVRMSE": 0.2 / number}, "protocol_fingerprint": "frozen",
                "feedback_surface": "validate"}}, track_id=self.tid)
        if (self.scenario.stage_report_only and number == 1) or (not self.scenario.stage_report_only and number >= 2):
            self.report()
        return {"ok": True, "id": job["id"]}


class FakeSession:
    def __init__(self, scenario, config, handler, event_handler):
        self.scenario, self.config, self.handler, self.event_handler = scenario, config, handler, event_handler
        self.tid = config.cwd.parent.name
        self.pid = 1000 + len(scenario.instances)
        self.thread_id = None
        self.state = "new"
        self.turns = 0
        self.interrupted = asyncio.Event()
        self.tool_task = None
        self.interrupt_calls = 0
        self.closed = False

    async def start(self, thread_id=None):
        self.scenario.starts.append((self.tid, thread_id))
        if self.scenario.start_failure == self.tid:
            raise CodexError("测试启动失败")
        self.thread_id = thread_id or f"session-{self.pid}"
        self.scenario.thread_turns.setdefault(self.thread_id, 0)
        self.state = "idle"
        return {"pid": self.pid, "thread_id": self.thread_id,
                "model": self.config.model or self.scenario.resolved_model,
                "reasoning_effort": self.config.effort or self.scenario.resolved_effort,
                "server": {"version": "fake-codex"}}

    async def turn(self, prompt):
        if self.closed or self.state == "failed":
            raise CodexError("测试实例已断开，必须重建进程")
        self.state = "running"
        self.interrupted.clear()
        self.turns += 1
        self.scenario.thread_turns[self.thread_id] += 1
        usage = {"total": {"totalTokens": self.turns * 123}, "last": {"totalTokens": 123}}
        self.scenario.active.add(self.tid)
        self.scenario.peak = max(self.scenario.peak, len(self.scenario.active))
        try:
            if self.scenario.emit_usage:
                await self.event_handler({"method": "thread/tokenUsage/updated", "params": {"tokenUsage": usage}})
            if self.scenario.crash_once and not self.scenario.crashed:
                self.scenario.crashed = True
                self.state = "failed"
                raise CodexError("测试连接中断")
            if "准备统一研究任务" in prompt:
                await self.handler("research_send_message", {"to": "all", "text": "同一研究任务"})
            elif "候选尚在运行" in prompt:
                self.scenario.main_overlap.set()
                self.scenario.coordination_started.set()
                if self.scenario.fail_coordination:
                    raise CodexError("测试主协调回合失败")
                if self.scenario.pause_coordination:
                    await self.interrupted.wait()
                    return TurnResult("interrupted", "", self.thread_id, str(self.turns), usage)
                if self.scenario.stage_team_report:
                    team = await self.handler("research_team", {})
                    await self.handler("research_team_report", {
                        "title": "阶段综合", "summary": "尚未结束", "body": "阶段反馈",
                        "evidence_ids": [j["id"] for j in team["jobs"]], "limitations": "候选仍在研究"})
                if self.scenario.send_refine and not self.scenario.refine_sent:
                    team = await self.handler("research_team", {})
                    refs = [j["id"] for j in team["jobs"] if j["track_id"] == "candidate-1"]
                    if len(refs) >= 2:
                        sent = await self.handler("research_send_message", {
                            "to": "candidate-1", "text": "refine: 用已保存验证反馈进行第三次实验",
                            "evidence_ids": refs})
                        assert sent["ok"]
                        self.scenario.refine_sent = True
                        if self.scenario.hold_review_for_refine:
                            await asyncio.wait_for(self.scenario.refine_executed.wait(), 5)
                await asyncio.sleep(0.02)
            elif "比较各候选" in prompt:
                team = await self.handler("research_team", {})
                refs = [j["id"] for j in team["jobs"]]
                if refs and self.scenario.final_team_report:
                    await self.handler("research_team_report", {
                        "title": "综合", "summary": "比较真实证据", "body": "候选比较",
                        "evidence_ids": refs, "limitations": "测试仅验证调度"})
            elif "补交最终轨迹报告" in prompt:
                await self.handler("fake_report", {})
            elif self.scenario.waiting:
                self.scenario.entered.set()
                await self.interrupted.wait()
                return TurnResult("interrupted", "", self.thread_id, str(self.turns), usage)
            else:
                inbox = await self.handler("research_messages", {})
                if any(m["id"] in inbox["new_message_ids"] and "refine:" in m["text"]
                       for m in inbox["messages"]):
                    self.scenario.refine_received = True
                if self.turns == 2 and self.scenario.second_turn_delay:
                    await asyncio.sleep(self.scenario.second_turn_delay)
                if self.scenario.require_main_overlap and self.turns == 2:
                    await asyncio.wait_for(self.scenario.main_overlap.wait(), 5)
                self.tool_task = asyncio.create_task(self.handler("fake_experiment", {}))
                stop = asyncio.create_task(self.interrupted.wait())
                try:
                    done, _ = await asyncio.wait([self.tool_task, stop], return_when=asyncio.FIRST_COMPLETED)
                    if stop in done:
                        return TurnResult("interrupted", "", self.thread_id, str(self.turns), usage)
                    await self.tool_task
                    if self.scenario.refine_received and self.tid == "candidate-1":
                        self.scenario.refine_executed.set()
                finally:
                    stop.cancel()
                    await asyncio.gather(stop, return_exceptions=True)
            return TurnResult("completed", "已记录", self.thread_id, str(self.turns), usage)
        finally:
            self.scenario.active.discard(self.tid)
            if self.state != "failed":
                self.state = "idle"

    async def interrupt(self):
        self.interrupt_calls += 1
        self.interrupted.set()

    async def close(self):
        self.closed = True
        self.interrupted.set()
        if self.tool_task and not self.tool_task.done():
            self.tool_task.cancel()
            await asyncio.gather(self.tool_task, return_exceptions=True)
        self.state = "closed"


def make_engine(tmp_path, scenario, **config):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "test@rev_0001",
                            "research_mode": "acceptance", "candidates": 0, "max_turns": 4,
                            "max_experiments_per_track": 4, **config},
                           {"fingerprint": "frozen"}, "test")
    engine = ResearchEngine(store, None, session_factory=scenario.session, tools_factory=scenario.tools)
    return store, run["run_id"], engine


def test_last_allowed_turn_report_is_completed(tmp_path):
    async def run():
        scenario = Scenario()
        store, rid, engine = make_engine(tmp_path, scenario, max_turns=2)
        await engine.launch(rid)
        assert len(store.records(rid, "jobs")) == 2
        assert len(store.records(rid, "reports")) == 1
        assert store.get_track(rid, "main")["status"] == "completed"
        assert store.get_run(rid)["status"] == "completed"
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_six_instances_overlap_and_leave_individual_and_team_reports(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.require_main_overlap = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=5, max_turns=5)
        await asyncio.wait_for(engine.launch(rid), 8)
        assert scenario.peak == 6
        assert len({i.thread_id for i in scenario.instances}) == 6
        assert len(store.records(rid, "reports")) == 6
        assert store.get_run(rid)["status"] == "completed"
        assert store.get_run(rid)["tokens_used"] == sum(engine._tokens(t["usage"]) for t in store.list_tracks(rid))
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_main_reserves_its_final_turn_for_team_report(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.second_turn_delay = 1.2
        store, rid, engine = make_engine(tmp_path, scenario, candidates=5, max_turns=2)
        await asyncio.wait_for(engine.launch(rid), 6)
        assert store.get_run(rid)["status"] == "completed"
        assert any(r.get("kind") == "team" for r in store.records(rid, "reports"))
        assert store.get_track(rid, "main")["turns"] == 2
    asyncio.run(run())


def test_pause_then_resume_uses_same_thread_and_new_process(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.waiting = True
        store, rid, engine = make_engine(tmp_path, scenario)
        task = engine.launch(rid)
        await asyncio.wait_for(scenario.entered.wait(), 3)
        saved_thread = store.get_track(rid, "main")["thread_id"]
        frozen = store.get_run(rid)["effective_backend"]
        assert frozen["model"] == "test-codex-model" and frozen["reasoning_effort"] == "high"
        store.control(rid, "pause")
        await asyncio.wait_for(task, 3)
        assert store.get_run(rid)["status"] == "paused"
        assert all(i.closed for i in scenario.instances)
        scenario.waiting = False
        scenario.resolved_model = "changed-global-default"
        scenario.resolved_effort = "low"
        store.control(rid, "resume")
        await asyncio.wait_for(engine.launch(rid), 3)
        assert scenario.starts[-1] == ("main", saved_thread)
        assert len(scenario.instances) == 2
        assert scenario.instances[-1].config.model == frozen["model"]
        assert scenario.instances[-1].config.effort == frozen["reasoning_effort"]
        assert store.get_run(rid)["effective_backend"] == frozen
        assert store.get_run(rid)["tokens_used"] == sum(i.turns for i in scenario.instances) * 123
        assert store.get_run(rid)["status"] == "completed"
    asyncio.run(run())


def test_independent_run_blocks_cross_candidate_feedback_after_experiments(tmp_path):
    async def run():
        store, rid, engine = make_engine(tmp_path, Scenario(), candidates=2)
        for tid in ("main", "candidate-1", "candidate-2"):
            store.create_track(rid, tid, "main" if tid == "main" else "candidate")
        assert engine._mediator(rid, "main", "research_send_message", {"to": "all", "text": "共同任务"})["ok"]
        job = store.add_record(rid, "jobs", {"status": "completed"}, track_id="candidate-1")
        args = {"to": "candidate-2", "text": "参考另一轨迹", "evidence_ids": [job["id"]]}
        assert engine._mediator(rid, "main", "research_send_message", args)["ok"] is False
        assert engine._mediator(rid, "candidate-1", "research_send_message", args)["ok"] is False
        assert len(engine._mediator(rid, "candidate-2", "research_messages", {})["messages"]) == 1
    asyncio.run(run())


def test_turn_result_usage_updates_global_budget_even_without_notification(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.emit_usage = False
        store, rid, engine = make_engine(tmp_path, scenario, max_turns=3)
        await engine.launch(rid)
        assert store.get_track(rid, "main")["usage"]["total"]["totalTokens"] == 246
        assert store.get_run(rid)["tokens_used"] == 246
    asyncio.run(run())


def test_dead_backend_is_recreated_before_retry(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    async def quick_sleep(_):
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", quick_sleep)
    async def run():
        scenario = Scenario()
        scenario.crash_once = True
        store, rid, engine = make_engine(tmp_path, scenario, max_turns=5)
        await engine.launch(rid)
        assert len(scenario.instances) == 2
        assert scenario.starts[1][1] == scenario.instances[0].thread_id
        assert store.get_run(rid)["status"] == "completed"
        assert all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_startup_failure_closes_every_allocated_session(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.start_failure = "main"
        store, rid, engine = make_engine(tmp_path, scenario, candidates=5)
        await engine.launch(rid)
        assert len(scenario.instances) == 6
        assert all(instance.closed for instance in scenario.instances)
        assert not engine.sessions
        assert store.get_run(rid)["status"] == "needs_input"
        assert all(t["pid"] is None for t in store.list_tracks(rid))
    asyncio.run(run())


def test_cancelled_run_leaves_no_candidate_or_monitor_tasks(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.waiting = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=5)
        task = engine.launch(rid)
        await asyncio.wait_for(scenario.entered.wait(), 3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                    and getattr(t.get_coro(), "__qualname__", "").startswith("ResearchEngine.")]
        try:
            assert not leftover, "运行已退出但仍有候选/主监视/控制协程存活"
            assert not engine.sessions and all(i.closed for i in scenario.instances)
        finally:
            for t in leftover:
                t.cancel()
            await asyncio.gather(*leftover, return_exceptions=True)
    asyncio.run(run())


def test_shutdown_drains_active_domain_tool_before_interrupting(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.block_tool = True
        store, rid, engine = make_engine(tmp_path, scenario)
        task = engine.launch(rid)
        closing = None
        try:
            for _ in range(200):
                if scenario.tool_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert scenario.tool_started.is_set()
            closing = asyncio.create_task(engine.close())
            await asyncio.sleep(0.05)
            assert not closing.done(), "领域线程仍在运行，不能提前关闭会话并返回已停止"
            assert scenario.instances[0].interrupt_calls == 0
        finally:
            scenario.tool_release.set()
            if closing:
                await closing
            await task
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


@pytest.mark.parametrize("strategy", ["top_k", "adaptive"])
def test_shared_strategy_processes_refine_after_candidate_has_reported(tmp_path, strategy):
    async def run():
        scenario = Scenario()
        scenario.send_refine = True
        scenario.hold_review_for_refine = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1,
                                         strategy=strategy, max_turns=6)
        await asyncio.wait_for(engine.launch(rid), 8)
        assert scenario.refine_sent and scenario.refine_received
        assert scenario.refine_executed.is_set()
        jobs = store.records(rid, "jobs", track_id="candidate-1")
        assert len(jobs) == 3, "先完成两次实验并报告，仍须执行主智能体新投递的第三次实验"
        message = next(m for m in store.records(rid, "messages") if not m["initial_task"])
        candidate = store.get_track(rid, "candidate-1")
        assert message["id"] in candidate["processed_message_ids"]
        assert set(j["id"] for j in jobs) <= set(store.get_run(rid)["coordinated_job_ids"])
        events = [e for e in store.events(rid, limit=1000)["events"] if e["kind"] == "coordination.completed"]
        assert len(events) == 2
        assert jobs[-1]["id"] not in events[0]["payload"]["job_ids"]
        assert store.get_run(rid)["status"] == "completed"
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_pause_interrupts_shared_candidate_waiting_for_main_review(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.pause_coordination = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1,
                                         strategy="adaptive", max_turns=6)
        task = engine.launch(rid)
        await asyncio.wait_for(scenario.coordination_started.wait(), 5)
        for _ in range(50):
            if store.get_track(rid, "candidate-1")["phase"] == "awaiting_coordination":
                break
            await asyncio.sleep(.02)
        assert store.get_track(rid, "candidate-1")["phase"] == "awaiting_coordination"
        store.control(rid, "pause")
        await asyncio.wait_for(task, 3)
        assert store.get_run(rid)["status"] == "paused"
        assert not store.get_run(rid).get("coordination_in_progress")
        assert store.get_track(rid, "candidate-1")["status"] == "paused"
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_shared_candidate_stops_waiting_when_main_has_only_final_turn_left(tmp_path):
    async def run():
        scenario = Scenario()
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1,
                                         strategy="top_k", max_turns=2)
        await asyncio.wait_for(engine.launch(rid), 4)
        assert len(store.records(rid, "jobs", track_id="candidate-1")) == 2
        assert store.get_run(rid)["status"] == "completed"
        assert any(r.get("kind") == "team" for r in store.records(rid, "reports"))
    asyncio.run(run())


def test_failed_main_review_releases_shared_candidate_waiters(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.fail_coordination = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1,
                                         strategy="adaptive", max_turns=6)
        await asyncio.wait_for(engine.launch(rid), 6)
        result = store.get_run(rid)
        assert result["status"] == "needs_input"
        assert "主协调回合失败" in result["error"]
        assert result["coordination_available"] is False
        assert not result.get("coordination_in_progress")
        assert not engine.sessions and all(i.closed for i in scenario.instances)
    asyncio.run(run())


def test_resume_does_not_repeat_main_initial_task_message(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.waiting = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1, max_turns=5)
        task = engine.launch(rid)
        await asyncio.wait_for(scenario.entered.wait(), 3)
        initial = [m for m in store.records(rid, "messages") if m.get("initial_task")]
        assert len(initial) == 1
        main_turns = store.get_track(rid, "main")["turns"]
        store.control(rid, "pause")
        await asyncio.wait_for(task, 3)
        scenario.waiting = False
        store.control(rid, "update", changes={"guidance": "恢复后关注负结果与模型复杂度"})
        store.control(rid, "resume")
        await asyncio.wait_for(engine.launch(rid), 5)
        messages = [m for m in store.records(rid, "messages") if m.get("initial_task")]
        assert [m["id"] for m in messages] == [initial[0]["id"]]
        prompts = [e["payload"]["prompt"] for e in store.events(rid, limit=1000)["events"]
                   if e["kind"] == "task.delivered" and e["track_id"] == "main"]
        assert sum("准备统一研究任务" in p for p in prompts) == 1
        assert "当前用户指导：恢复后关注负结果与模型复杂度" in prompts[-1]
        assert prompts[-1].count("恢复后关注负结果与模型复杂度") == 1
        assert store.get_track(rid, "main")["turns"] >= main_turns + 1
        assert store.get_run(rid)["status"] == "completed"
    asyncio.run(run())


def test_stage_report_does_not_complete_track_after_a_new_experiment(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.stage_report_only = True
        store, rid, engine = make_engine(tmp_path, scenario, max_turns=3,
                                         max_experiments_per_track=2)
        await engine.launch(rid)
        jobs = store.records(rid, "jobs")
        reports = store.records(rid, "reports")
        assert len(jobs) == 2, "补交报告不应额外启动实验"
        assert len(reports) == 2 and len(reports[0]["job_ids"]) == 1
        assert set(reports[-1]["job_ids"]) == {j["id"] for j in jobs}
        assert reports[-1]["created_at"] >= max(j["finished_at"] for j in jobs)
        assert scenario.report_turns == 1
        assert store.get_track(rid, "main")["turns"] == 3
        assert store.get_run(rid)["status"] == "completed"
    asyncio.run(run())


def test_resume_after_turn_budget_exhaustion_only_fills_missing_report(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.stage_report_only = True
        store, rid, engine = make_engine(tmp_path, scenario, max_turns=2,
                                         max_experiments_per_track=2)
        await engine.launch(rid)
        original_jobs = [j["id"] for j in store.records(rid, "jobs")]
        original_thread = store.get_track(rid, "main")["thread_id"]
        assert store.get_run(rid)["status"] == "budget_exhausted"
        assert engine._current_track_report(rid, "main") is None
        store.control(rid, "update", changes={"max_turns": 3})
        store.control(rid, "resume")
        await engine.launch(rid)
        assert [j["id"] for j in store.records(rid, "jobs")] == original_jobs
        assert scenario.starts[-1] == ("main", original_thread)
        assert scenario.instances[-1].turns == 1 and scenario.report_turns == 1
        assert store.get_run(rid)["tokens_used"] == sum(i.turns for i in scenario.instances) * 123
        assert store.get_run(rid)["status"] == "completed"
    asyncio.run(run())


def test_resume_reopens_legacy_completed_candidate_and_main_with_stale_reports(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.stage_report_only = True
        store, rid, engine = make_engine(tmp_path, scenario, candidates=1,
                                         max_turns=2, max_experiments_per_track=2)
        await engine.launch(rid)
        jobs = [j["id"] for j in store.records(rid, "jobs")]
        old_team = engine._current_team_report(rid)
        assert old_team is not None and store.get_run(rid)["status"] == "budget_exhausted"
        for track in store.list_tracks(rid):
            store.update_track(rid, track["track_id"], {"status": "completed"})
        scenario.fail_coordination = True  # 额度已满，恢复仅补报告，无需再调用中间协调回合。
        store.control(rid, "update", changes={"max_turns": 4})
        store.control(rid, "resume")
        await engine.launch(rid)
        assert len(scenario.instances) == 4, "候选与主都需要重建会话以补交最新报告"
        assert [j["id"] for j in store.records(rid, "jobs")] == jobs
        assert scenario.report_turns == 1
        new_team = engine._current_team_report(rid)
        assert new_team["id"] != old_team["id"]
        assert new_team["final_report_epoch"] != old_team["final_report_epoch"]
        assert new_team["created_at"] >= engine._current_track_report(rid, "candidate-1")["created_at"]
        assert len([m for m in store.records(rid, "messages") if m.get("initial_task")]) == 1
        assert store.get_run(rid)["status"] == "completed"
    asyncio.run(run())


def test_stage_team_report_cannot_replace_final_team_report(tmp_path):
    async def run():
        scenario = Scenario()
        scenario.stage_team_report = True
        scenario.final_team_report = False
        scenario.second_turn_delay = 1.2
        store, rid, engine = make_engine(tmp_path, scenario, candidates=2, max_turns=4)
        await asyncio.wait_for(engine.launch(rid), 6)
        teams = [r for r in store.records(rid, "reports") if r.get("kind") == "team"]
        assert teams and all(r["report_stage"] == "stage" for r in teams)
        assert engine._current_team_report(rid) is None
        assert store.get_track(rid, "main")["status"] != "completed"
        assert store.get_track(rid, "main")["turns"] == 4
        assert store.get_run(rid)["status"] == "budget_exhausted"
    asyncio.run(run())


def test_report_freshness_requires_all_settled_jobs_and_completion_time(tmp_path):
    _, _, engine = make_engine(tmp_path, Scenario())
    jobs = [{"id": "first", "status": "completed", "finished_at": 10},
            {"id": "second", "status": "failed", "finished_at": 20}]
    assert not engine._report_is_current({"job_ids": ["first"], "created_at": 30}, jobs)
    assert not engine._report_is_current({"job_ids": ["first", "second"], "created_at": 15}, jobs)
    assert engine._report_is_current({"job_ids": ["first", "second"], "created_at": 20}, jobs)
    assert not engine._report_is_current({"job_ids": ["first", "second"], "created_at": 30},
                                         [jobs[0], {**jobs[1], "status": "running"}])
    assert engine._report_is_current({"job_ids": ["first", "second"]},
                                     [{k: v for k, v in j.items() if k != "finished_at"} for j in jobs])


def test_final_team_report_can_reference_complete_track_reports(tmp_path):
    _, _, engine = make_engine(tmp_path, Scenario())
    jobs = [{"id": "first", "status": "completed", "finished_at": 10},
            {"id": "second", "status": "failed", "finished_at": 20}]
    track_reports = [{"id": "track-report", "job_ids": ["first", "second"]}]
    assert engine._report_is_current({"evidence_ids": ["track-report"], "created_at": 30},
                                     jobs, allow_report_refs=True, reports=track_reports)
    assert not engine._report_is_current({"evidence_ids": ["track-report"], "created_at": 30},
                                         jobs, allow_report_refs=True,
                                         reports=[{"id": "track-report", "job_ids": ["first"]}])
