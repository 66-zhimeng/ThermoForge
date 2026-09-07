"""V2 网页通过同一客户端操作后台；测试不启动真实 Codex 或研究。"""

from copy import deepcopy

from streamlit.testing.v1 import AppTest

from thermoforge_webui.screens import research_v2


class FakeClient:
    def __init__(self, *, available=True, with_run=True):
        self.available = available
        self.with_run = with_run
        self.started = []
        self.controls = []
        self.event_calls = []
        self.report_calls = []
        self.snapshot = {
            "run": {"run_id": "RUN-001", "status": "running", "version": 3,
                    "config": {"goal_id": "RG-0001", "max_experiments": 20, "token_budget": 200000},
                    "protocol": {"fingerprint": "test"}},
            "tracks": [{"track_id": "main", "role": "main", "pid": 101, "status": "running", "turns": 1},
                       *[{"track_id": f"candidate-{n}", "role": "candidate", "pid": 101+n,
                          "status": "waiting_experiment", "turns": 2} for n in range(1, 6)]],
            "jobs": [], "sources": [], "ideas": [], "findings": [], "decisions": [], "reports": []}

    def prepare(self):
        return {"available": self.available, "errors": [] if self.available else ["Codex 未认证"],
                "goals": [{"id": "RG-0001", "name": "冷机功率"}],
                "datasets": [{"ref": "CHILLER@rev_0001"}], "defaults": {}}

    def list_runs(self):
        return [self.snapshot["run"]] if self.with_run else []

    def get_run(self, run_id):
        return deepcopy(self.snapshot)

    def events(self, run_id, after=0, limit=100):
        self.event_calls.append((run_id, after, limit))
        return {"cursor": 1, "events": [{"id": "EVENT-1", "kind": "started", "message": "六条轨迹就绪"}] if after == 0 else []}

    def start(self, config, idempotency_key):
        self.started.append((config, idempotency_key))
        self.with_run = True
        return self.snapshot["run"]

    def control(self, run_id, action, expected_version=None, changes=None):
        self.controls.append((run_id, action, expected_version, changes))
        run = self.snapshot["run"]
        run["status"] = {"pause": "paused", "resume": "running", "cancel": "cancelled"}.get(action, run["status"])
        if action == "update":
            run["config"].update(changes or {})
        run["version"] += 1
        return self.snapshot["run"]

    def get_report(self, run_id, track_id=None):
        self.report_calls.append((run_id, track_id))
        from thermoforge_v2.reports import build_report
        return build_report(self.snapshot, track_id=track_id)


def run_page(monkeypatch, client):
    monkeypatch.setattr(research_v2, "_client", lambda: client)
    return AppTest.from_string(
        "from thermoforge_webui.screens.research_v2 import research_v2_page\nresearch_v2_page()",
        default_timeout=20).run()


def test_monitor_shows_six_real_track_records(monkeypatch):
    client = FakeClient()
    app = run_page(monkeypatch, client)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "实例记录").value == "6"
    assert next(m for m in app.metric if m.label == "执行中的轨迹").value == "1"
    assert "六条轨迹就绪" in " ".join(t.value for t in app.text)


def test_create_uses_actual_discovery_and_six_codex_configuration(monkeypatch):
    client = FakeClient(with_run=False)
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "启动后台研究").click().run()
    assert not app.exception
    config, key = client.started[0]
    assert config["goal_id"] == "RG-0001"
    assert config["dataset_ref"] == "CHILLER@rev_0001"
    assert config["candidates"] == 5
    assert config["research_mode"] == "autonomous"
    assert config["reuse_experiments"] is False
    assert config["max_turns"] >= 3
    assert config["experiment_workers"] == 1
    assert key
    worker = next(w for w in app.number_input if w.label == "实验并发数（当前固定串行）")
    assert worker.value == 1 and worker.disabled
    assert next(w for w in app.number_input if w.label == "每条轨迹最多执行片段").proto.min == 3


def test_shared_reuse_is_opt_in_and_reset_when_returning_to_independent(monkeypatch):
    client = FakeClient(with_run=False)
    app = run_page(monkeypatch, client)
    reuse = next(w for w in app.checkbox if w.label == "共享后复用相同实验结果")
    assert reuse.disabled and not reuse.value
    next(w for w in app.selectbox if w.label == "研究策略").set_value("top_k").run()
    reuse = next(w for w in app.checkbox if w.label == "共享后复用相同实验结果")
    assert not reuse.disabled and not reuse.value
    reuse.check().run()
    next(b for b in app.button if b.label == "启动后台研究").click().run()
    assert not app.exception
    assert client.started[0][0]["strategy"] == "top_k"
    assert client.started[0][0]["reuse_experiments"] is True
    next(w for w in app.selectbox if w.label == "研究策略").set_value("independent").run()
    reuse = next(w for w in app.checkbox if w.label == "共享后复用相同实验结果")
    assert reuse.disabled and not reuse.value
    next(b for b in app.button if b.label == "启动后台研究").click().run()
    assert not app.exception
    assert client.started[-1][0]["strategy"] == "independent"
    assert client.started[-1][0]["reuse_experiments"] is False


def test_monitor_traces_proposals_stops_and_reused_results(monkeypatch):
    client = FakeClient()
    client.snapshot["run"].update(research_stage="sharing")
    client.snapshot["run"]["config"].update(research_mode="autonomous", strategy="top_k")
    client.snapshot["proposals"] = [{
        "id": "PROPOSAL-1", "track_id": "candidate-1", "idea_id": "IDEA-1", "version": 2,
        "status": "committed", "purpose": "refine", "parent_job_ids": ["JOB-1"],
        "model": {"category": "data", "estimator": "ridge"}}]
    client.snapshot["stops"] = [{
        "id": "STOP-1", "track_id": "candidate-1", "reason": "预算内已完成预期假说检验",
        "evidence_ids": ["FINDING-1", "REPORT-1"], "created_at": 1}]
    client.snapshot["jobs"] = [{
        "id": "JOB-2", "track_id": "candidate-1", "proposal_id": "PROPOSAL-1", "idea_id": "IDEA-1",
        "status": "completed", "executed": False, "reused_from_job_id": "JOB-1",
        "result": {"experiment_id": "EXP-0001", "metrics": {"CVRMSE": .12}}}]
    app = run_page(monkeypatch, client)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "研究阶段").value == "共享证据与调整"
    assert next(m for m in app.metric if m.label == "已冻结方案").value == "1"
    assert next(m for m in app.metric if m.label == "已登记停止决定").value == "1"
    tables = " ".join(frame.value.to_json(force_ascii=False) for frame in app.dataframe)
    assert "PROPOSAL-1" in tables and "预算内已完成预期假说检验" in tables
    assert "复用既有结果" in tables and "FINDING-1" in tables


def test_unavailable_runtime_disables_start_but_keeps_monitor(monkeypatch):
    client = FakeClient(available=False)
    app = run_page(monkeypatch, client)
    assert not app.exception
    assert next(b for b in app.button if b.label == "启动后台研究").disabled
    assert next(m for m in app.metric if m.label == "实例记录").value == "6"


def test_pause_is_versioned_service_control(monkeypatch):
    client = FakeClient()
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "暂停研究").click().run()
    assert not app.exception
    assert client.controls[0] == ("RUN-001", "pause", 3, None)


def test_reports_render_from_backend_snapshot(monkeypatch):
    app = run_page(monkeypatch, FakeClient())
    next(b for b in app.button if b.label == "生成最新事实报告").click().run()
    assert not app.exception
    assert "负结果与未完成工作" in " ".join(t.value for t in app.markdown)


def test_connection_failure_is_visible_without_creating_work(monkeypatch):
    def unavailable():
        raise ConnectionError("服务不可达")
    monkeypatch.setattr(research_v2, "_client", unavailable)
    app = AppTest.from_string(
        "from thermoforge_webui.screens.research_v2 import research_v2_page\nresearch_v2_page()",
        default_timeout=20).run()
    assert not app.exception
    assert "服务不可达" in app.error[0].value


def test_pause_then_resume_keeps_run_and_uses_latest_version(monkeypatch):
    client = FakeClient()
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "暂停研究").click().run()
    assert next(m for m in app.metric if m.label == "运行状态").value == "已暂停"
    next(b for b in app.button if b.label == "恢复研究").click().run()
    assert not app.exception
    assert client.controls == [("RUN-001", "pause", 3, None), ("RUN-001", "resume", 4, None)]
    assert next(m for m in app.metric if m.label == "运行状态").value == "执行中"
    assert not client.started


def test_update_form_sends_only_mutable_guidance_and_budgets(monkeypatch):
    client = FakeClient()
    app = run_page(monkeypatch, client)
    next(w for w in app.text_area if w.label == "后续研究要求").set_value("先复现负结果")
    next(w for w in app.number_input if w.label == "调整全局实验上限").set_value(24)
    next(w for w in app.number_input if w.label == "调整全局 token 预算").set_value(250000)
    next(b for b in app.button if b.label == "保存后续配置").click().run()
    assert not app.exception
    assert client.controls == [("RUN-001", "update", 3, {
        "guidance": "先复现负结果", "max_experiments": 24, "token_budget": 250000})]
    assert client.snapshot["run"]["config"]["goal_id"] == "RG-0001"


def test_terminal_run_disables_control_but_can_export_report(monkeypatch):
    client = FakeClient()
    client.snapshot["run"]["status"] = "completed"
    app = run_page(monkeypatch, client)
    for label in ("暂停研究", "恢复研究", "取消研究", "保存后续配置"):
        assert next(b for b in app.button if b.label == label).disabled
    next(b for b in app.button if b.label == "生成最新事实报告").click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 3


def test_report_scope_switch_does_not_reuse_another_track_report(monkeypatch):
    client = FakeClient()
    client.snapshot["ideas"] = [
        {"id": "I-1", "track_id": "main", "statement": "主轨迹专有想法", "origin": "conjecture"},
        {"id": "I-2", "track_id": "candidate-1", "statement": "候选专有想法", "origin": "conjecture"}]
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "生成最新事实报告").click().run()
    next(w for w in app.selectbox if w.label == "报告范围").set_value("candidate-1").run()
    assert not app.get("download_button")
    next(b for b in app.button if b.label == "生成最新事实报告").click().run()
    assert not app.exception
    assert client.report_calls == [("RUN-001", None), ("RUN-001", "candidate-1")]
    report = app.session_state["v2_report_RUN-001_candidate-1"]
    assert [idea["id"] for idea in report["ideas"]] == ["I-2"]


def test_refresh_advances_event_cursor_without_duplicate_messages(monkeypatch):
    client = FakeClient()
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "刷新状态").click().run()
    assert not app.exception
    assert client.event_calls[0] == ("RUN-001", 0, 100)
    assert all(after == 1 for _, after, _ in client.event_calls[1:])
    assert sum("六条轨迹就绪" in t.value for t in app.text) == 1


def test_uncertain_start_reuses_idempotency_key(monkeypatch):
    client = FakeClient(with_run=False)
    original = client.start

    def uncertain(config, idempotency_key):
        if not client.started:
            client.started.append((config, idempotency_key))
            raise ConnectionError("响应丢失")
        return original(config, idempotency_key)

    client.start = uncertain
    app = run_page(monkeypatch, client)
    next(b for b in app.button if b.label == "启动后台研究").click().run()
    assert "响应丢失" in app.error[0].value
    next(b for b in app.button if b.label == "启动后台研究").click().run()
    assert not app.exception
    assert len(client.started) == 2
    assert client.started[0][1] == client.started[1][1]
