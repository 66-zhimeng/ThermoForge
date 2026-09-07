"""V2 真实内核接入：来源前置、身份隔离、冻结评价和作业幂等。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from phase2_helpers import FEATURES, TARGET, view_definition
from phase34_helpers import make_ctx
from thermoforge_research.tools import tf_goal_create
from thermoforge_v2.research_tools import ResearchTools, prepare_protocol
from thermoforge_v2.store import RunStore


@pytest.fixture()
def setup(tmp_path):
    ctx, ref = make_ctx(tmp_path, n_steps=600)
    goal = tf_goal_create(ctx, {
        "name": "V2 冷机研究", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES), "acceptance": {"cvrmse_max": 0.5},
    })
    assert goal["ok"], goal
    config = {"goal_id": goal["id"], "dataset_ref": ref, "research_mode": "acceptance", "candidates": 5,
              "max_experiments": 8, "max_experiments_per_track": 4, "seed": 79,
              "purge_seconds": 0, "embargo_seconds": 2700}
    protocol = prepare_protocol(ctx, config)
    store = RunStore(tmp_path / "v2")
    run = store.create_run(config, protocol, "test-run")
    store.update_run(run["run_id"], {"status": "running"})
    for track in ("candidate_1", "candidate_2"):
        store.create_track(run["run_id"], track, "candidate",
                           research_root=str(tmp_path / "tracks" / track),
                           workspace=str(tmp_path / "workspaces" / track))
    return ctx, store, run["run_id"], ResearchTools(store, ctx, run["run_id"], "candidate_1")


def idea(tools, **overrides):
    args = {"statement": "先拟合可复现的线性基线", "reason": "建立可比较起点",
            "prediction": "验证误差有限且可复现", "falsification": "验证误差超过阈值则否定",
            "origin": "conjecture"}
    args.update(overrides)
    result = tools.call("research_idea_create", args)
    assert result["ok"], result
    return result["id"]


def experiment(tools, idea_id, key="exp-1", alpha=.5):
    return tools.call("research_experiment_run", {
        "idea_id": idea_id,
        "model": {"category": "data", "estimator": "ridge", "hyperparameters": {"alpha": alpha}},
        "idempotency_key": key})


def test_protocol_freezes_comparable_data_seed_and_search_surface(setup):
    ctx, store, run_id, tools = setup
    p = tools.protocol
    assert p["runtime"]["random_seed"] == 79
    assert p["feedback_surface"] == "validate"
    assert p["modelability"]["evaluated_on"] == "train"
    assert prepare_protocol(ctx, store.get_run(run_id)["config"]) == p
    cfg = dict(store.get_run(run_id)["config"], validation={
        "temporal_split": {"train": .7, "validate": .15, "test": .15},
        "rolling_cv": {"enabled": True, "initial_train_fraction": .4}})
    with pytest.raises(ValueError, match="rolling_cv"):
        prepare_protocol(ctx, cfg)


def test_protocol_rejects_different_dataset_view_and_missing_validation(setup):
    ctx, store, run_id, _ = setup
    cfg = store.get_run(run_id)["config"]
    cfg["validation"] = {"temporal_split": {"train": .8, "validate": 0, "test": .2}}
    with pytest.raises(ValueError, match="validate"):
        prepare_protocol(ctx, cfg)
    view = view_definition(cfg["dataset_ref"])
    view["features"].append(TARGET)
    registered = ctx.ledger.register_view(view, actor="human")
    cfg["view_id"], cfg["validation"] = registered["id"], None
    with pytest.raises(ValueError, match="DD-16"):
        prepare_protocol(ctx, cfg)


def test_modelability_uses_resampled_view_and_selected_features(setup, monkeypatch):
    import thermoforge_v2.research_tools as module

    ctx, store, run_id, _ = setup
    cfg = store.get_run(run_id)["config"]
    view = view_definition(cfg["dataset_ref"])
    view.update(resolution="30min", features=list(FEATURES[:2]))
    cfg["view_id"] = ctx.ledger.register_view(view, actor="human")["id"]
    observed = {}
    original = module.build_modelability_report

    def inspect(vault, ref, **kwargs):
        table = vault.load_data(ref)
        observed.update(rows=table.num_rows, columns=table.column_names,
                        candidate_inputs=kwargs["candidate_inputs"])
        return original(vault, ref, **kwargs)

    monkeypatch.setattr(module, "build_modelability_report", inspect)
    protocol = prepare_protocol(ctx, cfg)
    assert observed["rows"] <= 210  # 600 点经30分钟重采样后300点，再取训练段。
    assert observed["candidate_inputs"] == list(FEATURES[:2])
    assert all(c == "timestamp" or c.split(".", 1)[1] in set(FEATURES[:2]) | {TARGET}
               for c in observed["columns"])
    assert protocol["view_definition"]["resolution"] == "30min"


def test_idea_requires_reason_and_real_references_but_allows_independent_conjectures(setup):
    _, _, _, tools = setup
    a = idea(tools)
    b = idea(tools)
    assert a != b
    missing = tools.call("research_idea_create", {"statement": "只有一句想法"})
    assert not missing["ok"]
    forged = tools.call("research_idea_create", {
        "statement": "文献假设", "reason": "来源", "prediction": "误差降低", "falsification": "反证",
        "origin": "literature", "source_ids": ["made-up-source"]})
    assert not forged["ok"]
    assert not tools.call("tf_preprocess_approve", {"actor": "human"})["ok"]


def test_track_boundary_rejects_foreign_idea_and_cross_track_sources(setup):
    ctx, store, run_id, tools = setup
    other = ResearchTools(store, ctx, run_id, "candidate_2")
    foreign = idea(other)
    assert not experiment(tools, foreign)["ok"]
    assert not tools.call("research_history", {"track_id": "candidate_2"})["ok"]
    assert tools.ctx.research_root != other.ctx.research_root


def test_two_real_experiments_are_traceable_idempotent_and_holdout_blind(setup):
    _, store, run_id, tools = setup
    first_idea = idea(tools)
    first = experiment(tools, first_idea)
    assert first["ok"], first
    assert first["summary"]["status"] == "completed", first
    feedback = first["summary"]["result"]
    assert feedback["metrics"]["CVRMSE"] is not None
    assert feedback["feedback_surface"] == "validate"
    assert not any(key in json.dumps(first) for key in ["physics_overall_rate", "artifacts", "rolling_cv", '"A"'])
    exp_dir = tools.ctx.research_root / "experiments" / feedback["experiment_id"]
    full = json.loads((exp_dir / "report.json").read_text(encoding="utf-8"))
    assert "A" in full["metrics"]["surfaces"]
    before = (exp_dir / "report.json").stat().st_mtime_ns
    repeat = experiment(tools, first_idea)
    assert repeat["id"] == first["id"] and repeat["summary"]["duplicate"]
    assert (exp_dir / "report.json").stat().st_mtime_ns == before
    second_idea = idea(tools, origin="history", parent_job_ids=[first["id"]],
                       parent_idea_ids=[first_idea], reason="根据第一次验证误差改变正则强度")
    second = experiment(tools, second_idea, key="exp-2", alpha=10)
    assert second["summary"]["status"] == "completed", second
    assert store.get_run(run_id)["experiments_reserved"] == 2
    assert len(tools.ctx.ledger.list_goals()) == 2
    assert not experiment(tools, first_idea, key="exp-1", alpha=3)["ok"]


def test_prep_and_run_cannot_override_protocol_or_actor(setup):
    _, _, _, tools = setup
    identifier = idea(tools)
    result = tools.call("research_experiment_run", {"idea_id": identifier,
        "model": {"category": "data", "estimator": "ridge"}, "idempotency_key": "one",
        "validation": {"temporal_split": {"train": .9, "validate": .1, "test": 0}}})
    assert not result["ok"]
    assert not tools.call("research_lab_submit", {"name": "../bad", "source": "print(1)"})["ok"]
    assert not tools.call("research_lab_submit", {"name": "bad", "source": "import subprocess"})["ok"]


def test_data_summary_omits_final_holdout_rows(setup):
    _, _, _, tools = setup
    response = tools.call("research_data_summary", {})
    assert response["ok"], response
    sections = response["summary"]["sections"]
    assert set(sections) == {"train", "validate"}
    assert all(len(section["samples"]) <= 12 for section in sections.values())
    end = tools.protocol["split_boundaries"]["b2"]
    assert all(sample["timestamp"] < end for section in sections.values() for sample in section["samples"])


def test_sources_distinguish_retrieved_excerpt_from_supplied_claim(setup, monkeypatch):
    _, _, _, tools = setup
    supplied = tools.call("research_source_register", {"title": "外部资料", "kind": "article",
        "url": "https://example.org/paper", "read_scope": "excerpt", "content": "一段摘录"})
    assert supplied["ok"] and supplied["summary"]["verification"] == "agent_supplied"
    monkeypatch.setattr("thermoforge_v2.research_tools.read_public_source", lambda url: {
        "url": url, "content": "x" * 17000, "read_scope": "fulltext", "verification": "retrieved"})
    retrieved = tools.call("research_source_read", {"title": "真正读取", "url": "https://example.org/read"})
    assert retrieved["ok"]
    assert retrieved["summary"]["read_scope"] == "excerpt"
    assert retrieved["summary"]["retrieved_scope"] == "fulltext"
    assert idea(tools, origin="literature", source_ids=[retrieved["id"]])
    tail = tools.call("research_source_excerpt", {"source_id": retrieved["id"], "offset": 16000})
    assert tail["summary"]["content"] == "x" * 1000


def test_public_reader_rejects_local_and_credential_urls(monkeypatch):
    from thermoforge_v2.research_tools import _public_url
    monkeypatch.setattr("socket.getaddrinfo", lambda *args: [(2, 1, 6, "", ("127.0.0.1", 80))])
    for url in ("http://127.0.0.1/secrets", "file:///C:/private", "https://user:pass@example.org"):
        with pytest.raises(ValueError):
            _public_url(url)


def test_lab_version_and_report_references_are_mandatory(setup):
    _, _, _, tools = setup
    identifier = idea(tools)
    unpinned = tools.call("research_experiment_run", {"idea_id": identifier,
        "model": {"category": "lab", "hyperparameters": {"lab": "model"}}, "idempotency_key": "lab"})
    assert not unpinned["ok"] and "name@vN" in unpinned["error"]
    invalid = tools.call("research_report_submit", {"title": "结论", "summary": "尚未实验", "body": "没有结果",
        "idea_ids": [], "job_ids": [], "limitations": "无证据"})
    assert not invalid["ok"]
    valid = tools.call("research_report_submit", {"title": "研究未完成报告", "summary": "仅有假设",
        "body": "提出假设但尚未实验，不声称效果", "idea_ids": [identifier], "job_ids": [], "limitations": "尚未实验"})
    assert valid["ok"]


def test_duplicate_concurrent_requests_reserve_exactly_one_experiment(setup, monkeypatch):
    import threading
    import thermoforge_v2.research_tools as module

    _, store, run_id, tools = setup
    identifier = idea(tools)
    entered, release = threading.Event(), threading.Event()

    def delayed(ctx, exp_id, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"ok": True, "summary": {"surfaces": {"validate": {
            "metrics": {"CVRMSE": .1}, "n_samples": 100}}}}

    monkeypatch.setattr(module, "tf_experiment_run", delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(experiment, tools, identifier)
        assert entered.wait(5)
        second = pool.submit(experiment, tools, identifier).result(timeout=5)
        assert second["summary"]["duplicate"]
        release.set()
        assert first.result(timeout=5)["ok"]
    assert store.get_run(run_id)["experiments_reserved"] == 1


def test_pause_after_reservation_does_not_start_training(setup, monkeypatch):
    ctx, store, run_id, _ = setup
    stale_running = store.get_run(run_id)

    class PauseOnAcquire:
        def __enter__(self):
            store.update_run(run_id, {"status": "paused"})

        def __exit__(self, *args):
            pass

    tools = ResearchTools(store, ctx, run_id, "candidate_1", PauseOnAcquire())
    # 普通 get_run 即便仍返回暂停前的快照，事务门也必须从数据库读真实状态。
    monkeypatch.setattr(store, "get_run", lambda rid: stale_running)
    monkeypatch.setattr("thermoforge_v2.research_tools.tf_experiment_run",
                        lambda *args, **kwargs: pytest.fail("暂停后不能训练"))
    result = experiment(tools, idea(tools))
    assert result["summary"]["status"] == "cancelled"
    assert result["summary"]["result"]["training_started"] is False


def test_read_responses_obey_envelope_size_limit(setup, monkeypatch):
    _, _, _, tools = setup
    monkeypatch.setattr("thermoforge_v2.research_tools.read_public_source", lambda url: {
        "url": url, "content": "论" * 90000, "read_scope": "fulltext", "verification": "retrieved"})
    response = tools.call("research_source_read", {"title": "大文献", "url": "https://example.org/read"})
    assert len(json.dumps(response, ensure_ascii=False).encode("utf-8")) <= 32 * 1024
    assert response["summary"]["delivered_chars"] == 6000


def test_lab_model_runs_autonomously_in_own_track(setup):
    from test_model_lab import MINI_SOURCE

    _, _, _, tools = setup
    submitted = tools.call("research_lab_submit", {"name": "mini_view", "source": MINI_SOURCE})
    assert submitted["ok"], submitted
    assert submitted["id"].endswith("@v1")
    run = tools.call("research_experiment_run", {"idea_id": idea(tools), "idempotency_key": "lab-1",
        "model": {"category": "lab", "hyperparameters": {"lab": submitted["id"]}}})
    assert run["summary"]["status"] == "completed", run
    assert run["summary"]["result"]["model"]["content_hash"] == submitted["summary"]["content_hash"]


def test_shared_evidence_requires_later_round_strategy_and_explicit_source_chain(setup):
    ctx, store, run_id, tools = setup
    other = ResearchTools(store, ctx, run_id, "candidate_2")
    source = other.call("research_source_register", {"title": "论文", "kind": "paper",
        "url": "https://example.org/paper", "read_scope": "abstract", "content": "论文摘要"})
    parent = idea(other, origin="literature", source_ids=[source["id"]])
    args = {"statement": "共享后改进", "reason": "来源启发", "prediction": "误差更低",
            "falsification": "同协议无改善", "origin": "literature",
            "source_ids": [source["id"]], "parent_idea_ids": [parent]}
    store.add_record(run_id, "messages", {"from": "main", "to": "candidate_1", "text": "共享证据",
        "evidence_ids": [source["id"], parent], "version": 3}, track_id="main")
    store.update_track(run_id, "candidate_1", {"turns": 2})
    assert not tools.call("research_idea_create", args)["ok"]  # independent 模式不共享
    config = store.get_run(run_id)["config"] | {"strategy": "top_k"}
    store.update_run(run_id, {"config": config})
    store.update_track(run_id, "candidate_1", {"turns": 1})
    assert not tools.call("research_idea_create", args)["ok"]  # 首轮不能偷看
    store.update_track(run_id, "candidate_1", {"turns": 2})
    child = tools.call("research_idea_create", args)
    assert child["ok"], child
    assert child["summary"]["source_ids"] == [source["id"]]
    assert child["summary"]["parent_idea_ids"] == [parent]
    assert not tools.call("research_idea_create", args | {"origin": "conjecture", "source_ids": []})["ok"]
    # 单独共享想法不隐式授权其资料；收件轨迹也必须一致。
    private_source = other.call("research_source_register", {"title": "未共享", "kind": "article",
        "url": "https://example.org/private", "read_scope": "metadata"})
    assert not tools.call("research_idea_create", args | {"source_ids": [private_source["id"]]})["ok"]


def test_initial_common_main_sources_are_usable_without_enabling_cross_candidate_sharing(setup):
    ctx, store, run_id, candidate = setup
    track_root = store.root / "tracks" / "main"
    store.create_track(run_id, "main", "main", research_root=str(track_root / "research"),
                       workspace=str(track_root / "workspace"))
    main = ResearchTools(store, ctx, run_id, "main")
    source = main.call("research_source_register", {"title": "共同初始文献", "kind": "paper",
        "url": "https://example.org/common", "read_scope": "abstract", "content": "所有候选统一可读的摘要"})
    args = {"statement": "基于共同资料独立提出假设", "reason": "统一初始证据",
            "prediction": "验证误差降低", "falsification": "无改善",
            "origin": "literature", "source_ids": [source["id"]]}
    assert not candidate.call("research_idea_create", args)["ok"]
    store.add_record(run_id, "messages", {"from": "main", "to": "all", "text": "共同任务",
        "initial_task": True, "evidence_ids": [source["id"]], "version": 1}, track_id="main")
    assert candidate.call("research_idea_create", args)["ok"]
    excerpt = candidate.call("research_source_excerpt", {"source_id": source["id"]})
    assert excerpt["ok"] and excerpt["summary"]["content"] == "所有候选统一可读的摘要"
    # 没有 initial_task 标记的后期主来源不能被当成初始资料。
    late = main.call("research_source_register", {"title": "后期分析", "kind": "paper",
        "url": "https://example.org/late", "read_scope": "abstract", "content": "后期资料"})
    store.add_record(run_id, "messages", {"from": "main", "to": "all", "text": "后期消息",
        "initial_task": False, "evidence_ids": [late["id"]], "version": 2}, track_id="main")
    assert not candidate.call("research_idea_create", args | {"source_ids": [late["id"]]})["ok"]
    other = ResearchTools(store, ctx, run_id, "candidate_2")
    foreign = other.call("research_source_register", {"title": "候选私有资料", "kind": "paper",
        "url": "https://example.org/candidate", "read_scope": "metadata"})
    store.add_record(run_id, "messages", {"from": "main", "to": "all", "text": "错误标记",
        "initial_task": True, "evidence_ids": [foreign["id"]], "version": 3}, track_id="main")
    assert not candidate.call("research_idea_create", args | {"source_ids": [foreign["id"]]})["ok"]


def test_main_can_reference_candidate_evidence_without_broadcasting_it(setup):
    ctx, store, run_id, candidate = setup
    track_root = store.root / "tracks" / "main"
    store.create_track(run_id, "main", "main", research_root=str(track_root / "research"),
                       workspace=str(track_root / "workspace"))
    main = ResearchTools(store, ctx, run_id, "main")
    source = candidate.call("research_source_register", {"title": "候选发现的论文", "kind": "paper",
        "url": "https://example.org/idea", "read_scope": "abstract", "content": "有依据的摘要"})
    parent = idea(candidate, origin="literature", source_ids=[source["id"]])
    child = idea(main, origin="literature", source_ids=[source["id"]], parent_idea_ids=[parent])
    assert child and store.records(run_id, "messages") == []
    assert main.call("research_source_excerpt", {"source_id": source["id"]})["ok"]
    # 协调读取权限不允许用别人的 idea_id 直接执行并占自己的预算。
    assert not experiment(main, parent)["ok"]
    assert not main.call("research_history", {"kind": "finalizations"})["ok"]
