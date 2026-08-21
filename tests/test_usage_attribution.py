"""用量归账聚合（thermoforge_research.usage）的测试。

不碰 Streamlit、不触网：tmp_path 造假的 planner_trace.json / 孤儿规划轮次 /
会话 JSONL / 假设 yaml，验证归因数学——均摊总量守恒、目标上卷不重复计数、
planner trace 精确归实验、孤儿轮次归目标、坏行容错。
"""

from __future__ import annotations

import json
from pathlib import Path

from thermoforge_research import usage


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n"
                for event in events),
        encoding="utf-8")


def _user(text: str, at: str = "2026-08-19T03:00:00+00:00") -> dict:
    return {"at": at, "type": "message", "role": "user", "content": text}


def _assistant(usage: dict | None = None, cost: float | None = None,
               reasoning: str | None = None, tool_calls: list | None = None,
               at: str = "2026-08-19T03:00:05+00:00") -> dict:
    event: dict = {"at": at, "type": "message", "role": "assistant",
                   "content": ""}
    if usage is not None:
        event["usage"] = usage
    if cost is not None:
        event["cost"] = cost
    if reasoning:
        event["reasoning"] = reasoning
    if tool_calls:
        event["tool_calls"] = tool_calls
    return event


def _tokens(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


# ---------------------------------------------------------------- 会话解析


def test_parse_session_splits_turns_and_collects(tmp_path: Path) -> None:
    path = tmp_path / "agent_sessions" / "s.jsonl"
    _write_jsonl(path, [
        _user("EXP-0001 怎么样"),
        _assistant(usage=_tokens(100, 20), cost=0.01, reasoning="先看报告",
                   tool_calls=[{"id": "c1", "name": "tf_experiment_get",
                                "arguments": '{"experiment_id": "EXP-0001"}'}]),
        {"at": "2026-08-19T03:00:06+00:00", "type": "tool_result",
         "tool": "tf_experiment_get", "ok": True, "id": "EXP-0001",
         "status": "OK"},
        _assistant(usage=_tokens(200, 30), reasoning="总结一下"),
        _user("换个话题"),
        _assistant(usage=_tokens(50, 10)),
    ])

    turns = usage.parse_session_file(path)

    assert len(turns) == 2
    first, second = turns
    assert first.question == "EXP-0001 怎么样"
    assert first.prompt_tokens == 300 and first.completion_tokens == 50
    assert first.cost == 0.01
    assert first.calls == 2
    assert first.reasonings == ["先看报告", "总结一下"]
    # tf_experiment_get 是只读工具：进 read_entities，不进 work_entities
    assert first.read_entities == {"EXP-0001"}
    assert first.work_entities == set()
    assert second.entities == set()


def test_parse_session_tolerates_broken_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        'not json\n'
        + json.dumps(_user("问")) + "\n"
        + json.dumps(_assistant(usage=_tokens(10, 5))) + "\n"
        '{"type": "message", "role": "assistant", "content": 1, ',
        encoding="utf-8")
    turns = usage.parse_session_file(path)
    assert len(turns) == 1
    assert turns[0].calls == 1


def test_usage_missing_still_counts_call(tmp_path: Path) -> None:
    """端点不返回用量时 usage 是 None，但调用次数照记（界面要展示）。"""
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_user("问"), _assistant()])
    turn = usage.parse_session_file(path)[0]
    assert turn.calls == 1
    assert turn.prompt_tokens == 0


# ---------------------------------------------------------------- 归因


def test_attribute_splits_evenly_and_conserves_total() -> None:
    """写/动作工具引用多个对象：均摊，总量守恒。"""
    turn = usage.TurnUsage(at="t", question="登记并运行", prompt_tokens=100,
                           completion_tokens=50, cost=0.3, calls=2,
                           reasonings=["r"],
                           work_entities={"EXP-0001", "RG-0001"})
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned)

    assert not unassigned
    for entity in ("EXP-0001", "RG-0001"):
        entry = per_entity[entity][0]
        assert entry.prompt_tokens == 50
        assert entry.completion_tokens == 25
        assert entry.cost == 0.15
        assert entry.calls == 1
        assert entry.share == 0.5
    total = sum(e.prompt_tokens for entries in per_entity.values()
                for e in entries)
    assert total == turn.prompt_tokens


def test_attribute_single_read_entity_gets_full_turn() -> None:
    """整轮只读查询围绕唯一对象：全额归给它（「EXP-0001 为什么差」）。"""
    turn = usage.TurnUsage(at="t", question="EXP-0001 为什么差",
                           prompt_tokens=80, completion_tokens=40, calls=2,
                           read_entities={"EXP-0001"})
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned)
    assert not unassigned
    assert per_entity["EXP-0001"][0].prompt_tokens == 80


def test_attribute_overview_turn_goes_unassigned() -> None:
    """总览/对比类轮次（只读 + 跨目标）不归账，防止总览问答摊到每个目标上。"""
    turn = usage.TurnUsage(at="t", question="整体进展怎么样",
                           prompt_tokens=5000, completion_tokens=300,
                           calls=5,
                           read_entities={"RG-0001", "RG-0002", "RG-0003"})
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned)
    assert per_entity == {}
    assert len(unassigned) == 1
    assert unassigned[0].prompt_tokens == 5000


def test_attribute_same_goal_reads_coalesce_to_goal() -> None:
    """只读轮次涉及多个对象但同属一个目标：归并到目标级，不进未归账。"""
    turn = usage.TurnUsage(at="t", question="读 RG-0001 和它的实验",
                           prompt_tokens=90, completion_tokens=45, calls=3,
                           read_entities={"RG-0001", "EXP-0001", "EXP-0002"})
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned,
                    experiment_refs={"EXP-0001": ("RG-0001", "H-0001"),
                                     "EXP-0002": ("RG-0001", "H-0002")})
    assert not unassigned
    assert list(per_entity) == ["RG-0001"]
    entry = per_entity["RG-0001"][0]
    assert entry.prompt_tokens == 90
    assert "同目标" in entry.detail


def test_attribute_cross_goal_reads_unassigned() -> None:
    """只读轮次横跨多个目标：无法归账，进未归账。"""
    turn = usage.TurnUsage(at="t", question="对比两个目标",
                           prompt_tokens=90, completion_tokens=45, calls=2,
                           read_entities={"EXP-0001", "EXP-0003"})
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned,
                    experiment_refs={"EXP-0001": ("RG-0001", None),
                                     "EXP-0003": ("RG-0002", None)})
    assert per_entity == {}
    assert len(unassigned) == 1


def test_attribute_unassigned_when_no_entity() -> None:
    turn = usage.TurnUsage(at="t", question="闲聊", prompt_tokens=10,
                           completion_tokens=5, calls=1)
    per_entity: dict[str, list[usage.UsageEntry]] = {}
    unassigned: list[usage.UsageEntry] = []
    usage.attribute([turn], per_entity, unassigned)
    assert per_entity == {}
    assert len(unassigned) == 1
    assert unassigned[0].cost is None  # 未定价是 None，不是 0


# ---------------------------------------------------------------- 汇总


def _make_research_root(root: Path) -> None:
    exp1 = root / "experiments" / "EXP-0001"
    exp1.mkdir(parents=True)
    (exp1 / "report.json").write_text(json.dumps({
        "experiment_id": "EXP-0001", "goal_id": "RG-0001",
        "hypothesis_id": "H-0001", "status": "completed"}),
        encoding="utf-8")
    (exp1 / "planner_trace.json").write_text(json.dumps({
        "round_index": 0, "prompt_summary": "已有 0 轮证据",
        "raw_reply": '{"statement": "s"}',
        "plan": {"statement": "s"}, "error": None, "attempts": 2,
        "reasoning": "选物理模型因为……",
        "cot": "端点返回的完整思维链",
        "usage": _tokens(1000, 200), "cost": None,
        "experiment_id": "EXP-0001"}), encoding="utf-8")

    exp2 = root / "experiments" / "EXP-0002"
    exp2.mkdir(parents=True)
    (exp2 / "report.json").write_text(json.dumps({
        "experiment_id": "EXP-0002", "goal_id": "RG-0001",
        "hypothesis_id": "H-0002", "status": "completed"}),
        encoding="utf-8")

    hyp_dir = root / "hypotheses"
    hyp_dir.mkdir(parents=True)
    (hyp_dir / "H-0001.yaml").write_text(
        "id: H-0001\nrefs:\n  goal_id: RG-0001\nstatement: 假设一\n",
        encoding="utf-8")
    (hyp_dir / "H-0002.yaml").write_text(
        "id: H-0002\nrefs:\n  goal_id: RG-0001\nstatement: 假设二\n",
        encoding="utf-8")


def test_build_book_planner_trace_binds_experiment(tmp_path: Path) -> None:
    _make_research_root(tmp_path)
    book = usage.build_book(tmp_path, [])

    assert book.planner_traces == 1
    assert book.experiment_refs["EXP-0001"] == ("RG-0001", "H-0001")
    assert book.hypothesis_refs == {"H-0001": "RG-0001", "H-0002": "RG-0001"}
    entry = book.per_entity["EXP-0001"][0]
    assert entry.source == "规划"
    assert entry.prompt_tokens == 1000
    # 端点 CoT 与计划自带的 reasoning 是两回事，都保留并贴标签
    assert entry.reasoning == ["端点返回的完整思维链", "选物理模型因为……"]
    assert entry.reasoning_labels == ["思维链（端点 CoT）", "规划理由"]
    assert entry.raw_reply == '{"statement": "s"}'
    assert "第 1 轮规划" in entry.detail


def test_orphan_planner_rounds_roll_into_goal(tmp_path: Path) -> None:
    """未产出实验的规划轮次（planner_rounds/）归到目标级并进总账。"""
    _make_research_root(tmp_path)
    rounds_dir = tmp_path / "planner_rounds" / "RG-0001"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "round_007_planner_stop_120000000000.json").write_text(
        json.dumps({
            "round_index": 6, "round_number": 7, "goal_id": "RG-0001",
            "orphan_kind": "planner_stop", "attempts": 1,
            "raw_reply": '{"stop": "没有可试的新假设"}',
            "reasoning": "", "cot": "没有新招可试，因为……",
            "usage": _tokens(500, 100), "cost": None}),
        encoding="utf-8")

    book = usage.build_book(tmp_path, [])
    assert book.orphan_traces == 1
    view = usage.goal_view(book, "RG-0001")

    orphan = next(e for e in view.direct.entries if e.source == "规划")
    assert "未产出实验" in orphan.detail and "模型主动停止" in orphan.detail
    assert orphan.reasoning == ["没有新招可试，因为……"]
    assert orphan.reasoning_labels == ["思维链（端点 CoT）"]
    # 总账 = 实验的 planner trace(1000) + 孤儿轮(500)
    assert view.total.prompt_tokens == 1500
    assert view.total.completion_tokens == 300


def test_goal_view_rolls_up_without_double_count(tmp_path: Path) -> None:
    _make_research_root(tmp_path)
    turns = [
        # 整轮只读围绕目标本身
        usage.TurnUsage(at="t1", question="目标进展", prompt_tokens=60,
                        completion_tokens=30, calls=1,
                        read_entities={"RG-0001"}),
        # 动作轮：登记了实验又提到目标——各半，上卷后总量仍守恒
        usage.TurnUsage(at="t2", question="登记并跑这个实验",
                        prompt_tokens=100, completion_tokens=50, calls=1,
                        work_entities={"EXP-0001", "RG-0001"}),
        # 整轮只读围绕另一个实验
        usage.TurnUsage(at="t3", question="EXP-0002", prompt_tokens=40,
                        completion_tokens=20, calls=1,
                        read_entities={"EXP-0002"}),
        # 与目标无关的闲聊
        usage.TurnUsage(at="t4", question="闲聊", prompt_tokens=7,
                        completion_tokens=3, calls=1),
    ]
    book = usage.build_book(tmp_path, turns)
    view = usage.goal_view(book, "RG-0001")

    # 总账 = planner(1000) + t1(60) + t2(100) + t3(40)，t4 在未归账
    assert view.total.prompt_tokens == 1200
    assert view.total.completion_tokens == 300
    assert view.direct.prompt_tokens == 110  # t1 全额的 60 + t2 的一半 50

    by_hyp = {v.entity_id: v for v in view.hypotheses}
    assert by_hyp["H-0001"].prompt_tokens == 1050  # planner 1000 + t2 一半 50
    assert by_hyp["H-0002"].prompt_tokens == 40    # t3 全额
    # 上卷 = 直接 + 各假设，互不重复
    assert (view.direct.prompt_tokens
            + sum(v.prompt_tokens for v in view.hypotheses)
            == view.total.prompt_tokens)

    unassigned = usage.unassigned_view(book)
    assert unassigned.prompt_tokens == 7
    assert unassigned.completion_tokens == 3


def test_experiment_view_merges_planner_and_copilot(tmp_path: Path) -> None:
    _make_research_root(tmp_path)
    turns = [usage.TurnUsage(at="t1", question="问", prompt_tokens=80,
                             completion_tokens=40, calls=1,
                             read_entities={"EXP-0001"})]
    book = usage.build_book(tmp_path, turns)
    view = usage.experiment_view(book, "EXP-0001")

    assert view.prompt_tokens == 1080
    assert view.completion_tokens == 240
    assert sorted(e.source for e in view.entries) == ["副驾", "规划"]
    assert view.calls == 3  # planner attempts=2 + 副驾 1 次调用


def test_empty_root_yields_empty_book(tmp_path: Path) -> None:
    book = usage.build_book(tmp_path, [])
    view = usage.goal_view(book, "RG-9999")
    assert view.total.entries == []
    assert view.total.cost is None
    assert usage.unassigned_view(book).entries == []
