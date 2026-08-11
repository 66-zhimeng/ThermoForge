"""Research Ledger 测试（research-loop.md §8、§2；conventions §7.9）。

创建/查询/引用关系（哪些实验支持某结论、哪些假设未验证、失败原因）、
每次状态转换留痕、实验完成后不可变（TFX-904）。
"""

from __future__ import annotations

import pytest

from thermoforge_research.errors import ResearchError
from thermoforge_research.ledger import ResearchLedger

ACTOR = "test-agent"


def _exp_definition(exp_id: str, goal_id: str, hyp_id: str,
                    view_id: str) -> dict:
    return {
        "experiment_id": exp_id,
        "goal_id": goal_id,
        "hypothesis_id": hyp_id,
        "dataset_view": view_id,
        "model": {"category": "data", "estimator": "ridge",
                  "hyperparameters": {"alpha": 0.5}},
        "target": "input_power",
        "validation": {
            "temporal_split": {"train": 0.7, "validate": 0.15, "test": 0.15},
        },
        "metrics": ["RMSE", "CVRMSE", "NMBE"],
        "runtime": {"environment_lock": "x" * 64, "random_seed": 1},
    }


@pytest.fixture()
def ledger(tmp_path):
    return ResearchLedger(tmp_path / "research")


def _seed_entities(led: ResearchLedger):
    goal = led.create_goal("冷水机输入功率模型", actor=ACTOR)
    view = led.register_view({"dataset": "DS@rev_0001", "features": ["f"]},
                             actor=ACTOR)
    hyp = led.create_hypothesis(goal["id"], "冷却水温度主导功率",
                                actor=ACTOR)
    return goal, view, hyp


def test_create_and_directory_layout(ledger, tmp_path):
    goal, view, hyp = _seed_entities(ledger)
    assert goal["id"] == "RG-0001"
    assert hyp["id"] == "H-0001"
    assert view["id"] == "VIEW-0001"
    root = tmp_path / "research"
    for sub in ("goals", "hypotheses", "experiments", "findings",
                "decisions", "models", "views", "ids"):
        assert (root / sub).is_dir(), sub
    assert (root / "goals" / "RG-0001.yaml").exists()
    assert (root / "hypotheses" / "H-0001.yaml").exists()
    assert (root / "ledger_index.json").exists()


def test_experiment_lifecycle_and_transition_trail(ledger):
    goal, view, hyp = _seed_entities(ledger)
    exp_id = ledger.allocator.allocate("EXP-")
    exp = ledger.register_experiment(
        _exp_definition(exp_id, goal["id"], hyp["id"], view["id"]),
        actor=ACTOR)
    assert exp["status"] == "created"

    ledger.transition(exp_id, "running", reason="runner 启动", actor=ACTOR,
                      inputs=["spec.json"])
    done = ledger.transition(
        exp_id, "completed", reason="实验完成", actor=ACTOR,
        inputs=["spec.json"], outputs=["report.json", "metrics.json"])
    trail = done["transitions"]
    assert [t["to"] for t in trail] == ["created", "running", "completed"]
    last = trail[-1]
    assert last["reason"] == "实验完成"
    assert last["actor"] == ACTOR
    assert last["inputs"] == ["spec.json"]
    assert last["outputs"] == ["report.json", "metrics.json"]
    assert last["at"]  # 时间戳留痕（§2）


def test_completed_experiment_immutable_tfx904(ledger):
    goal, view, hyp = _seed_entities(ledger)
    exp_id = ledger.allocator.allocate("EXP-")
    ledger.register_experiment(
        _exp_definition(exp_id, goal["id"], hyp["id"], view["id"]),
        actor=ACTOR)
    ledger.transition(exp_id, "running", reason="启动", actor=ACTOR)
    ledger.transition(exp_id, "completed", reason="完成", actor=ACTOR)
    with pytest.raises(ResearchError) as excinfo:
        ledger.transition(exp_id, "running", reason="试图重跑", actor=ACTOR)
    assert excinfo.value.code == "TFX-904"


def test_transition_requires_reason(ledger):
    goal, _, _ = _seed_entities(ledger)
    with pytest.raises(ValueError):
        ledger.transition(goal["id"], "stopped", reason="", actor=ACTOR)


def test_reference_queries(ledger):
    goal, view, hyp = _seed_entities(ledger)
    exp_id = ledger.allocator.allocate("EXP-")
    ledger.register_experiment(
        _exp_definition(exp_id, goal["id"], hyp["id"], view["id"]),
        actor=ACTOR)
    finding = ledger.create_finding(
        "低负荷区存在系统性偏差", actor=ACTOR, supported_by=[exp_id],
        hypothesis_id=hyp["id"])
    assert finding["id"] == "F-0001"
    # 哪些实验支持某个结论（§8）
    supporters = ledger.experiments_supporting(finding["id"])
    assert [e["id"] for e in supporters] == [exp_id]
    basis = ledger.goal_basis_evidence(goal["id"])
    assert basis["requires_basis"] is True
    assert {item["id"] for item in basis["candidates"]} == {
        exp_id, finding["id"],
    }
    # 哪些假设尚未验证（§8）
    assert [h["id"] for h in ledger.unverified_hypotheses()] == [hyp["id"]]
    ledger.transition(hyp["id"], "supported", reason="F-0001 支持",
                      actor=ACTOR, inputs=[finding["id"]])
    assert ledger.unverified_hypotheses() == []

    # 失败实验与失败原因（§8）
    exp2 = ledger.allocator.allocate("EXP-")
    ledger.register_experiment(
        _exp_definition(exp2, goal["id"], hyp["id"], view["id"]),
        actor=ACTOR)
    ledger.transition(exp2, "running", reason="启动", actor=ACTOR)
    ledger.transition(exp2, "failed", reason="子进程非零退出",
                      actor=ACTOR, error="TFX-901: 环境指纹不符")
    failed = ledger.failed_experiments(goal_id=goal["id"])
    assert failed[0]["experiment_id"] == exp2
    assert "TFX-901" in failed[0]["failure_reason"]


def test_goal_progress_and_models(ledger):
    goal, view, hyp = _seed_entities(ledger)
    exp_id = ledger.allocator.allocate("EXP-")
    ledger.register_experiment(
        _exp_definition(exp_id, goal["id"], hyp["id"], view["id"]),
        actor=ACTOR)
    model = ledger.register_model("ridge-baseline", exp_id, actor=ACTOR,
                                  metrics={"RMSE": 12.3})
    assert model["id"] == "M-0001"
    progress = ledger.goal_progress(goal["id"])
    assert progress["experiments"]["total"] == 1
    assert progress["experiments"]["by_status"] == {"created": 1}
    assert progress["hypotheses"]["unverified"] == 1
    assert progress["models"]["candidates"] == 1
    decision = ledger.create_decision(
        "验收阈值", "mape_max 0.05 → 0.10", actor=ACTOR,
        rationale="线性基线实测 8.9%（DD-13：阈值修订留痕）",
        references=[exp_id])
    assert decision["id"] == "D-0001"


def test_hypothesis_requires_evidence_after_first(ledger):
    goal, _, _ = _seed_entities(ledger)
    with pytest.raises(ValueError, match="basis"):
        ledger.create_hypothesis(goal["id"], "无理由的第二个假设",
                                 actor=ACTOR)


def test_new_goal_does_not_require_basis(ledger):
    goal = ledger.create_goal("fresh goal", actor=ACTOR)
    assert ledger.goal_basis_evidence(goal["id"]) == {
        "requires_basis": False,
        "candidates": [],
    }


def test_register_experiment_requires_existing_refs(ledger):
    goal, view, hyp = _seed_entities(ledger)
    exp_id = ledger.allocator.allocate("EXP-")
    with pytest.raises(ValueError):
        ledger.register_experiment(
            _exp_definition(exp_id, goal["id"], "H-9999", view["id"]),
            actor=ACTOR)


def test_finding_written_as_markdown(ledger, tmp_path):
    goal, view, hyp = _seed_entities(ledger)
    finding = ledger.create_finding("传感器漂移区间不应训练", actor=ACTOR)
    text = (tmp_path / "research" / "findings" / "F-0001.md").read_text(
        encoding="utf-8")
    assert text.startswith("---\n") and "传感器漂移" in text
