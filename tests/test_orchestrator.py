"""编排循环与停止条件（research-loop §2/§9、roadmap Phase 3 验收）。

停止条件各分支：验收达标 / 预算耗尽（TFX-906）/ 连续无信息增益 /
数据覆盖不足 / 必需变量缺失 / 需人工确认。
"""

from __future__ import annotations

import pytest

from thermoforge_research.orchestrator import (
    STOP_ACCEPTANCE_MET,
    STOP_BUDGET_EXHAUSTED,
    STOP_HUMAN_CONFIRMATION,
    STOP_INSUFFICIENT_COVERAGE,
    STOP_MISSING_VARIABLES,
    STOP_NO_GAIN,
    ResearchOrchestrator,
)
from thermoforge_research.tools import (
    tf_dataset_materialize,
    tf_goal_create,
)

from phase2_helpers import FEATURES, TARGET, view_definition
from phase34_helpers import make_ctx


def _goal(ctx, **overrides):
    doc = {
        "name": "冷水机输入功率模型",
        "object_model": "chiller.v1",
        "purpose": "optimization",
        "target": TARGET,
        "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 1.0},  # 默认宽松（ridge 近精确）
    }
    doc.update(overrides)
    env = tf_goal_create(ctx, doc)
    assert env["ok"], env["summary"]
    return env["id"]


def _view(ctx, ref):
    env = tf_dataset_materialize(ctx, view_definition(ref))
    assert env["ok"]
    return env["id"]


def _planner(view_id, model=None):
    """脚本化 planner：按轮次给出假设+实验计划，basis 引用上轮发现。"""
    def plan(round_index, evidence):
        basis = []
        if evidence["rounds"]:
            last = evidence["rounds"][-1]
            if last.get("finding_id"):
                basis = [last["finding_id"]]
        return {
            "statement": f"第 {round_index} 轮：ridge 基线假设",
            "basis": basis,
            "view_id": view_id,
            "model": model or {"category": "data", "estimator": "ridge",
                               "hyperparameters": {"alpha": 0.5}},
        }
    return plan


def test_stop_acceptance_met(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert result["ok"] is True and result["status"] == "PUBLISH"
    assert stop["reason"] == STOP_ACCEPTANCE_MET
    assert stop["evidence"]["candidate_experiment"] == "EXP-0001"
    assert ctx.ledger.get(goal_id)["status"] == "PUBLISH"
    # 状态机留痕（§2）：假设生成 → 设计 → 执行 → 分析 → 评审 → PUBLISH
    states = [t["to"] for t in ctx.ledger.transitions_of(goal_id)]
    for expected in ("HYPOTHESIS_GENERATION", "EXPERIMENT_DESIGN",
                     "EXPERIMENT_RUNNING", "RESULT_ANALYSIS", "MODEL_REVIEW",
                     "PUBLISH"):
        assert expected in states


def test_stop_budget_exhausted_tfx906(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, max_experiments=1,
                    acceptance={"cvrmse_max": 1e-12})  # 不可能达标
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_BUDGET_EXHAUSTED
    assert stop["diagnostic"]["code"] == "TFX-906"
    assert stop["diagnostic"]["level"] == "WARN"
    assert len(result["summary"]["rounds"]) == 1  # 只跑了预算内的一轮
    assert ctx.ledger.get(goal_id)["status"] == "STOPPED"


def test_stop_no_information_gain(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, acceptance={"cvrmse_max": 1e-12})
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref, no_gain_rounds=2)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_NO_GAIN
    # 确定性模型：首轮建立 best，连续两轮无改善后停止
    assert len(result["summary"]["rounds"]) == 3
    # 后续假设必须引用证据（§3）：第二轮起的 basis 非空
    rounds = result["summary"]["rounds"]
    assert all(r["status"] == "completed" for r in rounds)


def test_stop_missing_required_variables(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, candidate_inputs=[*FEATURES, "nonexistent_prop"])
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_MISSING_VARIABLES
    assert "nonexistent_prop" in stop["evidence"]["missing"]
    assert result["summary"]["rounds"] == []  # 预检即停，无实验


def test_stop_insufficient_data_coverage(tmp_path):
    ctx, ref = make_ctx(tmp_path, n_steps=50)  # 50 行 / ~0.5 天
    goal_id = _goal(ctx)
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)  # 默认下限 100 行 / 1 天
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_INSUFFICIENT_COVERAGE
    assert stop["evidence"]["rows"] == 50
    assert result["summary"]["rounds"] == []


def test_stop_human_confirmation_required(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, approval_required=["publish"])
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_HUMAN_CONFIRMATION
    assert stop["evidence"]["candidate_experiment"] == "EXP-0001"
    assert ctx.ledger.get(goal_id)["status"] == "STOPPED"


def test_planner_exhaustion_is_no_gain_stop(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, acceptance={"cvrmse_max": 1e-12})

    def empty_planner(round_index, evidence):
        return None  # 无可行假设

    orch = ResearchOrchestrator(ctx, goal_id, empty_planner, dataset_ref=ref)
    result = orch.run()
    assert result["summary"]["stop"]["reason"] == STOP_NO_GAIN
    assert "无可行假设" in result["summary"]["stop"]["detail"]
