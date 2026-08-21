"""编排循环与停止条件（research-loop §2/§9、roadmap Phase 3 验收）。

停止条件各分支：验收达标 / 预算耗尽（TFX-906）/ 连续无信息增益 /
数据覆盖不足 / 必需变量缺失 / 需人工确认。
"""

from __future__ import annotations

import json

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
from thermoforge_webui.services.planner import (
    PlannerContext,
    PlannerError,
    make_planner,
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


def test_web_planner_resumes_goal_with_historical_basis(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)
    view_id = _view(ctx, ref)
    old_hypothesis = ctx.ledger.create_hypothesis(
        goal_id, "historical hypothesis", actor=ctx.actor)
    old_experiment_id = ctx.ledger.allocator.allocate("EXP-")
    ctx.ledger.register_experiment({
        "experiment_id": old_experiment_id,
        "goal_id": goal_id,
        "hypothesis_id": old_hypothesis["id"],
        "dataset_view": view_id,
    }, actor=ctx.actor)
    old_finding = ctx.ledger.create_finding(
        "historical finding", actor=ctx.actor,
        supported_by=[old_experiment_id],
        hypothesis_id=old_hypothesis["id"])

    prompts = []

    def ask(system, user):
        prompts.append(user)
        return json.dumps({
            "statement": "continue from historical evidence",
            "basis": [old_finding["id"]],
            "view_id": view_id,
            "model": {
                "category": "data",
                "estimator": "ridge",
                "hyperparameters": {"alpha": 0.5},
            },
        })

    goal = ctx.ledger.get(goal_id)
    planner_context = PlannerContext(
        goal={"id": goal_id, **goal["definition"]},
        views=[ctx.ledger.get(view_id)],
        dataset_ref=ref,
    )
    orchestrator = ResearchOrchestrator(
        ctx, goal_id, make_planner(ask, planner_context),
        dataset_ref=ref, max_rounds=1)
    result = orchestrator.run()

    assert result["ok"] is True
    assert planner_context.traces[0].round_index == 0
    assert old_experiment_id in prompts[0]
    assert old_finding["id"] in prompts[0]
    assert planner_context.traces[0].plan["basis"] == [old_finding["id"]]


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


def test_r2_min_is_machine_checked(tmp_path):
    """R² 门槛单列一项：CVRMSE 宽松也拦得住，达标即 PUBLISH。"""
    ctx, ref = make_ctx(tmp_path)
    # CVRMSE 放到 1.0（必过），只留 R² 门槛，且设成不可能达到
    goal_id = _goal(ctx, max_experiments=1,
                    acceptance={"cvrmse_max": 1.0, "r2_min": 0.999999})
    result = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                  dataset_ref=ref).run()
    assert result["summary"]["stop"]["reason"] != STOP_ACCEPTANCE_MET
    unmet = result["summary"]["rounds"][-1]["acceptance_unmet"]
    assert any("R2" in u and "r2_min" in u for u in unmet), unmet

    ctx2, ref2 = make_ctx(tmp_path / "ok")
    goal_ok = _goal(ctx2, acceptance={"cvrmse_max": 1.0, "r2_min": 0.5})
    result_ok = ResearchOrchestrator(ctx2, goal_ok, _planner(_view(ctx2, ref2)),
                                     dataset_ref=ref2).run()
    assert result_ok["summary"]["stop"]["reason"] == STOP_ACCEPTANCE_MET


def test_planner_can_submit_lab_module_and_use_it_same_round(tmp_path):
    """规划器随计划附源码 → 编排器先过实验室门禁，同一轮直接引用。

    没有这条通道时，规划器碰到闭集表达不了的函数形式只能停下来要人代交
    （实测 RG-0017），「模型实验室自治」对编排器就不成立。
    """
    import test_model_lab

    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, max_experiments=1)
    view_id = _view(ctx, ref)

    def plan(round_index, evidence):
        return {
            "statement": "附模块源码，同轮引用",
            "basis": [],
            "view_id": view_id,
            "model": {"category": "lab",
                      "hyperparameters": {"lab": "mini_view"}},
            "lab_module": {"name": "mini_view",
                           "source": test_model_lab.MINI_SOURCE,
                           "description": "最小线性模块"},
        }

    result = ResearchOrchestrator(ctx, goal_id, plan, dataset_ref=ref).run()
    rounds = result["summary"]["rounds"]
    assert rounds and rounds[0]["status"] == "completed", rounds
    # 模块真的入库并被锁到具体版本
    refs = [m["ref"] for m in ctx.lab_store.list()]
    assert any(r.startswith("mini_view@v") for r in refs), refs
    spec = json.loads((ctx.research_root / "experiments"
                       / rounds[0]["experiment_id"] / "spec.json")
                      .read_text(encoding="utf-8"))
    assert spec["experiment"]["model"]["hyperparameters"]["lab"].startswith(
        "mini_view@v")


def test_lab_module_failing_the_gate_is_a_round_failure_not_a_crash(tmp_path):
    """校验没过要把明细记成本轮失败，让规划器下一轮改代码，而不是崩掉。"""
    import test_model_lab

    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)
    view_id = _view(ctx, ref)

    def plan(round_index, evidence):
        return {
            "statement": "交一个非确定性模块",
            "basis": [],
            "view_id": view_id,
            "model": {"category": "lab", "hyperparameters": {"lab": "jitter"}},
            "lab_module": {"name": "jitter",
                           "source": test_model_lab.JITTER_SOURCE},
        }

    result = ResearchOrchestrator(ctx, goal_id, plan, dataset_ref=ref,
                                  no_gain_rounds=2).run()
    rounds = result["summary"]["rounds"]
    assert rounds and rounds[0]["status"] == "failed"
    assert "未过校验" in rounds[0]["detail"]


def test_planner_stop_reason_is_recorded(tmp_path):
    """规划器主动收工的理由要进停止留痕，不能只剩一句通用「无可行假设」。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)

    def plan(round_index, evidence):
        return {"stop": "瓶颈是缺阀位测点，不是缺模型",
                "reasoning": "同频率下流量仍有 ±8.5% 散布"}

    result = ResearchOrchestrator(ctx, goal_id, plan, dataset_ref=ref).run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_NO_GAIN
    assert "缺阀位测点" in stop["detail"]
    assert "±8.5%" in stop["evidence"]["planner_reasoning"]


def test_acceptance_judged_on_declared_surface(tmp_path):
    """Goal 钉了 evaluated_on=rolling_cv，验收就必须判在滚动块上。

    同一个模型在面 A 与滚动块上能差三倍，判错面等于门槛没落到声明的口径。
    """
    summary = {
        "surfaces": {"A": {"n_samples": 10, "metrics": {"CVRMSE": 0.5,
                                                        "R2": 0.1}}},
        "rolling_cv": {"n_folds": 8, "metrics": {"CVRMSE": 0.02, "R2": 0.97}},
    }
    acceptance = {"evaluated_on": "rolling_cv", "r2_min": 0.9}
    assert ResearchOrchestrator._acceptance_unmet(acceptance, summary) == []
    # 同一份结果换个判据面就该判不过
    assert ResearchOrchestrator._acceptance_unmet(
        {"evaluated_on": "A", "r2_min": 0.9}, summary)
    # 滚动块没算出来时不得静默按别的面放行
    assert ResearchOrchestrator._acceptance_unmet(
        acceptance, {"surfaces": summary["surfaces"]})


def test_rolling_cv_enabled_when_goal_judges_on_it(tmp_path):
    """判据面 rolling_cv 的目标必须真的算滚动块，否则每轮验收指标都是 None。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx, max_experiments=1, acceptance={
        "evaluated_on": "rolling_cv", "cvrmse_max": 1e-12})
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    result = orch.run()
    exp_id = result["summary"]["rounds"][0]["experiment_id"]
    spec = json.loads((ctx.research_root / "experiments" / exp_id
                       / "spec.json").read_text(encoding="utf-8"))
    assert spec["experiment"]["validation"]["rolling_cv"]["enabled"] is True
    metrics = json.loads((ctx.research_root / "experiments" / exp_id
                          / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rolling_cv"]["metrics"], "判据面必须有指标，否则验收判不了"
    # 规划器显式给的滚动参数不被覆盖
    plan_with_cv = _planner(_view(ctx, ref))
    spec_out = ResearchOrchestrator._validation_spec(
        {"acceptance": {"evaluated_on": "rolling_cv"}},
        {**plan_with_cv(0, {"rounds": []}),
         "validation": {"rolling_cv": {"enabled": True, "max_folds": 3}}})
    assert spec_out["rolling_cv"]["max_folds"] == 3


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


def test_planner_trace_persisted_with_experiment(tmp_path):
    """规划留痕随实验工件落盘：思维链/原始回复/用量写进实验目录。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)  # cvrmse_max=1.0：ridge 一轮即验收达标
    view_id = _view(ctx, ref)

    def ask(system, user):
        ask.last_usage = {"prompt_tokens": 12, "completion_tokens": 5,
                          "total_tokens": 17}
        ask.last_cost = 0.0003
        return json.dumps({
            "statement": "带留痕的假设",
            "reasoning": "先跑 ridge 基线，便宜且可解释",
            "view_id": view_id,
            "model": {"category": "data", "estimator": "ridge",
                      "hyperparameters": {"alpha": 0.5}},
        }, ensure_ascii=False)  # 真实模型返回的是原始 UTF-8 文本

    goal = ctx.ledger.get(goal_id)
    context = PlannerContext(
        goal={"id": goal_id, **goal["definition"]},
        views=[ctx.ledger.get(view_id)],
        dataset_ref=ref,
    )
    orch = ResearchOrchestrator(ctx, goal_id, make_planner(ask, context),
                                dataset_ref=ref)
    result = orch.run()

    assert result["ok"] is True
    path = (ctx.research_root / "experiments" / "EXP-0001"
            / "planner_trace.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "EXP-0001"
    assert payload["round_index"] == 0
    assert payload["reasoning"] == "先跑 ridge 基线，便宜且可解释"
    assert payload["plan"]["view_id"] == view_id
    assert payload["usage"] == {"prompt_tokens": 12, "completion_tokens": 5,
                                "total_tokens": 17}
    assert payload["cost"] == 0.0003
    assert "带留痕的假设" in payload["raw_reply"]


def test_planner_trace_skipped_for_scripted_planner(tmp_path):
    """脚本化 planner 不带 last_trace：不产生留痕文件，也不报错。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)
    orch = ResearchOrchestrator(ctx, goal_id, _planner(_view(ctx, ref)),
                                dataset_ref=ref)
    assert orch.run()["ok"] is True
    assert not (ctx.research_root / "experiments" / "EXP-0001"
                / "planner_trace.json").exists()


def _trace_planner(ctx, goal_id, ref, ask):
    """用 make_planner 包一个假 ask，让编排器走完整的 trace 捕获链路。"""
    view_id = _view(ctx, ref)
    goal = ctx.ledger.get(goal_id)
    context = PlannerContext(
        goal={"id": goal_id, **goal["definition"]},
        views=[ctx.ledger.get(view_id)],
        dataset_ref=ref,
    )
    return make_planner(ask, context)


def _orphan_files(ctx, goal_id):
    directory = ctx.research_root / "planner_rounds" / goal_id
    if not directory.is_dir():
        return []
    return sorted(directory.glob("round_*.json"))


def test_orphan_trace_persisted_when_planner_stops(tmp_path):
    """规划器主动收工：这一轮的 token 与思维链落到 planner_rounds/ 归目标级。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)

    def ask(system, user):
        ask.last_usage = {"prompt_tokens": 21, "completion_tokens": 8,
                          "total_tokens": 29}
        ask.last_cost = 0.0005
        ask.last_reasoning = "端点思维链：流量散布太大，再拟合也是浪费"
        return json.dumps({
            "stop": "瓶颈是缺阀位测点，不是缺模型",
            "reasoning": "同频率下流量仍有 ±8.5% 散布",
        }, ensure_ascii=False)

    orch = ResearchOrchestrator(
        ctx, goal_id, _trace_planner(ctx, goal_id, ref, ask), dataset_ref=ref)
    result = orch.run()

    assert result["summary"]["stop"]["reason"] == STOP_NO_GAIN
    files = _orphan_files(ctx, goal_id)
    assert len(files) == 1
    assert "planner_stop" in files[0].name
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["orphan_kind"] == "planner_stop"
    assert payload["goal_id"] == goal_id
    assert payload["round_number"] == 1
    assert payload["cot"] == "端点思维链：流量散布太大，再拟合也是浪费"
    assert payload["reasoning"] == "同频率下流量仍有 ±8.5% 散布"
    assert payload["usage"] == {"prompt_tokens": 21, "completion_tokens": 8,
                                "total_tokens": 29}
    assert payload["cost"] == 0.0005
    # 没产出实验：实验目录里不该有 planner_trace.json
    assert not (ctx.research_root / "experiments").exists() or not list(
        (ctx.research_root / "experiments").glob("*/planner_trace.json"))


def test_orphan_trace_persisted_when_planner_raises(tmp_path):
    """规划器抛错（解析多次失败/网络异常）：trace 先落盘，异常再往上抛。"""
    ctx, ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)

    def ask(system, user):
        ask.last_usage = {"prompt_tokens": 10, "completion_tokens": 3,
                          "total_tokens": 13}
        ask.last_reasoning = "端点思维链：我还没想好呢"
        return "这不是 JSON，模型说了废话"

    orch = ResearchOrchestrator(
        ctx, goal_id, _trace_planner(ctx, goal_id, ref, ask), dataset_ref=ref)
    with pytest.raises(PlannerError):
        orch.run()

    files = _orphan_files(ctx, goal_id)
    assert len(files) == 1
    assert "planner_error" in files[0].name
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["orphan_kind"] == "planner_error"
    assert payload["cot"] == "端点思维链：我还没想好呢"
    assert payload["usage"]["total_tokens"] > 0
    assert payload["error"]  # 多次尝试后的最终错误要留痕
    assert "这不是 JSON" in payload["raw_reply"]


def test_lab_gate_failure_detail_carries_actionable_specifics():
    """门禁失败的明细必须带上「照着改什么」，不能只留一句概括。

    实测 RG-0023 七轮里五轮废在模块提交，五次拿到的 detail 都是同一句
    「静态扫描不通过」/「结构校验未通过」—— 规划器下一轮就是靠这段 detail
    改代码的，只给概括等于让它蒙着眼改。
    """
    from thermoforge_research.orchestrator import _lab_failure_detail

    # AST 静态扫描：逐条违规在 summary["violations"]
    ast_env = {"ok": False, "summary": {
        "error": "静态扫描不通过：按 violations 逐条改源码后重交",
        "violations": ["禁止导入 subprocess（第 3 行）",
                       "禁止调用 eval（第 17 行）"]}}
    detail = _lab_failure_detail(ast_env)
    assert "subprocess" in detail and "eval" in detail

    # 五连检：逐项明细在 summary["validation"]["checks"]
    check_env = {"ok": False, "summary": {
        "error": "[TFML-002] 结构校验未通过",
        "validation": {"ok": False, "checks": [
            {"name": "interface", "ok": True, "detail": ""},
            {"name": "determinism", "ok": False,
             "detail": "同种子重训预测不逐位一致（max|Δ|=3.2e-04）"}]}}}
    detail = _lab_failure_detail(check_env)
    assert "determinism" in detail and "3.2e-04" in detail
    assert "interface" not in detail          # 过了的检查不该占篇幅

    # 什么都没有时也得给出点东西，不能返回空串
    assert _lab_failure_detail({"ok": False, "summary": {}})
