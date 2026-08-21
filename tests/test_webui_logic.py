"""Web 控制台纯逻辑层测试：绘图数据准备、诊断处方、规划器校验、导出。

这些函数不碰 Streamlit、不碰网络，所以可以用合成数据直接测。图好不好看
要人看，但「残差算对没有、区间切在哪、模型编了个不存在的视图 ID 会不会
被挡住」必须能自动验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thermoforge_webui.charts import series
from thermoforge_webui.services import quality
from thermoforge_webui.services.experiments import ExperimentSummary
from thermoforge_webui.services.planner import (
    PlannerContext,
    PlannerError,
    make_planner,
    parse_plan,
    validate_plan,
)


# ---------------------------------------------------------------- 绘图数据


def _predictions(n: int = 100, surface: str = "A") -> pd.DataFrame:
    stamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "surface": [surface] * n,
        "object_id": ["chiller_01"] * n,
        "timestamp": stamps,
        "y_true": np.linspace(100.0, 200.0, n),
        "y_pred": np.linspace(100.0, 200.0, n) + 5.0,
    })


def test_prepare_predictions_computes_residual_and_filters_surface():
    frame = pd.concat([_predictions(50, "A"), _predictions(30, "validate")])
    result = series.prepare_predictions(frame, "A")
    assert result.total_points == 50
    assert result.surface == "A"
    # 残差 = 实测 − 预测，这里预测一律高 5
    assert result.frame["residual"].round(6).eq(-5.0).all()


def test_prepare_predictions_downsamples_but_keeps_total():
    frame = _predictions(1000)
    result = series.prepare_predictions(frame, "A", max_points=100)
    assert result.downsampled and result.stride == 10
    assert len(result.frame) <= 100
    # 抽稀只影响画的点，报告的总量必须是真实值
    assert result.total_points == 1000


def test_prepare_residuals_uses_full_data_not_downsampled():
    """残差分布必须用全量点统计，否则分布是假的。"""
    frame = _predictions(1000)
    frame.loc[frame.index[:500], "y_pred"] += 3.0  # 造出真实的分布宽度
    stats = series.prepare_residuals(frame, "A")
    assert stats is not None
    assert len(stats.values) == 1000
    assert stats.std > 0


def test_prepare_residuals_survives_constant_residual():
    """残差恒定（纯偏置）时直方图不能炸——一根柱子本身就是结论。"""
    stats = series.prepare_residuals(_predictions(200), "A")
    assert stats is not None
    assert stats.mean == pytest.approx(-5.0)
    assert stats.std == pytest.approx(0.0)
    assert stats.counts.sum() == 200


def test_prepare_predictions_empty_for_unknown_surface():
    result = series.prepare_predictions(_predictions(), "C")
    assert result.empty and result.total_points == 0


def test_prepare_split_reads_boundaries_and_gap():
    split = {
        "train_range": ["2025-01-01T00:00:00+00:00", "2025-03-01T00:00:00+00:00"],
        "validate_range": ["2025-03-01T00:45:00+00:00", "2025-04-01T00:00:00+00:00"],
        "test_range": ["2025-04-01T00:00:00+00:00", "2025-05-01T00:00:00+00:00"],
        "b1": "2025-03-01T00:00:00+00:00", "b2": "2025-04-01T00:00:00+00:00",
        "counts": {"train": 100, "validate": 20, "test": 20,
                   "purged_or_embargoed": 3},
        "purge_seconds": 0.0, "embargo_seconds": 2700.0,
    }
    timeline = series.prepare_split(split)
    assert timeline is not None
    assert [segment.name for segment in timeline.segments] == [
        "train", "validate", "test"]
    assert timeline.dropped == 3
    assert "45 分钟" in timeline.caption
    assert len(timeline.boundaries) == 2


def test_prepare_split_returns_none_without_ranges():
    assert series.prepare_split({"counts": {}}) is None


def test_split_timeline_figure_uses_date_axis():
    """回归：x 直接传 Timedelta 会让 plotly 把整条轴渲染成时长串
    （P255DT12H0M0S）；必须换算成 epoch 毫秒并声明 type="date"。"""
    from thermoforge_webui.charts import interactive

    split = {
        "train_range": ["2025-01-01T00:00:00+00:00", "2025-03-01T00:00:00+00:00"],
        "validate_range": ["2025-03-01T00:45:00+00:00", "2025-04-01T00:00:00+00:00"],
        "test_range": ["2025-04-01T00:00:00+00:00", "2025-05-01T00:00:00+00:00"],
        "b1": "2025-03-01T00:00:00+00:00", "b2": "2025-04-01T00:00:00+00:00",
        "counts": {"train": 100, "validate": 20, "test": 20},
        "purge_seconds": 0.0, "embargo_seconds": 2700.0,
    }
    fig = interactive.split_timeline_figure(series.prepare_split(split))
    payload = fig.to_plotly_json()
    assert payload["layout"]["xaxis"]["type"] == "date"
    train = payload["data"][0]
    assert train["base"] == [pd.Timestamp(split["train_range"][0]).value / 1e6]
    expected_ms = (pd.Timestamp(split["train_range"][1])
                   - pd.Timestamp(split["train_range"][0])).total_seconds() * 1000
    assert train["x"] == [expected_ms]
    shapes = payload["layout"]["shapes"]
    assert [s["x0"] for s in shapes] == [
        pd.Timestamp(split[key]).value / 1e6 for key in ("b1", "b2")]
    # 三段颜色必须两两不同（SURFACE_COLORS 漏了 "test" 会撞成训练灰）
    colors = [trace["marker"]["color"] for trace in payload["data"]]
    assert len(set(colors)) == 3


def _summary(experiment_id: str, cvrmse: float | None,
             r2: float | None = None, started: str = "2025-01-01") -> ExperimentSummary:
    metrics: dict[str, float | None] = {}
    if cvrmse is not None:
        metrics["CVRMSE"] = cvrmse
    if r2 is not None:
        metrics["R2"] = r2
    return ExperimentSummary(
        experiment_id=experiment_id, status="completed", goal_id="RG-0001",
        hypothesis_id="H-0001", dataset_view="VIEW-0001", model_label="data · ridge",
        started_at=started, duration_seconds=1.0, surface="A", metrics=metrics,
        physics_rate=0.0, error_code=None, failure_reason=None)


def test_metric_bars_sort_ascending_for_lower_is_better():
    bars = series.prepare_metric_bars(
        [_summary("EXP-1", 0.5), _summary("EXP-2", 0.1),
         _summary("EXP-3", 0.3)], "CVRMSE")
    assert bars is not None
    assert bars.labels == ["EXP-2", "EXP-3", "EXP-1"]
    assert bars.lower_is_better and bars.best_index == 0


def test_metric_bars_sort_descending_for_r2():
    bars = series.prepare_metric_bars(
        [_summary("EXP-1", 0.5, r2=0.2), _summary("EXP-2", 0.1, r2=0.9)], "R2")
    assert bars is not None
    assert bars.labels == ["EXP-2", "EXP-1"]  # R² 越大越好
    assert not bars.lower_is_better


def test_metric_bars_skip_experiments_missing_the_metric():
    """缺指标的实验不画成 0——那会看起来像「表现完美」。"""
    bars = series.prepare_metric_bars(
        [_summary("EXP-1", None), _summary("EXP-2", 0.2)], "CVRMSE")
    assert bars is not None and bars.labels == ["EXP-2"]


def test_metric_bars_none_when_no_experiment_has_metric():
    assert series.prepare_metric_bars([_summary("EXP-1", None)], "CVRMSE") is None


def test_progress_best_so_far_is_monotone():
    progress = series.prepare_progress([
        _summary("EXP-1", 0.5, started="2025-01-01"),
        _summary("EXP-2", 0.8, started="2025-01-02"),
        _summary("EXP-3", 0.3, started="2025-01-03"),
    ], "CVRMSE")
    assert progress is not None
    assert progress.values == [0.5, 0.8, 0.3]
    assert progress.best_so_far == [0.5, 0.5, 0.3]  # 越小越好，单调不增


# ---------------------------------------------------------------- 诊断处方


def test_modelability_blocker_becomes_blocker_finding():
    report = {"verdict": "FAIL", "target": "power",
              "candidate_inputs": ["load"],
              "checks": [{"name": "derivation_chain", "level": "blocker",
                          "passed": False, "summary": "load 是 power 的派生量",
                          "evidence": {"chain": ["load", "power"]}}]}
    result = quality.build_report(report, None)
    assert not result.passed
    assert len(result.blockers) == 1
    finding = result.blockers[0]
    assert finding.title == "派生链检查"
    # 阻断项必须给出「这不是预处理能解决的」，避免误导人去清洗
    assert "不是预处理能解决的" in finding.prescriptions[0].text
    assert not finding.prescriptions[0].actionable


def test_time_axis_finding_offers_registered_rule():
    profile = {"interval_stats": {"off_resolution_fraction": 0.05,
                                  "median_s": 900, "max_s": 3600},
               "variables": []}
    result = quality.build_report(None, profile)
    finding = next(f for f in result.findings if f.key == "time_axis")
    prescription = finding.prescriptions[0]
    assert prescription.rule_type == "repair_time_axis"
    assert prescription.actionable  # 规则库里注册过，可一键生成提案


def test_high_missing_rate_finding_refuses_imputation():
    profile = {"variables": [
        {"variable_id": "PLANT.chw_flow", "missing_rate": 0.35},
        {"variable_id": "PLANT.power", "missing_rate": 0.01}]}
    result = quality.build_report(None, profile)
    finding = next(f for f in result.findings if f.key == "missing")
    assert "chw_flow" in finding.detail
    # 填充值会被后续当成实测，破坏 source_kind 的实测/派生区分
    assert "不要在这里做填充" in finding.prescriptions[0].text


def test_findings_sorted_blockers_first():
    report = {"verdict": "FAIL", "checks": [
        {"name": "device_diversity", "level": "info", "passed": True,
         "summary": "ok"},
        {"name": "same_origin", "level": "blocker", "passed": False,
         "summary": "相关性 0.999"}]}
    result = quality.build_report(report, None)
    assert result.findings[0].severity == "blocker"


def test_clean_report_headline_says_ok():
    result = quality.build_report({"verdict": "PASS", "checks": []}, {})
    assert result.passed and "可以拿来建模" in result.headline


# ---------------------------------------------------------------- 规划器


def _context() -> PlannerContext:
    return PlannerContext(
        goal={"id": "RG-0001", "target": "power",
              "candidate_inputs": ["chw_flow"]},
        views=[{"id": "VIEW-0001",
                "definition": {"dataset": "D@rev_0001", "target": "power",
                               "features": ["chw_flow"]}}],
        dataset_ref="D@rev_0001")


def test_parse_plan_tolerates_code_fence_and_chatter():
    text = "好的，这是计划：\n```json\n{\"statement\": \"试试物理模型\"}\n```"
    assert parse_plan(text)["statement"] == "试试物理模型"


def test_parse_plan_raises_without_json():
    with pytest.raises(PlannerError):
        parse_plan("我觉得应该先看看数据")


def test_validate_plan_rejects_invented_view_id():
    plan = {"statement": "试试", "view_id": "VIEW-9999",
            "model": {"category": "data", "estimator": "ridge"}}
    result, error = validate_plan(plan, _context(), 0)
    assert result is None and "VIEW-9999" in error


def test_validate_plan_rejects_unimplemented_estimator():
    plan = {"statement": "试试", "view_id": "VIEW-0001",
            "model": {"category": "data", "estimator": "lightgbm"}}
    result, error = validate_plan(plan, _context(), 0)
    assert result is None and "lightgbm" in error


def test_validate_plan_rejects_repeat_of_an_earlier_round():
    """确定性实验重跑必然得到逐位一样的指标，重复计划要当场拦住。"""
    from thermoforge_webui.services.planner import PlannerTrace

    context = _context()
    trace = PlannerTrace(round_index=0, prompt_summary="")
    trace.plan = {"statement": "第一轮", "view_id": "VIEW-0001",
                  "model": {"category": "data", "estimator": "ridge",
                            "hyperparameters": {"alpha": 1.0}}}
    context.traces.append(trace)

    # 超参键序不同但内容相同 → 仍算重复
    repeat = {"statement": "再试一次", "basis": ["EXP-0001"],
              "view_id": "VIEW-0001",
              "model": {"category": "data", "estimator": "ridge",
                        "hyperparameters": {"alpha": 1.0}}}
    result, error = validate_plan(repeat, context, 1)
    assert result is None and "第 0 轮等价" in error

    # 跨进程补进来的历史实验用 round_index<0 标记，报错文案不能出现「第 -1 轮」
    history = _context()
    old = PlannerTrace(round_index=-1, prompt_summary="历史")
    old.plan = dict(trace.plan)
    history.traces.append(old)
    result, error = validate_plan(repeat, history, 1)
    assert result is None and "此前跑过" in error and "-1" not in error

    # 换了超参量级就不是重复
    changed = {**repeat, "model": {"category": "data", "estimator": "ridge",
                                   "hyperparameters": {"alpha": 5.0}}}
    assert validate_plan(changed, context, 1)[0] is not None


def test_duplicate_detection_normalizes_inputs_and_lab_ref():
    """写法不同但跑起来一样的超参，必须算同一个计划。

    实测 RG-0021 十一轮里六轮是重复：`inputs` 换了分隔符或条目顺序、
    `lab` 有时写裸名有时写 name@vN，逐字符比对全都漏掉了。
    """
    from thermoforge_webui.services.planner import PlannerTrace

    context = _context()
    context.lab_modules = [{"ref": "m@v1", "runnable": True}]
    trace = PlannerTrace(round_index=0, prompt_summary="")
    trace.plan = {"view_id": "VIEW-0001", "model": {
        "category": "lab",
        "hyperparameters": {"lab": "m@v1", "inputs": "a=a;b=b;c=c"}}}
    context.traces.append(trace)

    def plan_with(hyperparameters):
        return {"statement": "再来", "basis": ["EXP-0001"],
                "view_id": "VIEW-0001",
                "model": {"category": "lab",
                          "hyperparameters": hyperparameters}}

    # 逗号分隔、条目乱序、lab 写裸名 —— 三种写法都是同一个实验
    for hp in ({"lab": "m@v1", "inputs": "a=a,b=b,c=c"},
               {"lab": "m@v1", "inputs": "c=c;a=a;b=b"},
               {"lab": "m", "inputs": "b=b;c=c;a=a"}):
        result, error = validate_plan(plan_with(hp), context, 1)
        assert result is None and "等价" in (error or ""), hp

    # 真的多了一列就不是重复
    result, _ = validate_plan(
        plan_with({"lab": "m@v1", "inputs": "a=a;b=b;c=c;d=d"}), context, 1)
    assert result is not None


def test_validate_plan_requires_basis_after_first_round():
    plan = {"statement": "试试", "view_id": "VIEW-0001", "basis": [],
            "model": {"category": "data", "estimator": "ridge"}}
    assert validate_plan(plan, _context(), 0)[0] is not None  # 首轮可以没有
    result, error = validate_plan(plan, _context(), 1)
    assert result is None and "basis" in error


def test_validate_plan_requires_real_basis_when_resuming_goal():
    plan = {"statement": "继续优化", "view_id": "VIEW-0001", "basis": [],
            "model": {"category": "data", "estimator": "ridge"}}
    evidence = {
        "basis_required": True,
        "basis_candidates": [{"id": "EXP-0001", "kind": "experiment"}],
    }
    result, error = validate_plan(plan, _context(), 0, evidence=evidence)
    assert result is None and "EXP-0001" in error

    plan["basis"] = ["EXP-9999"]
    result, error = validate_plan(plan, _context(), 0, evidence=evidence)
    assert result is None and "不可用" in error

    plan["basis"] = ["EXP-0001"]
    result, error = validate_plan(plan, _context(), 0, evidence=evidence)
    assert error is None and result["basis"] == ["EXP-0001"]


def test_validate_plan_rejects_basis_when_catalog_is_empty():
    plan = {"statement": "首轮", "view_id": "VIEW-0001",
            "basis": ["EXP-9999"],
            "model": {"category": "data", "estimator": "ridge"}}
    evidence = {"basis_required": False, "basis_candidates": []}
    result, error = validate_plan(plan, _context(), 0, evidence=evidence)
    assert result is None and "EXP-9999" in error


def test_validate_plan_accepts_hybrid_with_physics_and_residual():
    plan = {"statement": "物理主干 + 残差", "view_id": "VIEW-0001",
            "basis": ["EXP-0001"],
            "model": {"category": "hybrid", "physics": "cooling_balance_v2",
                      "residual": "xgboost"}}
    result, error = validate_plan(plan, _context(), 1)
    assert error is None and result["model"]["category"] == "hybrid"


def test_validate_plan_keeps_the_stop_reason():
    """规划器主动收工时，它给的理由必须带出去——那常常就是结论本身。"""
    result, error = validate_plan(
        {"stop": "瓶颈是缺阀位测点，不是缺模型", "reasoning": "残差随阻力走"},
        _context(), 2)
    assert error is None
    assert result["stop"] == "瓶颈是缺阀位测点，不是缺模型"
    assert result["reasoning"] == "残差随阻力走"


def test_planner_retries_once_with_the_error_fed_back():
    replies = [
        '{"statement": "试试", "view_id": "VIEW-BAD", '
        '"model": {"category": "data", "estimator": "ridge"}}',
        '{"statement": "改用真实视图", "view_id": "VIEW-0001", '
        '"model": {"category": "data", "estimator": "ridge"}}',
    ]
    prompts: list[str] = []

    def ask(system: str, user: str) -> str:
        prompts.append(user)
        return replies[len(prompts) - 1]

    context = _context()
    plan = make_planner(ask, context)(0, {})
    assert plan is not None and plan["view_id"] == "VIEW-0001"
    assert "VIEW-BAD" in prompts[1]  # 第二次把错误原样喂回去了
    assert context.traces[-1].attempts == 2


def test_planner_gives_up_after_max_attempts():
    def ask(system: str, user: str) -> str:
        return '{"statement": "x", "view_id": "VIEW-BAD", ' \
               '"model": {"category": "data", "estimator": "ridge"}}'

    with pytest.raises(PlannerError):
        make_planner(ask, _context())(0, {})


def test_planner_accumulates_usage_across_attempts():
    """规划器经 ask 的函数属性读用量；修复重试是真实调用，要累计。"""
    replies = [
        '{"statement": "试试", "view_id": "VIEW-BAD", '
        '"model": {"category": "data", "estimator": "ridge"}}',
        '{"statement": "改用真实视图", "view_id": "VIEW-0001", '
        '"model": {"category": "data", "estimator": "ridge"}}',
    ]
    calls: list[str] = []

    def ask(system: str, user: str) -> str:
        calls.append(user)
        ask.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}
        ask.last_cost = 0.001
        return replies[len(calls) - 1]

    context = _context()
    plan = make_planner(ask, context)(0, {})
    assert plan is not None
    trace = context.traces[-1]
    assert trace.attempts == 2
    assert trace.usage == {"prompt_tokens": 200, "completion_tokens": 40}
    assert trace.cost == pytest.approx(0.002)


def test_planner_without_usage_attributes_stays_none():
    """测试桩 ask 不带 last_usage/last_cost 时，trace 用量为 None 而不是报错。"""
    def ask(system: str, user: str) -> str:
        return '{"statement": "x", "view_id": "VIEW-0001", ' \
               '"model": {"category": "data", "estimator": "ridge"}}'

    context = _context()
    plan = make_planner(ask, context)(0, {})
    assert plan is not None
    assert context.traces[-1].usage is None
    assert context.traces[-1].cost is None


# ---------------------------------------------------------------- 用量绘图数据


def test_prepare_usage_bars_and_cumulative_cost():
    entries = [
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20},
         "cost": 0.001},
        {"usage": {"prompt_tokens": 50, "completion_tokens": 10},
         "cost": 0.0005},
        {"usage": None, "cost": None},  # 端点没给用量的调用也占位
    ]
    bars = series.prepare_usage(entries, ["Q1·1", "Q1·2", "Q1·3"])
    assert bars.labels == ["Q1·1", "Q1·2", "Q1·3"]
    assert bars.prompt_tokens == [100, 50, 0]
    assert bars.completion_tokens == [20, 10, 0]
    assert bars.cumulative_cost == pytest.approx([0.001, 0.0015, 0.0015])
    assert bars.total_prompt == 150 and bars.total_completion == 30
    assert bars.total_cost == pytest.approx(0.0015)
    # 全部没有 cost（未配置单价）→ 不画金额线
    no_cost = series.prepare_usage(
        [{"usage": {"prompt_tokens": 1, "completion_tokens": 1}}])
    assert no_cost.cumulative_cost is None and no_cost.total_cost is None


def _lab_context() -> PlannerContext:
    """三张视图共享一个数据集，特征集逐张变宽 —— RG-0028 的真实形状。"""
    def view(vid, features):
        return {"id": vid, "definition": {"dataset": "D@rev_0001",
                                          "target": "power",
                                          "features": features}}
    context = PlannerContext(
        goal={"id": "RG-0001", "target": "power",
              "candidate_inputs": ["frequency", "tower_freq_mean",
                                   "ambient_t"]},
        views=[view("VIEW-0001", ["frequency"]),
               view("VIEW-0002", ["frequency", "tower_freq_mean"]),
               view("VIEW-0003", ["frequency", "tower_freq_mean",
                                  "ambient_t"])],
        dataset_ref="D@rev_0001")
    context.lab_modules = [
        {"ref": "freq_only@v1", "name": "freq_only", "version": 1,
         "runnable": True, "input_roles": ["frequency"]},
        {"ref": "with_ambient@v1", "name": "with_ambient", "version": 1,
         "runnable": True, "input_roles": ["frequency", "ambient_t"]},
        {"ref": "typo@v1", "name": "typo", "version": 1, "runnable": True,
         "input_roles": ["fan_freq_mean"]},
    ]
    return context


def _lab_plan(view_id: str, lab: str, inputs: str | None = None) -> dict:
    hyperparameters = {"lab": lab}
    if inputs:
        hyperparameters["inputs"] = inputs
    return {"statement": "试试", "basis": ["EXP-0001"], "view_id": view_id,
            "model": {"category": "lab", "hyperparameters": hyperparameters}}


def test_swapping_view_is_not_a_new_experiment_for_a_lab_module():
    """lab 模块只读 INPUT_ROLES 声明的列，换视图跑出来逐位相同。

    实测 RG-0028：`ct_fan_tower_shared_b`（只声明 frequency）在三张视图上
    跑了四遍，CVRMSE 全是 0.0645，而查重因为 view_id 不同全部放行 —— 七轮
    里三轮就这么烧掉了。
    """
    from thermoforge_webui.services.planner import PlannerTrace

    context = _lab_context()
    trace = PlannerTrace(round_index=0, prompt_summary="")
    trace.plan = _lab_plan("VIEW-0001", "freq_only@v1")
    context.traces.append(trace)

    for view_id in ("VIEW-0001", "VIEW-0002", "VIEW-0003"):
        result, error = validate_plan(
            _lab_plan(view_id, "freq_only@v1"), context, 1)
        assert result is None, f"{view_id} 应判为重复"
        assert "等价" in error

    # 换成真读了新列的模块，才算换实验
    result, _ = validate_plan(
        _lab_plan("VIEW-0003", "with_ambient@v1"), context, 1)
    assert result is not None


def test_lab_module_declaring_a_column_no_view_has_is_rejected():
    """五连检拿模块自己声明的列名造合成数据，列名编错了它照样全绿。

    实测 RG-0028 的 EXP-0153/0155/0156 三轮全崩在
    `KeyError: 'fan_freq_mean'`（真实列名是 `tower_freq_mean`）——这道检查
    要在实验之前拦下来，而不是等子进程炸。
    """
    context = _lab_context()
    result, error = validate_plan(_lab_plan("VIEW-0003", "typo@v1"),
                                  context, 1)
    assert result is None
    assert "fan_freq_mean" in error and "tower_freq_mean" in error

    # 用 inputs 把角色映射到真实列名，就该放行
    result, _ = validate_plan(
        _lab_plan("VIEW-0003", "typo@v1", "fan_freq_mean=tower_freq_mean"),
        context, 1)
    assert result is not None


def test_lab_roles_are_read_from_a_same_round_submission():
    """同轮提交 + 立刻引用的模块还不在清单里，得从源码读 INPUT_ROLES。"""
    context = _lab_context()
    plan = _lab_plan("VIEW-0001", "brand_new@v1")
    plan["lab_module"] = {
        "name": "brand_new",
        "source": "INPUT_ROLES = ['ambient_t']\nMODEL_FORMAT = 'x'\n",
    }
    result, error = validate_plan(plan, context, 1)
    assert result is None and "ambient_t" in error   # VIEW-0001 没这一列
