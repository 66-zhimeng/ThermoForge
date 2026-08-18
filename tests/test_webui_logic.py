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


def test_validate_plan_stop_returns_none_without_error():
    result, error = validate_plan({"stop": "没有新想法了"}, _context(), 2)
    assert result is None and error is None


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
