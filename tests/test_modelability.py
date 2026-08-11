"""可建模性报告（gap-analysis G3、data-survey F1/F3/F6）。

五类检查各自的正/负例、blocker/warning/info 分级、FAIL 门禁
（orchestrator 不进入建模）、工具信封约定。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thermoforge_data.importer import default_registry
from thermoforge_research.modelability import (
    ModelabilityConfig,
    build_modelability_report,
    check_derivation_chain,
    check_device_diversity,
    check_operating_coverage,
    check_physical_plausibility,
    check_same_origin,
)
from thermoforge_research.orchestrator import (
    STOP_MODELABILITY_FAILED,
    ResearchOrchestrator,
)
from thermoforge_research.tools import (
    tf_dataset_materialize,
    tf_dataset_modelability,
    tf_goal_create,
)

from phase2_helpers import FEATURES, TARGET, view_definition
from phase34_helpers import make_ctx

CFG = ModelabilityConfig()
REG = default_registry()
N = 300
BASE = np.linspace(0.0, 10.0, N)


def _noisy(scale=0.01, seed=7):
    return np.random.default_rng(seed).normal(0.0, scale, N)


# ---------------------------------------------------------------- 1. 派生链


def test_derivation_chain_circular_target_blocker():
    """目标 load 由 current_percent 派生（F1）→ 候选含之即 blocker。"""
    out = check_derivation_chain("load", ["current_percent"],
                                 object_model="chiller.v2", registry=REG)
    assert out["level"] == "blocker" and not out["passed"]
    finding = out["evidence"]["findings"][0]
    assert finding["kind"] == "circular_target"
    assert finding["chain"] == ["load", "current_percent"]  # 完整链路证据


def test_derivation_chain_clean_passes():
    out = check_derivation_chain("power", ["current_percent"],
                                 object_model="chiller.v2", registry=REG)
    assert out["level"] == "info" and out["passed"]


def test_derivation_chain_derived_input_warns_with_evidence():
    """派生候选（与目标无交叉）→ warning + 链路证据（whitelist 做准入拦截）。"""
    out = check_derivation_chain("power", ["load"],
                                 object_model="chiller.v2", registry=REG)
    assert out["level"] == "warning" and out["passed"]
    finding = out["evidence"]["findings"][0]
    assert finding["kind"] == "derived_input"
    assert "current_percent" in finding["chain"]


# ---------------------------------------------------------------- 2. 同源


def test_same_origin_high_correlation_blocker():
    t = BASE + _noisy(0.05)
    x = 2.0 * t + _noisy(0.02)  # r ≈ 1
    out = check_same_origin({"y": t, "x": x}, "y", ["x"], config=CFG)
    assert out["level"] == "blocker" and not out["passed"]
    assert out["evidence"]["target_correlations"][0]["candidate"] == "x"


def test_same_origin_identical_to_target_blocker():
    t = BASE.copy()
    out = check_same_origin({"y": t, "x": t.copy()}, "y", ["x"], config=CFG)
    assert out["level"] == "blocker"
    assert out["evidence"]["target_correlations"][0]["reason"] == "identical"


def test_same_origin_normal_correlation_info():
    t = BASE + _noisy(0.1)
    x = np.random.default_rng(3).normal(0, 1, N)  # 无关
    out = check_same_origin({"y": t, "x": x}, "y", ["x"], config=CFG)
    assert out["level"] == "info" and out["passed"]


def test_same_origin_redundant_pair_warning():
    t = np.random.default_rng(2).normal(0, 1, N)  # 与候选无关
    dup = BASE.copy()
    out = check_same_origin({"y": t, "a": dup, "b": dup.copy()},
                            "y", ["a", "b"], config=CFG)
    assert out["level"] == "warning" and out["passed"]
    assert out["evidence"]["redundant_pairs"][0]["reason"] == "identical"


# ---------------------------------------------------------------- 3. 设备区分度


def test_device_diversity_identical_series_warning():
    temps = BASE.copy()
    per_object = {
        "CH-01": {"evap_chw_supply_temp": temps, "power": BASE + _noisy(1)},
        "CH-02": {"evap_chw_supply_temp": temps.copy(),
                  "power": BASE + _noisy(1, seed=8)},
    }
    out = check_device_diversity(per_object, config=CFG)
    assert out["level"] == "warning" and out["passed"]
    assert out["evidence"]["findings"][0]["reason"] == "identical"


def test_device_diversity_distinct_series_info():
    per_object = {
        "CH-01": {"power": BASE + _noisy(1)},
        "CH-02": {"power": BASE[::-1] + _noisy(1, seed=9)},
    }
    out = check_device_diversity(per_object, config=CFG)
    assert out["level"] == "info" and out["passed"]


def test_device_diversity_single_instance_na():
    out = check_device_diversity({"PLANT": {"total_power": BASE}}, config=CFG)
    assert out["level"] == "info" and "无需" in out["summary"]


# ---------------------------------------------------------------- 4. 物理自洽


def _hvac_series(cop=5.0, delta_t=5.0):
    flow = np.full(N, 3000.0)
    supply = np.full(N, 17.5)
    ret = supply + delta_t
    q = flow * 998.0 / 3600.0 * 4.186 * delta_t
    power = q / cop
    return {"chw_flow": flow, "chw_supply_temp": supply,
            "chw_return_temp": ret, "total_power": power}


def test_physical_plausibility_in_range_info():
    out = check_physical_plausibility(_hvac_series(cop=10.0), "total_power",
                                      config=CFG)
    assert out["level"] == "info" and out["passed"]
    cop = next(q for q in out["evidence"]["quantities"]
               if q["quantity"] == "cop_estimate")
    assert cop["p50"] == pytest.approx(10.0)
    assert "内置参数化默认" in cop["range_source"]  # 范围来源写明


def test_physical_plausibility_out_of_range_warning():
    out = check_physical_plausibility(_hvac_series(cop=200.0), "total_power",
                                      config=CFG)
    assert out["level"] == "warning" and out["passed"]  # 范围参数化 → 不阻断
    cop = next(q for q in out["evidence"]["quantities"]
               if q["quantity"] == "cop_estimate")
    assert cop["out_of_range_fraction"] > CFG.ratio_outlier_tol


def test_physical_plausibility_custom_range_override():
    cfg = ModelabilityConfig(quantity_ranges={"cop_estimate": (2.0, 8.0)})
    out = check_physical_plausibility(_hvac_series(cop=10.0), "total_power",
                                      config=cfg)
    cop = next(q for q in out["evidence"]["quantities"]
               if q["quantity"] == "cop_estimate")
    assert cop["range"] == [2.0, 8.0]
    assert cop["range_source"] == "config.quantity_ranges"
    assert out["level"] == "warning"  # F6：范围参数化，不写死 5~7


# ---------------------------------------------------------------- 5. 工况覆盖


def test_operating_coverage_blocker_when_target_mostly_missing():
    t = np.full(N, np.nan)
    t[:10] = 1.0  # 3.3% 可用
    out = check_operating_coverage({"y": t}, "y", None, config=CFG)
    assert out["level"] == "blocker" and not out["passed"]


def test_operating_coverage_ok_with_load_bins():
    t = np.sort(np.abs(np.random.default_rng(1).normal(100, 30, N)))
    ts = np.arange(N, dtype=np.float64) * 900.0
    out = check_operating_coverage({"y": t}, "y", ts, config=CFG)
    assert out["level"] == "info" and out["passed"]
    bins = out["evidence"]["load_bins"]
    assert [b["bin"] for b in bins] == ["low", "mid", "high"]
    assert sum(b["n_samples"] for b in bins) == N
    assert out["evidence"]["time_span_days"] == pytest.approx(
        (N - 1) * 900 / 86400)


# ---------------------------------------------------------------- 报告装配与门禁


def test_report_verdict_fail_on_blocker(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    report = build_modelability_report(
        ctx.vault, ref, target=TARGET,
        candidate_inputs=[*FEATURES, TARGET],  # 目标自身入候选 → 完全相同
        object_model="chiller.v1",
    )
    assert report["verdict"] == "FAIL"
    assert report["blockers"] == ["same_origin"]


def test_report_verdict_pass_on_clean_goal(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    report = build_modelability_report(
        ctx.vault, ref, target=TARGET, candidate_inputs=list(FEATURES),
        object_model="chiller.v1",
    )
    assert report["verdict"] == "PASS"
    assert report["blockers"] == []
    # 合成数据三台温度逐点相同 → 设备区分度 warning 但不阻断
    assert "device_diversity" in report["warnings"]
    levels = {c["name"]: c["level"] for c in report["checks"]}
    assert levels["derivation_chain"] == "info"
    assert levels["operating_coverage"] == "info"


def _goal_doc(**overrides):
    doc = {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5},
    }
    doc.update(overrides)
    return doc


def test_tool_envelope_and_artifact(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal = tf_goal_create(ctx, _goal_doc())
    env = tf_dataset_modelability(ctx, ref, goal_id=goal["id"])
    assert env["ok"] is True and env["status"] == "PASS"
    assert set(env) == {"ok", "tool", "id", "status", "inputs", "summary",
                        "diagnostics", "artifacts", "truncated"}
    artifact = env["artifacts"][0]
    assert artifact["kind"] == "modelability_report"
    with open(artifact["path"], encoding="utf-8") as fp:
        full = json.load(fp)
    assert full["kind"] == "modelability_report"
    assert len(full["checks"]) == 5  # 完整报告落 artifact


def test_orchestrator_gate_blocks_modeling_on_fail(tmp_path):
    """FAIL 时不允许登记实验：停止原因 modelability_failed，零实验。"""
    ctx, ref = make_ctx(tmp_path)
    goal = tf_goal_create(ctx, _goal_doc(
        candidate_inputs=[*FEATURES, TARGET]))  # 目标自身 → 同源 blocker
    assert goal["ok"]

    def planner(round_index, evidence):  # 不应被调用
        raise AssertionError("FAIL 门禁下不得进入假设/实验")

    orch = ResearchOrchestrator(ctx, goal["id"], planner, dataset_ref=ref)
    result = orch.run()
    stop = result["summary"]["stop"]
    assert stop["reason"] == STOP_MODELABILITY_FAILED
    assert stop["evidence"]["blockers"] == ["same_origin"]
    assert result["summary"]["rounds"] == []
    # 未登记任何实验
    assert ctx.ledger.goal_progress(goal["id"])["experiments"]["total"] == 0
    states = [t["to"] for t in ctx.ledger.transitions_of(goal["id"])]
    assert "MODELABILITY_ASSESSMENT" in states
    assert "EXPERIMENT_RUNNING" not in states


def test_orchestrator_gate_pass_allows_modeling(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal = tf_goal_create(ctx, _goal_doc())
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    calls = []

    def planner(round_index, evidence):
        calls.append(round_index)
        return None  # 无可行假设 → 证明已通过门禁进入循环

    orch = ResearchOrchestrator(ctx, goal["id"], planner, dataset_ref=ref)
    result = orch.run()
    assert result["summary"]["stop"]["reason"] == "no_information_gain"
    assert calls == [0]  # 到达 planner，门禁已通过（planner 使用零基索引）
    states = [t["to"] for t in ctx.ledger.transitions_of(goal["id"])]
    assert "MODELABILITY_ASSESSMENT" in states
    assert "BASELINE_MODELING" in states
