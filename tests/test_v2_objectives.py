"""Research success uses frozen requirements, never missing or held-out evidence."""
from copy import deepcopy

import pytest

from thermoforge_v2.contracts import RunConfig
from thermoforge_v2.objectives import build_objective_contract, evaluate_objective
from thermoforge_v2.store import RunStore


def run_doc(acceptance=None, **options):
    config = RunConfig(goal_id="RG-0001", dataset_ref="D@rev_0001", **options).model_dump(mode="json")
    protocol = {"fingerprint": "frozen", "goal_definition": {"acceptance": acceptance or {}},
                "feedback_surface": "validate"}
    return {"run_id": "RUN-one", "config": config, "protocol": protocol,
            "objective_contract": build_objective_contract(config, protocol)}


def job(jid="J1", value=4.0, *, metric="CVRMSE", model=None, track="candidate-1", **fields):
    return {"id": jid, "run_id": "RUN-one", "track_id": track,
            "status": "completed", "executed": True,
            "request": {"model": model or {"category": "data", "estimator": "ridge"}},
            "result": {"protocol_fingerprint": "frozen", "feedback_surface": "validate",
                       "metrics": {metric: value}}, **fields}


def reviewed(*jids):
    return [{"id": "F-one", "run_id": "RUN-one", "track_id": "candidate-1",
             "protocol_fingerprint": "frozen", "job_ids": list(jids),
             "statement": "本次实测误差见实验。", "interpretation": "该测试支持本验证段的拟合。",
             "limitations": "未评价未来设备和分布漂移。"}]


@pytest.mark.parametrize("acceptance, mode", [({}, "explore"), ({"cvrmse_max": 5}, "target"),
                                            ({"extrapolation_required": True}, "target")])
def test_auto_contract_uses_frozen_requirements_without_mutating_protocol(acceptance, mode):
    run = run_doc(acceptance)
    before = deepcopy(run["protocol"])
    contract = build_objective_contract(run["config"], run["protocol"])
    assert contract["requested_mode"] is None
    assert contract["objective_mode"] == mode
    assert contract["min_improvement"] == 0.01
    assert contract["improvement_unit"] == "relative_fraction"
    assert contract["review_required"] is True
    assert run["protocol"] == before
    assert build_objective_contract(run["config"], before) == contract


def test_threshold_review_and_execution_are_distinct():
    run = run_doc({"evaluated_on": "validate", "cvrmse_max": 5})
    assessment = evaluate_objective(run, [job()], [])
    assert assessment["per_job"][0]["execution_succeeded"]
    assert assessment["threshold_status"] == "pass"
    assert assessment["acceptance_status"] == "pass"
    assert assessment["review_status"] == "unknown"
    assert assessment["best_eligible"] is None
    assert assessment["recommended_action"] == "review"
    reviewed_result = evaluate_objective(run, [job()], reviewed("J1"))
    assert reviewed_result["best_eligible"]["job_id"] == "J1"
    assert reviewed_result["recommended_action"] == "conclude"
    assert reviewed_result["requires_scientific_decision"] is True


@pytest.mark.parametrize("surface", ["A", "C", "auto", "rolling_cv"])
def test_validate_threshold_never_establishes_external_acceptance(surface):
    run = run_doc({"evaluated_on": surface, "cvrmse_max": 5})
    result = evaluate_objective(run, [job()], reviewed("J1"))
    assert result["threshold_status"] == "pass"
    assert result["best_eligible"]["job_id"] == "J1"
    assert result["acceptance_status"] == "unknown"
    assert any("validate 不能证明" in item for item in result["missing_evidence"])


@pytest.mark.parametrize("constraint", [{"physics_violation_rate_max": 0.01},
                                       {"inference_latency_ms_max": 2},
                                       {"extrapolation_required": True}])
def test_missing_constraints_are_unknown_not_passed(constraint):
    run = run_doc({"evaluated_on": "validate", "cvrmse_max": 5, **constraint})
    measured = job()
    # Unscoped global/test outputs must not become search evidence.
    measured["result"].update(physics_overall_rate=0, inference_latency_ms=0.1,
                              extrapolation=True, surfaces={"C": {"n_samples": 100}})
    result = evaluate_objective(run, [measured], reviewed("J1"))
    assert result["threshold_status"] == "pass"
    assert result["per_job"][0]["constraint_status"] == "unknown"
    assert result["acceptance_status"] == "unknown"
    assert result["best_eligible"] is None


def test_evidence_scope_and_measured_constraint_failure():
    run = run_doc({"evaluated_on": "validate", "physics_violation_rate_max": 0.01})
    measured = job()
    measured["result"]["constraint_evidence"] = {
        "physics_violation_rate": {"value": 0, "verified": True, "evaluated_on": "A"}}
    assert evaluate_objective(run, [measured], reviewed("J1"))["acceptance_status"] == "unknown"
    measured["result"]["constraint_evidence"]["physics_violation_rate"].update(value=0.2, evaluated_on="validate")
    result = evaluate_objective(run, [measured], reviewed("J1"))
    assert result["acceptance_status"] == "fail"
    assert result["best_eligible"] is None


def test_baseline_is_predeclared_model_first_real_result_not_best_candidate():
    base = {"category": "data", "estimator": "ridge", "hyperparameters": {"alpha": 1}}
    run = run_doc(objective_mode="optimize", baseline_model=base)
    jobs = [job("BEST", 3, model={"category": "data", "estimator": "linear"}, finished_at=1),
            job("BASE", 5, model=base, finished_at=2), job("BASE-REPEAT", 4, model=base, finished_at=3)]
    result = evaluate_objective(run, jobs, reviewed("BEST", "BASE", "BASE-REPEAT"))
    assert result["baseline"]["job_id"] == "BASE"
    assert result["best_observed"]["job_id"] == "BEST"
    assert result["best_eligible"]["job_id"] == "BEST"
    assert result["best_eligible"]["relative_improvement"] == pytest.approx(0.4)
    assert result["improvement_status"] == "pass"
    missing = evaluate_objective(run_doc(objective_mode="optimize"), jobs, reviewed("BEST"))
    assert missing["baseline"]["status"] == "not_configured"
    assert missing["improvement_status"] == "unknown"
    assert missing["best_eligible"] is None


def test_baseline_requires_matching_protocol_and_visible_real_execution():
    base = {"category": "data", "estimator": "ridge"}
    run = run_doc(objective_mode="optimize", baseline_model=base)
    wrong = job("WRONG", 9, model=base)
    wrong["result"]["protocol_fingerprint"] = "different"
    reused = job("REUSED", 10, model=base, executed=False, reused_from_job_id="HIDDEN")
    candidate = job("CANDIDATE", 1, model={"category": "data", "estimator": "linear"})
    result = evaluate_objective(run, [wrong, reused, candidate], reviewed("CANDIDATE"), "candidate-1")
    assert result["baseline"]["status"] == "missing"
    assert result["improvement_status"] == "unknown"
    assert result["best_observed"]["job_id"] == "CANDIDATE"
    assert result["best_eligible"] is None


def test_track_projection_does_not_leak_other_candidate_rank():
    run = run_doc({"cvrmse_max": 5})
    result = evaluate_objective(run, [job("OWN", 4), job("OTHER", 1, track="candidate-2")],
                                reviewed("OWN", "OTHER"), "candidate-1")
    assert result["best_observed"]["job_id"] == "OWN"
    assert [row["job_id"] for row in result["per_job"]] == ["OWN"]


def test_signed_nmbe_absolute_target_and_r2_improvement_direction():
    nmbe = evaluate_objective(run_doc({"evaluated_on": "validate", "nmbe_abs_max": 5}),
                              [job(metric="NMBE", value=-6)], reviewed("J1"))
    assert nmbe["threshold_status"] == "fail"
    assert nmbe["best_observed"]["value"] == 6
    base = {"category": "data", "estimator": "linear"}
    run = run_doc({"r2_min": 0.9}, objective_mode="optimize", baseline_model=base)
    result = evaluate_objective(run, [job("BASE", metric="R2", value=0.8, model=base),
                                      job("J1", metric="R2", value=0.85)], reviewed("J1", "BASE"))
    assert result["primary_metric"]["direction"] == "max"
    assert result["improvement_status"] == "pass"
    assert result["threshold_status"] == "fail"  # Improvement does not mean absolute target met.


def test_zero_baseline_relative_gain_is_unknown_and_reuse_does_not_count_stagnation():
    base = {"category": "data", "estimator": "linear"}
    run = run_doc(objective_mode="optimize", baseline_model=base)
    result = evaluate_objective(run, [job("BASE", 0, model=base), job("J1", 0),
                                      job("REUSED", 0, executed=False, reused_from_job_id="J1")], reviewed("J1", "BASE"))
    assert result["improvement_status"] == "unknown"
    assert result["rounds_without_meaningful_gain"] == 1
    assert result["stagnant"] is False


def test_negative_result_and_stagnation_do_not_create_a_scientific_stop():
    run = run_doc({"evaluated_on": "validate", "cvrmse_max": 1})
    jobs = [job("J1", 4), job("J2", 4.2), job("J3", 5)]
    before = deepcopy((run, jobs))
    result = evaluate_objective(run, jobs, reviewed("J1", "J2", "J3"))
    assert result["threshold_status"] == "fail"
    assert result["stagnant"] is True
    assert result["recommended_action"] == "declare_limitations"
    assert result["best_observed"]["job_id"] == "J1"
    assert result["best_eligible"] is None
    assert (run, jobs) == before
    assert "stops" not in run


def test_exploration_needs_scientific_judgment_beyond_a_scored_and_reviewed_job():
    result = evaluate_objective(run_doc(objective_mode="explore"), [job()], reviewed("J1"))
    assert result["best_eligible"]["job_id"] == "J1"
    assert result["recommended_action"] == "continue"
    assert result["requires_scientific_decision"] is True


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "0"])
def test_invalid_metrics_never_pass(value):
    result = evaluate_objective(run_doc({"cvrmse_max": 5}), [job(value=value)], reviewed("J1"))
    assert result["best_eligible"] is None
    assert result["best_observed"] is None
    assert result["threshold_status"] == "unknown"


def test_incomplete_review_or_failed_job_is_not_qualified():
    run = run_doc({"evaluated_on": "validate", "cvrmse_max": 5})
    finding = reviewed("J1")[0]
    finding["limitations"] = ""
    result = evaluate_objective(run, [job()], [finding])
    assert result["review_status"] == "unknown"
    assert result["best_eligible"] is None
    failed = evaluate_objective(run, [job(status="failed")], reviewed("J1"))
    assert failed["best_observed"] is None


def test_contract_is_frozen_across_control_and_not_retrofitted_to_old_run(tmp_path):
    store = RunStore(tmp_path)
    run = run_doc({"evaluated_on": "validate", "cvrmse_max": 5})
    created = store.create_run(run["config"], run["protocol"], "once")
    updated = store.control(created["run_id"], "update", changes={"token_budget": 2000000})
    assert updated["objective_contract"] == created["objective_contract"]
    assert updated["protocol"] == created["protocol"]
    with pytest.raises(ValueError, match="不可修改"):
        store.update_run(created["run_id"], {"objective_contract": {}})
    old = deepcopy(run)
    old.pop("objective_contract")
    result = evaluate_objective(old, [job()], reviewed("J1"))
    assert result["contract_available"] is False
    assert result["acceptance_status"] == "unknown"
    assert "objective_contract" not in old


@pytest.mark.parametrize("changes", [{"min_improvement": float("inf")}, {"min_improvement": -0.1},
                                     {"min_improvement": 1.1}, {"objective_mode": "auto"},
                                     {"baseline_model": {"category": "invalid"}}])
def test_invalid_objective_configuration(changes):
    with pytest.raises(ValueError):
        run_doc(**changes)


def test_lab_baseline_rejected_until_cross_track_source_identity_can_be_frozen():
    baseline = {"category": "lab", "hyperparameters": {"lab": "same-name@v1"}}
    with pytest.raises(ValueError, match="共同基线暂不支持 lab"):
        run_doc(objective_mode="optimize", baseline_model=baseline)
    with pytest.raises(ValueError, match="共同基线暂不支持 lab"):
        build_objective_contract({"baseline_model": baseline}, {"fingerprint": "frozen"})


@pytest.mark.parametrize("baseline", [
    {"category": "hybrid", "physics": "same-name@v1", "residual": "xgboost"},
    {"category": "hybrid", "physics": "cooling_balance_v1", "residual": "same-name@v1"},
    {"category": "data", "estimator": "same-name@v1"},
    {"category": "data", "estimator": "ridge", "hyperparameters": {"lab": "same-name@v1"}},
])
def test_baseline_cannot_hide_local_lab_reference_in_builtin_fields(baseline):
    with pytest.raises(ValueError, match="共同基线"):
        run_doc(objective_mode="optimize", baseline_model=baseline)
