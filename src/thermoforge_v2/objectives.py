"""Frozen research objectives and conservative, search-only evidence assessment.

These pure functions never read holdout artifacts or turn a measurement into an
agent's scientific conclusion. Missing constraints remain unknown. A registered
finding supplies an evidence review, not an independent replication.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from thermoforge_core.contracts.experiment import ModelSpec
from .contracts import validate_baseline_model


METRICS = {"CVRMSE": "min", "RMSE": "min", "MAE": "min", "MAPE": "min",
           "NMBE": "min", "R2": "max"}
_LIMITS = (("cvrmse_max", "CVRMSE", "le"), ("r2_min", "R2", "ge"),
           ("mape_max", "MAPE", "le"), ("nmbe_abs_max", "NMBE", "abs_le"))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _combined(statuses, empty="not_configured"):
    if not statuses:
        return empty
    if "fail" in statuses:
        return "fail"
    return "unknown" if "unknown" in statuses else "pass"


def _model(spec):
    try:
        return ModelSpec.model_validate(spec).model_dump(mode="json")
    except (ValueError, TypeError):
        return None


def build_objective_contract(config: dict, protocol: dict) -> dict:
    """Freeze requested goals without altering the experiment protocol hash.

    A baseline is a model chosen before search, not the best observed candidate.
    Its first successful exact execution is identified from visible job records.
    No baseline is invented when that execution is absent.
    """
    acceptance = deepcopy((protocol.get("goal_definition") or {}).get("acceptance") or {})
    thresholds = [{"key": key, "metric": metric, "operator": operator, "limit": acceptance[key]}
                  for key, metric, operator in _LIMITS if _number(acceptance.get(key)) is not None]
    constraints = []
    for key, metric in (("physics_violation_rate_max", "physics_violation_rate"),
                        ("inference_latency_ms_max", "inference_latency_ms")):
        if _number(acceptance.get(key)) is not None:
            constraints.append({"key": key, "metric": metric, "operator": "le", "limit": acceptance[key]})
    if acceptance.get("extrapolation_required"):
        constraints.append({"key": "extrapolation_required", "metric": "extrapolation", "operator": "verified"})
    requested = config.get("objective_mode")
    mode = requested or ("target" if thresholds or constraints else "explore")
    metric = config.get("objective_metric") or (thresholds[0]["metric"] if thresholds else "CVRMSE")
    if mode not in {"target", "optimize", "explore"} or metric not in METRICS:
        raise ValueError("未知研究目标模式或主指标")
    minimum = _number(config.get("min_improvement", 0.01))
    if minimum is None or not 0 <= minimum <= 1:
        raise ValueError("min_improvement 必须是 0 到 1 的有限相对改善比例")
    baseline = _model(config["baseline_model"]) if config.get("baseline_model") else None
    if config.get("baseline_model") and baseline is None:
        raise ValueError("共同基线必须是有效 ModelSpec")
    if baseline:
        validate_baseline_model(ModelSpec.model_validate(baseline))
    contract = {
        "version": 1, "objective_mode": mode, "requested_mode": requested,
        "protocol_fingerprint": protocol.get("fingerprint"),
        "search_surface": "validate", "acceptance_surface": acceptance.get("evaluated_on", "auto"),
        "primary_metric": {"name": metric, "direction": METRICS[metric],
                           "transform": "absolute" if metric == "NMBE" else "identity"},
        "thresholds": thresholds, "constraints": constraints,
        "baseline_model": baseline, "baseline_job_id": None,
        "baseline_policy": "first_successful_exact_model" if baseline else "not_configured",
        "min_improvement": minimum, "improvement_unit": "relative_fraction",
        "stagnation_rounds": int(config.get("stagnation_rounds", 2)),
        "review_required": bool(config.get("review_required", True)),
        "review_definition": "对所选实验登记包含解释和限制的发现；这是证据审阅，不是独立复现或统计保证。",
        "acceptance_note": "validate 仅为搜索反馈；A、C、auto 或 rolling_cv 验收须在冻结选择后由外部评价确认。",
    }
    contract["fingerprint"] = hashlib.sha256(json.dumps(
        contract, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return contract


def _check_threshold(spec, metrics):
    value = _number(metrics.get(spec["metric"]))
    if value is None:
        return {**spec, "value": None, "status": "unknown"}
    comparable = abs(value) if spec["operator"] == "abs_le" else value
    passed = comparable >= spec["limit"] if spec["operator"] == "ge" else comparable <= spec["limit"]
    return {**spec, "value": value, "status": "pass" if passed else "fail"}


def _constraints(specs, result):
    # The present runner does not expose validated physics/latency evidence to
    # agents. Only explicit, surface-labelled server evidence can establish it.
    evidence = result.get("constraint_evidence") or {}
    checks = []
    for spec in specs:
        item = evidence.get(spec["metric"]) or {}
        value, state = None, "unknown"
        if spec["metric"] != "extrapolation" and isinstance(item, dict):
            if item.get("evaluated_on") == "validate" and item.get("verified") is True:
                value = _number(item.get("value"))
                if value is not None and value >= 0:
                    state = "pass" if value <= spec["limit"] else "fail"
        # Extrapolation requires held-out evidence and therefore cannot be
        # established in this search-only function, even by an agent assertion.
        checks.append({**spec, "value": value, "status": state})
    return checks


def _order(indexed):
    index, job = indexed
    timestamp = _number(job.get("finished_at"))
    if timestamp is None:
        timestamp = _number(job.get("created_at"))
    return (timestamp if timestamp is not None else float("-inf"), index)


def _gain(baseline, value, direction):
    delta = baseline - value if direction == "min" else value - baseline
    # Relative change against zero has no finite definition. Preserve the
    # signed absolute delta while reporting the relative comparison unknown.
    return delta, delta / abs(baseline) if baseline != 0 else None


def evaluate_objective(run: dict, jobs: list[dict], findings: list[dict],
                       track_id: str | None = None) -> dict[str, Any]:
    """Assess visible evidence only; never infer missing historical contracts.

    Callers must apply their visibility policy before passing jobs/findings.
    Track selection narrows candidate results but may use a visible common
    baseline. No mutations or agent-authored findings/stops are created.
    """
    contract = run.get("objective_contract")
    if not isinstance(contract, dict):
        return {"contract_available": False, "objective_mode": None, "primary_metric": None,
                "baseline": {"status": "not_configured", "job_id": None, "value": None},
                "best_observed": None, "best_eligible": None, "per_job": [],
                "threshold_status": "unknown", "improvement_status": "unknown",
                "review_status": "unknown", "acceptance_status": "unknown",
                "missing_evidence": ["历史运行没有冻结研究目标契约，不能追认达标或改善。"],
                "recommended_action": "declare_limitations", "requires_scientific_decision": True}
    mode = contract["objective_mode"]
    metric = contract["primary_metric"]
    expected = contract.get("protocol_fingerprint")
    run_fingerprint = (run.get("protocol") or {}).get("fingerprint")
    rows, valid_jobs = [], []
    reviews = {jid for finding in findings
               if finding.get("run_id", run.get("run_id")) == run.get("run_id")
               and finding.get("protocol_fingerprint") == expected
               and all(isinstance(finding.get(key), str) and finding[key].strip()
                       for key in ("statement", "interpretation", "limitations"))
               for jid in (finding.get("job_ids") or [])}
    for _, job in sorted(enumerate(jobs), key=_order):
        result = job.get("result") or {}
        reasons = []
        if job.get("run_id", run.get("run_id")) != run.get("run_id"):
            continue
        if job.get("status") != "completed":
            reasons.append("实验未成功完成")
        if not expected or expected != run_fingerprint or result.get("protocol_fingerprint") != expected:
            reasons.append("缺少或不匹配的冻结协议")
        if result.get("feedback_surface") != "validate":
            reasons.append("缺少 validate 搜索评价面")
        if job.get("reused_from_job_id") or job.get("executed") is False:
            reasons.append("复用记录不作为新增实测或独立复核")
        metrics = result.get("metrics") or {}
        raw = _number(metrics.get(metric["name"]))
        value = abs(raw) if raw is not None and metric["transform"] == "absolute" else raw
        if value is None:
            reasons.append("缺少有效主指标")
        threshold_checks = [_check_threshold(spec, metrics) for spec in contract["thresholds"]]
        constraint_checks = _constraints(contract["constraints"], result)
        threshold = _combined([c["status"] for c in threshold_checks])
        constraints = _combined([c["status"] for c in constraint_checks], "pass")
        if reasons:
            threshold = "unknown" if threshold_checks else "not_configured"
            constraints = "unknown" if constraint_checks else "pass"
        review = ("not_required" if not contract["review_required"] else
                  "pass" if job.get("id") in reviews else "unknown")
        row = {"job_id": job.get("id"), "track_id": job.get("track_id"),
               "execution_succeeded": job.get("status") == "completed", "comparable": not reasons,
               "value": value, "raw_value": raw, "metric": metric["name"],
               "threshold_status": threshold, "constraint_status": constraints,
               "threshold_checks": threshold_checks, "constraint_checks": constraint_checks,
               "review_status": review, "excluded_reasons": reasons}
        if not reasons:
            valid_jobs.append((job, row))
        if track_id is None or job.get("track_id") == track_id:
            rows.append(row)
    baseline = {"status": "not_configured", "job_id": None, "value": None,
                "reason": "没有预先指定共同基线模型，不能把当前最好结果当作基线。"}
    if contract.get("baseline_model"):
        baseline = {"status": "missing", "job_id": None, "value": None,
                    "reason": "尚无可见的同协议、预定模型真实成功执行，未配置可比较的基线结果。"}
        for job, row in valid_jobs:
            spec = (job.get("request") or {}).get("model") or ((job.get("result") or {}).get("model") or {}).get("spec")
            if _model(spec) == contract["baseline_model"]:
                baseline = {"status": "available", "job_id": job["id"], "value": row["value"],
                            "reason": "预先指定共同基线模型的首次同协议真实成功执行。"}
                break
    observed = [row for row in rows if row["comparable"]]
    sign = 1 if metric["direction"] == "min" else -1
    best = min(observed, key=lambda row: sign * row["value"], default=None)
    for row in rows:
        row.update(absolute_improvement=None, relative_improvement=None, improvement_status="unknown")
        if row["comparable"] and baseline["status"] == "available":
            delta, gain = _gain(baseline["value"], row["value"], metric["direction"])
            row.update(absolute_improvement=delta, relative_improvement=gain)
            if gain is not None:
                row["improvement_status"] = "pass" if delta > 0 and gain >= contract["min_improvement"] else "fail"
    eligible = [row for row in observed if row["constraint_status"] == "pass"
                and row["review_status"] in {"pass", "not_required"}
                and (mode != "target" or row["threshold_status"] == "pass"
                     or (not contract["thresholds"] and bool(contract["constraints"])))
                and (mode != "optimize" or row["improvement_status"] == "pass")]
    best_eligible = min(eligible, key=lambda row: sign * row["value"], default=None)
    selected = best_eligible or best
    threshold = selected["threshold_status"] if selected else "unknown"
    review = selected["review_status"] if selected else ("unknown" if contract["review_required"] else "not_required")
    improvement = (selected["improvement_status"] if selected else "unknown") if mode == "optimize" else "not_applicable"
    acceptance_status = "not_configured" if not contract["thresholds"] and not contract["constraints"] else "unknown"
    if selected and contract["acceptance_surface"] == "validate" and acceptance_status != "not_configured":
        statuses = [selected["constraint_status"]]
        if contract["thresholds"]:
            statuses.append(threshold)
        acceptance_status = _combined(statuses)
    incumbent, rounds_without_gain = None, 0
    for row in observed:
        if incumbent is None:
            incumbent = row["value"]
            continue
        delta, gain = _gain(incumbent, row["value"], metric["direction"])
        meaningful = delta > 0 and gain is not None and gain >= contract["min_improvement"]
        rounds_without_gain = 0 if meaningful else rounds_without_gain + 1
        if meaningful:
            incumbent = row["value"]
    stagnant = rounds_without_gain >= contract["stagnation_rounds"]
    missing = []
    if not observed:
        missing.append("尚无可比较的真实 validate 主指标。")
    if mode == "optimize" and baseline["status"] != "available":
        missing.append(baseline["reason"])
    if mode == "optimize" and baseline["status"] == "available" and baseline["value"] == 0:
        missing.append("基线为零，不能计算相对改善；需另行预先定义适用的改善尺度。")
    if mode == "target" and not contract["thresholds"] and not contract["constraints"]:
        missing.append("达标模式未配置阈值或约束，不能宣称达标。")
    if selected:
        missing.extend("缺少验收证据：" + check["key"] for check in
                       selected["threshold_checks"] + selected["constraint_checks"] if check["status"] == "unknown")
    if review == "unknown":
        missing.append("尚未登记所选实验的解释与限制；证据审阅未完成。")
    if acceptance_status == "unknown" and contract["acceptance_surface"] != "validate":
        missing.append("最终验收面为 " + str(contract["acceptance_surface"]) + "，validate 不能证明外部验收达标。")
    if best_eligible and mode != "explore":
        action = "conclude"
    elif selected and review == "unknown":
        action = "review"
    elif stagnant or (mode == "optimize" and baseline["status"] != "available"):
        action = "declare_limitations"
    else:
        action = "continue"
    return {"contract_available": True, "objective_mode": mode, "primary_metric": deepcopy(metric),
            "search_surface": "validate", "acceptance_surface": contract["acceptance_surface"],
            "baseline": baseline, "best_observed": deepcopy(best), "best_eligible": deepcopy(best_eligible),
            "threshold_status": threshold, "improvement_status": improvement,
            "review_status": review, "acceptance_status": acceptance_status, "per_job": rows,
            "stagnant": stagnant, "rounds_without_meaningful_gain": rounds_without_gain,
            "min_improvement": contract["min_improvement"], "missing_evidence": missing,
            "recommended_action": action, "requires_scientific_decision": True,
            "note": "这是系统事实评估与下一步建议，不是智能体研究结论或停止决定；证据审阅不等于独立复现。"}
