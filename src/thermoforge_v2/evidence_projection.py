"""Bounded research evidence views, with no artifact or held-out surface passthrough.

These functions do not grant access. Callers must resolve each record through their
identity boundary before projecting it. Source bodies are read by source_excerpt.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

METRICS = ("RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2")
COMMON = ("id", "track_id", "created_at", "updated_at", "protocol_fingerprint")
FIELDS = {
    "sources": ("title", "kind", "url", "doi", "read_scope", "retrieved_scope", "verification",
                "retrieved_at", "content_sha256", "history_ids", "content_truncated"),
    "ideas": ("statement", "reason", "prediction", "falsification", "origin", "source_ids",
              "parent_idea_ids", "parent_job_ids", "strategy"),
    "proposals": ("idea_id", "model", "version", "status", "purpose", "expected_cost",
                  "parent_job_ids", "finding_ids", "experiment_fingerprint", "lab_content_hash", "research_stage"),
    "jobs": ("idea_id", "proposal_id", "status", "experiment_id", "experiment_fingerprint", "executed",
             "reused_from_job_id", "started_at", "finished_at", "request", "result"),
    "findings": ("statement", "interpretation", "limitations", "job_ids", "idea_ids", "failure_category"),
    "reports": ("kind", "title", "summary", "body", "limitations", "idea_ids", "job_ids", "finding_ids",
                "evidence_ids", "report_stage", "final_report_epoch", "status"),
    "stops": ("reason", "evidence_ids", "job_ids", "research_stage"),
    "messages": ("from", "to", "text", "evidence_ids", "version", "initial_task", "research_stage"),
    "decisions": ("action", "reason", "evidence_ids", "parent_idea_ids"),
}


def numeric_metrics(value: Any) -> dict[str, float | None]:
    """Allow registered finite numeric metrics, never arbitrary nested payloads."""
    if not isinstance(value, Mapping):
        return {}
    return {key: float(item) if item is not None else None
            for key in METRICS if key in value
            and ((item := value[key]) is None or (
                isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)))}


def _model(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {key: value[key] for key in ("category", "estimator", "physics", "residual")
              if isinstance(value.get(key), str)}
    hp = value.get("hyperparameters")
    if isinstance(hp, Mapping):
        result["hyperparameters"] = {str(k): v for k, v in hp.items()
            if isinstance(v, (str, bool, int, float)) or v is None}
    return result


def _diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("train", "validate"):
        item = value.get(key)
        if isinstance(item, Mapping):
            result[key] = {k: item[k] for k in ("status", "n_samples", "reason")
                           if isinstance(item.get(k), (str, int, float))}
            result[key]["metrics"] = numeric_metrics(item.get("metrics"))
    bias = value.get("bias")
    if isinstance(bias, Mapping):
        result["bias"] = {key: bias[key] for key in ("status", "direction", "convention", "reason")
                          if isinstance(bias.get(key), str)}
        if "NMBE" in numeric_metrics({"NMBE": bias.get("nmbe")}):
            result["bias"]["nmbe"] = numeric_metrics({"NMBE": bias.get("nmbe")})["NMBE"]
    gap = value.get("generalization_gap")
    if isinstance(gap, Mapping):
        result["generalization_gap"] = {key: gap[key] for key in ("status", "reason", "convention")
                                       if isinstance(gap.get(key), str)}
        result["generalization_gap"]["validate_minus_train"] = numeric_metrics(gap.get("validate_minus_train"))
    physics = value.get("physics")
    if isinstance(physics, Mapping):
        result["physics"] = {key: physics[key] for key in ("status", "reason")
                             if isinstance(physics.get(key), str)}
    return result


def safe_feedback(value: Any) -> dict[str, Any]:
    """Project already settled feedback without trusting arbitrary result keys."""
    if not isinstance(value, Mapping):
        return {}
    result = {key: value[key] for key in (
        "experiment_id", "protocol_fingerprint", "feedback_surface", "n_samples", "duration_seconds",
        "failure_category", "error_code", "error", "training_started", "reservation_consumed", "reason")
        if isinstance(value.get(key), (str, int, float, bool))}
    # This projector is a second guard for old/imported records. No other surface
    # may masquerade as search feedback, even when its metrics use known names.
    if value.get("feedback_surface") in (None, "validate"):
        result["metrics"] = numeric_metrics(value.get("metrics"))
        if "diagnostics" in value:
            result["diagnostics"] = _diagnostics(value["diagnostics"])
        evidence = value.get("constraint_evidence")
        if isinstance(evidence, Mapping):
            verified = {}
            for key in ("physics_violation_rate", "inference_latency_ms"):
                item = evidence.get(key)
                if not isinstance(item, Mapping):
                    continue
                number = item.get("value")
                if (item.get("evaluated_on") == "validate" and item.get("verified") is True
                        and isinstance(number, (int, float)) and not isinstance(number, bool)
                        and math.isfinite(number) and number >= 0):
                    verified[key] = {"evaluated_on": "validate", "verified": True, "value": number}
            if verified:
                result["constraint_evidence"] = verified
    else:
        result["feedback_surface"] = "unavailable"
    model = value.get("model")
    if isinstance(model, Mapping):
        result["model"] = {key: model[key] for key in ("lab_ref", "content_hash")
                           if isinstance(model.get(key), str)}
        if "spec" in model:
            result["model"]["spec"] = _model(model["spec"])
    return result


def project_record(kind: str, record: Mapping[str, Any], fields: list[str] | None = None) -> dict[str, Any]:
    """Complete safe metadata and agent-authored text, without size compaction."""
    if kind not in FIELDS:
        raise ValueError("不允许读取该类记录")
    allowed = set(COMMON + FIELDS[kind])
    if fields is not None:
        if not isinstance(fields, list) or any(not isinstance(v, str) for v in fields):
            raise ValueError("fields 必须为字段名列表")
        if set(fields) - allowed:
            raise ValueError("不允许读取字段: " + ", ".join(sorted(set(fields) - allowed)))
        allowed = set(fields) | {"id"}
    result = {key: value for key, value in record.items() if key in allowed}
    if kind == "jobs":
        if "result" in result:
            result["result"] = safe_feedback(result["result"])
        if "request" in result:
            req = result["request"] if isinstance(result["request"], Mapping) else {}
            result["request"] = {key: req[key] for key in (
                "protocol_fingerprint", "proposal_id", "experiment_fingerprint", "lab_content_hash")
                if isinstance(req.get(key), str)}
            result["request"]["model"] = _model(req.get("model"))
    elif kind == "proposals" and "model" in result:
        result["model"] = _model(result["model"])
    if kind == "sources":
        result["content_chars"] = len(record.get("content") or "")
        result["read_hint"] = "正文按需调用 research_source_excerpt(source_id, offset)。"
    return result


def compact_record(kind: str, record: Mapping[str, Any], fields: list[str] | None = None) -> dict[str, Any]:
    """Bounded whitelist view for history, shared evidence and incremental briefs."""
    projected = project_record(kind, record, fields)
    truncated = False

    text_limit = 320

    def bound(value: Any, depth: int = 0) -> Any:
        nonlocal truncated
        if depth > 8:
            truncated = True
            return None
        if isinstance(value, str):
            if len(value) > text_limit:
                truncated = True
                return value[:text_limit] + "…"
            return value
        if isinstance(value, Mapping):
            items = list(value.items())
            if len(items) > 24:
                truncated = True
            return {str(k): bound(v, depth + 1) for k, v in items[:24]}
        if isinstance(value, (list, tuple)):
            if len(value) > 12:
                truncated = True
            return [bound(item, depth + 1) for item in value[:12]]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value if value is None or isinstance(value, (bool, int, float)) else None

    # Reports already carry a summary. Their long narrative must not be repeated
    # on every history/team call. The full body remains addressable by ID.
    if kind == "reports" and "body" in projected and fields is None:
        projected.pop("body")
        truncated = True
    result = bound(projected)
    while len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 12 * 1024 and text_limit > 20:
        text_limit //= 2
        result = bound(projected)
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 12 * 1024:
        result = {key: result[key] for key in ("id", "track_id", "status", "title", "summary") if key in result}
        truncated = True
    # Cursor anchors and individual lookup IDs must never be shortened.
    if "id" in projected:
        result["id"] = projected["id"]
    if truncated:
        result["truncated"] = True
        result["read_hint"] = ("消息正文按需调用 research_messages(record_ids=[id], offset=0)。" if kind == "messages" else
            "按需调用 research_history(kind, record_ids=[id], compact=false, fields=[字段名])；长文本以单个字段和 offset 分页。")
    return result
