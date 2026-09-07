"""Fact-only research flow for exported reports, with explicit evidence boundaries.

The graph describes recorded relationships rather than inferring dialogue or causality
from timestamps. It never opens experiment artifacts or exposes private model reasoning.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


_KINDS = {"sources": "source", "ideas": "idea", "proposals": "proposal", "jobs": "job",
          "findings": "finding", "stops": "stop", "decisions": "decision",
          "reports": "report", "messages": "message"}
_REFERENCES = {"source_ids": ("source", "想法来源"),
               "parent_idea_ids": ("parent_idea", "承接想法"),
               "parent_job_ids": ("parent_experiment", "历史实验反馈"),
               "history_ids": ("history", "历史来源依据"),
               "idea_ids": ("idea", "引用想法"),
               "job_ids": ("experiment", "引用实验"),
               "finding_ids": ("finding", "引用发现"),
               "evidence_ids": ("evidence", "引用证据"),
               "report_ids": ("report", "引用报告")}
_COMMON = ("id", "track_id", "created_at", "updated_at", "protocol_fingerprint")
_FIELDS = {
    "source": ("kind", "title", "url", "doi", "read_scope", "verification", "retrieved_at",
               "content_sha256", "history_ids"),
    "idea": ("statement", "origin", "reason", "prediction", "falsification", "strategy",
             "source_ids", "parent_idea_ids", "parent_job_ids"),
    "proposal": ("idea_id", "model", "version", "status", "purpose", "expected_cost",
                 "experiment_fingerprint", "parent_job_ids", "research_stage", "finding_ids", "lab_content_hash"),
    "job": ("idea_id", "experiment_id", "status", "request", "started_at", "finished_at",
            "error", "failure_category", "proposal_id", "experiment_fingerprint",
            "duplicate_of_job_id", "reused_from_job_id", "executed"),
    "stop": ("reason", "evidence_ids", "job_ids", "research_stage"),
    "finding": ("statement", "interpretation", "limitations", "failure_category",
                "idea_ids", "job_ids"),
    "decision": ("action", "reason", "evidence_ids", "parent_idea_ids", "finding_ids", "job_ids"),
    "report": ("kind", "title", "summary", "body", "limitations", "report_stage",
               "idea_ids", "job_ids", "finding_ids", "evidence_ids", "report_ids"),
    "message": ("from", "to", "text", "initial_task", "evidence_ids", "research_stage"),
}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [deepcopy(dict(row)) for row in value or [] if isinstance(row, Mapping)]


def _ids(record: Mapping[str, Any], field: str) -> list[str]:
    value = record.get(field)
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_metrics(value: Any) -> dict[str, float | int]:
    return {str(key): item for key, item in _object(value).items()
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)}


def _clean(value: Any) -> Any:
    """Ensure JSON portability even when partial tool results contain NaN metrics."""
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def execution_label(job: Mapping[str, Any]) -> str:
    """Describe recorded execution, without counting reused data as a new measurement."""
    if job.get("reused_from_job_id"):
        return "复用既有结果（非独立复现）"
    if job.get("executed") is True:
        return "重复配置，分别执行" if job.get("duplicate_of_job_id") else "实际执行"
    if job.get("executed") is False:
        return "未执行"
    return "执行方式未记录"


def _label(kind: str, row: dict[str, Any]) -> str:
    if kind == "job":
        result = _object(row.get("result"))
        result_model = _object(result.get("model"))
        model = _object(result_model.get("spec")) or result_model
        if not model:
            model = _object(_object(row.get("request")).get("model"))
        return " · ".join(str(value) for value in (
            row.get("experiment_id") or result.get("experiment_id") or row.get("id"),
            model.get("estimator")) if value)
    if kind == "message":
        return "共同研究任务" if row.get("initial_task") else "协调消息"
    if kind == "proposal":
        stage = "首轮独立冻结提案" if row.get("version") == 1 else "修订提案"
        if row.get("status") != "committed":
            stage = "待冻结提案"
        return f"{stage} · v{row.get('version', '?')}"
    if kind == "stop":
        return "停止研究 · " + str(row.get("reason") or "未记录理由")
    if kind == "decision":
        return "路线决策 · " + str(row.get("action") or "未记录")
    return str(row.get("title") or row.get("statement") or row.get("summary") or row.get("id"))


def build_flow(snapshot: Mapping[str, Any], track_id: str | None = None) -> dict[str, Any]:
    """Build a serializable lane graph using only the supplied report snapshot.

    ``metrics`` on a job node always means finite, successful, explicitly labelled
    validate feedback. Final holdout measurements remain on a separate external node.
    A track-only view can cite outside records by ID without exposing their contents.
    """
    run = _object(snapshot.get("run"))
    tracks = _rows(snapshot.get("tracks"))
    if track_id is not None and not any(t.get("track_id") == track_id for t in tracks):
        raise ValueError(f"没有这条研究轨迹：{track_id}")
    selected_tracks = [t for t in tracks if track_id is None or t.get("track_id") == track_id]
    rows = {name: _rows(snapshot.get(name)) for name in _KINDS}
    all_records = {str(row["id"]): row for values in rows.values() for row in values if row.get("id")}
    selected = {name: [row for row in values
                       if track_id is None or row.get("track_id") == track_id]
                for name, values in rows.items()}
    if track_id is not None:
        source_ids = {ref for idea in selected["ideas"] for ref in _ids(idea, "source_ids")}
        selected["sources"] = [row for row in rows["sources"]
                               if row.get("track_id") == track_id or row.get("id") in source_ids]
        selected["messages"] = [row for row in rows["messages"]
                                if row.get("from") == track_id or row.get("to") in {track_id, "all"}]

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    lanes = []
    missing = []
    node_map: dict[str, dict[str, Any]] = {}
    edge_keys = set()

    def add(node: dict[str, Any]) -> None:
        if node["id"] not in node_map:
            node_map[node["id"]] = node
            nodes.append(node)

    def reference(ref: str, target: str, relation: str, label: str) -> None:
        if ref not in node_map:
            outside = ref in all_records
            status = "outside_scope" if outside else "missing"
            add({"id": ref, "kind": "missing", "track_id": node_map[target].get("track_id"),
                 "label": "范围外引用" if outside else "引用记录缺失", "status": status,
                 "detail": {"reference_id": ref, "reason": status}})
        if node_map[ref]["kind"] == "missing":
            item = {"record_id": target, "reference_id": ref, "reason": node_map[ref]["status"]}
            if item not in missing:
                missing.append(item)
        key = (ref, target, relation)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append({"from": ref, "to": target, "relation": relation, "label": label})

    for track in selected_tracks:
        tid = str(track["track_id"])
        role = track.get("role", "candidate")
        label = "主智能体" if role == "main" else tid.replace("candidate-", "候选 ")
        lanes.append({"id": tid, "track_id": tid, "label": label, "role": role,
                      "status": track.get("status", "unknown")})
        detail = {key: track[key] for key in ("track_id", "role", "status", "phase", "turns", "error",
                                             "created_at", "updated_at") if key in track}
        backend = _object(track.get("backend"))
        detail["backend"] = {key: backend[key] for key in ("model", "server", "native_subagents") if key in backend}
        add({"id": f"track:{tid}", "kind": "track", "track_id": tid, "label": label,
             "status": track.get("status", "unknown"), "detail": detail})

    for name, kind in _KINDS.items():
        for row in selected[name]:
            if not row.get("id"):
                continue
            detail = {key: row[key] for key in (*_COMMON, *_FIELDS[kind]) if key in row}
            node = {"id": str(row["id"]), "kind": kind, "track_id": row.get("track_id"),
                    "label": _label(kind, row), "status": row.get("status") or
                    ("submitted" if kind == "report" else "registered"), "detail": detail}
            if kind == "job":
                detail["execution_label"] = execution_label(row)
                result = _object(row.get("result"))
                detail["result"] = {key: result[key] for key in ("experiment_id", "protocol_fingerprint",
                    "feedback_surface", "n_samples", "duration_seconds", "failure_category", "error") if key in result}
                result_model = _object(result.get("model"))
                if result_model:
                    # Preserve executed model metadata without forwarding arbitrary result fields.
                    detail["result"]["model"] = deepcopy(result_model)
                metrics = _finite_metrics(result.get("metrics"))
                successful = row.get("status") in {"completed", "succeeded"}
                validated = result.get("feedback_surface") == "validate"
                node["metrics"] = metrics if successful and validated else {}
                detail["result"]["metrics"] = node["metrics"]
                if not successful:
                    detail["metric_note"] = "实验未成功完成，不将缺失结果当成零分或假说证伪。"
                elif not validated:
                    detail["metric_note"] = "未明确记录 validate 评价面，指标不作为研究反馈展示。"
                elif not metrics:
                    detail["metric_note"] = "没有有效的实测验证指标。"
                expected = _object(run.get("protocol")).get("fingerprint")
                detail["comparable"] = bool(expected and result.get("protocol_fingerprint") == expected
                                             and successful and validated and metrics)
            add(node)

    for name, values in selected.items():
        for row in values:
            target = str(row.get("id") or "")
            if not target:
                continue
            for field, (relation, label) in _REFERENCES.items():
                for ref in _ids(row, field):
                    reference(ref, target, relation, label)
            if isinstance(row.get("idea_id"), str) and row["idea_id"]:
                reference(row["idea_id"], target, "idea", "检验想法")
            for field, relation, label in (
                ("proposal_id", "proposal", "按冻结提案执行"),
                ("reused_from_job_id", "result_reuse", "复用结果，非独立复现"),
                ("duplicate_of_job_id", "duplicate_configuration", "相同配置登记，非启发关系"),
            ):
                if isinstance(row.get(field), str) and row[field]:
                    reference(row[field], target, relation, label)

    for message in selected["messages"]:
        mid = str(message.get("id") or "")
        if not mid:
            continue
        sender = message.get("from")
        if isinstance(sender, str) and sender:
            # In a candidate-only report, the sender is context, never a copy of its research.
            sender_id = f"track:{sender}"
            if sender_id not in node_map:
                add({"id": sender_id, "kind": "track", "track_id": sender,
                     "label": "主智能体" if sender == "main" else sender,
                     "status": "context", "detail": {"track_id": sender, "context_only": True}})
            reference(sender_id, mid, "message_sent", "发送任务" if message.get("initial_task") else "发送消息")
        recipients = [t for t in selected_tracks if t.get("track_id") != sender and
                      (message.get("to") == t.get("track_id") or
                       (message.get("to") == "all" and t.get("role") != "main"))]
        for recipient in recipients:
            reference(mid, f"track:{recipient['track_id']}", "message_to",
                      "共同任务投递" if message.get("initial_task") else "消息投递")

    final = _object(snapshot.get("final_evaluation"))
    if final and (track_id is None or final.get("track_id_selected") == track_id):
        fid = str(final.get("id") or f"final_evaluation:{run.get('run_id', 'unknown')}")
        fields = ("id", "status", "reason", "job_id", "experiment_id", "track_id_selected", "created_at",
                  "selection_reason", "selection_metric", "protocol_fingerprint", "report_sha256", "surfaces", "physics")
        detail = {key: final[key] for key in fields if key in final}
        detail["visibility"] = "external_only"
        detail["feedback_to_agents"] = False
        add({"id": fid, "kind": "final_evaluation", "track_id": None,
             "label": "最终留出评价 · 仅外部查看", "status": final.get("status", "unknown"), "detail": detail})
        if isinstance(final.get("job_id"), str) and final["job_id"]:
            reference(final["job_id"], fid, "final_evaluation", "冻结候选后评价")

    config = _object(run.get("config"))
    notes = ["连线只表示已登记的引用或软件消息；空间顺序不证明因果，未记录的候选讨论不会被补画。",
             "想法显示已登记的研究理由、预测与反证条件；不依赖模型内部思维链。",
             "实验卡片只显示成功实验明确标记为 validate 的反馈；留出评价独立展示且没有回到研究的反馈边。"]
    if selected["proposals"]:
        notes.append("提案保留实验前冻结的模型和版本；历史实验到修订提案的连线来自明确登记的父实验引用。")
    if any(row.get("reused_from_job_id") for row in selected["jobs"]):
        notes.append("复用请求展示原实验的已有指标，不计为新增实测或独立复现；重复配置标记本身不表示候选互相启发。")
    if run.get("research_stage"):
        notes.append(f"已登记研究阶段：{run['research_stage']}。跨候选共享以软件登记的消息和证据引用为准。")
    if config.get("experiment_workers") == 1:
        notes.append("本次实验执行器并发数为 1：多个智能体可并行研究，训练任务按队列串行执行。")
    elif config.get("experiment_workers") is not None:
        notes.append(f"本次实验执行器并发上限为 {config['experiment_workers']}，不代表每项任务实际同时执行。")
    if config.get("strategy") == "independent":
        notes.append("本次采用 independent 策略；候选独立研究，主智能体汇总证据。")
    return _clean({"schema_version": "1.0", "run_id": run.get("run_id"), "track_id": track_id,
                   "research_stage": run.get("research_stage"),
                   "lanes": lanes, "nodes": nodes, "edges": edges,
                   "missing_references": missing, "notes": notes})
