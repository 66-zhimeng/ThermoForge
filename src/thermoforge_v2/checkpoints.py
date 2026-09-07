"""Durable research facts, independent of a successful final model response.

Only the control-plane snapshot is used. This module never opens experiment
artifacts, evaluates holdouts, adds agent reports, or changes scientific state.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping
import uuid

from .evidence_projection import safe_feedback


SETTLED = frozenset({"completed", "succeeded", "failed", "error", "cancelled", "interrupted", "timeout"})
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_NOTICE = ("系统依据已登记记录自动汇总；这不是智能体撰写的研究解释，也不替代发现、报告或停止决定。"
           "指标排名只使用 validate 反馈，诊断限于 train/validate；未读取或评价最终留出。")


def _rows(snapshot, name):
    return [deepcopy(dict(row)) for row in snapshot.get(name) or [] if isinstance(row, Mapping)]


def _object(value):
    return dict(value) if isinstance(value, Mapping) else {}


def _pick(value, names):
    return {name: deepcopy(value[name]) for name in names if name in value}


def _clean(value):
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def _validate_snapshot(snapshot):
    """Project only structured research feedback, even for an external snapshot."""
    safe = {name: _rows(snapshot, name) for name in (
        "tracks", "ideas", "proposals", "jobs", "findings", "stops", "reports", "decisions", "messages")}
    safe["run"] = deepcopy(_object(snapshot.get("run")))
    safe["sources"] = [_pick(row, ("id", "track_id", "kind", "title", "url", "doi", "read_scope",
        "verification", "retrieved_at", "content_sha256", "history_ids", "created_at"))
        for row in _rows(snapshot, "sources")]
    # Event payloads may contain external-only finalizations. A sequence number is
    # sufficient for checkpoint provenance; no payload is forwarded.
    safe["events"] = [_pick(row, ("seq",)) for row in _rows(snapshot, "events")]
    for job in safe["jobs"]:
        result = _object(job.get("result"))
        projected = safe_feedback(result)
        metrics = _object(projected.get("metrics"))
        projected["metrics"] = {str(k): v for k, v in metrics.items()
            if result.get("feedback_surface") == "validate" and job.get("status") in {"completed", "succeeded"}
            and isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)}
        if result.get("feedback_surface") != "validate":
            projected.pop("diagnostics", None)
        job["result"] = projected
    return safe


def _references(report, reports):
    """Resolve explicit report references without implying an unrecorded review."""
    by_id = {r.get("id"): r for r in reports}
    seen, pending = set(), [report]
    refs = set()
    while pending:
        current = pending.pop()
        for field in ("job_ids", "evidence_ids", "report_ids"):
            for ref in current.get(field) or []:
                if not isinstance(ref, str):
                    continue
                refs.add(ref)
                if ref in by_id and ref not in seen:
                    seen.add(ref)
                    pending.append(by_id[ref])
    return refs


def _job_fact(job):
    result = _object(job.get("result"))
    fact = _pick(job, ("id", "track_id", "idea_id", "proposal_id", "status", "executed",
        "reused_from_job_id", "duplicate_of_job_id", "experiment_fingerprint", "started_at", "finished_at"))
    fact.update(result=_clean(result), artifact_ids={
        "experiment_id": result.get("experiment_id") or job.get("experiment_id"),
        "lab_ref": _object(result.get("model")).get("lab_ref"),
        "content_hash": _object(result.get("model")).get("content_hash")})
    return fact


def _regressions(jobs, proposals, primary):
    metric = primary.get("name")
    if not metric:
        return []
    job_map = {j.get("id"): j for j in jobs}
    proposal_map = {p.get("id"): p for p in proposals}
    regressions = []
    for job in jobs:
        result = _object(job.get("result"))
        value = _object(result.get("metrics")).get(metric)
        if value is None:
            continue
        for parent_id in proposal_map.get(job.get("proposal_id"), {}).get("parent_job_ids") or []:
            parent_result = _object(job_map.get(parent_id, {}).get("result"))
            before = _object(parent_result.get("metrics")).get(metric)
            if before is None or not result.get("protocol_fingerprint") or (
                    result.get("protocol_fingerprint") != parent_result.get("protocol_fingerprint")):
                continue
            transformed = abs(value) if primary.get("transform") in {"abs", "absolute"} else value
            transformed_before = abs(before) if primary.get("transform") in {"abs", "absolute"} else before
            worse = transformed < transformed_before if primary.get("direction") == "max" else transformed > transformed_before
            if worse:
                regressions.append({"kind": "metric_regression", "job_id": job.get("id"),
                    "track_id": job.get("track_id"), "parent_job_id": parent_id, "metric": metric,
                    "before": before, "after": value, "delta": value - before,
                    "interpretation": "相同协议下主指标相较明确引用的父实验退化；这项观察本身不证明假说被证伪。"})
    return regressions


def build_checkpoint(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build a rolling, fact-only brief without mutating or completing the run."""
    from .objectives import evaluate_objective

    safe = _validate_snapshot(snapshot)
    run, jobs, findings, reports = safe["run"], safe["jobs"], safe["findings"], safe["reports"]
    objective = evaluate_objective(run, jobs, findings)
    settled = [j for j in jobs if j.get("status") in SETTLED]
    negatives = [{"kind": j.get("status"), "job_id": j.get("id"), "track_id": j.get("track_id"),
        "reason": _object(j.get("result")).get("error") or j.get("error"),
        "failure_category": _object(j.get("result")).get("failure_category") or j.get("failure_category"),
        "interpretation": "执行没有产生成功验证结果；不能据此认定假说被证伪。"}
        for j in settled if j.get("status") not in {"completed", "succeeded"}]
    negatives += [{"kind": "agent_finding", "finding_id": f.get("id"), "track_id": f.get("track_id"),
        "reason": f.get("statement"), "interpretation": f.get("interpretation"),
        "failure_category": f.get("failure_category")}
        for f in findings if f.get("failure_category") not in {None, "", "none"}]
    # Historical runs lack frozen objectives; a descriptive CVRMSE comparison
    # still preserves observed regressions without assigning a new objective.
    comparison_metric = _object(objective.get("primary_metric")) or {
        "name": "CVRMSE", "direction": "min", "transform": "identity"}
    negatives += _regressions(jobs, safe["proposals"], comparison_metric)
    tracks = []
    for track in safe["tracks"]:
        tid = track.get("track_id")
        main = track.get("role") == "main" and (bool((run.get("config") or {}).get("candidates"))
            or any(t.get("role") == "candidate" for t in safe["tracks"]))
        own_jobs = jobs if main else [j for j in jobs if j.get("track_id") == tid]
        own_settled = [j for j in own_jobs if j.get("status") in SETTLED]
        own_ideas = [i for i in safe["ideas"] if i.get("track_id") == tid]
        own_findings = [f for f in findings if f.get("track_id") == tid]
        analyses = findings if main else own_findings
        covered = {j for f in analyses for j in f.get("job_ids") or []}
        kind = "team" if main else "track"
        own_reports = [r for r in reports if r.get("track_id") == tid and r.get("kind") == kind]
        latest = own_reports[-1] if own_reports else None
        refs = _references(latest, reports) if latest else set()
        report_missing = [j["id"] for j in own_settled if j.get("id") not in refs]
        stops = [s for s in safe["stops"] if s.get("track_id") == tid]
        def timestamp(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None

        latest_result = max((v for j in own_settled if (v := timestamp(j.get("finished_at"))) is not None), default=None)
        report_time = timestamp((latest or {}).get("created_at"))
        final_current = bool(latest and latest.get("report_stage") == "final" and run.get("final_report_epoch")
                             and latest.get("final_report_epoch") == run.get("final_report_epoch"))
        tracks.append({"track_id": tid, "role": track.get("role"), "status": track.get("status"),
            "job_scope": "team" if main else "track",
            "hypotheses": [_pick(i, ("id", "statement", "origin", "reason", "prediction", "falsification",
                "source_ids", "parent_idea_ids", "parent_job_ids")) for i in own_ideas],
            "sources": [_pick(s, ("id", "title", "kind", "read_scope", "url", "doi", "history_ids"))
                for s in safe["sources"] if s.get("track_id") == tid or s.get("id") in {
                    sid for i in own_ideas for sid in i.get("source_ids") or []}],
            "settled_jobs": [_job_fact(j) for j in own_settled],
            "findings": [_pick(f, ("id", "statement", "interpretation", "limitations", "failure_category", "job_ids"))
                for f in own_findings],
            "latest_agent_report": _pick(latest, ("id", "title", "summary", "kind", "report_stage", "created_at",
                "job_ids", "finding_ids", "evidence_ids", "report_ids")) if latest else None,
            "stop_ids": [s.get("id") for s in stops],
            "objective": objective if main else evaluate_objective(run, jobs, findings, track_id=tid),
            "missing": {"unsettled_job_ids": [j.get("id") for j in own_jobs if j.get("status") not in SETTLED],
                "finding_job_ids": [j.get("id") for j in own_settled if j.get("id") not in covered],
                "agent_report": latest is None, "report_job_ids": report_missing,
                "report_predates_latest_result": latest_result is not None and (report_time is None or report_time < latest_result),
                "final_team_report": main and not final_current,
                "stop_decision": not main and (run.get("config") or {}).get("research_mode") == "autonomous" and not bool(stops)},
            "negative_results": [n for n in negatives if main or n.get("track_id") == tid]})
    checkpoint = {"schema_version": "thermoforge.research-checkpoint.v2", "run_id": run.get("run_id"),
        "status": run.get("status"), "generated_by": "system", "is_agent_report": False,
        "evidence_scope": "train_validate_only", "comparison_surface": "validate", "notice": _NOTICE,
        "protocol_fingerprint": _object(run.get("protocol")).get("fingerprint"),
        "event_cursor": max((e.get("seq", 0) for e in safe["events"] if isinstance(e.get("seq"), int)), default=0),
        "objective": objective, "counts": {"tracks": len(tracks), "settled_jobs": len(settled),
            "successful_jobs": sum(j.get("status") in {"completed", "succeeded"} for j in settled),
            "findings": len(findings), "agent_reports": len(reports), "stops": len(safe["stops"])},
        "tracks": tracks, "negative_results": negatives}
    checkpoint = _clean(checkpoint)
    checkpoint["checkpoint_id"] = "CHECKPOINT-" + hashlib.sha256(_json(checkpoint).encode("utf-8")).hexdigest()[:20]
    checkpoint["generated_at"] = datetime.now(timezone.utc).isoformat()
    return checkpoint


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("检查点的运行、轨迹和实验 ID 必须是安全的路径组件")
    return value


def _io_path(path):
    """Use Windows long-path support only at the filesystem boundary.

    Canonical artifact paths remain ordinary absolute paths for GUI, browser and
    manifest consumers. Windows otherwise rejects sufficiently deep temporary
    files even when their parent directories were created successfully.
    """
    path = Path(path)
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


@contextmanager
def _run_lock(directory):
    directory = _io_path(directory)
    key = str(directory).casefold() if os.name == "nt" else str(directory.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".export.lock").open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write(path, content):
    path = _io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_checkpoint(store, run_id: str, reason: str) -> dict[str, Any]:
    """Persist job facts and current readable reports, then publish latest.json.

    The run lock spans snapshot acquisition and the complete export. Each export
    writes a new immutable version, including when the underlying facts match a
    previous export. latest.json is published last and references only that
    version, so a failed or concurrent export cannot invalidate an old manifest.
    Top-level copies are convenience views, never canonical artifact references.
    This separate directory never overwrites an external holdout report.
    """
    from .reports import build_report, render_html, render_markdown

    directory = Path(store.root) / "runs" / _component(run_id) / "checkpoints"
    with _run_lock(directory):
        snapshot = _validate_snapshot(store.snapshot(run_id))
        checkpoint = build_checkpoint(snapshot)
        snapshot["checkpoint"] = checkpoint
        version_id = checkpoint["checkpoint_id"] + "-" + uuid.uuid4().hex[:12]
        version = directory / "versions" / version_id
        _io_path(version).mkdir(parents=True, exist_ok=False)
        artifacts, aliases = [], []

        def write(relative, content, kind):
            target = version / relative
            _atomic_write(target, content)
            aliases.append((directory / relative, content))
            artifacts.append({"kind": kind, "path": os.path.abspath(target),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})

        for job in snapshot["jobs"]:
            if job.get("status") not in SETTLED:
                continue
            idea = next((i for i in snapshot["ideas"] if i.get("id") == job.get("idea_id")), None)
            fact = {"schema_version": "thermoforge.experiment-fact.v2", "run_id": run_id,
                "generated_by": "system", "is_agent_report": False, "evidence_scope": "train_validate_only",
                "comparison_surface": "validate",
                "notice": _NOTICE, "job": _job_fact(job), "hypothesis": idea,
                "source_ids": (idea or {}).get("source_ids", [])}
            write(Path("jobs") / (_component(job.get("id")) + ".json"), _json(fact), "experiment_fact")
        for track_id in [None, *[t.get("track_id") for t in snapshot["tracks"]]]:
            stem = "team" if track_id is None else _component(track_id)
            if track_id == "team":
                raise ValueError("轨迹 ID team 与综合检查点文件名冲突")
            report = build_report(snapshot, track_id=track_id)
            report["checkpoint_id"] = checkpoint["checkpoint_id"]
            for ext, content in (("json", _json(report)), ("md", render_markdown(report)), ("html", render_html(report))):
                write(f"{stem}.{ext}", content, "research_report")
            if track_id is None:
                write("team-flow.json", _json(report["flow"]), "research_flow")
                flow_report = {**report, "title": "ThermoForge V2 持续研究流程", "sections": []}
                write("team-flow.html", render_html(flow_report), "research_flow")
        checkpoint.update(reason=str(reason), artifacts=artifacts, version_id=version_id)
        manifest = _json(checkpoint)
        _atomic_write(version / "manifest.json", manifest)
        for alias, content in aliases:
            _atomic_write(alias, content)
        _atomic_write(directory / "latest.json", manifest)
        return {"checkpoint_id": checkpoint["checkpoint_id"], "version_id": version_id,
            "generated_at": checkpoint["generated_at"],
            "reason": checkpoint["reason"], "event_cursor": checkpoint["event_cursor"],
            "path": os.path.abspath(directory / "latest.json"), "artifacts": artifacts,
            "generated_by": "system", "is_agent_report": False, "evidence_scope": "train_validate_only",
            "comparison_surface": "validate"}
