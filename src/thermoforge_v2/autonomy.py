"""Transactional research stages and immutable pre-experiment proposals."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid

from .contracts import V2Error


SETTLED = {"completed", "failed", "cancelled", "interrupted"}
EXEMPT = {"completed", "failed", "cancelled", "budget_exhausted"}


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _records(db, run_id, kind, track_id=None):
    query = "SELECT data FROM records WHERE run_id=? AND kind=?"
    args = [run_id, kind]
    if track_id is not None:
        query += " AND track_id=?"
        args.append(track_id)
    return [json.loads(r[0]) for r in db.execute(query + " ORDER BY rowid", args)]


class AutonomyMixin:
    def _report_reserve(self, db, run, tracks):
        """Estimate remaining report calls; usage.last is a measurement, not a cost guarantee."""
        budget = int(run["config"]["token_budget"])
        used = max(0, int(run.get("tokens_used") or 0))
        remaining = max(0, budget - used)
        stopped = {item["track_id"] for item in _records(db, run["run_id"], "stops")}

        def measured_last(track):
            last = (track.get("usage") or {}).get("last") or {}
            if not isinstance(last, dict):
                return 0

            def counter(name):
                value = last.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
                return int(value) if isinstance(value, float) and math.isfinite(value) and value >= 0 else 0

            # Cached input is already included in inputTokens; never add it twice.
            return max(counter("totalTokens"), counter("inputTokens") + counter("outputTokens"))

        observed = {track["track_id"]: measured_last(track) for track in tracks}
        candidates = [track for track in tracks if track.get("role") == "candidate"]
        largest_context = max((observed[track["track_id"]] for track in candidates), default=0)
        calls = {}
        for track in tracks:
            tid = track["track_id"]
            if tid in stopped:
                continue
            estimate = observed[tid]
            if track.get("role") == "main":
                estimate = max(estimate, largest_context)
            calls[tid] = estimate
        floor = (budget + 3) // 4
        usage_reserve = (sum(calls.values()) * 24 + 4) // 5  # ceil(4 calls × 1.2 margin)
        reserve = max(floor, usage_reserve)
        closure = run.get("research_closure") or {}
        existing = closure.get("reason") == "token_report_reserve"
        increased = existing and budget > int(closure.get("token_budget") or 0)
        should_close = remaining <= reserve
        eligible = run.get("status") not in {"completed", "cancelled"}
        if eligible and ((not existing and should_close) or increased):
            if increased and not should_close:
                previous = closure
                run.pop("research_closure", None)
                self._write_run(db, run)
                self._event(db, run["run_id"], "research.closure_released", {
                    "reason": "token_budget_increased", "previous_token_budget": previous["token_budget"],
                    "token_budget": budget, "tokens_used": used, "reserve_tokens": reserve})
            else:
                run["research_closure"] = {
                    "reason": "token_report_reserve", "token_budget": budget, "tokens_used": used,
                    "tokens_remaining": remaining, "reserve_tokens": reserve,
                    "minimum_reserve_tokens": floor, "usage_reserve_tokens": usage_reserve,
                    "estimated_calls_per_track": 4, "safety_multiplier": 1.2,
                    "observed_last_call_tokens_by_track": observed,
                    "estimated_call_tokens_by_track": calls,
                    "estimation_method": "max(25% budget, 4 calls per unfinished track with 20% margin); main uses at least largest candidate context",
                    "estimate_note": "报告额度为基于实际最后一次用量的保守估算，不保证足够；未知用量只使用预算比例。",
                    "triggered_at": closure.get("triggered_at", time.time()),
                    "evaluated_at": time.time(),
                }
                self._write_run(db, run)
                self._event(db, run["run_id"], "research.closure_reassessed" if increased else "research.closure_started",
                            run["research_closure"])
        current = run.get("research_closure") or {}
        closing = current.get("reason") == "token_report_reserve"
        return {"closing_for_tokens": closing,
                "report_token_reserve": max(reserve, int(current.get("reserve_tokens") or 0)) if closing else reserve,
                "tokens_remaining": remaining}

    def _require_research_budget(self, db, run):
        state = self._advance_autonomy(db, run)
        if state["closing_for_tokens"]:
            # This guard precedes any new proposal/job write. Preserve the closure
            # decision and event even though the caller receives a rejected request.
            db.commit()
            raise V2Error("TFV2-REPORT-RESERVE", "已为报告收尾预留 token；请保存发现、完整报告和停止依据，不再提交新方案或实验")
        return state

    def _append_autonomy(self, db, run_id, track_id, kind, body):
        prefix = {"proposals": "PROPOSAL", "stops": "STOP"}[kind]
        doc = {**body, "id": prefix + "-" + uuid.uuid4().hex[:16], "run_id": run_id,
               "track_id": track_id, "created_at": time.time()}
        db.execute("INSERT INTO records VALUES(?,?,?,?,?)", (doc["id"], run_id, kind, track_id, _encode(doc)))
        self._event(db, run_id, kind + ".created", {"id": doc["id"]}, track_id)
        return doc

    def _advance_autonomy(self, db, run):
        if run["config"].get("research_mode", "acceptance") != "autonomous":
            return {"enabled": False, "stage": "acceptance", "proposal_barrier_open": True,
                    "sharing_ready": False, "closing_for_tokens": False, "report_token_reserve": 0,
                    "tokens_remaining": max(0, int(run["config"]["token_budget"]) - int(run.get("tokens_used") or 0))}
        rid = run["run_id"]
        tracks = [json.loads(r[0]) for r in db.execute("SELECT data FROM tracks WHERE run_id=?", (rid,))]
        expected = ([f"candidate-{i}" for i in range(1, run["config"]["candidates"] + 1)]
                    if run["config"]["candidates"] else ["main"])
        by_track = {t["track_id"]: t for t in tracks}
        proposed = {p["track_id"] for p in _records(db, rid, "proposals")}
        settled = {j["track_id"] for j in _records(db, rid, "jobs") if j["status"] in SETTLED}
        stopped = {s["track_id"] for s in _records(db, rid, "stops")}
        exempt = {t["track_id"] for t in tracks if t["status"] in EXEMPT} | stopped
        ready = all(t in by_track and t in proposed | exempt for t in expected)
        stage = run.get("research_stage", "independent_proposals")
        if stage == "independent_proposals" and ready:
            stage = "independent_experiments"
        if (stage == "independent_experiments" and run["config"]["strategy"] != "independent"
                and all(t in settled | exempt for t in expected) and settled):
            stage = "sharing"
        if stage != run.get("research_stage"):
            run["research_stage"] = stage
            self._write_run(db, run)
            self._event(db, rid, "research.stage_changed", {"stage": stage})
        return {"enabled": True, "stage": stage, "proposal_barrier_open": stage != "independent_proposals",
                "sharing_ready": stage == "sharing", **self._report_reserve(db, run, tracks)}

    def autonomy_state(self, run_id):
        with self._db() as db:
            return self._advance_autonomy(db, self._get(db, "runs", "id", run_id))

    def commit_proposal(self, run_id, track_id, idea_id, model, details, idempotency_key):
        if not isinstance(idempotency_key, str) or not 0 < len(idempotency_key) <= 200:
            raise V2Error("TFV2-INPUT", "提案需要有效幂等键")
        purpose = details.get("purpose", "explore")
        if purpose not in {"explore", "refine", "replicate"}:
            raise V2Error("TFV2-INPUT", "无效提案目的")
        explicit = {key: details.get(key) for key in ("purpose", "expected_cost", "lab_content_hash", "experiment_fingerprint")}
        digest = hashlib.sha256(_encode({"idea_id": idea_id, "model": model, "details": explicit}).encode()).hexdigest()
        with self._db() as db:
            run = self._get(db, "runs", "id", run_id)
            proposals = _records(db, run_id, "proposals", track_id)
            existing = next((p for p in proposals if p.get("idempotency_key") == idempotency_key), None)
            if existing:
                if existing["request_hash"] != digest:
                    raise V2Error("TFV2-CONFLICT", "提案幂等键已用于其他内容")
                return existing | {"fresh": False}
            if run["status"] != "running" or _records(db, run_id, "stops", track_id):
                raise V2Error("TFV2-STATE", "轨迹已停止或研究不接受提案")
            self._require_research_budget(db, run)
            if not db.execute("SELECT 1 FROM tracks WHERE run_id=? AND id=?", (run_id, track_id)).fetchone():
                raise V2Error("TFV2-NOT-FOUND", track_id)
            idea = next((i for i in _records(db, run_id, "ideas", track_id) if i["id"] == idea_id), None)
            if not idea or idea.get("protocol_fingerprint") != run["protocol"]["fingerprint"]:
                raise V2Error("TFV2-INPUT", "提案需引用本轨迹同协议的已登记想法")
            if any(p["idea_id"] == idea_id for p in proposals):
                raise V2Error("TFV2-IMMUTABLE", "想法已有冻结提案；修订时创建有历史依据的新想法")
            jobs = _records(db, run_id, "jobs", track_id)
            if any(j["status"] not in SETTLED for j in jobs):
                raise V2Error("TFV2-STATE", "先等待正在执行的实验反馈")
            if proposals and not any(j.get("proposal_id") == proposals[-1]["id"] for j in jobs):
                raise V2Error("TFV2-STATE", "先执行上一份冻结提案，或记录不能执行的原因并停止")
            parents = list(idea.get("parent_job_ids") or [])
            findings = _records(db, run_id, "findings", track_id)
            if jobs:
                latest = jobs[-1]
                if latest["id"] not in parents:
                    raise V2Error("TFV2-INPUT", "修订提案必须引用自己最近一次真实实验反馈")
                if not any(latest["id"] in (f.get("job_ids") or []) for f in findings):
                    raise V2Error("TFV2-INPUT", "修订前先保存最近实验的发现与结果分析")
            from .experiment_identity import experiment_fingerprint
            fingerprint = experiment_fingerprint(run["protocol"]["fingerprint"], model, details.get("lab_content_hash"))
            if details.get("experiment_fingerprint") not in {None, fingerprint}:
                raise V2Error("TFV2-CONFLICT", "提案实验指纹不一致")
            if purpose != "replicate" and any(p["experiment_fingerprint"] == fingerprint for p in proposals):
                raise V2Error("TFV2-DUPLICATE", "相同实验方案已提交；有复现理由时使用 replicate，否则修订方法或参数")
            doc = self._append_autonomy(db, run_id, track_id, "proposals", {
                "idea_id": idea_id, "model": model, "version": len(proposals) + 1, "status": "committed",
                "purpose": purpose, "expected_cost": details.get("expected_cost"),
                "parent_job_ids": parents, "finding_ids": [f["id"] for f in findings
                    if set(f.get("job_ids") or []) & set(parents)],
                "lab_content_hash": details.get("lab_content_hash"), "experiment_fingerprint": fingerprint,
                "protocol_fingerprint": run["protocol"]["fingerprint"],
                "idempotency_key": idempotency_key, "request_hash": digest,
                "research_stage": run.get("research_stage", "independent_proposals"),
            })
            self._advance_autonomy(db, run)
            return doc | {"fresh": True}

    def request_research_stop(self, run_id, track_id, reason, evidence_ids):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 12000:
            raise V2Error("TFV2-INPUT", "停止需给出具体研究理由")
        with self._db() as db:
            run = self._get(db, "runs", "id", run_id)
            previous = _records(db, run_id, "stops", track_id)
            if previous:
                return previous[-1]
            if run["status"] != "running":
                raise V2Error("TFV2-STATE", "研究当前不接受停止决定")
            own = {r["id"]: r for kind in ("ideas", "jobs", "findings", "reports", "proposals", "sources")
                   for r in _records(db, run_id, kind, track_id)}
            if not isinstance(evidence_ids, list) or not evidence_ids or any(i not in own for i in evidence_ids):
                raise V2Error("TFV2-INPUT", "停止决定必须引用本轨迹已登记证据")
            jobs = _records(db, run_id, "jobs", track_id)
            if any(j["status"] not in SETTLED for j in jobs):
                raise V2Error("TFV2-STATE", "实验尚未结束，不能提前结题")
            required = {j["id"] for j in jobs}
            reports = _records(db, run_id, "reports", track_id)
            latest = max((j.get("finished_at", 0) for j in jobs), default=0)
            analyses = _records(db, run_id, "findings", track_id)
            report = next((r for r in reversed(reports) if r.get("kind") == "track"
                           and required.issubset(r.get("job_ids") or []) and r["created_at"] >= latest
                           and {j["idea_id"] for j in jobs}.issubset(r.get("idea_ids") or [])
                           and all(any(f["id"] in (r.get("finding_ids") or []) and j["id"] in (f.get("job_ids") or [])
                                       for f in analyses) for j in jobs)), None)
            if report is None:
                raise V2Error("TFV2-INPUT", "停止前提交覆盖全部已结算实验、想法及发现分析的轨迹报告")
            doc = self._append_autonomy(db, run_id, track_id, "stops", {
                "reason": reason.strip(), "evidence_ids": list(dict.fromkeys(evidence_ids + [report["id"]])),
                "job_ids": sorted(required), "research_stage": run.get("research_stage"),
            })
            self._advance_autonomy(db, run)
        self.save_checkpoint(run_id, "research.stopped")
        return doc
