"""Bounded, recoverable delivery; acknowledge only bytes from completed turns."""
from __future__ import annotations

import hashlib
import json

from .evidence_access import authorized_record
from .evidence_projection import compact_record

KINDS = ("jobs", "findings", "proposals", "ideas", "reports", "stops", "sources")
MAX_CONTEXT_BYTES = 32 * 1024


def byte_size(value):
    return len(json.dumps(value, ensure_ascii=False, default=str).encode())


def visible_messages(store, run, track):
    rid, tid = run["run_id"], track["track_id"]
    rows = [m for m in store.records(rid, "messages") if m.get("to") in {tid, "all"}]
    autonomous = run["config"].get("research_mode", "acceptance") == "autonomous"
    sharing = store.autonomy_state(rid)["sharing_ready"] if autonomous else track["turns"] >= 2
    if autonomous and tid != "main":
        proposals = {p["id"] for p in store.records(rid, "proposals", tid) if p.get("status") == "committed"}
        sharing = sharing and any(j.get("proposal_id") in proposals and j["status"] in {
            "completed", "failed", "cancelled", "interrupted"} for j in store.records(rid, "jobs", tid))
    if tid != "main" and (not sharing or run["config"]["strategy"] == "independent"):
        rows = [m for m in rows if m.get("initial_task") is True]
    return rows


def visible_objective_evidence(store, run, tid):
    """Own results plus explicitly authorized exact baseline, never a peer rank."""
    from .objectives import _model
    rid, state = run["run_id"], store.autonomy_state(run["run_id"])
    if tid == "main" and (not state["enabled"] or state["sharing_ready"] or run.get("final_report_phase")):
        return store.records(rid, "jobs"), store.records(rid, "findings")
    jobs, findings = store.records(rid, "jobs", tid), store.records(rid, "findings", tid)
    baseline = (run.get("objective_contract") or {}).get("baseline_model")
    if baseline:
        own_ids = {j["id"] for j in jobs}
        for job in store.records(rid, "jobs"):
            if job["id"] in own_ids:
                continue
            model = (job.get("request") or {}).get("model") or ((job.get("result") or {}).get("model") or {}).get("spec")
            if _model(model) != baseline:
                continue
            try:
                jobs.append(authorized_record(store, rid, tid, "jobs", job["id"], allow_shared=True))
            except ValueError:
                pass
    return jobs, findings


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _message_signature(message):
    return _signature([message.get("text", ""), message.get("evidence_ids") or []])


def merge_message_ranges(*maps):
    merged = {}
    for values in maps:
        for identifier, item in (values or {}).items():
            previous = merged.get(identifier)
            if previous is None or previous.get("signature") != item.get("signature"):
                previous = {"signature": item["signature"], "total_chars": item["total_chars"], "ranges": [],
                            "reference_count": item.get("reference_count", 0), "reference_ranges": []}
            updated = dict(previous)
            for field in ("ranges", "reference_ranges"):
                compacted = []
                for start, end in sorted(previous.get(field, []) + item.get(field, [])):
                    if compacted and start <= compacted[-1][1]:
                        compacted[-1][1] = max(compacted[-1][1], end)
                    else:
                        compacted.append([start, end])
                updated[field] = compacted
            merged[identifier] = updated
    return merged


def _first_unread(message, ranges, *, references=False):
    item = ranges.get(message["id"], {})
    if item.get("signature") != _message_signature(message):
        return 0
    end = 0
    for start, stop in item.get("reference_ranges" if references else "ranges", []):
        if start > end:
            break
        end = max(end, stop)
    return min(end, len(message.get("evidence_ids") or []) if references else len(message.get("text", "")))


def _processed(track, message, ranges):
    known = ranges.get(message["id"])
    if known is not None and known.get("signature") == _message_signature(message):
        return (_first_unread(message, ranges) >= len(message.get("text", ""))
                and _first_unread(message, ranges, references=True) >= len(message.get("evidence_ids") or []))
    return known is None and message["id"] in (track.get("processed_message_ids") or [])


def _message_page(message, offset, chars, *, evidence_offset=0):
    body = message.get("text", "")
    start, end = min(offset, len(body)), min(offset + chars, len(body))
    references = message.get("evidence_ids") or []
    evidence_start, evidence_end = min(evidence_offset, len(references)), min(evidence_offset + 100, len(references))
    result = compact_record("messages", message, fields=["from", "to", "version", "initial_task"])
    result.update(text=body[start:end], text_page={"offset": start, "next_offset": end,
                  "total_chars": len(body), "has_more": end < len(body)})
    result.update(evidence_ids=references[evidence_start:evidence_end], evidence_page={"offset": evidence_start,
        "next_offset": evidence_end, "total": len(references), "has_more": evidence_end < len(references)})
    result["read_hint"] = "完整消息用 research_messages(record_ids=[id], offset=text_page.next_offset, evidence_offset=evidence_page.next_offset)；完整投递后确认处理。"
    receipt = {message["id"]: {"signature": _message_signature(message), "total_chars": len(body), "ranges": [[start, end]],
                              "reference_count": len(references), "reference_ranges": [[evidence_start, evidence_end]]}}
    return result, receipt


def read_message_page(store, run, track, args):
    """Authorized old or unread messages with a byte budget and pending receipts."""
    if set(args) - {"include_read", "record_ids", "offset", "evidence_offset"}:
        raise ValueError("未知消息参数")
    include_read = args.get("include_read", False)
    if not isinstance(include_read, bool):
        raise ValueError("include_read 必须为布尔值")
    visible = visible_messages(store, run, track)
    receipts = merge_message_ranges((track.get("context_ack") or {}).get("message_ranges"), track.get("delivered_message_ranges"))
    ids = args.get("record_ids")
    if ids is not None:
        if not isinstance(ids, list) or not ids or len(ids) > 50 or any(not isinstance(v, str) for v in ids):
            raise ValueError("record_ids 必须为 1–50 个消息 ID")
        by_id = {m["id"]: m for m in visible}
        if any(identifier not in by_id for identifier in ids):
            raise ValueError("消息 ID 不存在或当前不可见")
        rows = [by_id[identifier] for identifier in dict.fromkeys(ids)]
    else:
        committed = (track.get("context_ack") or {}).get("message_ranges") or {}
        tool_delivered = set(track.get("delivered_message_ids") or [])
        rows = visible[-30:] if include_read else [m for m in visible
            if not _processed(track, m, committed) and m["id"] not in tool_delivered]
    offset = args.get("offset")
    if offset is not None and (ids is None or len(ids) != 1 or not isinstance(offset, int) or isinstance(offset, bool) or offset < 0):
        raise ValueError("offset 需要单个 record_ids 和非负整数")
    evidence_offset = args.get("evidence_offset")
    if evidence_offset is not None and (ids is None or len(ids) != 1 or not isinstance(evidence_offset, int)
                                        or isinstance(evidence_offset, bool) or evidence_offset < 0):
        raise ValueError("evidence_offset 需要单个 record_ids 和非负整数")
    response = {"ok": True, "messages": [], "new_message_ids": [], "remaining_message_ids": [], "more_messages": 0,
                "read_hint": "未完整投递的消息继续按 ID/offset 读取；没有 record_ids 时继续读取未读消息。"}
    new_receipts = {}
    for message in rows[:30]:
        start = offset if offset is not None else (0 if include_read or ids is not None else _first_unread(message, receipts))
        if offset is None and ids is None and _processed(track, message, receipts):
            # A direct brief is not an acknowledgment. Preserve the legacy tool
            # read of a short, still-unconfirmed message when explicitly asked.
            start = 0
        ref_start = evidence_offset if evidence_offset is not None else (0 if include_read or ids is not None
                    else _first_unread(message, receipts, references=True))
        chars = 6000
        projected, receipt = _message_page(message, start, chars, evidence_offset=ref_start)
        response["messages"].append(projected)
        while len(response["messages"]) == 1 and byte_size(response) > MAX_CONTEXT_BYTES - 6000 and chars > 1:
            chars //= 2
            projected, receipt = _message_page(message, start, chars, evidence_offset=ref_start)
            response["messages"][-1] = projected
        if byte_size(response) > MAX_CONTEXT_BYTES - 6000:
            response["messages"].pop()
            break
        new_receipts = merge_message_ranges(new_receipts, receipt)
        if not _processed(track, message, (track.get("context_ack") or {}).get("message_ranges") or {}):
            response["new_message_ids"].append(message["id"])
    remaining = rows[len(response["messages"]):]
    response["remaining_message_ids"], response["more_messages"] = [m["id"] for m in remaining[:50]], len(remaining)
    all_pending = merge_message_ranges(track.get("delivered_message_ranges"), new_receipts)
    all_coverage = merge_message_ranges((track.get("context_ack") or {}).get("message_ranges"), all_pending)
    full_ids = [m["id"] for m in rows[:len(response["messages"])] if _processed(track, m, all_coverage)]
    return response, {"delivered_message_ranges": all_pending,
        "delivered_message_ids": list(dict.fromkeys((track.get("delivered_message_ids") or []) + full_ids))}


def build_team_page(store, run, tracks, args):
    """Page the main agent's already-authorized team view without a huge replay."""
    from .strategy import strategy_advice
    offset = args.get("offset", 0)
    if set(args) - {"offset"} or not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("团队分页 offset 必须为非负整数")
    rid = run["run_id"]
    kinds = ("jobs", "findings", "ideas", "sources", "reports", "decisions", "proposals", "stops")
    rows = {kind: store.records(rid, kind) for kind in kinds}
    flattened = sorted(((kind, row) for kind, records in rows.items() for row in records),
                       key=lambda pair: (pair[1].get("created_at", 0), pair[1]["id"]))
    advice = strategy_advice(rows["jobs"], run["config"], run["protocol"]["fingerprint"])
    comparison = advice.get("comparison") or {}
    advice = {key: advice[key] for key in ("configured_strategy", "suggested_action", "stagnant", "advice_scope",
        "reason", "requires_scientific_decision", "duplicate_experiment_count") if key in advice} | {
        "comparison": {key: comparison[key] for key in ("metric", "surface", "lower_is_better", "reason") if key in comparison},
        "top_k": (comparison.get("top_k") or [])[:5], "ranked_count": len(comparison.get("ranked") or [])}
    response = {"ok": True, **{kind: [] for kind in kinds},
        "tracks": [{key: (str(t[key])[:320] if key == "error" and t[key] is not None else t[key])
                    for key in ("track_id", "status", "phase", "turns", "usage", "error") if key in t} for t in tracks],
        "record_counts": {kind: len(records) for kind, records in rows.items()},
        "offset": offset, "next_offset": offset, "has_more": False, "remaining_record_ids": [],
        "read_hint": "继续 research_team(offset=next_offset) 获取未投递证据；按 ID 的完整记录用 research_history(record_ids,fields,compact=false)。",
        "research_closure": run.get("research_closure"), "research_stage": store.autonomy_state(rid)["stage"],
        "strategy": advice}
    end = min(offset, len(flattened))
    for kind, row in flattened[end:end + 50]:
        response[kind].append(compact_record(kind, row))
        if byte_size(response) > MAX_CONTEXT_BYTES - 5000:
            response[kind].pop()
            break
        end += 1
    response.update(next_offset=end, has_more=end < len(flattened),
                    remaining_record_ids=[row["id"] for _, row in flattened[end:end + 50]])
    return response


def build_turn_context(store, rid, tid, *, limit=16):
    run, track = store.get_run(rid), store.get_track(rid, tid)
    state, protocol = store.autonomy_state(rid), run["protocol"]
    ack, pending = track.get("context_ack") or {}, []
    previous = ack.get("records") or {}
    team_visible = tid == "main" and (not state["enabled"] or state["sharing_ready"] or run.get("final_report_phase"))
    for kind in KINDS:
        for row in store.records(rid, kind, None if team_visible else tid):
            projected = compact_record(kind, row)
            key, digest = kind + ":" + row["id"], _signature(projected)
            if previous.get(key) != digest:
                pending.append((kind, row, projected, key, digest))
    pending.sort(key=lambda p: (p[1].get("finished_at") or p[1].get("created_at") or 0, p[1]["id"]), reverse=True)
    goal = protocol.get("goal_definition") or {}
    hint = {"fingerprint": protocol.get("fingerprint"), "feedback_surface": "validate", "holdout_feedback_allowed": False}
    if ack.get("protocol_fingerprint") != protocol.get("fingerprint") or not ack:
        hint.update(goal={k: goal[k] for k in ("goal_id", "name", "purpose", "target", "candidate_inputs", "acceptance") if k in goal},
                    metrics=protocol.get("metrics"), validation=protocol.get("validation"),
                    note="这是协议摘要；实现模型前按需 research_protocol 获取完整接口。")
    config, own_jobs = run["config"], store.records(rid, "jobs", tid)
    envelope = {"protocol": hint, "objective_contract": run.get("objective_contract"), "research_stage": state["stage"],
        "budget": {"tokens_remaining": max(0, config["token_budget"] - run["tokens_used"]),
                   "turns_remaining": max(0, config["max_turns"] - track["turns"]),
                   "experiments_remaining": max(0, min(config["max_experiments"] - run["experiments_reserved"],
                                                       config["max_experiments_per_track"] - len(own_jobs))),
                   "closure": run.get("research_closure")},
        "messages": [], "new_evidence": [], "more_evidence": len(pending), "more_messages": 0,
        "note": "软件投递的事实与已保存解释，不是新实验结论。无需重复查询已知内容；截断证据按 ID 查，长消息按 ID/offset 查。"}
    assessment = {}
    if run.get("objective_contract"):
        from .objectives import evaluate_objective
        jobs, findings = visible_objective_evidence(store, run, tid)
        assessment = evaluate_objective(run, jobs, findings, track_id=None if team_visible else tid)
        envelope["objective_status"] = {k: v for k, v in assessment.items() if k != "per_job"}
    if byte_size(envelope) > 16 * 1024:
        envelope["protocol"] = {"fingerprint": protocol.get("fingerprint"), "feedback_surface": "validate",
            "holdout_feedback_allowed": False, "truncated": True, "read_hint": "通过 research_protocol 获取完整冻结协议。"}
        if byte_size(envelope) > 16 * 1024:
            contract = run.get("objective_contract") or {}
            envelope["objective_contract"] = {k: contract.get(k) for k in ("fingerprint", "objective_mode", "primary_metric", "min_improvement")}
            envelope["objective_contract"]["truncated"] = True
            envelope["objective_status"] = {k: assessment.get(k) for k in ("recommended_action", "acceptance_status", "baseline")}
    ranges = ack.get("message_ranges") or {}
    messages = [m for m in visible_messages(store, run, track) if not _processed(track, m, ranges)]
    message_receipts, message_ids = {}, []
    first_evidence_bytes = byte_size({"kind": pending[0][0], "record": pending[0][2]}) if pending else 0
    message_budget = min(24 * 1024, MAX_CONTEXT_BYTES - first_evidence_bytes - 1024)
    for message in messages[:20]:
        projected, receipt = _message_page(message, _first_unread(message, ranges), 1000,
                                           evidence_offset=_first_unread(message, ranges, references=True))
        envelope["messages"].append(projected)
        if byte_size(envelope) > message_budget:
            envelope["messages"].pop()
            break
        message_receipts = merge_message_ranges(message_receipts, receipt)
        message_ids.append(message["id"])
    envelope["more_messages"] = len(messages) - len(message_ids)
    new_ack = {}
    for kind, _, projected, key, digest in pending[:max(1, limit)]:
        envelope["new_evidence"].append({"kind": kind, "record": projected})
        if byte_size(envelope) > MAX_CONTEXT_BYTES - 512:
            envelope["new_evidence"].pop()
            break
        new_ack[key] = digest
    envelope["more_evidence"] = len(pending) - len(new_ack)
    return envelope, {"records": new_ack, "message_ids": message_ids, "message_ranges": message_receipts,
                      "protocol_fingerprint": protocol.get("fingerprint")}


def acknowledge_context(track, delivered):
    old = track.get("context_ack") or {}
    ranges = merge_message_ranges(old.get("message_ranges"), delivered.get("message_ranges"), track.get("delivered_message_ranges"))
    complete = [identifier for identifier, item in ranges.items()
                if item["ranges"] and item["ranges"][0][0] == 0 and item["ranges"][0][1] >= item["total_chars"]
                and (item.get("reference_count", 0) == 0 or item.get("reference_ranges")
                     and item["reference_ranges"][0][0] == 0 and item["reference_ranges"][0][1] >= item["reference_count"])]
    legacy = delivered.get("message_ids", []) if "message_ranges" not in delivered else []
    return {"context_ack": {"records": {**(old.get("records") or {}), **delivered["records"]},
                            "message_ranges": ranges, "protocol_fingerprint": delivered["protocol_fingerprint"]},
            "processed_message_ids": list(dict.fromkeys((track.get("processed_message_ids") or [])
                + complete + legacy + (track.get("delivered_message_ids") or [])))}
