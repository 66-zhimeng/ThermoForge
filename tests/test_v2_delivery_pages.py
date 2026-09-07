"""Long-message receipts and research pages are bounded and lossless on retry."""
import asyncio
import json

import pytest

from test_v2_turn_context import setup_tracks
from thermoforge_v2.codex import TurnResult
from thermoforge_v2.turn_context import acknowledge_context, build_turn_context, byte_size, visible_objective_evidence


def message(store, rid, text, **extra):
    return store.add_record(rid, "messages", {"from": "main", "to": "all", "initial_task": True,
        "text": text, **extra}, track_id="main")


def commit_context(store, rid, delivered, tid="candidate-1"):
    return store.update_track(rid, tid, acknowledge_context(store.get_track(rid, tid), delivered))


def test_partial_message_is_not_consumed_and_old_tail_is_addressable_by_id(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    text = "研究任务" * 4000
    first = message(store, rid, text)
    for i in range(40):
        message(store, rid, f"Later {i}")
    body, receipt = build_turn_context(store, rid, "candidate-1")
    assert body["messages"][0]["text"] == text[:1000]
    current = commit_context(store, rid, receipt)
    assert first["id"] not in current["processed_message_ids"]
    parts, offset = [text[:1000]], 1000
    for _ in range(3):
        response = engine._mediator(rid, "candidate-1", "research_messages", {"record_ids": [first["id"]], "offset": offset})
        assert response["ok"] and byte_size(response) <= 32 * 1024
        page = response["messages"][0]
        parts.append(page["text"])
        offset = page["text_page"]["next_offset"]
        if not page["text_page"]["has_more"]:
            break
    assert "".join(parts) == text
    assert first["id"] not in store.get_track(rid, "candidate-1")["processed_message_ids"]
    assert first["id"] in commit_context(store, rid, receipt)["processed_message_ids"]
    # Even after acknowledgment and more than 30 later messages, the ID is valid.
    response = engine._mediator(rid, "candidate-1", "research_messages", {"record_ids": [first["id"]], "offset": 15990})
    assert response["messages"][0]["text"] == text[15990:]


@pytest.mark.parametrize("status", ["completed", "interrupted", "exception"])
def test_tool_pages_are_acknowledged_only_with_completed_turn(tmp_path, status):
    async def run():
        store, rid, engine = setup_tracks(tmp_path)
        original = message(store, rid, "字" * 5000)

        class Session:
            async def turn(self, prompt):
                reply = engine._mediator(rid, "candidate-1", "research_messages", {})
                assert reply["messages"][0]["text_page"]["offset"] == 1000
                assert reply["messages"][0]["text_page"]["has_more"] is False
                if status == "exception":
                    raise RuntimeError("connection lost")
                return TurnResult(status, "Response", "T", "U")

        engine.sessions[rid, "candidate-1"] = Session()
        if status == "exception":
            with pytest.raises(RuntimeError):
                await engine._turn(rid, "candidate-1", "Read")
        else:
            await engine._turn(rid, "candidate-1", "Read")
        track = store.get_track(rid, "candidate-1")
        assert (original["id"] in (track.get("processed_message_ids") or [])) is (status == "completed")
        following, _ = build_turn_context(store, rid, "candidate-1")
        if status == "completed":
            assert not following["messages"]
        else:
            assert following["messages"][0]["text_page"]["offset"] == 0
    asyncio.run(run())


def test_out_of_order_pages_do_not_ack_a_gap_and_reference_pages_are_required(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    record = message(store, rid, "字" * 14000, evidence_ids=[f"EVIDENCE-{i}" for i in range(230)])
    _, receipt = build_turn_context(store, rid, "candidate-1")
    commit_context(store, rid, receipt)
    # Missing [1000, 6000) must prevent consumption even if the tail was fetched.
    for offset, evidence_offset in ((6000, 100), (12000, 200)):
        reply = engine._mediator(rid, "candidate-1", "research_messages", {
            "record_ids": [record["id"]], "offset": offset, "evidence_offset": evidence_offset})
        assert reply["ok"] and byte_size(reply) <= 32 * 1024
    assert record["id"] not in commit_context(store, rid, receipt)["processed_message_ids"]
    engine._mediator(rid, "candidate-1", "research_messages", {"record_ids": [record["id"]], "offset": 1000})
    assert record["id"] in commit_context(store, rid, receipt)["processed_message_ids"]


def test_message_id_reads_do_not_bypass_visibility(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    private = message(store, rid, "PRIVATE", to="candidate-2")
    late = message(store, rid, "LATE PRIVATE", initial_task=False)
    for identifier in (private["id"], late["id"], "missing"):
        response = engine._mediator(rid, "candidate-1", "research_messages", {"record_ids": [identifier]})
        assert not response["ok"] and "PRIVATE" not in json.dumps(response)


def test_four_byte_unicode_message_is_returned_under_byte_limit(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    original = message(store, rid, "🌎" * 16000, evidence_ids=[f"EVIDENCE-{i:040d}" for i in range(100)])
    response = engine._mediator(rid, "candidate-1", "research_messages", {"record_ids": [original["id"]]})
    assert response["ok"] and byte_size(response) <= 32 * 1024
    assert response["messages"][0]["text"]


def test_total_context_byte_budget_preserves_all_pending_records(tmp_path):
    store, rid, _ = setup_tracks(tmp_path)
    ids = {store.add_record(rid, "ideas", {key: "论" * 12000 for key in
        ("statement", "reason", "prediction", "falsification")}, track_id="candidate-1")["id"] for _ in range(25)}
    for _ in range(10):
        message(store, rid, "长消息" * 5000)
    seen = set()
    for _ in range(40):
        body, receipt = build_turn_context(store, rid, "candidate-1")
        assert byte_size(body) <= 32 * 1024
        selected = {row["record"]["id"] for row in body["new_evidence"]}
        assert not selected & seen
        seen |= selected
        commit_context(store, rid, receipt)
        if seen == ids:
            break
    assert seen == ids


def test_team_tool_pages_all_records_under_byte_budget(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    store.update_run(rid, {"final_report_phase": True})
    expected = {store.add_record(rid, "ideas", {"statement": "论" * 12000, "reason": "文" * 12000},
                                 track_id="candidate-1")["id"] for _ in range(110)}
    seen, offset = set(), 0
    for _ in range(30):
        response = engine._mediator(rid, "main", "research_team", {"offset": offset})
        assert response["ok"] and byte_size(response) <= 32 * 1024
        ids = {row["id"] for row in response["ideas"]}
        assert not ids & seen
        seen |= ids
        assert response["record_counts"]["ideas"] == 110
        if not response["has_more"]:
            break
        assert response["next_offset"] > offset
        offset = response["next_offset"]
    assert seen == expected


def test_candidate_objective_uses_only_explicitly_shared_baseline(tmp_path):
    baseline = {"category": "data", "estimator": "ridge"}
    store, rid, _ = setup_tracks(tmp_path, objective_mode="optimize", baseline_model=baseline)
    own = store.add_record(rid, "jobs", {"status": "completed", "request": {"model": {"category": "data", "estimator": "linear"}}}, track_id="candidate-1")
    main = store.add_record(rid, "jobs", {"status": "completed", "request": {"model": baseline}}, track_id="main")
    peer = store.add_record(rid, "jobs", {"status": "completed", "request": {"model": baseline}}, track_id="candidate-2")
    run = store.get_run(rid)
    assert [r["id"] for r in visible_objective_evidence(store, run, "candidate-1")[0]] == [own["id"]]
    message(store, rid, "共同基线", evidence_ids=[main["id"], peer["id"]])
    jobs, _ = visible_objective_evidence(store, run, "candidate-1")
    assert {r["id"] for r in jobs} == {own["id"], main["id"]}


def test_coordination_writes_return_compact_receipts_but_preserve_full_records(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    source = store.add_record(rid, "sources", {"title": "Common paper", "content": "PRIVATE_SOURCE_BODY" * 5000}, track_id="main")
    response = engine._mediator(rid, "main", "research_send_message", {
        "to": "all", "text": "任务" * 8000, "evidence_ids": [source["id"]]})
    assert response["ok"] and byte_size(response) <= 32 * 1024
    assert "PRIVATE_SOURCE_BODY" not in json.dumps(response) and "evidence" not in response["summary"]
    assert len(store.records(rid, "messages")[0]["text"]) == 16000
    decision = engine._mediator(rid, "main", "research_decision", {
        "action": "retain", "reason": "研究解释" * 4000, "evidence_ids": [source["id"]]})
    assert decision["ok"] and byte_size(decision) <= 32 * 1024
    assert len(store.records(rid, "decisions")[0]["reason"]) == 16000
    report = engine._mediator(rid, "main", "research_team_report", {
        "title": "综合报告", "summary": "结果说明" * 25000, "body": "全文" * 50000,
        "limitations": "研究局限" * 25000, "evidence_ids": [source["id"]]})
    assert report["ok"] and byte_size(report) <= 32 * 1024
    assert "body" not in report["summary"] and report["summary"]["truncated"]
    assert len(store.records(rid, "reports")[0]["body"]) == 100000
