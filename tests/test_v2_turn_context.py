"""Direct delivery must reduce repeated context without acknowledging lost work."""
import asyncio
import json

import pytest

from thermoforge_v2.codex import TurnResult
from thermoforge_v2.turn_context import acknowledge_context, build_turn_context
from test_v2_engine import Scenario, make_engine


def setup_tracks(tmp_path, **config):
    store, rid, engine = make_engine(tmp_path, Scenario(), research_mode="autonomous", candidates=2, **config)
    store.begin_run(rid)
    for tid, role in (("main", "main"), ("candidate-1", "candidate"), ("candidate-2", "candidate")):
        store.create_track(rid, tid, role)
    return store, rid, engine


def test_context_obeys_independent_visibility_and_main_stage_gate(tmp_path):
    store, rid, _ = setup_tracks(tmp_path)
    own = store.add_record(rid, "ideas", {"statement": "OWN-IDEA"}, track_id="candidate-1")
    peer = store.add_record(rid, "ideas", {"statement": "PRIVATE-PEER"}, track_id="candidate-2")
    body, _ = build_turn_context(store, rid, "candidate-1")
    assert own["id"] in json.dumps(body) and "PRIVATE-PEER" not in json.dumps(body)
    main, _ = build_turn_context(store, rid, "main")
    assert peer["id"] not in json.dumps(main)
    store.update_run(rid, {"final_report_phase": True})
    main, _ = build_turn_context(store, rid, "main")
    assert peer["id"] in json.dumps(main)


def test_record_updates_are_redelivered_but_unchanged_long_reports_are_not(tmp_path):
    store, rid, _ = setup_tracks(tmp_path)
    job = store.add_record(rid, "jobs", {"status": "reserved", "idea_id": "I"}, track_id="candidate-1")
    store.add_record(rid, "reports", {"kind": "track", "summary": "Short finding", "body": "LARGE" * 30000},
                     track_id="candidate-1")
    first, delivered = build_turn_context(store, rid, "candidate-1")
    assert len(json.dumps(first)) < 16000
    assert job["id"] in json.dumps(first)
    store.update_track(rid, "candidate-1", acknowledge_context(store.get_track(rid, "candidate-1"), delivered))
    second, _ = build_turn_context(store, rid, "candidate-1")
    assert second["new_evidence"] == []
    store.update_record(rid, "jobs", job["id"], {"status": "completed", "result": {
        "protocol_fingerprint": "frozen", "feedback_surface": "validate", "metrics": {"CVRMSE": 0.1}}})
    third, _ = build_turn_context(store, rid, "candidate-1")
    assert third["new_evidence"][0]["record"]["id"] == job["id"]
    assert third["new_evidence"][0]["record"]["result"]["metrics"] == {"CVRMSE": 0.1}


def test_bounded_context_keeps_undelivered_evidence_pending(tmp_path):
    store, rid, _ = setup_tracks(tmp_path)
    expected = {store.add_record(rid, "ideas", {"statement": f"Idea {i}"}, track_id="candidate-1")["id"]
                for i in range(37)}
    seen = set()
    for _ in range(4):
        body, delivered = build_turn_context(store, rid, "candidate-1", limit=12)
        rows = {r["record"]["id"] for r in body["new_evidence"]}
        assert not rows & seen
        seen |= rows
        store.update_track(rid, "candidate-1", acknowledge_context(store.get_track(rid, "candidate-1"), delivered))
    assert seen == expected


@pytest.mark.parametrize("status", ["completed", "interrupted", "exception"])
def test_turn_ack_only_after_success_and_does_not_consume_messages_arriving_mid_turn(tmp_path, status):
    async def run():
        store, rid, engine = setup_tracks(tmp_path)
        message = store.add_record(rid, "messages", {"from": "main", "to": "all", "text": "FIRST",
            "initial_task": True}, track_id="main")
        prompts = []

        class Session:
            async def turn(self, prompt):
                prompts.append(prompt)
                store.add_record(rid, "messages", {"from": "main", "to": "all", "text": "LATER",
                    "initial_task": True}, track_id="main")
                if status == "exception":
                    raise RuntimeError("Disconnected")
                return TurnResult(status, "Result", "T", "U")

        engine.sessions[rid, "candidate-1"] = Session()
        if status == "exception":
            with pytest.raises(RuntimeError, match="Disconnected"):
                await engine._turn(rid, "candidate-1", "Proceed")
        else:
            await engine._turn(rid, "candidate-1", "Proceed")
        assert "FIRST" in prompts[0] and "LATER" not in prompts[0]
        processed = store.get_track(rid, "candidate-1").get("processed_message_ids") or []
        assert (message["id"] in processed) is (status == "completed")
        following, _ = build_turn_context(store, rid, "candidate-1")
        assert [m["text"] for m in following["messages"]] == (["LATER"] if status == "completed" else ["FIRST", "LATER"])
    asyncio.run(run())


def test_usage_events_are_incremental_and_duplicate_notifications_do_not_add_cost(tmp_path):
    store, rid, engine = setup_tracks(tmp_path)
    store.update_track(rid, "candidate-1", {"usage_generation": "G", "usage_base": {}, "usage_process": {}})
    raw = {"total": {"totalTokens": 120, "inputTokens": 100, "cachedInputTokens": 80,
                      "outputTokens": 20, "reasoningOutputTokens": 10}}
    engine._record_usage(rid, "candidate-1", raw, "G")
    engine._record_usage(rid, "candidate-1", raw, "G")
    engine._record_usage(rid, "candidate-1", {"total": {"totalTokens": 130, "outputTokens": 30}}, "G")
    rows = [e["payload"] for e in store.events(rid, limit=1000)["events"] if e["kind"] == "model.usage_increment"]
    assert [e["delta"]["totalTokens"] for e in rows] == [120, 10]
    assert rows[0]["delta"]["cachedInputTokens"] == 80
    assert store.get_run(rid)["tokens_used"] == 130
