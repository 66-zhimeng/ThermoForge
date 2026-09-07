"""The single authorization boundary for own and explicitly shared research facts."""
from __future__ import annotations


def authorized_record(store, run_id, track_id, kind, record_id, *, allow_shared=False):
    own = store.records(run_id, kind, track_id=track_id)
    for item in own:
        if item["id"] == record_id:
            return item
    if allow_shared:
        run, track = store.get_run(run_id), store.get_track(run_id, track_id)
        autonomous = run["config"].get("research_mode", "acceptance") == "autonomous"
        foreign = next((item for item in store.records(run_id, kind) if item["id"] == record_id), None)
        if foreign is not None and track.get("role") == "main":
            if not autonomous or run.get("final_report_phase") or store.autonomy_state(run_id)["sharing_ready"]:
                return foreign
        messages = [m for m in store.records(run_id, "messages")
                    if m.get("from") == "main" and m.get("to") in {track_id, "all"}
                    and record_id in (m.get("evidence_ids") or [])]
        if foreign is not None and foreign.get("track_id") == "main" and any(
                m.get("initial_task") is True for m in messages):
            return foreign
        sharing = store.autonomy_state(run_id)["sharing_ready"] if autonomous else int(track.get("turns", 0)) >= 2
        if autonomous and track.get("role") == "candidate":
            proposals = {p["id"] for p in store.records(run_id, "proposals", track_id=track_id)
                         if p.get("status") == "committed"}
            independent_done = any(j.get("proposal_id") in proposals and j.get("status") in {
                "completed", "failed", "cancelled", "interrupted"}
                for j in store.records(run_id, "jobs", track_id=track_id))
            sharing = sharing and independent_done
        if run["config"]["strategy"] != "independent" and sharing and messages and foreign is not None:
            return foreign
    raise ValueError(f"{kind} 引用不存在或不属于当前轨迹: {record_id}")
