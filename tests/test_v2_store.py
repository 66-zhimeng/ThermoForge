"""V2 状态恢复、幂等提交、并发预算和不可变证据。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from thermoforge_v2.contracts import RunConfig, V2Error
from thermoforge_v2.store import RunStore


def setup_run(tmp_path, **config):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001", **config},
                           {"fingerprint": "frozen"}, "start-one")
    store.update_run(run["run_id"], {"status": "running"})
    for i in range(6):
        store.create_track(run["run_id"], f"t{i}", "candidate")
    return store, run["run_id"]


def test_idempotent_start_and_durable_state(tmp_path):
    store, rid = setup_run(tmp_path)
    same = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001"},
                            {"fingerprint": "frozen"}, "start-one")
    assert same["run_id"] == rid
    assert RunStore(store.root).get_run(rid)["status"] == "running"
    with pytest.raises(V2Error, match="不同"):
        store.create_run({"goal_id": "RG-0001", "dataset_ref": "changed"}, {}, "start-one")


def test_atomic_budget_across_six_tracks(tmp_path):
    store, rid = setup_run(tmp_path, max_experiments=3)

    def reserve(i):
        try:
            return store.reserve_job(rid, f"t{i}", "idea", {"model": "linear"}, "one")
        except V2Error as exc:
            assert exc.code == "TFV2-BUDGET"
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [j for j in pool.map(reserve, range(6)) if j]
    assert len(jobs) == 3
    assert store.get_run(rid)["experiments_reserved"] == 3
    assert len(store.records(rid, "jobs")) == 3


def test_duplicate_job_never_trains_or_charges_twice(tmp_path):
    store, rid = setup_run(tmp_path)
    first = store.reserve_job(rid, "t0", "idea", {"model": "linear"}, "one")
    repeated = store.reserve_job(rid, "t0", "idea", {"model": "linear"}, "one")
    assert first["fresh"] is True and repeated["fresh"] is False
    assert first["id"] == repeated["id"]
    store.settle_job(first["id"], "failed", {"error": "training failed"})
    store.settle_job(first["id"], "completed", {"score": 0})
    assert store.records(rid, "jobs")[0]["status"] == "failed"
    assert store.get_run(rid)["experiments_settled"] == 1
    with pytest.raises(V2Error, match="另一请求"):
        store.reserve_job(rid, "t0", "idea", {"model": "changed"}, "one")


def test_pause_update_resume_and_optimistic_version(tmp_path):
    store, rid = setup_run(tmp_path)
    version = store.get_run(rid)["version"]
    assert store.control(rid, "pause", version)["status"] == "pausing"
    with pytest.raises(V2Error, match="状态已改变"):
        store.control(rid, "cancel", version)
    with pytest.raises(V2Error, match="不接受"):
        store.reserve_job(rid, "t0", "idea", {}, "one")
    store.update_run(rid, {"status": "paused"})
    store.control(rid, "update", changes={"guidance": "检查失败原因"})
    assert store.control(rid, "resume")["status"] == "queued"
    with pytest.raises(V2Error, match="仅可调整"):
        store.control(rid, "update", changes={"candidates": 1})


def test_source_and_protocol_are_immutable(tmp_path):
    store, rid = setup_run(tmp_path)
    source = store.add_record(rid, "sources", {"title": "paper"}, track_id="t0")
    with pytest.raises(V2Error, match="改写历史"):
        store.update_record(rid, "sources", source["id"], {"title": "other"})
    with pytest.raises(V2Error, match="不可修改"):
        store.update_run(rid, {"protocol": {}})
    with pytest.raises(V2Error, match="不能覆盖"):
        store.add_record(rid, "sources", {}, record_id=source["id"])


def test_lease_and_event_cursor(tmp_path):
    store, rid = setup_run(tmp_path)
    assert store.lease(rid, "worker1")
    assert not store.lease(rid, "worker2")
    store.release_lease(rid, "worker2")
    assert not store.lease(rid, "worker2")
    store.release_lease(rid, "worker1")
    assert store.lease(rid, "worker2")
    page = store.events(rid, limit=1)
    store.event(rid, "progress", {"text": "done"}, "t0")
    rest = store.events(rid, page["cursor"])
    assert all(e["seq"] > page["cursor"] for e in rest["events"])


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RunConfig(goal_id="RG-0001", dataset_ref="x", candidates=-1)
    with pytest.raises(ValueError):
        RunConfig(goal_id="RG-0001", dataset_ref="x", y_floor=float("inf"))
