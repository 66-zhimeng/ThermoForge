"""V2 状态恢复、幂等提交、并发预算和不可变证据。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3

import pytest

from thermoforge_v2.contracts import RunConfig, V2Error
from thermoforge_v2.store import RunStore


def setup_run(tmp_path, **config):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001", "research_mode": "acceptance", **config},
                           {"fingerprint": "frozen"}, "start-one")
    store.update_run(run["run_id"], {"status": "running"})
    for i in range(6):
        store.create_track(run["run_id"], f"t{i}", "candidate")
    return store, run["run_id"]


def test_idempotent_start_and_durable_state(tmp_path):
    store, rid = setup_run(tmp_path)
    same = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001", "research_mode": "acceptance"},
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
    with pytest.raises(ValueError):
        RunConfig(goal_id="RG-0001", dataset_ref="x", experiment_workers=2)


def test_queued_pause_wins_over_delayed_scheduler(tmp_path):
    store = RunStore(tmp_path / "state")
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "D@rev_0001", "research_mode": "acceptance"},
                           {"fingerprint": "frozen"}, "one")
    rid = run["run_id"]
    store.control(rid, "pause")
    assert not store.begin_run(rid)["started"]
    assert store.get_run(rid)["status"] == "paused"
    store.control(rid, "resume")
    assert store.begin_run(rid)["started"]
    assert not store.begin_run(rid)["started"]


def test_reserved_job_does_not_start_after_pause(tmp_path):
    store, rid = setup_run(tmp_path)
    job = store.reserve_job(rid, "t0", "idea", {}, "one")
    store.control(rid, "pause")
    result = store.start_job(job["id"])
    assert not result["started"]
    assert result["result"]["training_started"] is False
    assert result["result"]["reservation_consumed"] is True
    assert store.get_run(rid)["experiments_settled"] == 1
    assert not store.start_job(job["id"])["started"]
    assert store.get_run(rid)["experiments_settled"] == 1


def legacy_run(tmp_path, **overrides):
    """写入旧版完整 config/hash，不通过当前 create_run 自动补新字段。"""
    store = RunStore(tmp_path / "legacy-state")
    request = {"goal_id": "RG-0001", "dataset_ref": "D@rev_0001",
               "guidance": "保留初始验收范围", **overrides}
    config = RunConfig.model_validate(request | {"research_mode": "acceptance"}).model_dump(mode="json")
    config.pop("research_mode")
    config.pop("reuse_experiments")

    def encode_old(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))

    request_hash = hashlib.sha256(encode_old(config).encode()).hexdigest()
    run = {"run_id": "RUN-legacy", "config": config, "protocol": {"fingerprint": "legacy-frozen"},
           "status": "running", "version": 3, "created_at": 1.0, "updated_at": 1.0,
           "error": None, "experiments_reserved": 0, "experiments_settled": 0, "tokens_used": 0}
    with sqlite3.connect(store.path) as db:
        db.execute("INSERT INTO runs VALUES(?,?,?,?)",
                   (run["run_id"], "legacy-start", request_hash, encode_old(run)))
    return RunStore(store.root), request, run, request_hash


def retry_legacy(store, operation, request):
    if operation == "find_request":
        return store.find_request(request, "legacy-start")
    return store.create_run(request, {"fingerprint": "new-environment-must-not-replace-old"}, "legacy-start")


@pytest.mark.parametrize("operation", ["find_request", "create_run"])
def test_legacy_start_retry_uses_old_hash_without_migrating_research_mode(tmp_path, operation):
    store, request, original, old_hash = legacy_run(tmp_path)
    repeated = retry_legacy(store, operation, request)
    assert repeated == original
    assert "research_mode" not in repeated["config"]
    assert "reuse_experiments" not in repeated["config"]
    assert repeated["protocol"]["fingerprint"] == "legacy-frozen"
    assert store.autonomy_state(repeated["run_id"])["enabled"] is False
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT request_hash FROM runs WHERE id=?", (original["run_id"],)).fetchone()[0] == old_hash
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


@pytest.mark.parametrize("operation", ["find_request", "create_run"])
def test_legacy_guidance_update_keeps_acceptance_and_original_start_retry(tmp_path, operation):
    # 旧验收允许两回合；重新套用自主默认值会拒绝这份原本有效的运行。
    store, request, original, old_hash = legacy_run(tmp_path, max_turns=2)
    updated = store.control(original["run_id"], "update", expected_version=original["version"],
                            changes={"guidance": "补充已有实验的失败分析"})
    assert updated["config"]["research_mode"] == "acceptance"
    assert updated["config"]["reuse_experiments"] is False
    assert updated["config"]["max_turns"] == 2
    assert updated["config"]["guidance"] == "补充已有实验的失败分析"
    assert store.autonomy_state(original["run_id"])["enabled"] is False
    reopened = RunStore(store.root)
    repeated = retry_legacy(reopened, operation, request)
    assert repeated == updated
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT request_hash FROM runs WHERE id=?", (original["run_id"],)).fetchone()[0] == old_hash


@pytest.mark.parametrize("operation", ["find_request", "create_run"])
@pytest.mark.parametrize("new_options", [{"research_mode": "autonomous"}, {"reuse_experiments": True}])
def test_explicit_autonomy_or_reuse_cannot_reuse_legacy_start_key(tmp_path, operation, new_options):
    store, request, original, _ = legacy_run(tmp_path, strategy="top_k")
    with pytest.raises(V2Error) as conflict:
        retry_legacy(store, operation, request | new_options)
    assert conflict.value.code == "TFV2-CONFLICT"
    assert store.get_run(original["run_id"]) == original


@pytest.mark.parametrize("operation", ["find_request", "create_run"])
def test_legacy_compatibility_does_not_accept_changed_start_request(tmp_path, operation):
    store, request, original, _ = legacy_run(tmp_path)
    with pytest.raises(V2Error) as conflict:
        retry_legacy(store, operation, request | {"dataset_ref": "OTHER@rev_0001"})
    assert conflict.value.code == "TFV2-CONFLICT"
    assert store.get_run(original["run_id"]) == original
