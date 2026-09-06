"""真实恢复发现的预算回归：进程计数重置，但持久研究总用量不重置。"""

from thermoforge_v2.store import RunStore
from thermoforge_v2.usage import begin_process, merge_usage


def raw(total, input_count=None, cached=None):
    counters = {"totalTokens": total}
    if input_count is not None:
        counters.update(inputTokens=input_count, outputTokens=total - input_count)
    if cached is not None:
        counters["cachedInputTokens"] = cached
    return {"total": counters, "last": counters.copy()}


def update(track, value):
    patch = merge_usage(track, value, track["usage_generation"])
    if patch:
        track.update(patch)
    return track


def test_duplicate_and_late_counter_never_charge_twice_or_move_backwards():
    track = begin_process({})
    update(track, raw(10, 8, 5))
    update(track, raw(20, 16, 10))
    update(track, raw(20, 16, 10))
    update(track, raw(10, 8, 5))
    assert track["usage"]["total"] == raw(20, 16, 10)["total"]


def test_three_processes_accumulate_offsets_without_adding_cache_twice():
    track = begin_process({})
    update(track, raw(100, 80, 50))
    track.update(begin_process(track))
    update(track, raw(20, 18, 12))
    update(track, raw(30, 27, 18))
    assert track["usage"]["total"] == raw(130, 107, 68)["total"]
    track.update(begin_process(track))
    update(track, raw(5, 4, 2))
    assert track["usage"]["total"] == raw(135, 111, 70)["total"]


def test_retired_process_notifications_are_ignored():
    track = begin_process({})
    old = track["usage_generation"]
    update(track, raw(100))
    track.update(begin_process(track))
    assert merge_usage(track, raw(100000), old) is None
    assert track["usage"]["total"]["totalTokens"] == 100


def test_unknown_usage_does_not_fabricate_a_zero_measurement():
    track = begin_process({})
    assert merge_usage(track, {}, track["usage_generation"]) is None
    assert "usage" not in track
    update(track, {"total": {"inputTokens": 8, "outputTokens": 2}})
    assert track["usage"]["total"]["totalTokens"] == 10


def test_process_offset_survives_service_store_reconstruction(tmp_path):
    store = RunStore(tmp_path)
    run = store.create_run({"goal_id": "RG-0001", "dataset_ref": "test@rev_0001"},
                           {"fingerprint": "frozen"}, "one")
    rid = run["run_id"]
    track = store.create_track(rid, "main", "main")
    track.update(begin_process(track))
    update(track, raw(811672))
    store.update_track(rid, "main", {k: v for k, v in track.items() if k.startswith("usage")})
    recovered = RunStore(tmp_path).get_track(rid, "main")
    recovered.update(begin_process(recovered))
    update(recovered, raw(215694))
    assert recovered["usage"]["total"]["totalTokens"] == 1027366
