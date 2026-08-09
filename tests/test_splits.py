"""数据切分与泄漏检测测试（implementation-notes.md §4、§13.5）。

- 时间边界切分：分位时间点 → 向下对齐 resolution → [t0,b1)/[b1,b2)/[b2,end]。
- purge / embargo 生效。
- 含滚动特征的合成数据未 purge 时被 TFX-903 捕获（§13.5 测试基线 5）。
- 设备留出与设备维度泄漏。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from thermoforge_research.errors import ResearchError
from thermoforge_research.splits import (
    DEFAULT_EMBARGO_SECONDS,
    build_test_surfaces,
    check_no_leakage,
    equipment_holdout_split,
    temporal_split,
)

UTC = timezone.utc
RES = 900  # 15 min


def _ts(n: int, start_minute: int = 0) -> list[datetime]:
    base = datetime(2026, 1, 1, 0, start_minute, tzinfo=UTC)
    return [base + timedelta(seconds=RES * i) for i in range(n)]


def test_boundaries_align_down_to_resolution():
    # 起点不对齐 resolution（:07 开始），边界必须向下对齐到 15 min 整数倍
    ts = _ts(100, start_minute=7)
    split = temporal_split(ts, RES, embargo_seconds=0)
    b1 = datetime.fromisoformat(split.boundaries["b1"])
    b2 = datetime.fromisoformat(split.boundaries["b2"])
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    assert b1.timestamp() % RES == 0
    assert b2.timestamp() % RES == 0
    # b1 不超过 70% 分位原始时间点（向下对齐）
    raw = sorted(ts)[int(0.7 * 100)]
    assert b1 <= raw < b1 + timedelta(seconds=RES)
    assert b1 > epoch


def test_intervals_are_half_open_and_partitioned():
    ts = _ts(200)
    split = temporal_split(ts, RES, embargo_seconds=0)
    b1 = datetime.fromisoformat(split.boundaries["b1"])
    b2 = datetime.fromisoformat(split.boundaries["b2"])
    train_ts = {ts[i] for i in split.train_idx}
    val_ts = {ts[i] for i in split.validate_idx}
    test_ts = {ts[i] for i in split.test_idx}
    assert max(train_ts) < b1 <= min(val_ts)
    assert max(val_ts) < b2 <= min(test_ts)
    assert len(train_ts | val_ts | test_ts) == 200  # 无 purge/embargo 时无丢失
    assert split.boundaries["counts"]["purged_or_embargoed"] == 0


def test_purge_and_embargo_take_effect():
    ts = _ts(200)
    purge = 4 * RES
    embargo = 2 * RES
    split = temporal_split(ts, RES, purge_seconds=purge,
                           embargo_seconds=embargo)
    b1 = datetime.fromisoformat(split.boundaries["b1"])
    train_max = max(ts[i] for i in split.train_idx)
    val_min = min(ts[i] for i in split.validate_idx)
    assert train_max < b1 - timedelta(seconds=purge)  # purge 剔除训练末尾
    assert val_min >= b1 + timedelta(seconds=embargo)  # embargo 跳过验证起始
    counts = split.boundaries["counts"]
    assert counts["purged_or_embargoed"] > 0
    assert (
        counts["train"] + counts["validate"] + counts["test"]
        + counts["purged_or_embargoed"] == 200
    )


def test_default_embargo_is_45min():
    assert DEFAULT_EMBARGO_SECONDS == 45 * 60
    ts = _ts(200)
    split = temporal_split(ts, RES)  # 默认 embargo
    b1 = datetime.fromisoformat(split.boundaries["b1"])
    val_min = min(ts[i] for i in split.validate_idx)
    assert val_min >= b1 + timedelta(seconds=DEFAULT_EMBARGO_SECONDS)


def test_leakage_time_overlap_detected():
    ts = _ts(10)
    with pytest.raises(ResearchError) as excinfo:
        check_no_leakage(ts, [0, 1, 2, 5], [5, 6, 7])  # 共享索引 5
    assert excinfo.value.code == "TFX-903"


def test_rolling_feature_without_purge_caught():
    """§13.5：含滚动窗口特征的合成数据，未 purge 时必须被 TFX-903 捕获。"""
    n = 400
    ts = _ts(n)
    window = 8 * RES  # 滚动窗口 8 步（2 h）
    # 未 purge 的切分：训练末尾紧贴验证起点
    split = temporal_split(ts, RES, purge_seconds=0, embargo_seconds=0)
    with pytest.raises(ResearchError) as excinfo:
        check_no_leakage(ts, split.train_idx, split.validate_idx,
                         min_gap_seconds=window)
    assert excinfo.value.code == "TFX-903"
    # 正确 purge 后同一检查通过
    split_ok = temporal_split(ts, RES, purge_seconds=window, embargo_seconds=0)
    check_no_leakage(ts, split_ok.train_idx, split_ok.validate_idx,
                     min_gap_seconds=window)


def test_device_overlap_detected():
    ts = _ts(10)
    objects = ["CH-01"] * 5 + ["CH-02"] * 5
    with pytest.raises(ResearchError) as excinfo:
        check_no_leakage(ts, [0, 1], [3, 4], object_ids=objects)  # CH-01 重叠
    assert excinfo.value.code == "TFX-903"
    # 无设备重叠时通过（时间也不重叠）
    check_no_leakage(ts, [0, 1], [7, 8],
                     object_ids=["CH-01"] * 5 + ["CH-03"] * 5)


def test_equipment_holdout_split():
    objects = ["CH-01", "CH-02", "CH-01", "CH-03"]
    train, held = equipment_holdout_split(objects, ["CH-03"])
    assert train == [0, 1, 2] and held == [3]
    with pytest.raises(ValueError):
        equipment_holdout_split(objects, ["CH-99"])


def test_test_surfaces_three_planes():
    ts = _ts(100)
    objects = ["CH-01"] * 100 + ["CH-02"] * 100  # 长表：两台设备同一时间轴
    ts_long = ts + ts
    split = temporal_split(ts, RES, embargo_seconds=0)
    surfaces = build_test_surfaces(ts_long, objects, split,
                                   holdout_objects=["CH-02"])
    assert surfaces["A"] == split.test_idx
    assert surfaces["B"] and surfaces["C"]
    b2 = datetime.fromisoformat(split.boundaries["b2"])
    assert all(ts_long[i] < b2 for i in surfaces["B"])
    assert all(ts_long[i] >= b2 for i in surfaces["C"])
    # 未启用留出时只有面 A
    only_a = build_test_surfaces(ts_long, objects, split)
    assert set(only_a) == {"A"}


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        temporal_split(_ts(10), RES, train=0.5, validate=0.3, test=0.3)
    with pytest.raises(ValueError):
        temporal_split([], RES)
