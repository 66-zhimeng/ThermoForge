"""数据内容指纹与环境指纹测试（conventions.md §5.2、§5.3）。"""

from datetime import datetime, timedelta, timezone

import pytest

from thermoforge_core.fingerprint import (
    Column,
    FingerprintError,
    content_sha256,
    environment_lock,
)

TZ8 = timezone(timedelta(hours=8))
BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=TZ8)


def _timestamps(n: int) -> list[datetime]:
    return [BASE + timedelta(seconds=60 * i) for i in range(n)]


def _columns() -> list[Column]:
    return [
        Column("CH-01.evap_chw_flow", "float", [521.2, 522.1, 520.8, 519.5]),
        Column("CH-01.evap_chw_supply_temp", "float", [6.8, 6.8, 6.9, 7.0]),
        Column("CH-01.input_power", "float", [412.5, 411.7, 413.2, 415.0]),
        Column("CH-01.running", "boolean", [True, True, False, True]),
        Column("CH-01.mode", "string", ["auto", "auto", None, "manual"]),
        Column("CH-01.alarm_count", "integer", [0, None, 1, 0]),
    ]


def test_column_order_irrelevant():
    cols = _columns()
    shuffled = list(reversed(cols))
    assert content_sha256(_timestamps(4), cols) == content_sha256(_timestamps(4), shuffled)


def test_row_order_irrelevant():
    ts = _timestamps(4)
    order = [2, 0, 3, 1]
    ts_shuffled = [ts[i] for i in order]
    cols_shuffled = [
        Column(c.variable_id, c.dtype, [c.values[i] for i in order]) for c in _columns()
    ]
    assert content_sha256(ts, _columns()) == content_sha256(ts_shuffled, cols_shuffled)


def test_value_change_changes_fingerprint():
    cols = _columns()
    modified = [
        Column(c.variable_id, c.dtype, [999.9 if i == 0 else v for i, v in enumerate(c.values)])
        if c.variable_id == "CH-01.input_power"
        else c
        for c in cols
    ]
    assert content_sha256(_timestamps(4), cols) != content_sha256(_timestamps(4), modified)


def test_timestamp_change_changes_fingerprint():
    ts = _timestamps(4)
    ts_shifted = [t + timedelta(seconds=1) for t in ts]
    assert content_sha256(ts, _columns()) != content_sha256(ts_shifted, _columns())


def test_missing_value_encodings():
    # None 在 float / integer / boolean / string 各类型下均可编码
    ts = _timestamps(2)
    cols = [
        Column("A.f", "float", [None, 1.5]),
        Column("A.i", "integer", [None, 3]),
        Column("A.b", "boolean", [None, True]),
        Column("A.s", "string", [None, "x"]),
    ]
    digest = content_sha256(ts, cols)
    assert len(digest) == 64


def test_nan_treated_as_missing():
    ts = _timestamps(2)
    a = [Column("A.f", "float", [None, 1.5])]
    b = [Column("A.f", "float", [float("nan"), 1.5])]
    assert content_sha256(ts, a) == content_sha256(ts, b)


def test_negative_zero_normalized():
    ts = _timestamps(2)
    a = [Column("A.f", "float", [-0.0, 1.5])]
    b = [Column("A.f", "float", [0.0, 1.5])]
    assert content_sha256(ts, a) == content_sha256(ts, b)


def test_inf_rejected():
    with pytest.raises(FingerprintError):
        content_sha256(_timestamps(1), [Column("A.f", "float", [float("inf")])])


def test_naive_timestamp_rejected():
    with pytest.raises(FingerprintError):
        content_sha256([datetime(2026, 1, 1)], [Column("A.f", "float", [1.0])])


def test_environment_lock_deterministic_and_order_insensitive():
    packages_a = [
        {"name": "numpy", "version": "2.3.1", "sha256": "aa"},
        {"name": "pandas", "version": "2.3.0", "sha256": "bb"},
    ]
    packages_b = list(reversed(packages_a))
    lock_a = environment_lock("3.12.11", "win_amd64", packages_a)
    lock_b = environment_lock("3.12.11", "win_amd64", packages_b)
    assert lock_a == lock_b
    assert lock_a != environment_lock("3.12.12", "win_amd64", packages_a)
    assert len(lock_a) == 64
