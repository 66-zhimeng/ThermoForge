"""时间与时区测试（conventions.md §3）。"""

from datetime import datetime, timezone

import pytest

from thermoforge_core.timeutil import (
    ResolutionError,
    TimestampNaiveError,
    TimestampParseError,
    TimezoneError,
    format_timestamp_utc,
    is_iana_timezone,
    parse_time_resolution,
    parse_timestamp,
    validate_iana_timezone,
)


def test_iana_timezone_valid():
    tz = validate_iana_timezone("Asia/Shanghai")
    assert str(tz) == "Asia/Shanghai"
    assert is_iana_timezone("UTC")


@pytest.mark.parametrize("name", ["CST", "UTC+8", "GMT+8", "北京时间", ""])
def test_iana_timezone_invalid(name):
    # §3.2：禁止缩写与固定偏移
    assert not is_iana_timezone(name)
    with pytest.raises(TimezoneError) as exc:
        validate_iana_timezone(name)
    assert exc.value.code == "TFDC-204"


def test_parse_timestamp_with_offset_to_utc():
    dt = parse_timestamp("2026-01-01T00:00:00+08:00")
    assert dt == datetime(2025, 12, 31, 16, 0, 0, tzinfo=timezone.utc)
    assert dt.tzinfo is not None


def test_parse_timestamp_z_suffix():
    dt = parse_timestamp("2026-01-01T00:00:00Z")
    assert dt == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_timestamp_naive_rejected():
    # §3.1：无时区信息不得猜测，报 TFDC-502
    with pytest.raises(TimestampNaiveError) as exc:
        parse_timestamp("2026-01-01T00:00:00")
    assert exc.value.code == "TFDC-502"


def test_parse_timestamp_unparseable():
    with pytest.raises(TimestampParseError) as exc:
        parse_timestamp("2026/01/01 00:00")
    assert exc.value.code == "TFDC-501"


def test_parse_timestamp_non_string_rejected():
    # Excel 日期类型单元格读出 datetime → TFDC-502（§3.1）
    with pytest.raises(TimestampNaiveError):
        parse_timestamp(datetime(2026, 1, 1))


def test_format_roundtrip():
    dt = parse_timestamp("2026-06-01T12:30:45.123456+08:00")
    assert parse_timestamp(format_timestamp_utc(dt)) == dt


@pytest.mark.parametrize(
    "text,seconds",
    [("60s", 60), ("5min", 300), ("1h", 3600), ("1d", 86400)],
)
def test_time_resolution_valid(text, seconds):
    assert parse_time_resolution(text) == seconds


@pytest.mark.parametrize("text", ["PT60S", "10m", "1.5h", "min", "60 s", ""])
def test_time_resolution_invalid(text):
    with pytest.raises(ResolutionError) as exc:
        parse_time_resolution(text)
    assert exc.value.code == "TFDC-205"
