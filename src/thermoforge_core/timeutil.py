"""时间与时区处理（conventions.md §3）。

- `manifest.timezone` 必须是 IANA 时区名，禁止 CST / UTC+8 等缩写或固定偏移。
- timestamp 为带偏移量的 ISO 8601 文本，解析后立即转 UTC。
- `time_resolution` 采用简写，正则 `^[0-9]+(s|min|h|d)$`（§3.4 [草案]）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIME_RESOLUTION_RE = re.compile(r"^[0-9]+(s|min|h|d)$")

_RESOLUTION_UNIT_SECONDS = {"s": 1, "min": 60, "h": 3600, "d": 86400}


class TimezoneError(ValueError):
    """TFDC-204 TIMEZONE_UNKNOWN：非 IANA 时区名。"""

    code = "TFDC-204"


class TimestampParseError(ValueError):
    """TFDC-501 TIMESTAMP_UNPARSEABLE。"""

    code = "TFDC-501"


class TimestampNaiveError(ValueError):
    """TFDC-502 TIMESTAMP_NAIVE：无时区信息。"""

    code = "TFDC-502"


class ResolutionError(ValueError):
    """TFDC-205 RESOLUTION_INVALID。"""

    code = "TFDC-205"


def validate_iana_timezone(name: str) -> ZoneInfo:
    """校验 IANA 时区名，返回 ZoneInfo；非法时报 TFDC-204。"""
    if not isinstance(name, str) or not name:
        raise TimezoneError(f"非 IANA 时区名: {name!r}")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TimezoneError(f"非 IANA 时区名: {name!r}") from exc


def is_iana_timezone(name: str) -> bool:
    try:
        validate_iana_timezone(name)
        return True
    except TimezoneError:
        return False


def parse_timestamp(text: str) -> datetime:
    """解析带偏移量的 ISO 8601 文本时间戳，返回 UTC `datetime`。

    - 无法解析：TFDC-501。
    - 无时区信息（naive）：TFDC-502，不得猜测时区（§3.1）。
    """
    if not isinstance(text, str):
        raise TimestampNaiveError(f"时间戳必须是文本格式的 ISO 8601 字符串，得到 {type(text).__name__}")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimestampParseError(f"无法解析的时间戳: {text!r}") from exc
    if dt.tzinfo is None:
        raise TimestampNaiveError(f"时间戳缺少时区偏移量: {text!r}")
    return dt.astimezone(timezone.utc)


def format_timestamp_utc(dt: datetime) -> str:
    """UTC `datetime` → ISO 8601 字符串（微秒精度，+00:00 偏移）。"""
    if dt.tzinfo is None:
        raise TimestampNaiveError("只能格式化带时区的 datetime")
    return dt.astimezone(timezone.utc).isoformat()


def parse_time_resolution(text: str) -> int:
    """解析 time_resolution 简写（`60s` / `5min` / `1h` / `1d`），返回秒数。"""
    if not isinstance(text, str):
        raise ResolutionError(f"非法 time_resolution: {text!r}")
    m = TIME_RESOLUTION_RE.fullmatch(text)
    if not m:
        raise ResolutionError(
            f"非法 time_resolution: {text!r}（规则 {TIME_RESOLUTION_RE.pattern}）"
        )
    unit = m.group(1)
    number = int(text[: -len(unit)])
    return number * _RESOLUTION_UNIT_SECONDS[unit]
