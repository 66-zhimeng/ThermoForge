"""数据内容指纹与环境指纹（conventions.md §5.2、§5.3）。

`content_sha256` 对规范化后的逻辑数据内容计算，与 Excel 文件字节无关：
列按 variable_id 字典序、行按 timestamp 升序，缺失值编码为 8×0xFF，
不写入任何 NaN 位模式（§5.2 规则 4）。

实现说明：`timestamp` 本身作为第一列参与列摘要（dtype 记为 `timestamp`，
值按 UTC 微秒 int64 编码）。§5.2 未明示这一点，但不纳入时间轴时
「同一组值配到不同时间轴」会得到相同指纹，违背去重语义。
"""

from __future__ import annotations

import hashlib
import math
import platform
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, sha256_hex

_DTYPES = ("float", "integer", "boolean", "string", "timestamp")

_MISSING_BYTES = b"\xff" * 8  # §5.2 规则 4：缺失值统一编码
_MISSING_STRING_LEN = 0xFFFFFFFF


class FingerprintError(ValueError):
    """指纹输入数据不合法。"""


@dataclass(frozen=True)
class Column:
    """一列待指纹数据。`values` 中 None 表示缺失。"""

    variable_id: str
    dtype: str  # float / integer / boolean / string
    values: Sequence[Any]


def _encode_value(dtype: str, value: Any) -> bytes:
    if value is None:
        if dtype == "string":
            return struct.pack("<I", _MISSING_STRING_LEN)
        return _MISSING_BYTES
    if dtype == "float":
        v = float(value)
        if math.isnan(v):
            # 导入层不产生 NaN（§4.2）；防御性按缺失编码，不写入 NaN 位模式
            return _MISSING_BYTES
        if math.isinf(v):
            raise FingerprintError("inf 不允许进入指纹计算（TFDC-602）")
        if v == 0.0:
            v = 0.0  # -0.0 规范化为 0.0（§4.3）
        return struct.pack("<d", v)
    if dtype == "integer":
        return struct.pack("<q", int(value))
    if dtype == "boolean":
        if not isinstance(value, bool):
            raise FingerprintError(f"boolean 列出现非布尔值: {value!r}")
        return b"\x01" if value else b"\x00"
    if dtype == "string":
        data = str(value).encode("utf-8")
        return struct.pack("<I", len(data)) + data
    if dtype == "timestamp":
        if value.tzinfo is None:
            raise FingerprintError("timestamp 必须带时区（UTC 微秒编码）")
        micros = int(value.astimezone(timezone.utc).timestamp() * 1_000_000)
        return struct.pack("<q", micros)
    raise FingerprintError(f"未知 dtype: {dtype!r}")


def _column_digest(name: str, dtype: str, values: Iterable[Any]) -> bytes:
    """列摘要：sha256(variable_id ‖ 0x00 ‖ dtype ‖ 0x00 ‖ 值字节流)。"""
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(dtype.encode("utf-8"))
    h.update(b"\x00")
    for value in values:
        h.update(_encode_value(dtype, value))
    return h.digest()


def content_sha256(timestamps: Sequence[datetime], columns: Sequence[Column]) -> str:
    """计算数据内容指纹（§5.2）。

    - 行按 timestamp 升序排序后再编码。
    - 列按 variable_id 字典序排序；timestamp 列固定为最前。
    - 列/行顺序不影响结果；任一值改动必然改变结果。
    """
    if not timestamps:
        raise FingerprintError("timestamps 不能为空")
    n = len(timestamps)
    order = sorted(range(n), key=lambda i: timestamps[i])
    digests = [_column_digest("timestamp", "timestamp", (timestamps[i] for i in order))]
    for col in sorted(columns, key=lambda c: c.variable_id):
        if col.dtype not in _DTYPES or col.dtype == "timestamp":
            raise FingerprintError(f"列 {col.variable_id} 的 dtype 非法: {col.dtype!r}")
        if len(col.values) != n:
            raise FingerprintError(f"列 {col.variable_id} 长度与 timestamps 不一致")
        digests.append(_column_digest(col.variable_id, col.dtype, (col.values[i] for i in order)))
    h = hashlib.sha256()
    for digest in digests:
        h.update(digest)
    return h.hexdigest()


def environment_lock(
    python_version: str,
    platform_tag: str,
    packages: Sequence[Mapping[str, str]],
) -> str:
    """环境指纹（§5.3）：sha256(canonical_json({python, platform, packages}))。

    `packages` 为 [{name, version, sha256}, ...]，按 name 排序后参与哈希。
    CPU 型号、核数、BLAS 等信息按 §5.3 记录但不纳入哈希。
    """
    sorted_packages = sorted(packages, key=lambda p: p["name"])
    doc = {
        "python": python_version,
        "platform": platform_tag,
        "packages": [dict(p) for p in sorted_packages],
    }
    return sha256_hex(canonical_json(doc))


def current_python_version() -> str:
    return platform.python_version()


def current_platform_tag() -> str:
    """§5.3 风格的平台标签，如 win_amd64 / linux_x86_64。"""
    return f"{sys.platform}_{platform.machine().lower()}"
