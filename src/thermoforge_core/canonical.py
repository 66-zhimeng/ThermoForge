"""Canonical JSON 与 SHA-256（conventions.md §5.1、§5）。

规则（§5.1 [草案]，此处为唯一实现来源）：
1. UTF-8 编码，不转义非 ASCII（ensure_ascii=False）。
2. 对象键按 Unicode 码点升序排序（json.dumps(sort_keys=True) 即码点序）。
3. 分隔符无空格：`,` 与 `:`。
4. 浮点数保留最短往返表示（repr 语义），`5.0` 不得写成 `5`。
5. 值为 null 的键在序列化前一律删除。
6. 数组默认保序；仅对契约显式声明为集合语义的字段按字典序排序。
7. 排除字段：description / name_zh / name_en / created_at / author / comment / tags。

所有文本写入必须显式 encoding="utf-8"、newline="\\n"（implementation-notes §10.3）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, FrozenSet, Iterable

# §5.1 规则 7：不影响语义的排除字段（在任意嵌套层级均排除）
EXCLUDED_KEYS: FrozenSet[str] = frozenset(
    {"description", "name_zh", "name_en", "created_at", "author", "comment", "tags"}
)


def _normalize(obj: Any, set_fields: FrozenSet[str]) -> Any:
    """递归规范化：删 null 键、排除无语义字段、集合语义数组排序。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in EXCLUDED_KEYS:
                continue
            if value is None:
                continue  # 规则 5：null 键与缺失等价
            normalized = _normalize(value, set_fields)
            if key in set_fields and isinstance(normalized, list):
                # 规则 6：仅契约显式声明的集合语义字段按字典序排序。
                # 以元素的 canonical JSON 表示排序，字符串元素即字典序。
                normalized = sorted(normalized, key=_element_sort_key)
            out[key] = normalized
        return out
    if isinstance(obj, list):
        return [_normalize(item, set_fields) for item in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("NaN / inf 不允许进入 canonical JSON")
        if obj == 0.0:
            return 0.0  # §4.3：-0.0 规范化为 0.0
        return obj
    return obj


def _element_sort_key(element: Any) -> str:
    return json.dumps(element, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json(obj: Any, set_fields: Iterable[str] = ()) -> str:
    """把 YAML/JSON 解析出的数据结构序列化为 canonical JSON 字符串。

    `set_fields`：声明为集合语义的字段名集合（如 {"objects", "metrics"}），
    这些字段的数组值在序列化前按字典序排序；其余数组严格保序。
    """
    normalized = _normalize(obj, frozenset(set_fields))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(data: str | bytes) -> str:
    """SHA-256，小写十六进制全值（conventions.md §5）。"""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any, set_fields: Iterable[str] = ()) -> str:
    """canonical JSON 的 SHA-256 全值。前 16 位仅用于展示，比较必须用全值。"""
    return sha256_hex(canonical_json(obj, set_fields))
