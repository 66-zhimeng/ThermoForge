"""命名与标识符规则（conventions.md §1.1–1.2）。

所有正则与校验函数是契约的唯一实现来源；
`object_id` 禁止包含 `.`，`variable_id` 按第一个 `.` 切分。
"""

from __future__ import annotations

import re

# conventions.md §1.1
PROPERTY_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
OBJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VARIABLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.[a-z][a-z0-9_]{0,63}$")
OBJECT_MODEL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.v[0-9]+$")
DATASET_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

# 数据集 revision（conventions.md §6 / data-contract.md §6）
DATASET_REVISION_RE = re.compile(r"^rev_[0-9]{4,}$")


class NamingError(ValueError):
    """标识符不符合 conventions.md §1.1 的命名规则。"""


def is_property_code(value: str) -> bool:
    return bool(PROPERTY_CODE_RE.fullmatch(value))


def is_object_id(value: str) -> bool:
    return bool(OBJECT_ID_RE.fullmatch(value))


def is_variable_id(value: str) -> bool:
    return bool(VARIABLE_ID_RE.fullmatch(value))


def is_object_model_id(value: str) -> bool:
    return bool(OBJECT_MODEL_ID_RE.fullmatch(value))


def is_dataset_id(value: str) -> bool:
    return bool(DATASET_ID_RE.fullmatch(value))


def _validate(value: str, pattern: re.Pattern[str], kind: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise NamingError(f"非法 {kind}: {value!r}（规则 {pattern.pattern}）")
    return value


def validate_property_code(value: str) -> str:
    return _validate(value, PROPERTY_CODE_RE, "property_code")


def validate_object_id(value: str) -> str:
    return _validate(value, OBJECT_ID_RE, "object_id")


def validate_variable_id(value: str) -> str:
    return _validate(value, VARIABLE_ID_RE, "variable_id")


def validate_object_model_id(value: str) -> str:
    return _validate(value, OBJECT_MODEL_ID_RE, "object_model_id")


def validate_dataset_id(value: str) -> str:
    return _validate(value, DATASET_ID_RE, "dataset_id")


def split_variable_id(variable_id: str) -> tuple[str, str]:
    """按**第一个** `.` 切分 variable_id（conventions.md §1.1）。

    `property_code` 不含 `.`，而 `object_id` 若违规包含 `.` 会导致
    从右侧切分时的静默错配，因此契约固定从左侧第一个 `.` 切分。
    """
    validate_variable_id(variable_id)
    object_id, property_code = variable_id.split(".", 1)
    return object_id, property_code


def variable_id_for(object_id: str, property_code: str) -> str:
    """由 object_id 与 property_code 合成 variable_id。"""
    validate_object_id(object_id)
    validate_property_code(property_code)
    return f"{object_id}.{property_code}"
