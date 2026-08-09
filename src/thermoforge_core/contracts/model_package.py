"""模型包契约（model-package.md §2–5）。

签名使用 property_code 表达输入输出（两层命名，DD-11），部署绑定时
解析为 variable_id；模型内部不得依赖现场地址。版本状态机见 §5：

    candidate → validated → approved → production → deprecated → retired
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..naming import is_object_model_id, is_property_code
from ..units import normalize_unit

# 输入/输出列表的顺序是特征顺序语义（implementation-notes §8.1），
# 不按集合语义排序
SET_SEMANTIC_FIELDS: frozenset[str] = frozenset()

SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")

MODEL_STATUSES: tuple[str, ...] = (
    "candidate",
    "validated",
    "approved",
    "production",
    "deprecated",
    "retired",
)


def can_transition(from_status: str, to_status: str) -> bool:
    """状态机（model-package.md §5）：只允许沿链前进，不得回退或原地停留。"""
    try:
        i = MODEL_STATUSES.index(from_status)
        j = MODEL_STATUSES.index(to_status)
    except ValueError:
        return False
    return j > i


class SignaturePort(BaseModel):
    """签名中的一个输入/输出端口。"""

    model_config = ConfigDict(extra="forbid")

    property_code: str
    unit: str
    dtype: Literal["float", "integer", "boolean", "string"]
    required: bool = True

    @field_validator("property_code")
    @classmethod
    def _property_code_valid(cls, v: str) -> str:
        if not is_property_code(v):
            raise ValueError(f"非法 property_code: {v!r}")
        return v

    @field_validator("unit")
    @classmethod
    def _unit_registered(cls, v: str) -> str:
        return normalize_unit(v)


class ModelSignature(BaseModel):
    """模型签名（model-package.md §3）。"""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: str
    object_model: str
    object_model_compat: str | None = None  # 如 ">=1.0,<2.0"（conventions §6 [草案]）
    inputs: list[SignaturePort] = Field(min_length=1)
    outputs: list[SignaturePort] = Field(min_length=1)

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not SEMVER_RE.fullmatch(v):
            raise ValueError(f"非法语义化版本: {v!r}")
        return v

    @field_validator("object_model")
    @classmethod
    def _object_model_valid(cls, v: str) -> str:
        if not is_object_model_id(v):
            raise ValueError(f"非法 object_model_id: {v!r}")
        return v


class InputConstraint(BaseModel):
    """单个输入的工程范围与超范围策略（implementation-notes §9.2）。"""

    model_config = ConfigDict(extra="forbid")

    property_code: str
    min_value: float | None = None
    max_value: float | None = None
    out_of_range: Literal["reject", "clamp", "passthrough_with_flag"] = "reject"

    @field_validator("property_code")
    @classmethod
    def _property_code_valid(cls, v: str) -> str:
        if not is_property_code(v):
            raise ValueError(f"非法 property_code: {v!r}")
        return v


class Constraints(BaseModel):
    """`constraints.yaml`（model-package.md §4）。"""

    model_config = ConfigDict(extra="forbid")

    inputs: list[InputConstraint] = Field(default_factory=list)
    output_min_value: float | None = None
    output_max_value: float | None = None
    missing_policy: str | None = None
    physical_constraints: list[str] = Field(default_factory=list)
    object_model_compat: str | None = None
    validated_domain: str | None = None
    forbidden_extrapolation: str | None = None


class ModelPackage(BaseModel):
    """模型包元数据（对应 contracts/model-package/schema.json）。"""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["TFMP"] = "TFMP"
    contract_version: str = Field(default="1.0", pattern=r"^[0-9]+\.[0-9]+$")
    model_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: str
    status: Literal[
        "candidate", "validated", "approved", "production", "deprecated", "retired"
    ]
    signature: ModelSignature
    constraints: Constraints = Field(default_factory=Constraints)
    goal_id: str | None = Field(default=None, pattern=r"^RG-[0-9]{4,}$")
    experiment_id: str | None = Field(default=None, pattern=r"^EXP-[0-9]{4,}$")
    description: str | None = None
    created_at: str | None = None
    author: str | None = None

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not SEMVER_RE.fullmatch(v):
            raise ValueError(f"非法语义化版本: {v!r}")
        return v
