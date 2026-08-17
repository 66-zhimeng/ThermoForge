"""Research Goal 契约（research-loop.md §1）。

`candidate_inputs` 是封闭白名单（DD-16）：建模只允许使用其中列出的变量
计算 target。条目可以是 `property_code`（本对象）或 `object.property`
两层命名（跨设备取数）。

按 DD-14 / implementation-notes §5.2，NMBE 纳入必报指标与验收条件。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..naming import (
    is_object_model_id,
    is_property_code,
    is_variable_id,
)

# 集合语义字段（conventions.md §5.1 规则 6）
SET_SEMANTIC_FIELDS: frozenset[str] = frozenset({"candidate_inputs", "approval_required"})

GOAL_ID_PATTERN = r"^RG-[0-9]{4,}$"


def _set_semantics() -> dict:
    return {"json_schema_extra": {"x-tf-set-semantics": True}}


class ModelTypes(BaseModel):
    """允许的建模路线（research-loop.md §4）。"""

    model_config = ConfigDict(extra="forbid")

    physics: bool = True
    data: bool = True
    hybrid: bool = True


class Acceptance(BaseModel):
    """硬性验收条件。NMBE 按 DD-14 纳入，单独设限。

    `evaluated_on` 决定门槛判在哪个口径上，默认 `auto`（C→A→validate，§4.3）。
    这不是可有可无的修饰：面 A 是「训练截止后隔一段再考」，衡量的是**漂移**；
    `rolling_cv` 每折用最近数据重训，衡量的是**定期重训下的精度**。同一个
    模型在两者上能差三倍（实测冷机 hybrid：面 A 15.9%、滚动 4.9%）。
    门槛从哪个口径的基线推出来，就必须判在哪个口径上。
    """

    model_config = ConfigDict(extra="forbid")

    evaluated_on: Literal["auto", "C", "A", "validate", "rolling_cv"] = "auto"
    cvrmse_max: float | None = Field(default=None, ge=0)
    mape_max: float | None = Field(default=None, ge=0)
    nmbe_abs_max: float | None = Field(default=None, ge=0)
    physics_violation_rate_max: float | None = Field(default=None, ge=0, le=1)
    inference_latency_ms_max: float | None = Field(default=None, gt=0)
    extrapolation_required: bool = False


class ResearchGoal(BaseModel):
    """Research Goal：一次持续研究的根对象。"""

    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(pattern=GOAL_ID_PATTERN)
    name: str
    object_model: str
    purpose: str
    target: str
    candidate_inputs: list[str] = Field(min_length=1, **_set_semantics())
    model_types: ModelTypes = Field(default_factory=ModelTypes)
    acceptance: Acceptance
    max_experiments: int | None = Field(default=None, ge=1)
    max_duration_days: int | None = Field(default=None, ge=1)
    compute_budget_hours: float | None = Field(default=None, gt=0)
    approval_required: list[str] = Field(default_factory=list, **_set_semantics())
    description: str | None = None
    created_at: str | None = None
    author: str | None = None

    @field_validator("object_model")
    @classmethod
    def _object_model_valid(cls, v: str) -> str:
        if not is_object_model_id(v):
            raise ValueError(f"非法 object_model_id: {v!r}")
        return v

    @field_validator("target")
    @classmethod
    def _target_valid(cls, v: str) -> str:
        if not is_property_code(v):
            raise ValueError(f"target 必须是 property_code: {v!r}")
        return v

    @field_validator("candidate_inputs")
    @classmethod
    def _candidate_inputs_valid(cls, v: list[str]) -> list[str]:
        for item in v:
            if not (is_property_code(item) or is_variable_id(item)):
                raise ValueError(
                    f"candidate_inputs 条目必须是 property_code 或 object.property: {item!r}"
                )
        if len(set(v)) != len(v):
            raise ValueError("candidate_inputs 存在重复条目")
        return v
