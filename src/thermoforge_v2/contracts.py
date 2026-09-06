"""V2 控制契约：运行配置、状态及边界校验，不修改冻结的 V1 数据契约。"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class V2Error(ValueError):
    """可通过 GUI/CLI/MCP 统一展示的领域错误。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(pattern=r"^RG-[0-9]{4,}$")
    dataset_ref: str = Field(min_length=1)
    view_id: str | None = Field(default=None, pattern=r"^VIEW-[0-9]{4,}$")
    candidates: int = Field(default=5, ge=0, le=16)
    model: str | None = None
    reasoning_effort: str | None = None
    max_experiments: int = Field(default=20, ge=1, le=10000)
    max_experiments_per_track: int = Field(default=4, ge=1, le=1000)
    max_turns: int = Field(default=8, ge=1, le=200)
    token_budget: int = Field(default=200000, ge=1000)
    strategy: Literal["independent", "top_k", "adaptive"] = "independent"
    top_k: int = Field(default=2, ge=1, le=16)
    stagnation_rounds: int = Field(default=2, ge=1, le=20)
    experiment_workers: int = Field(default=1, ge=1, le=16)
    experiment_timeout_seconds: float = Field(default=300, ge=10, le=7200)
    turn_timeout_seconds: float = Field(default=900, ge=10, le=7200)
    max_failures: int = Field(default=3, ge=1, le=10)
    seed: int = 42
    guidance: str = Field(default="", max_length=16000)
    validation: dict[str, Any] | None = None
    purge_seconds: float = Field(default=2700, ge=0)
    embargo_seconds: float = Field(default=2700, ge=0)
    y_floor: float = Field(default=1e-6, gt=0)

    @field_validator("model", "reasoning_effort")
    @classmethod
    def clean_optional(cls, value):
        return value.strip() or None if value is not None else None

    @model_validator(mode="after")
    def finite_numbers(self):
        for name in ("experiment_timeout_seconds", "turn_timeout_seconds",
                     "purge_seconds", "embargo_seconds", "y_floor"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须是有限数")
        return self


TERMINAL_RUN_STATES = frozenset({"completed", "cancelled", "failed", "budget_exhausted"})
ACTIVE_RUN_STATES = frozenset({"queued", "running", "pausing", "cancelling"})
MUTABLE_CONFIG = frozenset({"guidance", "max_experiments", "max_experiments_per_track",
                            "max_turns", "token_budget"})
