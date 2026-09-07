"""V2 控制契约：运行配置、状态及边界校验，不修改冻结的 V1 数据契约。"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thermoforge_core.contracts.experiment import ModelSpec
from thermoforge_research.model_catalog import DATA_ESTIMATORS, HYBRID_RESIDUALS, PHYSICS_MODELS

from .profile import CODEX_EFFORT, CODEX_MODEL


class V2Error(ValueError):
    """可通过 GUI/CLI/MCP 统一展示的领域错误。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_baseline_model(model: ModelSpec) -> ModelSpec:
    """Only globally named built-ins currently have comparable source identity."""
    if model.category == "lab" or "lab" in model.hyperparameters:
        raise ValueError("共同基线暂不支持 lab 模型；跨轨迹源码身份需先冻结内容哈希")
    if model.category in {"physics", "hybrid"} and model.physics not in PHYSICS_MODELS:
        raise ValueError("共同基线必须使用已登记的内置物理模型")
    if model.category == "hybrid" and model.residual not in HYBRID_RESIDUALS:
        raise ValueError("共同基线必须使用已登记的内置残差模型")
    if model.category == "data" and (model.estimator or model.residual) not in DATA_ESTIMATORS:
        raise ValueError("共同基线必须使用已登记的内置数据模型")
    return model


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(pattern=r"^RG-[0-9]{4,}$")
    dataset_ref: str = Field(min_length=1)
    view_id: str | None = Field(default=None, pattern=r"^VIEW-[0-9]{4,}$")
    candidates: int = Field(default=5, ge=0, le=16)
    model: str | None = CODEX_MODEL
    reasoning_effort: str | None = CODEX_EFFORT
    max_experiments: int = Field(default=20, ge=1, le=10000)
    max_experiments_per_track: int = Field(default=4, ge=1, le=1000)
    max_turns: int = Field(default=8, ge=1, le=200)
    token_budget: int = Field(default=1000000, ge=1000)
    strategy: Literal["independent", "top_k", "adaptive"] = "independent"
    research_mode: Literal["autonomous", "acceptance"] = "autonomous"
    objective_mode: Literal["target", "optimize", "explore"] | None = None
    objective_metric: Literal["CVRMSE", "RMSE", "MAE", "MAPE", "NMBE", "R2"] | None = None
    baseline_model: ModelSpec | None = None
    min_improvement: float = Field(default=0.01, ge=0, le=1)
    review_required: bool = True
    reuse_experiments: bool = False
    top_k: int = Field(default=2, ge=1, le=16)
    stagnation_rounds: int = Field(default=2, ge=1, le=20)
    experiment_workers: int = Field(default=1, ge=1, le=1)
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
        if self.baseline_model is not None:
            validate_baseline_model(self.baseline_model)
        if self.research_mode == "autonomous" and self.max_turns < 3:
            raise ValueError("自主研究至少需要三次执行回合：冻结提案、实验、结题")
        if self.reuse_experiments and self.strategy == "independent":
            raise ValueError("独立对照不复用其他候选的实验；共享模式可显式开启复用")
        if self.candidates and self.max_turns < 2:
            raise ValueError("多候选研究至少需要两轮主智能体预算（准备与结题）")
        for name in ("experiment_timeout_seconds", "turn_timeout_seconds",
                     "purge_seconds", "embargo_seconds", "y_floor", "min_improvement"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须是有限数")
        return self


TERMINAL_RUN_STATES = frozenset({"completed", "cancelled", "failed", "budget_exhausted"})
ACTIVE_RUN_STATES = frozenset({"queued", "running", "pausing", "cancelling"})
MUTABLE_CONFIG = frozenset({"guidance", "max_experiments", "max_experiments_per_track",
                            "max_turns", "token_budget"})

# Defaults introduced with objective contracts. Historical idempotency hashes
# must be checked without these fields, but never discard an explicit change.
OBJECTIVE_CONFIG_DEFAULTS = {"objective_mode": None, "objective_metric": None,
                             "baseline_model": None, "min_improvement": 0.01,
                             "review_required": True}
