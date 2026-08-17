"""Experiment Contract（research-loop.md §5）。

每次实验必须有独立、不可变的定义；指标集按 DD-14 纳入 NMBE。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..naming import is_property_code

# 集合语义字段（conventions.md §5.1 规则 6）
SET_SEMANTIC_FIELDS: frozenset[str] = frozenset({"metrics"})

EXPERIMENT_ID_PATTERN = r"^EXP-[0-9]{4,}$"
METRICS = ("RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2")


class ModelSpec(BaseModel):
    """模型规格：类别 + 路线细节（物理方程版本 / 残差学习器等）。

    `category="lab"` 引用模型实验室（thermoforge_research.model_lab）中
    已批准的模块：`hyperparameters.lab` 给出 `name` 或 `name@vN`，
    其余标量超参原样透传给模块的 `build_model()`。
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal["physics", "data", "hybrid", "lab"]
    physics: str | None = None
    residual: str | None = None
    estimator: str | None = None
    hyperparameters: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _category_consistent(self) -> "ModelSpec":
        if self.category == "hybrid" and not (self.physics and self.residual):
            raise ValueError("hybrid 实验必须同时声明 physics 与 residual")
        if self.category == "physics" and not self.physics:
            raise ValueError("physics 实验必须声明 physics")
        if self.category == "data" and not (self.estimator or self.residual):
            raise ValueError("data 实验必须声明 estimator")
        if self.category == "lab":
            if self.physics or self.residual or self.estimator:
                raise ValueError(
                    "lab 实验不得声明 physics/residual/estimator"
                    "（模型定义全部在实验室模块内）"
                )
            if not str(self.hyperparameters.get("lab") or "").strip():
                raise ValueError(
                    "lab 实验必须在 hyperparameters.lab 声明模型实验室引用"
                    "（name 或 name@vN，且须已批准）"
                )
        return self


class TemporalSplit(BaseModel):
    """时间顺序切分比例（边界时间点由运行器计算并记录，implementation-notes §4.1）。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    train: float = Field(gt=0, lt=1)
    # 契约 YAML 中的键是 `validate`（research-loop.md §5），
    # 与 BaseModel.validate 冲突，故用别名
    validate_: float = Field(ge=0, lt=1, alias="validate")
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "TemporalSplit":
        total = self.train + self.validate_ + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"temporal_split 比例之和必须为 1，得到 {total}")
        return self


class EquipmentHoldout(BaseModel):
    """留一设备验证（research-loop.md §6）。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    holdout_objects: list[str] = Field(default_factory=list)


class RollingCV(BaseModel):
    """滚动原点时序交叉验证（research-loop.md §6 的合规 CV 形态）。

    不配置或 `enabled=false` 时不启用（向后兼容：旧实验定义行为不变）。
    fold 边界与 temporal_split 同规则（时间边界 → 向下对齐 resolution），
    fold 语义见 `thermoforge_research.splits.rolling_origin_splits`。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["expanding", "sliding"] = "expanding"
    initial_train_fraction: float | None = Field(default=None, gt=0, lt=1)
    initial_train_seconds: float | None = Field(default=None, gt=0)
    horizon_seconds: float = Field(default=86400.0, gt=0)  # [草案] 默认 1 天
    step_seconds: float | None = Field(default=None, gt=0)  # 默认 = horizon
    max_folds: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _initial_window_given(self) -> "RollingCV":
        if self.enabled and (
            (self.initial_train_fraction is None)
            == (self.initial_train_seconds is None)
        ):
            raise ValueError(
                "rolling_cv 启用时 initial_train_fraction 与 "
                "initial_train_seconds 必须恰给其一"
            )
        return self


class Validation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_split: TemporalSplit
    equipment_holdout: EquipmentHoldout = Field(default_factory=EquipmentHoldout)
    rolling_cv: RollingCV = Field(default_factory=RollingCV)


class PhysicsTests(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class Runtime(BaseModel):
    """运行环境声明：环境指纹与随机种子（TFX-901/902）。"""

    model_config = ConfigDict(extra="forbid")

    environment_lock: str
    random_seed: int


class Experiment(BaseModel):
    """实验定义（对应 contracts/experiment/schema.json）。"""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    goal_id: str = Field(pattern=r"^RG-[0-9]{4,}$")
    hypothesis_id: str = Field(pattern=r"^H-[0-9]{4,}$")
    dataset_view: str = Field(pattern=r"^VIEW-[0-9]{4,}$")
    model: ModelSpec
    target: str
    validation: Validation
    metrics: list[Literal["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2"]] = Field(
        min_length=1, json_schema_extra={"x-tf-set-semantics": True}
    )
    physics_tests: PhysicsTests = Field(default_factory=PhysicsTests)
    runtime: Runtime
    description: str | None = None
    created_at: str | None = None
    author: str | None = None

    @field_validator("target")
    @classmethod
    def _target_valid(cls, v: str) -> str:
        if not is_property_code(v):
            raise ValueError(f"target 必须是 property_code: {v!r}")
        return v

    @field_validator("metrics")
    @classmethod
    def _metrics_unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("metrics 存在重复条目")
        return v
