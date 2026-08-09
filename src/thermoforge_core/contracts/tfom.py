"""TFOM v1 物模型契约（data-contract.md §3、conventions.md §1/§2）。

`object_model_id` 形如 `chiller.v1`，由 `model_id` 与 `version` 的 MAJOR 组成；
属性语义、单位或约束变化即升 N，不允许原地修改（conventions.md §6）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..naming import PROPERTY_CODE_RE, is_property_code
from ..units import UNIT_TABLE, normalize_unit

DTYPES = ("float", "integer", "boolean", "string")
ROLES = ("state", "control", "disturbance", "target", "context", "derived")
QUANTITY_KINDS = tuple(sorted({u.quantity_kind for u in UNIT_TABLE}))

# TFOM 的 properties 是 dict（以 property_code 为键），无集合语义词段
SET_SEMANTIC_FIELDS: frozenset[str] = frozenset()


class TfomProperty(BaseModel):
    """物模型属性（TFOM property）。"""

    model_config = ConfigDict(extra="forbid")

    unit: str
    dtype: Literal["float", "integer", "boolean", "string"]
    role: Literal["state", "control", "disturbance", "target", "context", "derived"]
    name_zh: str | None = None
    name_en: str | None = None
    quantity_kind: Literal[
        "temperature", "temperature_difference", "power", "energy",
        "volume_flow", "mass_flow", "pressure", "percent", "dimensionless",
        "frequency", "time",
    ] | None = None
    expression: str | None = None
    min_value: float | None = None
    max_value: float | None = None

    @field_validator("unit")
    @classmethod
    def _unit_registered(cls, v: str) -> str:
        # 单位必须是登记在册的规范单位或别名，存规范形式（TFDC-401）
        return normalize_unit(v)

    @model_validator(mode="after")
    def _check_consistency(self) -> "TfomProperty":
        if self.role == "derived" and not self.expression:
            raise ValueError(f"role=derived 的属性必须给出 expression")
        if self.role != "derived" and self.expression:
            raise ValueError("仅 role=derived 的属性允许携带 expression")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value 不得大于 max_value")
        if self.quantity_kind is not None:
            # 声明的 quantity_kind 必须能解释该单位（如温差上下文允许 Cel→K）
            normalize_unit(self.unit, self.quantity_kind)
        return self


class ObjectModel(BaseModel):
    """TFOM 物模型定义（对应 contracts/tfom/schema.json）。"""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["TFOM"] = "TFOM"
    contract_version: str = Field(default="1.0", pattern=r"^[0-9]+\.[0-9]+$")
    model_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    properties: dict[str, TfomProperty]

    @field_validator("properties")
    @classmethod
    def _property_codes_valid(
        cls, v: dict[str, TfomProperty]
    ) -> dict[str, TfomProperty]:
        for code in v:
            if not is_property_code(code):
                raise ValueError(f"非法 property_code: {code!r}")
        if not v:
            raise ValueError("properties 不得为空")
        return v

    @property
    def object_model_id(self) -> str:
        """`name.vN`，N 取 version 的 MAJOR。"""
        major = self.version.split(".", 1)[0]
        return f"{self.model_id}.v{major}"
