"""TFDC-XLSX v1.0 各表 record 结构（data-contract.md §4）。

对应 manifest / objects / variables / parameters / relations / bindings 表；
`TfdcDataset` 是整册容器，做跨表引用校验（TFDC-301/304/305/307 等）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from ..naming import (
    is_dataset_id,
    is_object_id,
    is_object_model_id,
    is_property_code,
    is_variable_id,
    split_variable_id,
)
from ..timeutil import TIME_RESOLUTION_RE, is_iana_timezone
from ..units import normalize_unit

# 整册定义哈希时的集合语义字段（conventions.md §5.1 规则 6）
SET_SEMANTIC_FIELDS: frozenset[str] = frozenset(
    {"objects", "variables", "parameters", "relations", "bindings"}
)

AGGREGATIONS = ("mean", "sum", "min", "max", "first", "last", "median")


def _set_semantics() -> dict:
    return {"json_schema_extra": {"x-tf-set-semantics": True}}


class TfdcManifest(BaseModel):
    """`manifest` 表（key/value 两列表解析后的结构）。"""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["TFDC"]
    contract_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    dataset_id: str
    dataset_version: int = Field(ge=1)
    site_id: str
    timezone: str
    time_resolution: str
    object_model_version: str | None = None
    source_system: str | None = None
    created_at: AwareDatetime | None = None
    description: str | None = None

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_valid(cls, v: str) -> str:
        if not is_dataset_id(v):
            raise ValueError(f"非法 dataset_id: {v!r}")
        return v

    @field_validator("timezone")
    @classmethod
    def _timezone_iana(cls, v: str) -> str:
        if not is_iana_timezone(v):
            raise ValueError(f"非 IANA 时区名（TFDC-204）: {v!r}")
        return v

    @field_validator("time_resolution")
    @classmethod
    def _resolution_valid(cls, v: str) -> str:
        if not TIME_RESOLUTION_RE.fullmatch(v):
            raise ValueError(f"非法 time_resolution（TFDC-205）: {v!r}")
        return v


class ObjectRecord(BaseModel):
    """`objects` 表一行。"""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_model_id: str
    object_name: str | None = None
    parent_id: str | None = None
    system_id: str | None = None

    @field_validator("object_id", "parent_id")
    @classmethod
    def _object_id_valid(cls, v: str | None) -> str | None:
        if v is not None and not is_object_id(v):
            raise ValueError(f"非法 object_id: {v!r}")
        return v

    @field_validator("object_model_id")
    @classmethod
    def _model_id_valid(cls, v: str) -> str:
        if not is_object_model_id(v):
            raise ValueError(f"非法 object_model_id: {v!r}")
        return v


class VariableRecord(BaseModel):
    """`variables` 表一行。"""

    model_config = ConfigDict(extra="forbid")

    variable_id: str
    object_id: str
    property_code: str
    unit: str
    dtype: Literal["float", "integer", "boolean", "string"]
    role: Literal["state", "control", "disturbance", "target", "context", "derived"]
    source_kind: Literal["measured", "derived", "estimated", "manual"]
    name_zh: str | None = None
    name_en: str | None = None
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    sample_period: str | None = None
    aggregation: Literal["mean", "sum", "min", "max", "first", "last", "median"] | None = None
    description: str | None = None

    @field_validator("unit")
    @classmethod
    def _unit_registered(cls, v: str) -> str:
        return normalize_unit(v)

    @model_validator(mode="after")
    def _variable_id_consistent(self) -> "VariableRecord":
        if not is_variable_id(self.variable_id):
            raise ValueError(f"非法 variable_id（TFDC-304）: {self.variable_id!r}")
        object_id, property_code = split_variable_id(self.variable_id)
        if object_id != self.object_id or property_code != self.property_code:
            raise ValueError(
                f"variable_id 与 object_id/property_code 不一致: {self.variable_id!r}"
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value 不得大于 max_value")
        return self


class ParameterRecord(BaseModel):
    """`parameters` 表一行（设备额定参数和固定参数）。"""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    parameter_code: str
    value: float
    unit: str

    @field_validator("object_id")
    @classmethod
    def _object_id_valid(cls, v: str) -> str:
        if not is_object_id(v):
            raise ValueError(f"非法 object_id: {v!r}")
        return v

    @field_validator("parameter_code")
    @classmethod
    def _parameter_code_valid(cls, v: str) -> str:
        if not is_property_code(v):
            raise ValueError(f"非法 parameter_code: {v!r}")
        return v

    @field_validator("unit")
    @classmethod
    def _unit_registered(cls, v: str) -> str:
        return normalize_unit(v)


class RelationRecord(BaseModel):
    """`relations` 表一行（对象之间的系统拓扑关系）。"""

    model_config = ConfigDict(extra="forbid")

    from_object: str
    relation: str
    to_object: str
    port_from: str | None = None
    port_to: str | None = None
    medium: str | None = None
    direction: Literal["forward", "reverse"] | None = None

    @field_validator("from_object", "to_object")
    @classmethod
    def _object_id_valid(cls, v: str) -> str:
        if not is_object_id(v):
            raise ValueError(f"非法 object_id: {v!r}")
        return v


class BindingRecord(BaseModel):
    """`bindings` 表一行（现场点位绑定，只存在于 Adapter 边界）。"""

    model_config = ConfigDict(extra="forbid")

    variable_id: str
    adapter: str
    source_ref: str

    @field_validator("variable_id")
    @classmethod
    def _variable_id_valid(cls, v: str) -> str:
        if not is_variable_id(v):
            raise ValueError(f"非法 variable_id（TFDC-304）: {v!r}")
        return v


class TfdcDataset(BaseModel):
    """TFDC-XLSX 整册结构：四张必需表 + 三张可选表，含跨表引用校验。"""

    model_config = ConfigDict(extra="forbid")

    manifest: TfdcManifest
    objects: list[ObjectRecord] = Field(min_length=1, **_set_semantics())
    variables: list[VariableRecord] = Field(min_length=1, **_set_semantics())
    parameters: list[ParameterRecord] = Field(default_factory=list, **_set_semantics())
    relations: list[RelationRecord] = Field(default_factory=list, **_set_semantics())
    bindings: list[BindingRecord] = Field(default_factory=list, **_set_semantics())

    @model_validator(mode="after")
    def _cross_reference(self) -> "TfdcDataset":
        object_ids = [o.object_id for o in self.objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("objects 中 object_id 重复")
        known = set(object_ids)
        # 注意：parent_id 可能指向未列入 objects 的上层系统或站点
        # （data-contract.md §4 示例即如此），成员资格与成环检查（TFDC-309）
        # 属于 Phase 1 导入器的语义校验，不在 record 结构层强制
        seen_vars: set[str] = set()
        for var in self.variables:
            if var.variable_id in seen_vars:
                raise ValueError(f"variable_id 重复声明（TFDC-307）: {var.variable_id!r}")
            seen_vars.add(var.variable_id)
            if var.object_id not in known:
                raise ValueError(f"variables 引用不存在的对象（TFDC-301）: {var.object_id!r}")
        for rec in self.parameters:
            if rec.object_id not in known:
                raise ValueError(f"parameters 引用不存在的对象（TFDC-301）: {rec.object_id!r}")
        for rec in self.relations:
            for ref in (rec.from_object, rec.to_object):
                if ref not in known:
                    raise ValueError(f"relations 引用非法（TFDC-308）: {ref!r}")
        for rec in self.bindings:
            if rec.variable_id not in seen_vars:
                raise ValueError(f"bindings 引用未声明的 variable_id: {rec.variable_id!r}")
        return self
