"""声明式派生数据集（I-49：预处理能力工具化进编排层）。

`examples/chiller_power/derive_plant.py` 证明了派生数据集这条路走得通，
但它是一个写死的脚本：源列、算子、目标属性全部硬编码，Agent 无法复用。
本模块把那条路参数化——

Agent 只提供「哪些源变量、走哪个算子、产出什么属性」的**参数**，
执行一律由下面登记在册的算子完成，**不接受任何自由代码**
（与 `preprocess.py` 同一原则：DD-02 方案 C）。

产出走 `import_parsed` 标准管线（全部 TFDC 校验 + 指纹），作为新数据集
的不可变 revision 落 vault，lineage 记录完整派生规则与源 revision，
因此派生结果同样可追溯、可复现、可去重。

算子（全部为确定性纯函数，无随机源）::

    sum            逐点求和；任一源缺失 → 缺失
    mean           逐点平均；任一源缺失 → 缺失
    weighted_mean  按 weights 加权平均（权重和归一）；任一源缺失 → 缺失
    first          取第一个非缺失的源（多路冗余测点取其一）
    diff           sources[0] - sources[1]
    count_true     布尔源中 true 的个数 → integer
    any_true       任一源为 true → boolean
    sum_where      sources 与 conditions 配对，条件为 true 才计入；
                   条件为 true 但值缺失 → 整体缺失（不凑数）

`sum_where` 是「运行设备的功率之和」这类跨对象聚合的通用形式，也是
Dataset View 长表机制表达不了、必须走派生数据集的那一类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_core.contracts.tfom import ObjectModel
from thermoforge_core.naming import is_property_code

from .importer import TfomRegistry, default_registry

DERIVATION_VERSION = "tf_dataset_derive v1"

# 算子 → (期望的 dtype, 是否要求源单位一致, 是否要求布尔源)
OPS: dict[str, tuple[str, bool, bool]] = {
    "sum": ("float", True, False),
    "mean": ("float", True, False),
    "weighted_mean": ("float", True, False),
    "first": ("float", True, False),
    "diff": ("float", True, False),
    "count_true": ("integer", False, True),
    "any_true": ("boolean", False, True),
    "sum_where": ("float", True, False),
}

UNITLESS = "1"


class DeriveError(Exception):
    """派生规则不合法（TFPP- 域之外的参数错误，转成信封 ok=False）。"""


@dataclass(frozen=True)
class DeriveRule:
    """一条派生规则：一个算子 + 一组源变量 → 一个新属性。"""

    property_code: str
    op: str
    sources: tuple[str, ...]
    unit: str | None = None
    dtype: str | None = None
    role: str = "state"
    source_kind: str = "estimated"
    conditions: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    description: str = ""

    @classmethod
    def parse(cls, doc: Mapping[str, Any]) -> "DeriveRule":
        prop = str(doc.get("property_code") or "").strip()
        if not is_property_code(prop):
            raise DeriveError(
                f"property_code 不合法（lower_snake_case）: {prop!r}")
        op = str(doc.get("op") or "").strip()
        if op not in OPS:
            raise DeriveError(
                f"未知算子 {op!r}；允许：{', '.join(sorted(OPS))}")
        sources = tuple(str(s) for s in (doc.get("sources") or ()))
        if not sources:
            raise DeriveError(f"{prop}: sources 不能为空")
        conditions = tuple(str(s) for s in (doc.get("conditions") or ()))
        weights = tuple(float(w) for w in (doc.get("weights") or ()))
        if op == "sum_where" and len(conditions) != len(sources):
            raise DeriveError(
                f"{prop}: sum_where 要求 conditions 与 sources 一一对应"
                f"（{len(conditions)} vs {len(sources)}）")
        if op == "weighted_mean":
            if len(weights) != len(sources):
                raise DeriveError(
                    f"{prop}: weighted_mean 要求 weights 与 sources 等长")
            if sum(weights) <= 0:
                raise DeriveError(f"{prop}: weights 之和必须为正")
        if op == "diff" and len(sources) != 2:
            raise DeriveError(f"{prop}: diff 需要恰好 2 个源")
        return cls(
            property_code=prop, op=op, sources=sources,
            unit=(str(doc["unit"]) if doc.get("unit") else None),
            dtype=(str(doc["dtype"]) if doc.get("dtype") else None),
            role=str(doc.get("role") or "state"),
            source_kind=str(doc.get("source_kind") or "estimated"),
            conditions=conditions, weights=weights,
            description=str(doc.get("description") or ""),
        )

    def spec_text(self) -> str:
        """写进 lineage 的人类可读派生式。"""
        if self.op == "sum_where":
            pairs = ", ".join(f"{s} where {c}"
                              for s, c in zip(self.sources, self.conditions))
            return f"sum({pairs})"
        if self.op == "weighted_mean":
            pairs = ", ".join(f"{w}*{s}"
                              for w, s in zip(self.weights, self.sources))
            return f"weighted_mean({pairs})"
        return f"{self.op}({', '.join(self.sources)})"


# ---------------------------------------------------------------- 算子实现


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise DeriveError(f"源变量在该数据版本中不存在: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)


def _boolean(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise DeriveError(f"源变量在该数据版本中不存在: {col}")
    return df[col].eq(True).to_numpy(bool)


def _apply(df: pd.DataFrame, rule: DeriveRule) -> np.ndarray | list[Any]:
    """执行一条规则。缺失一律用 NaN 承载，落盘前再转 None。"""
    op = rule.op
    if op in ("count_true", "any_true"):
        stacked = np.vstack([_boolean(df, s) for s in rule.sources])
        if op == "count_true":
            return stacked.sum(axis=0).astype(np.int64)
        return stacked.any(axis=0)

    cols = np.vstack([_numeric(df, s) for s in rule.sources])
    finite = np.isfinite(cols)

    if op == "first":
        out = np.full(cols.shape[1], np.nan)
        for row, mask in zip(cols, finite):  # 后面的不覆盖前面已填的
            fill = mask & ~np.isfinite(out)
            out[fill] = row[fill]
        return out
    if op == "diff":
        both = finite.all(axis=0)
        return np.where(both, cols[0] - cols[1], np.nan)
    if op == "sum_where":
        conds = np.vstack([_boolean(df, c) for c in rule.conditions])
        counted = conds.sum(axis=0)
        # 条件为 true 但取值缺失 → 整体缺失，绝不用部分数据凑和
        broken = (conds & ~finite).any(axis=0)
        total = np.where(conds, np.nan_to_num(cols, nan=0.0), 0.0).sum(axis=0)
        total = np.where(counted == 0, 0.0, total)
        return np.where(broken, np.nan, total)

    both = finite.all(axis=0)
    if op == "sum":
        return np.where(both, cols.sum(axis=0), np.nan)
    if op == "mean":
        return np.where(both, cols.mean(axis=0), np.nan)
    weights = np.array(rule.weights, dtype=np.float64).reshape(-1, 1)
    weighted = (cols * weights).sum(axis=0) / weights.sum()
    return np.where(both, weighted, np.nan)


def apply_rules(df: pd.DataFrame,
                rules: Sequence[DeriveRule]) -> pd.DataFrame:
    """源宽表 → 派生帧（确定性纯函数）。"""
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    for rule in rules:
        out[rule.property_code] = _apply(df, rule)
    return out


# ---------------------------------------------------------------- 元数据


def _infer_unit(rule: DeriveRule,
                variables: Mapping[str, Mapping[str, Any]]) -> str:
    """未声明单位时从源变量继承，并校验一致性。"""
    if rule.unit:
        return rule.unit
    _dtype, needs_same_unit, _needs_bool = OPS[rule.op]
    if not needs_same_unit:
        return UNITLESS
    units = {str(variables[s]["unit"]) for s in rule.sources
             if s in variables}
    if not units:
        return UNITLESS
    if len(units) > 1:
        raise DeriveError(
            f"{rule.property_code}: 源变量单位不一致 {sorted(units)}，"
            "请在规则里显式声明 unit（换算须在导入边界完成）")
    return units.pop()


def build_object_model(model_id: str, rules: Sequence[DeriveRule],
                       units: Mapping[str, str],
                       dtypes: Mapping[str, str]) -> ObjectModel:
    """按派生规则生成 TFOM（object_model_id 形如 `name.vN`）。

    派生对象的物模型由规则唯一确定，因此就地生成而不要求预先注册。
    契约只允许 `role=derived` 的属性携带 `expression`（tfom.py），
    所以派生式默认写进 `name_zh`，与 plant.v1 的写法一致；完整规则另
    记于数据集 lineage。
    """
    properties: dict[str, Any] = {}
    for rule in rules:
        prop: dict[str, Any] = {
            "unit": units[rule.property_code],
            "dtype": dtypes[rule.property_code],
            "role": rule.role,
            "name_zh": (f"{rule.description}（{rule.spec_text()}）"
                        if rule.description else rule.spec_text()),
        }
        if rule.role == "derived":
            prop["expression"] = rule.spec_text()
        properties[rule.property_code] = prop
    name, _sep, major = model_id.rpartition(".v")
    return ObjectModel.model_validate({
        "model_id": name,
        "version": f"{int(major)}.0",
        "properties": properties,
    })


def build_dataset(dataset_id: str, object_id: str, model_id: str,
                  rules: Sequence[DeriveRule], units: Mapping[str, str],
                  dtypes: Mapping[str, str],
                  source_manifest: Mapping[str, Any],
                  object_name: str | None,
                  description: str | None) -> TfdcDataset:
    """派生数据集的 TFDC 元数据（站点/时区/分辨率继承源数据版本）。"""
    manifest = TfdcManifest(
        contract="TFDC", contract_version="1.0", dataset_id=dataset_id,
        dataset_version=1,
        site_id=str(source_manifest.get("site_id") or "NA"),
        timezone=str(source_manifest.get("timezone") or "Asia/Shanghai"),
        time_resolution=str(source_manifest.get("time_resolution") or "900s"),
        source_system="derived",
        description=description or f"{object_id} 的派生数据集",
    )
    objects = [ObjectRecord(object_id=object_id, object_model_id=model_id,
                            object_name=object_name or object_id)]
    resolution = manifest.time_resolution
    variables = [
        VariableRecord(
            variable_id=f"{object_id}.{rule.property_code}",
            object_id=object_id, property_code=rule.property_code,
            unit=units[rule.property_code],
            dtype=dtypes[rule.property_code], role=rule.role,
            source_kind=rule.source_kind,  # type: ignore[arg-type]
            name_zh=rule.description or rule.property_code,
            nullable=True, sample_period=resolution,
            aggregation="mean",
        )
        for rule in rules
    ]
    return TfdcDataset(manifest=manifest, objects=objects,
                       variables=variables)


def resolve_metadata(
    rules: Sequence[DeriveRule],
    source_variables: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """确定每个派生属性的单位与 dtype。"""
    by_id = {str(v["variable_id"]): v for v in source_variables}
    units: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    for rule in rules:
        missing = [s for s in (*rule.sources, *rule.conditions)
                   if s not in by_id]
        if missing:
            raise DeriveError(
                f"{rule.property_code}: 源变量不存在 {missing}")
        default_dtype, _same_unit, needs_bool = OPS[rule.op]
        if needs_bool:
            bad = [s for s in rule.sources
                   if str(by_id[s]["dtype"]) != "boolean"]
            if bad:
                raise DeriveError(
                    f"{rule.property_code}: {rule.op} 要求布尔源，"
                    f"但这些不是：{bad}")
        if rule.op == "sum_where":
            bad = [c for c in rule.conditions
                   if str(by_id[c]["dtype"]) != "boolean"]
            if bad:
                raise DeriveError(
                    f"{rule.property_code}: conditions 必须是布尔变量：{bad}")
        units[rule.property_code] = _infer_unit(rule, by_id)
        dtypes[rule.property_code] = rule.dtype or default_dtype
    return units, dtypes


def divergence_report(df: pd.DataFrame,
                      rules: Sequence[DeriveRule]) -> dict[str, Any]:
    """对 `first`（多路冗余测点取一）复核各路一致性。

    取其一的前提是各路本来就一致；不复核就等于把「假设」当「事实」，
    因此这里把最大偏差记进 lineage，由使用者判断该假设成不成立。
    """
    out: dict[str, Any] = {}
    for rule in rules:
        if rule.op != "first" or len(rule.sources) < 2:
            continue
        cols = [pd.to_numeric(df[s], errors="coerce").to_numpy(np.float64)
                for s in rule.sources]
        base = cols[0]
        worst, compared = 0.0, 0
        for other in cols[1:]:
            mask = np.isfinite(base) & np.isfinite(other)
            if mask.any():
                worst = max(worst, float(np.max(np.abs(base[mask]
                                                       - other[mask]))))
                compared = max(compared, int(mask.sum()))
        out[rule.property_code] = {"max_abs_diff": worst,
                                   "compared_samples": compared}
    return out


def build_registry(model: ObjectModel,
                   base: TfomRegistry | None = None) -> TfomRegistry:
    """把生成的 TFOM 合并进注册表，供 import_parsed 校验使用。"""
    models = dict((base or default_registry())._models)
    models[model.object_model_id] = model
    return TfomRegistry(models)


def lineage(source_ref: str, rules: Sequence[DeriveRule],
            divergence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapter": "thermoforge_data.derive",
        "derivation_version": DERIVATION_VERSION,
        "derived_from": source_ref,
        "derivation_spec": {r.property_code: r.spec_text() for r in rules},
        "redundancy_check": dict(divergence),
    }


def to_columns(frame: pd.DataFrame, object_id: str,
               dtypes: Mapping[str, str]) -> dict[str, list[Any]]:
    """派生帧 → import_parsed 的列字典。

    契约口径：缺失是 null 而不是 NaN（TFDC-602），布尔/整数列同样要在
    这里转成 None，否则 NaN 会被当成非有限值报错。
    """
    columns: dict[str, list[Any]] = {}
    for prop in frame.columns:
        if prop == "timestamp":
            continue
        values = frame[prop].tolist()
        dtype = dtypes.get(prop, "float")
        out: list[Any] = []
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(None)
            elif dtype == "integer":
                out.append(int(v))
            elif dtype == "boolean":
                out.append(bool(v))
            else:
                out.append(float(v))
        columns[f"{object_id}.{prop}"] = out
    return columns
