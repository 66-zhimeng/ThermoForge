"""数据预处理规则库与执行器（I-49、risks R8、design-decisions DD-02 方案 C）。

核心原则：Agent 只能**提出规则参数**，执行必须是预先注册的确定性变换——
不允许 Agent 生成自由代码。两个原人工脚本固化为库内变换：

- `scripts/fill_cooling_tower.py` → `fill_from_header_divide_by_count`
- `scripts/fix_chwp.py` → `repair_time_axis` + `fix_header_labels`

审批门禁：`status=proposed` 的规则不得作用于产生 vault revision 的正式导入
（`apply_ruleset(..., require_approved=True)`）；`actor=human` 审批置为
approved 后方可，审批全程留痕（`ApprovalRecord`）。

错误码：预处理域使用 `TFPP-` 前缀。conventions.md §7 错误码表当前无预处理
小节（冻结文档不得改），本模块自带小注册表，待契约增补后并入（issues I-53）。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from thermoforge_core.canonical import canonical_json, sha256_hex

from .legacy import LegacySheet, _parse_multi_header

# ---------------------------------------------------------------- 错误码（预处理域）

TFPP_RULE_TYPE_UNKNOWN = "TFPP-001"      # 规则类型未在规则库注册
TFPP_RULE_NOT_APPROVED = "TFPP-002"      # 未审批规则作用于正式导入
TFPP_RULESET_NOT_FOUND = "TFPP-003"      # 规则集不存在
TFPP_RULE_TARGET_UNKNOWN = "TFPP-004"    # 规则目标 sheet/列不存在
TFPP_RULESET_VERSION_CONFLICT = "TFPP-005"  # 同版本规则集内容变更
TFPP_APPROVAL_FORBIDDEN = "TFPP-006"     # 非 human 审批

RULE_STATUSES = ("proposed", "approved", "deprecated")


class PreprocessError(RuntimeError):
    """预处理错误，携带 TFPP-xxx 错误码（机器可判定）。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


# ---------------------------------------------------------------- 规则 Schema


class ApprovalRecord(BaseModel):
    """审批留痕。"""

    model_config = ConfigDict(extra="forbid")

    actor: str
    action: Literal["approve", "deprecate"]
    at: str  # ISO 8601 UTC
    note: str | None = None


class PreprocessRule(BaseModel):
    """一条预处理规则：类型 + 目标 + 参数 + 审批状态。"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    rule_type: str
    sheet: str  # 目标工作表
    params: dict[str, Any] = Field(default_factory=dict)
    status: Literal["proposed", "approved", "deprecated"] = "proposed"
    proposer: str = "agent"
    rationale: str | None = None
    approvals: list[ApprovalRecord] = Field(default_factory=list)

    @field_validator("rule_type")
    @classmethod
    def _rule_type_registered(cls, v: str) -> str:
        if v not in RULE_LIBRARY:
            raise ValueError(
                f"规则类型未在规则库注册: {v!r}（已注册 {sorted(RULE_LIBRARY)}）"
            )
        return v


class RuleSet(BaseModel):
    """版本化规则集：ruleset_id + version + 规则列表 + 内容哈希。

    内容哈希只覆盖变换语义（rule_id/rule_type/sheet/params），
    status/approvals/rationale 等留痕元数据不参与（审批不改变内容指纹）。
    """

    model_config = ConfigDict(extra="forbid")

    ruleset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    version: int = Field(ge=1)
    rules: list[PreprocessRule] = Field(min_length=1)

    @field_validator("rules")
    @classmethod
    def _rule_ids_unique(cls, v: list[PreprocessRule]) -> list[PreprocessRule]:
        ids = [r.rule_id for r in v]
        if len(set(ids)) != len(ids):
            raise ValueError("rule_id 在规则集内重复")
        return v

    def content_hash(self) -> str:
        doc = [
            {
                "rule_id": r.rule_id,
                "rule_type": r.rule_type,
                "sheet": r.sheet,
                "params": r.params,
            }
            for r in sorted(self.rules, key=lambda r: r.rule_id)
        ]
        return sha256_hex(canonical_json(doc))

    @property
    def ref(self) -> str:
        return f"{self.ruleset_id}@v{self.version}"


# ---------------------------------------------------------------- 各规则类型的参数模型


class FillFromHeaderDivideByCountParams(BaseModel):
    """`fill_from_header_divide_by_count` 参数（冷却塔填充规则的泛化）。"""

    model_config = ConfigDict(extra="forbid")

    source_sheet: str  # 总管所在表（如 冷却水总管）
    source_instance: str  # 总管实例（如 cw_A1）
    value_map: dict[str, str]  # 目标属性 → 总管属性（如 supply_t → t_supply）
    divide_prop: str  # 均分属性（如 instant_flow）
    divide_source_prop: str  # 均分的来源属性（如 f）
    running_sheet: str  # 运行状态所在表（如 冷却塔风机）
    running_prop: str  # 运行状态属性（如 status_run）
    member_pattern: str = "{instance}_f"  # 成员实例前缀（ct_01 → ct_01_f_*）
    leave_empty: list[str] = Field(default_factory=list)  # 始终留空（supply_p）
    idle_value: float = 0.0  # 实例未运行时的填充值
    round_decimals: int | None = None  # 写出小数位（复现脚本的 round(v, 10)）


class RepairTimeAxisParams(BaseModel):
    """`repair_time_axis` 参数（冷冻水泵时间轴修复的泛化）。"""

    model_config = ConfigDict(extra="forbid")

    reference_sheet: str  # 参考时间轴所在表
    on_duplicate: Literal["keep_first"] = "keep_first"
    fill: Literal["blank"] = "blank"  # 补齐行只写时间戳（不插值，data-contract §6）


class FixHeaderLabelsParams(BaseModel):
    """`fix_header_labels` 参数：A 列标签修正（1 起始行号 → 标签）。"""

    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str]


_PARAMS_MODELS: dict[str, type[BaseModel]] = {
    "fill_from_header_divide_by_count": FillFromHeaderDivideByCountParams,
    "repair_time_axis": RepairTimeAxisParams,
    "fix_header_labels": FixHeaderLabelsParams,
}


def validate_rule_params(rule: PreprocessRule) -> BaseModel:
    """按规则类型校验参数（propose 时即失败，而非执行时）。"""
    model = _PARAMS_MODELS[rule.rule_type]
    return model.model_validate(rule.params)


# ---------------------------------------------------------------- 规则库：确定性变换


def _col_index(sheet: LegacySheet) -> dict[tuple[str, str], int]:
    """(instance, property) → 列号（复用 legacy 的 4 行表头解析）。"""
    cols, _ = _parse_multi_header(sheet.header_rows)
    return {(inst, prop): idx for idx, inst, _m, prop in cols}


def _ts_index(sheet: LegacySheet) -> dict[Any, int]:
    """时间戳 → 行号（取首次出现）。"""
    out: dict[Any, int] = {}
    for i, row in enumerate(sheet.data_rows):
        ts = row[0] if row else None
        if ts is not None and ts not in out:
            out[ts] = i
    return out


def _rule_fill_from_header_divide_by_count(
    sheets: dict[str, LegacySheet], rule: PreprocessRule
) -> dict[str, Any]:
    """从总管列取值填入实例列；流量类属性按运行实例数均分；未运行填 idle_value。

    固化的原脚本语义（fill_cooling_tower.py）：运行实例的 supply/return 取总管
    对应列，instant_flow = 总管 f ÷ 运行实例总数；未运行实例全部填 0；
    leave_empty 列恒为空；round_decimals 复现脚本 round(v, 10) 的写出精度。
    """
    params = FillFromHeaderDivideByCountParams.model_validate(rule.params)
    for name in (rule.sheet, params.source_sheet, params.running_sheet):
        if name not in sheets:
            raise PreprocessError(
                TFPP_RULE_TARGET_UNKNOWN, f"工作表不存在: {name!r}"
            )
    target = sheets[rule.sheet]
    source = sheets[params.source_sheet]
    running_sheet = sheets[params.running_sheet]
    t_cols = _col_index(target)
    s_cols = _col_index(source)
    r_cols = _col_index(running_sheet)
    s_rows = _ts_index(source)
    r_rows = _ts_index(running_sheet)

    instances = sorted({inst for inst, _ in t_cols})
    members = {
        inst: sorted(
            i for i, p in r_cols
            if p == params.running_prop
            and i.startswith(params.member_pattern.format(instance=inst))
        )
        for inst in instances
    }
    filled_cells = 0

    def _src(row_map, cols, ts, inst, prop):
        i = row_map.get(ts)
        j = cols.get((inst, prop))
        if i is None or j is None:
            return None
        row = (source if cols is s_cols else running_sheet).data_rows[i]
        return row[j] if j < len(row) else None

    for i, row in enumerate(target.data_rows):
        ts = row[0] if row else None
        if ts is None:
            continue
        running = {
            inst: any(
                _src(r_rows, r_cols, ts, m, params.running_prop) == 1
                for m in members[inst]
            )
            for inst in instances
        }
        n_running = sum(running.values())
        new_row = list(row)
        for inst in instances:
            props = [p for ii, p in t_cols if ii == inst]
            for prop in props:
                j = t_cols[(inst, prop)]
                if prop in params.leave_empty:
                    value = None
                elif prop == params.divide_prop:
                    f = _src(s_rows, s_cols, ts, params.source_instance,
                             params.divide_source_prop)
                    if running[inst] and n_running and isinstance(
                            f, (int, float)) and not isinstance(f, bool):
                        value = f / n_running
                    elif running[inst]:
                        value = None
                    else:
                        value = params.idle_value
                elif prop in params.value_map:
                    if running[inst]:
                        value = _src(s_rows, s_cols, ts, params.source_instance,
                                     params.value_map[prop])
                    else:
                        value = params.idle_value
                else:
                    continue  # 未在规则中声明的列不动
                if isinstance(value, float) and params.round_decimals is not None:
                    value = round(value, params.round_decimals)
                while j >= len(new_row):
                    new_row.append(None)
                if new_row[j] != value:
                    filled_cells += 1
                new_row[j] = value
        target.data_rows[i] = tuple(new_row)
    return {
        "instances": instances,
        "filled_cells": filled_cells,
        "rows": len(target.data_rows),
    }


def _rule_repair_time_axis(
    sheets: dict[str, LegacySheet], rule: PreprocessRule
) -> dict[str, Any]:
    """去重复时间戳（保留首次出现）+ 按参考表时间轴补齐空行 / 剔除非轴行。

    固化的原脚本语义（fix_chwp.py）：干净前缀逐行保留，重复粘贴块删除，
    尾部按规范时间轴补齐仅含时间戳的空行（不插值）。
    """
    params = RepairTimeAxisParams.model_validate(rule.params)
    if rule.sheet not in sheets or params.reference_sheet not in sheets:
        raise PreprocessError(
            TFPP_RULE_TARGET_UNKNOWN,
            f"工作表不存在: {rule.sheet!r} 或 {params.reference_sheet!r}",
        )
    target = sheets[rule.sheet]
    reference = sheets[params.reference_sheet]
    width = max((len(r) for r in target.data_rows), default=1)

    ref_axis: list[Any] = []
    seen_ref: set[Any] = set()
    for row in reference.data_rows:
        ts = row[0] if row else None
        if ts is not None and ts not in seen_ref:
            seen_ref.add(ts)
            ref_axis.append(ts)

    kept: dict[Any, tuple[Any, ...]] = {}
    duplicates = 0
    for row in target.data_rows:
        ts = row[0] if row else None
        if ts is None:
            duplicates += 1
            continue
        if ts in kept:
            duplicates += 1  # 重复时间戳：保留首次出现
        else:
            kept[ts] = row

    out_rows: list[tuple[Any, ...]] = []
    off_axis = 0
    blanks = 0
    for ts in ref_axis:
        row = kept.pop(ts, None)
        if row is None:
            blanks += 1
            out_rows.append(tuple([ts] + [None] * (width - 1)))
        else:
            padded = tuple(list(row) + [None] * (width - len(row)))
            out_rows.append(padded)
    off_axis = len(kept)  # 不在参考轴上的行剔除
    target.data_rows = out_rows
    return {
        "input_rows": len(target.data_rows) + duplicates + off_axis - blanks,
        "duplicates_removed": duplicates,
        "off_axis_removed": off_axis,
        "blank_rows_appended": blanks,
        "output_rows": len(out_rows),
    }


def _rule_fix_header_labels(
    sheets: dict[str, LegacySheet], rule: PreprocessRule
) -> dict[str, Any]:
    """修正多级表头的 A 列标签（如「设备名称/物模型」两行标签互换）。"""
    params = FixHeaderLabelsParams.model_validate(rule.params)
    if rule.sheet not in sheets:
        raise PreprocessError(
            TFPP_RULE_TARGET_UNKNOWN, f"工作表不存在: {rule.sheet!r}"
        )
    sheet = sheets[rule.sheet]
    changes: dict[str, dict[str, Any]] = {}
    for row_no_s, label in params.labels.items():
        row_no = int(row_no_s)
        if not (1 <= row_no <= len(sheet.header_rows)):
            raise PreprocessError(
                TFPP_RULE_TARGET_UNKNOWN,
                f"表头行号超出范围: {row_no}（{rule.sheet}）",
            )
        old = sheet.header_rows[row_no - 1][0]
        changes[row_no_s] = {"from": old, "to": label}
        sheet.header_rows[row_no - 1][0] = label
    return {"labels_fixed": changes}


RULE_LIBRARY: dict[str, Callable[[dict[str, LegacySheet], PreprocessRule], dict[str, Any]]] = {
    "fill_from_header_divide_by_count": _rule_fill_from_header_divide_by_count,
    "repair_time_axis": _rule_repair_time_axis,
    "fix_header_labels": _rule_fix_header_labels,
}


# ---------------------------------------------------------------- 执行器


def _sheet_digest(sheet: LegacySheet) -> str:
    """表内容指纹（规则输入/输出留痕用）。"""
    h = hashlib.sha256()
    h.update(sheet.name.encode("utf-8"))
    for row in [*sheet.header_rows, *sheet.data_rows]:
        h.update(b"\x02")
        for v in row:
            if v is None:
                h.update(b"\x00")
            elif isinstance(v, datetime):
                h.update(b"\x01" + v.isoformat().encode("utf-8"))
            elif isinstance(v, bool):
                h.update(b"\x03" + (b"1" if v else b"0"))
            elif isinstance(v, float):
                h.update(b"\x04" + repr(v).encode("ascii"))
            else:
                h.update(b"\x05" + str(v).encode("utf-8"))
    return h.hexdigest()


@dataclass
class RuleExecution:
    """单条规则的执行留痕（rule_id、参数、输入/输出指纹）。"""

    rule_id: str
    rule_type: str
    sheet: str
    status: str
    skipped: bool
    input_sha256: str | None
    output_sha256: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "sheet": self.sheet,
            "status": self.status,
            "skipped": self.skipped,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "detail": self.detail,
        }


def apply_ruleset(
    sheets: dict[str, LegacySheet],
    ruleset: RuleSet,
    *,
    require_approved: bool = False,
) -> list[RuleExecution]:
    """按规则集顺序执行确定性变换（就地修改 `sheets`）。

    `require_approved=True`（产生 vault revision 的正式导入）时，
    非 approved 规则报 TFPP-002 且不做任何修改；deprecated 规则跳过并留痕。
    """
    if require_approved:
        # deprecated 规则本就跳过不执行，门禁只拦截 proposed
        not_approved = [r.rule_id for r in ruleset.rules
                        if r.status == "proposed"]
        if not_approved:
            raise PreprocessError(
                TFPP_RULE_NOT_APPROVED,
                f"规则未审批，不得作用于正式导入: {not_approved}"
                f"（需 actor=human 审批）",
            )
    executions: list[RuleExecution] = []
    for rule in ruleset.rules:
        if rule.status == "deprecated":
            executions.append(RuleExecution(
                rule.rule_id, rule.rule_type, rule.sheet, rule.status,
                skipped=True, input_sha256=None, output_sha256=None,
                detail={"note": "deprecated 规则跳过"},
            ))
            continue
        if rule.sheet not in sheets:
            raise PreprocessError(
                TFPP_RULE_TARGET_UNKNOWN,
                f"规则目标工作表不存在: {rule.sheet!r}",
            )
        before = _sheet_digest(sheets[rule.sheet])
        detail = RULE_LIBRARY[rule.rule_type](sheets, rule)
        after = _sheet_digest(sheets[rule.sheet])
        executions.append(RuleExecution(
            rule.rule_id, rule.rule_type, rule.sheet, rule.status,
            skipped=False, input_sha256=before, output_sha256=after,
            detail=detail,
        ))
    return executions


# ---------------------------------------------------------------- 规则集存储


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuleStore:
    """规则集的版本化存储（`<root>/rulesets/<id>.v<N>.yaml`）。

    同一 `(ruleset_id, version)` 的内容不可变：content_hash 不同而版本相同
    报 TFPP-005；仅 status/approvals 等留痕元数据允许原地更新。
    """

    def __init__(self, root: str | Path):
        self.dir = Path(root) / "rulesets"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ruleset_id: str, version: int) -> Path:
        return self.dir / f"{ruleset_id}.v{version}.yaml"

    def save(self, ruleset: RuleSet) -> None:
        path = self._path(ruleset.ruleset_id, ruleset.version)
        if path.exists():
            existing = self.load(ruleset.ruleset_id, ruleset.version)
            if existing.content_hash() != ruleset.content_hash():
                raise PreprocessError(
                    TFPP_RULESET_VERSION_CONFLICT,
                    f"规则集 {ruleset.ref} 已存在且内容不同，"
                    "请递增 version",
                )
        doc = ruleset.model_dump(mode="json")
        doc["content_hash"] = ruleset.content_hash()
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            yaml.safe_dump(doc, fp, allow_unicode=True, sort_keys=True)
        os.replace(tmp, path)

    def load(self, ruleset_id: str, version: int | None = None) -> RuleSet:
        if version is None:
            version = self.latest_version(ruleset_id)
        path = self._path(ruleset_id, version)
        if not path.exists():
            raise PreprocessError(
                TFPP_RULESET_NOT_FOUND,
                f"规则集不存在: {ruleset_id}@v{version}",
            )
        with open(path, encoding="utf-8") as fp:
            doc = yaml.safe_load(fp)
        doc.pop("content_hash", None)
        return RuleSet.model_validate(doc)

    def latest_version(self, ruleset_id: str) -> int:
        versions = [
            int(p.stem.rsplit(".v", 1)[1])
            for p in self.dir.glob(f"{ruleset_id}.v*.yaml")
        ]
        if not versions:
            raise PreprocessError(
                TFPP_RULESET_NOT_FOUND, f"规则集不存在: {ruleset_id}"
            )
        return max(versions)

    def list_rulesets(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.dir.glob("*.v*.yaml")):
            with open(path, encoding="utf-8") as fp:
                doc = yaml.safe_load(fp)
            out.append({
                "ruleset_id": doc["ruleset_id"],
                "version": doc["version"],
                "content_hash": doc.get("content_hash"),
                "rules": [
                    {"rule_id": r["rule_id"], "rule_type": r["rule_type"],
                     "sheet": r["sheet"], "status": r["status"]}
                    for r in doc["rules"]
                ],
            })
        return out
