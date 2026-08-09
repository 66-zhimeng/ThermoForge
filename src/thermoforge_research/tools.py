"""Agent 工具层（architecture.md §3/§6、implementation-notes.md §11）。

architecture §6 工具清单的确定性实现，薄封装 Phase 1/2/4 能力：

- 数据：tf_dataset_import / list / get / schema / profile / query /
  sample / materialize / compare
- 研究：tf_goal_create / tf_research_status / tf_hypothesis_create /
  tf_experiment_plan / tf_experiment_run / tf_experiment_get /
  tf_model_compare / tf_model_publish

约定：

- 每个工具返回统一信封（`envelope.py`）：ok/tool/id/status/inputs/
  summary/diagnostics/artifacts/truncated；有副作用的工具返回稳定 ID
  （dataset ref、RG-/H-/EXP-/VIEW-/model@version），Agent 后续引用
  ID 而非重新描述输入。
- Agent 只拿到摘要与有限样本：sample 硬上限 200 行，profile 分位数
  点位固定，query 只返回聚合统计；响应体 32 KB 上限截断。
- 已知错误（VaultError / ResearchError / ModelRegistryError / 契约
  校验错误）不抛出，转为 ok=False 的信封；诊断携带原错误码。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from pydantic import ValidationError

from thermoforge_core.contracts.experiment import Experiment
from thermoforge_core.contracts.research_goal import ResearchGoal
from thermoforge_core.timeutil import parse_timestamp
from thermoforge_data.importer import TfomRegistry, default_registry, import_parsed, import_xlsx
from thermoforge_data.legacy import convert_legacy_tables, load_workbook_model
from thermoforge_data.preprocess import (
    TFPP_APPROVAL_FORBIDDEN,
    ApprovalRecord,
    PreprocessError,
    RuleSet,
    RuleStore,
    apply_ruleset,
    validate_rule_params,
)
from thermoforge_data.vault import DataVault, VaultError
from thermoforge_data.views import (
    materialize_view,
    validate_view_definition,
    view_hash,
)
from thermoforge_research.envelope import (
    SAMPLE_MAX_ROWS,
    diagnostic_dicts,
    finalize_envelope,
    make_envelope,
)
from thermoforge_research.ledger import ResearchLedger
from thermoforge_research.runner import run_experiment
from thermoforge_research.splits import DEFAULT_EMBARGO_SECONDS
from thermoforge_runtime.errors import ModelRegistryError
from thermoforge_runtime.package import boundary_rows, build_model_package
from thermoforge_runtime.registry import ModelRegistry

from .errors import ResearchError
from .modelability import ModelabilityConfig, build_modelability_report
from .whitelist import check_candidate_inputs, check_view_within_whitelist

QUERY_AGGREGATIONS = ("count", "missing", "mean", "min", "max", "std")

# tf_dataset_sample 未指定列时的默认上限（防 32KB 截断掩盖列选择失误）
SAMPLE_DEFAULT_MAX_COLUMNS = 12


class ToolContext:
    """工具调用的运行上下文（控制面持有一次，逐工具传入）。

    ::

        ctx = ToolContext(vault_root="vault", research_root="research",
                          models_root="models")
        env = tf_dataset_import(ctx, "data/workbook.xlsx")
    """

    def __init__(
        self,
        *,
        vault_root: str | Path,
        research_root: str | Path,
        models_root: str | Path | None = None,
        tfom_registry: TfomRegistry | None = None,
        actor: str = "agent",
    ):
        self.vault_root = Path(vault_root)
        self.research_root = Path(research_root)
        self.models_root = (
            Path(models_root) if models_root else self.research_root.parent / "models"
        )
        self.tfom_registry = tfom_registry
        self.actor = actor
        self.artifacts_root = self.research_root / "tool_artifacts"
        self._vault: DataVault | None = None
        self._ledger: ResearchLedger | None = None
        self._registry: ModelRegistry | None = None
        self._preprocess_store: RuleStore | None = None

    @property
    def vault(self) -> DataVault:
        if self._vault is None:
            self._vault = DataVault(self.vault_root)
        return self._vault

    @property
    def ledger(self) -> ResearchLedger:
        if self._ledger is None:
            self._ledger = ResearchLedger(self.research_root)
        return self._ledger

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = ModelRegistry(self.models_root)
        return self._registry

    @property
    def view_cache_root(self) -> Path:
        return self.research_root / "view_cache"

    @property
    def preprocess_store(self) -> RuleStore:
        if self._preprocess_store is None:
            self._preprocess_store = RuleStore(self.research_root / "preprocess")
        return self._preprocess_store


def _finish(ctx: ToolContext, env: dict[str, Any]) -> dict[str, Any]:
    return finalize_envelope(env, ctx.artifacts_root)


def _error_envelope(
    ctx: ToolContext, tool: str, exc: Exception,
    *, inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """已知错误 → ok=False 信封（诊断携带原错误码，不用自由文本猜测）。"""
    code = getattr(exc, "code", None) or type(exc).__name__
    return _finish(ctx, make_envelope(
        tool, ok=False, status="FAILED", inputs=inputs,
        summary={"error": str(exc)},
        diagnostics=[{"code": code, "level": "ERROR", "count": 1,
                      "message": str(exc)[:500]}],
    ))


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {"kind": kind, "path": str(path),
            "bytes": path.stat().st_size if path.exists() else 0}


def _rev_dir(ctx: ToolContext, ref: str) -> Path:
    return ctx.vault.resolve(ref).path


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


# ---------------------------------------------------------------- 数据工具


def tf_dataset_import(
    ctx: ToolContext,
    path: str | Path,
    *,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """导入 TFDC-XLSX 并落 Data Vault，返回稳定 `dataset_id@rev_NNNN`。"""
    tool = "tf_dataset_import"
    result = import_xlsx(path, registry=ctx.tfom_registry)
    diagnostics = diagnostic_dicts(result.diagnostics)
    if not result.ok:
        return _finish(ctx, make_envelope(
            tool, ok=False, status="FAILED",
            inputs={"path": str(path)},
            summary={"error": "存在 ERROR 级诊断，未写入 vault"},
            diagnostics=diagnostics,
        ))
    try:
        ref = ctx.vault.store(result, source_path=path, lineage=lineage)
    except VaultError as exc:
        return _error_envelope(ctx, tool, exc, inputs={"path": str(path)})
    rev = _rev_dir(ctx, ref)
    has_warn = any(d["level"] in ("WARN", "REJECT") for d in diagnostics)
    table = result.table
    assert table is not None and result.dataset is not None
    ts = table.column("timestamp").to_pylist()
    return _finish(ctx, make_envelope(
        tool, ok=True, id=ref,
        status="IMPORTED_WITH_WARNINGS" if has_warn else "IMPORTED",
        inputs={"path": str(path), "importer_version": result.importer_version},
        summary={
            "dataset_ref": ref,
            "rows": table.num_rows,
            "variables": len(result.dataset.variables),
            "objects": len(result.dataset.objects),
            "time_range": [ts[0].isoformat() if ts else None,
                           ts[-1].isoformat() if ts else None],
            "degradations": list(result.degradations),
        },
        diagnostics=diagnostics,
        artifacts=[_artifact(rev / "quality.json", "quality_report"),
                   _artifact(rev / "profile.json", "profile"),
                   _artifact(rev / "manifest.json", "manifest")],
    ))


def tf_dataset_list(ctx: ToolContext) -> dict[str, Any]:
    """列出全部数据集及 revision 摘要。"""
    datasets = ctx.vault.list_datasets()
    return _finish(ctx, make_envelope(
        "tf_dataset_list",
        summary={"datasets": datasets, "count": len(datasets)},
    ))


def tf_dataset_get(ctx: ToolContext, ref: str) -> dict[str, Any]:
    """单个数据版本的清单、质量与谱系摘要。"""
    tool = "tf_dataset_get"
    try:
        rev = _rev_dir(ctx, ref)
    except VaultError as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref})
    manifest = _read_json(rev / "manifest.json")
    quality = _read_json(rev / "quality.json")
    lineage = _read_json(rev / "lineage.json")
    return _finish(ctx, make_envelope(
        tool, id=ref, status="OK", inputs={"ref": ref},
        summary={
            "manifest": manifest,
            "quality": {
                "imported_at": quality.get("imported_at"),
                "degradations": quality.get("degradations", []),
                "diagnostics": quality.get("diagnostics", []),
            },
            "lineage": lineage,
        },
        diagnostics=quality.get("diagnostics", []),
        artifacts=[_artifact(rev / "quality.json", "quality_report"),
                   _artifact(rev / "profile.json", "profile")],
    ))


def tf_dataset_schema(ctx: ToolContext, ref: str) -> dict[str, Any]:
    """数据版本的变量 Schema（variable_id/unit/dtype/role/范围）。

    宽表数据集的完整 variables 明细始终落 artifact；信封 summary 携带
    紧凑清单（variable_ids/property_codes，供白名单与预检使用）+
    前 50 条明细（§11 响应上限）。
    """
    tool = "tf_dataset_schema"
    try:
        variables = ctx.vault.load_variables(ref)
        objects = ctx.vault.load_objects(ref)
    except VaultError as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref})
    out_dir = ctx.artifacts_root / "schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{ref.replace('@', '_')}-schema.json"
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump({"dataset_ref": ref, "variables": variables,
                   "objects": objects}, fp, ensure_ascii=False,
                  sort_keys=True, indent=2)
        fp.write("\n")
    os.replace(tmp, artifact_path)
    return _finish(ctx, make_envelope(
        tool, id=ref, status="OK", inputs={"ref": ref},
        summary={"variable_count": len(variables),
                 "variable_ids": sorted(v["variable_id"] for v in variables),
                 "property_codes": sorted({v["property_code"]
                                           for v in variables}),
                 "objects": objects,
                 "variables": variables[:50],
                 "variables_in_artifact": len(variables)},
        artifacts=[_artifact(artifact_path, "schema")],
    ))


def tf_dataset_profile(ctx: ToolContext, ref: str) -> dict[str, Any]:
    """数据画像摘要：分位数点位固定 min/p1/p25/p50/p75/p99/max（§11）。"""
    tool = "tf_dataset_profile"
    try:
        rev = _rev_dir(ctx, ref)
    except VaultError as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref})
    profile = _read_json(rev / "profile.json")
    variables = []
    for v in profile.get("variables", []):
        entry = {
            "variable_id": v["variable_id"],
            "unit": v.get("unit"),
            "dtype": v.get("dtype"),
            "role": v.get("role"),
            "count": v.get("count"),
            "missing_rate": v.get("missing_rate"),
        }
        if "quantiles" in v:
            q = v["quantiles"]
            entry["distribution"] = {
                "min": v.get("min"),
                "p01": q.get("p01"),
                "p25": q.get("p25"),
                "p50": q.get("p50"),
                "p75": q.get("p75"),
                "p99": q.get("p99"),
                "max": v.get("max"),
            }
            entry["outlier_count_iqr"] = v.get("outlier_count_iqr")
            entry["constant"] = v.get("constant")
        variables.append(entry)
    return _finish(ctx, make_envelope(
        tool, id=ref, status="OK", inputs={"ref": ref},
        summary={
            "dataset_id": profile.get("dataset_id"),
            "record_count": profile.get("record_count"),
            "time_range": profile.get("time_range"),
            "interval_stats": profile.get("interval_stats"),
            "gap_count": profile.get("gap_count"),
            "range_violations": profile.get("range_violations"),
            "coverage": profile.get("coverage"),
            "quantile_points": ["min", "p01", "p25", "p50", "p75", "p99", "max"],
            "variables": variables,
        },
        diagnostics=profile.get("diagnostics_summary", []),
        artifacts=[_artifact(rev / "profile.json", "profile")],
    ))


def _load_frame(
    ctx: ToolContext, ref: str,
    variable_ids: Sequence[str] | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """加载 canonical 数据并按时间窗过滤（工具内部，不暴露给 Agent）。"""
    table = ctx.vault.load_data(ref)
    df = table.to_pandas()
    if start is not None:
        df = df[df["timestamp"] >= parse_timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] < parse_timestamp(end)]
    if variable_ids:
        known = set(table.column_names)
        unknown = [v for v in variable_ids if v not in known]
        if unknown:
            raise VaultError("TFV-802", f"变量在该数据版本中不存在: {unknown}")
        df = df[["timestamp", *variable_ids]]
    return df


def tf_dataset_query(
    ctx: ToolContext,
    ref: str,
    *,
    variable_ids: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    aggregations: Sequence[str] = ("count", "missing", "mean", "min", "max"),
) -> dict[str, Any]:
    """聚合查询：只返回统计量，不返回原始行（§6：摘要不刷屏）。"""
    tool = "tf_dataset_query"
    bad = [a for a in aggregations if a not in QUERY_AGGREGATIONS]
    if bad:
        return _finish(ctx, make_envelope(
            tool, ok=False, status="FAILED", inputs={"ref": ref},
            summary={"error": f"未登记的聚合: {bad}（允许 {QUERY_AGGREGATIONS}）"},
        ))
    try:
        df = _load_frame(ctx, ref, variable_ids, start, end)
    except (VaultError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref})
    stats: dict[str, Any] = {}
    for var in variable_ids:
        col = df[var]
        non_null = col.dropna()
        entry: dict[str, Any] = {}
        for agg in aggregations:
            if agg == "count":
                entry["count"] = int(len(non_null))
            elif agg == "missing":
                entry["missing"] = int(len(col) - len(non_null))
            elif not len(non_null):
                entry[agg] = None
            elif agg == "mean":
                entry["mean"] = float(non_null.mean())
            elif agg == "min":
                entry["min"] = float(non_null.min())
            elif agg == "max":
                entry["max"] = float(non_null.max())
            elif agg == "std":
                entry["std"] = float(non_null.std()) if len(non_null) > 1 else 0.0
        stats[var] = entry
    return _finish(ctx, make_envelope(
        tool, id=ref, status="OK",
        inputs={"ref": ref, "start": start, "end": end,
                "aggregations": list(aggregations)},
        summary={"matched_rows": int(len(df)), "stats": stats},
    ))


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def tf_dataset_sample(
    ctx: ToolContext,
    ref: str,
    *,
    n: int = 50,
    variable_ids: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """有限样本：硬上限 200 行（§11），等距抽样（确定性）。"""
    tool = "tf_dataset_sample"
    capped = int(n) > SAMPLE_MAX_ROWS
    take = min(max(int(n), 1), SAMPLE_MAX_ROWS)
    try:
        df = _load_frame(ctx, ref, variable_ids, start, end)
    except (VaultError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref})
    if variable_ids is None and len(df.columns) > SAMPLE_DEFAULT_MAX_COLUMNS + 1:
        df = df.iloc[:, : SAMPLE_DEFAULT_MAX_COLUMNS + 1]
    total = len(df)
    if total > take:
        idx = [round(i * (total - 1) / (take - 1)) for i in range(take)] \
            if take > 1 else [0]
        df = df.iloc[sorted(set(idx))]
    rows = [
        {k: _json_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    env = make_envelope(
        tool, ok=True, id=ref, status="OK",
        inputs={"ref": ref, "requested": int(n), "start": start, "end": end},
        summary={
            "rows": rows,
            "returned": len(rows),
            "total_matched": int(total),
            "cap": SAMPLE_MAX_ROWS,
        },
    )
    if capped:
        env["truncated"] = True  # 策略性截断：请求超硬上限
    return _finish(ctx, env)


def tf_dataset_materialize(
    ctx: ToolContext,
    view_definition: Mapping[str, Any],
) -> dict[str, Any]:
    """登记并物化 Dataset View，返回稳定 VIEW-ID 与 view_hash。"""
    tool = "tf_dataset_materialize"
    doc = dict(view_definition)
    try:
        validate_view_definition(doc)
        mv = materialize_view(ctx.vault, doc, ctx.view_cache_root)
        view = ctx.ledger.register_view(
            {**doc, "view_hash": mv.view_hash}, actor=ctx.actor,
            reason="工具层登记 Dataset View",
        )
    except (VaultError, ValueError, KeyError) as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"dataset": doc.get("dataset")})
    return _finish(ctx, make_envelope(
        tool, id=view["id"], status="MATERIALIZED",
        inputs={"dataset": doc.get("dataset"), "view_hash": mv.view_hash},
        summary={
            "view_id": view["id"],
            "view_hash": mv.view_hash,
            "rows": mv.table.num_rows,
            "columns": mv.table.column_names,
            "cache_reused": mv.reused,
        },
        artifacts=[_artifact(mv.path / "data.parquet", "view_data"),
                   _artifact(mv.path / "view.json", "view_definition")],
    ))


def tf_dataset_compare(ctx: ToolContext, ref_a: str, ref_b: str) -> dict[str, Any]:
    """两个数据版本的对比：行数/时间范围/变量增删/数值漂移（有界）。"""
    tool = "tf_dataset_compare"
    try:
        info_a, info_b = ctx.vault.resolve(ref_a), ctx.vault.resolve(ref_b)
        vars_a = {v["variable_id"]: v for v in ctx.vault.load_variables(ref_a)}
        vars_b = {v["variable_id"]: v for v in ctx.vault.load_variables(ref_b)}
    except VaultError as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"ref_a": ref_a, "ref_b": ref_b})
    df_a = ctx.vault.load_data(ref_a).to_pandas()
    df_b = ctx.vault.load_data(ref_b).to_pandas()
    common = sorted(set(vars_a) & set(vars_b))
    drift = []
    for var in common:
        if vars_a[var].get("dtype") not in ("float", "integer"):
            continue
        a = pd.to_numeric(df_a[var], errors="coerce").dropna()
        b = pd.to_numeric(df_b[var], errors="coerce").dropna()
        if not len(a) or not len(b):
            continue
        mean_a, mean_b = float(a.mean()), float(b.mean())
        drift.append({
            "variable_id": var, "mean_a": mean_a, "mean_b": mean_b,
            "rel_change": (abs(mean_b - mean_a) / max(abs(mean_a), 1e-12)),
        })
    drift.sort(key=lambda d: -d["rel_change"])
    return _finish(ctx, make_envelope(
        tool, status="OK", inputs={"ref_a": ref_a, "ref_b": ref_b},
        summary={
            "content_identical": info_a.content_sha256 == info_b.content_sha256,
            "rows": [int(len(df_a)), int(len(df_b))],
            "time_range_a": [str(df_a["timestamp"].min()),
                             str(df_a["timestamp"].max())],
            "time_range_b": [str(df_b["timestamp"].min()),
                             str(df_b["timestamp"].max())],
            "variables_added": sorted(set(vars_b) - set(vars_a)),
            "variables_removed": sorted(set(vars_a) - set(vars_b)),
            "variables_common": len(common),
            "drift_top": drift[:20],
        },
    ))


# ---------------------------------------------------------------- 研究工具


def tf_dataset_modelability(
    ctx: ToolContext,
    ref: str,
    *,
    goal_id: str | None = None,
    target: str | None = None,
    candidate_inputs: Sequence[str] | None = None,
    object_model: str | None = None,
) -> dict[str, Any]:
    """可建模性报告（G3 语义层检查）。FAIL = 存在 blocker，不得进入建模。

    target/candidate_inputs 可由 `goal_id` 从账本中的 Research Goal 定义
    读取；完整报告始终落 artifact，信封只带摘要（§11）。
    """
    tool = "tf_dataset_modelability"
    try:
        if goal_id is not None:
            goal = ctx.ledger.get(goal_id)
            definition = goal.get("definition") or {}
            target = target or definition.get("target")
            candidate_inputs = candidate_inputs or definition.get(
                "candidate_inputs")
            object_model = object_model or definition.get("object_model")
        if not target or not candidate_inputs:
            raise ValueError("必须提供 goal_id 或显式 target + candidate_inputs")
        report = build_modelability_report(
            ctx.vault, ref, target=str(target),
            candidate_inputs=[str(c) for c in candidate_inputs],
            object_model=object_model, registry=ctx.tfom_registry,
        )
    except (VaultError, KeyError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs={"ref": ref,
                                                       "goal_id": goal_id})
    out_dir = ctx.artifacts_root / "modelability"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{ref.replace('@', '_').replace('/', '_')}-modelability.json"
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(report, fp, ensure_ascii=False, sort_keys=True, indent=2)
        fp.write("\n")
    os.replace(tmp, artifact_path)
    ok = report["verdict"] == "PASS"
    return _finish(ctx, make_envelope(
        tool, ok=ok, id=ref, status=report["verdict"],
        inputs={"ref": ref, "goal_id": goal_id, "target": report["target"]},
        summary={
            "verdict": report["verdict"],
            "blockers": report["blockers"],
            "warnings": report["warnings"],
            "checks": [
                {"name": c["name"], "level": c["level"],
                 "passed": c["passed"], "summary": c["summary"]}
                for c in report["checks"]
            ],
        },
        artifacts=[_artifact(artifact_path, "modelability_report")],
    ))


def tf_goal_create(
    ctx: ToolContext,
    definition: Mapping[str, Any],
    *,
    dataset_ref: str | None = None,
) -> dict[str, Any]:
    """创建 Research Goal（契约校验 + 稳定 RG-ID）。

    `dataset_ref` 提供时做白名单机器校验（DD-16）：candidate_inputs
    必须存在于数据版本且不得为派生量（data-survey §F1 循环论证）。
    """
    tool = "tf_goal_create"
    doc = dict(definition)
    doc.setdefault("goal_id", ctx.ledger.allocator.allocate("RG-"))
    try:
        goal = ResearchGoal(**doc)
        if dataset_ref is not None:
            violations = check_candidate_inputs(
                goal.candidate_inputs, target=goal.target,
                variables=ctx.vault.load_variables(dataset_ref),
            )
            if violations:
                raise ValueError(
                    "candidate_inputs 白名单校验失败: " + "; ".join(violations)
                )
        entity = ctx.ledger.create_goal(
            goal.name, actor=ctx.actor,
            definition=goal.model_dump(mode="json"),
            entity_id=goal.goal_id, reason="工具层创建 Research Goal",
        )
    except (ValidationError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs={"name": doc.get("name")})
    return _finish(ctx, make_envelope(
        tool, id=entity["id"], status="CREATED",
        inputs={"goal_id": entity["id"], "dataset_ref": dataset_ref},
        summary={
            "goal_id": entity["id"],
            "name": goal.name,
            "target": goal.target,
            "object_model": goal.object_model,
            "candidate_inputs": list(goal.candidate_inputs),
            "acceptance": goal.acceptance.model_dump(mode="json",
                                                     exclude_none=True),
            "budgets": {
                "max_experiments": goal.max_experiments,
                "max_duration_days": goal.max_duration_days,
                "compute_budget_hours": goal.compute_budget_hours,
            },
            "approval_required": list(goal.approval_required),
        },
    ))


def tf_research_status(
    ctx: ToolContext,
    goal_id: str | None = None,
) -> dict[str, Any]:
    """研究进展：目标状态、假设/实验/模型计数、预算消耗。"""
    tool = "tf_research_status"
    if goal_id is None:
        goals = ctx.ledger.list_goals()
        return _finish(ctx, make_envelope(
            tool, status="OK",
            summary={"goals": [
                {"goal_id": g["id"], "name": g.get("name"),
                 "status": g["status"]}
                for g in goals
            ]},
        ))
    try:
        goal = ctx.ledger.get(goal_id)
        progress = ctx.ledger.goal_progress(goal_id)
    except (KeyError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs={"goal_id": goal_id})
    definition = goal.get("definition") or {}
    spent = progress["experiments"]["total"]
    budget: dict[str, Any] = {
        "experiments_used": spent,
        "experiments_max": definition.get("max_experiments"),
    }
    if definition.get("max_experiments") is not None:
        budget["experiments_remaining"] = max(
            0, int(definition["max_experiments"]) - spent)
    created = definition.get("created_at") or goal.get("created_at")
    if created and definition.get("max_duration_days"):
        elapsed = (datetime.now(timezone.utc)
                   - parse_timestamp(created)).total_seconds() / 86400.0
        budget["days_elapsed"] = elapsed
        budget["days_max"] = definition["max_duration_days"]
    return _finish(ctx, make_envelope(
        tool, id=goal_id, status=str(goal["status"]).upper(),
        inputs={"goal_id": goal_id},
        summary={
            "progress": progress,
            "budget": budget,
            "unverified_hypotheses": [
                h["id"] for h in ctx.ledger.unverified_hypotheses(goal_id)],
            "failed_experiments": ctx.ledger.failed_experiments(goal_id),
        },
    ))


def tf_hypothesis_create(
    ctx: ToolContext,
    goal_id: str,
    statement: str,
    *,
    basis: Sequence[str] = (),
) -> dict[str, Any]:
    """创建假设；非首个假设必须引用已有证据（research-loop §3）。"""
    tool = "tf_hypothesis_create"
    try:
        hyp = ctx.ledger.create_hypothesis(
            goal_id, statement, actor=ctx.actor, basis=list(basis),
            reason="工具层创建假设",
        )
    except ValueError as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"goal_id": goal_id, "basis": list(basis)})
    return _finish(ctx, make_envelope(
        tool, id=hyp["id"], status="UNVERIFIED",
        inputs={"goal_id": goal_id, "basis": list(basis)},
        summary={"hypothesis_id": hyp["id"], "statement": statement,
                 "basis": hyp["refs"].get("basis", [])},
    ))


def tf_experiment_plan(
    ctx: ToolContext,
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    """登记实验定义（契约校验 + 稳定 EXP-ID，不执行）。"""
    tool = "tf_experiment_plan"
    doc = dict(definition)
    doc.setdefault("experiment_id", ctx.ledger.allocator.allocate("EXP-"))
    try:
        exp = Experiment(**doc)
        # DD-16 封闭白名单的机器校验：View 特征必须 ⊆ candidate_inputs，
        # 目标必须与 goal.target 一致（goal/view 均在账本时可查）
        goal_entity = ctx.ledger.get(exp.goal_id)
        view_entity = ctx.ledger.get(exp.dataset_view)
        candidate = (goal_entity.get("definition") or {}).get("candidate_inputs")
        goal_target = (goal_entity.get("definition") or {}).get("target")
        view_def = view_entity.get("definition") or {}
        if candidate and view_def:
            violations = check_view_within_whitelist(
                features=[str(f) for f in view_def.get("features", [])],
                view_target=(str(view_def["target"])
                             if view_def.get("target") else None),
                view_objects=[str(o) for o in view_def.get("objects", [])],
                candidate_inputs=[str(c) for c in candidate],
                goal_target=str(goal_target),
            )
            if violations:
                raise ValueError("DD-16 白名单校验失败: " + "; ".join(violations))
        entity = ctx.ledger.register_experiment(
            exp.model_dump(by_alias=True, mode="json"), actor=ctx.actor,
            reason="工具层登记实验计划",
        )
    except (ValidationError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"goal_id": doc.get("goal_id")})
    return _finish(ctx, make_envelope(
        tool, id=entity["id"], status="PLANNED",
        inputs={"goal_id": exp.goal_id, "hypothesis_id": exp.hypothesis_id,
                "dataset_view": exp.dataset_view},
        summary={
            "experiment_id": entity["id"],
            "model": exp.model.model_dump(mode="json"),
            "target": exp.target,
            "random_seed": exp.runtime.random_seed,
        },
    ))


def _experiment_artifacts(ctx: ToolContext, exp_id: str) -> list[dict[str, Any]]:
    exp_dir = ctx.research_root / "experiments" / exp_id
    out = []
    for name, kind in (("report.json", "experiment_report"),
                       ("metrics.json", "metrics"),
                       ("predictions.parquet", "predictions"),
                       ("split.json", "split"),
                       ("physics_report.json", "physics_report")):
        path = exp_dir / name
        if path.exists():
            out.append(_artifact(path, kind))
    if (exp_dir / "model").is_dir():
        out.append({"kind": "model_dir", "path": str(exp_dir / "model"),
                    "bytes": 0})
    return out


def _metrics_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    surfaces = {
        name: {"n_samples": surf.get("n_samples"),
               "metrics": surf.get("metrics")}
        for name, surf in (metrics.get("surfaces") or {}).items()
    }
    physics = report.get("physics") or {}
    return {
        "surfaces": surfaces,
        "physics_overall_rate": physics.get("overall_rate"),
        "duration_seconds": report.get("duration_seconds"),
        "environment_lock": report.get("environment_lock"),
    }


def tf_experiment_run(
    ctx: ToolContext,
    experiment_id: str,
    *,
    purge_seconds: float = 0.0,
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
    y_floor: float | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """执行已登记的实验（子进程隔离），返回结构化指标摘要。"""
    tool = "tf_experiment_run"
    try:
        entity = ctx.ledger.get(experiment_id)
        exp = Experiment(**entity["definition"])
        report = run_experiment(
            exp, research_root=ctx.research_root, vault_root=ctx.vault_root,
            ledger=ctx.ledger, purge_seconds=purge_seconds,
            embargo_seconds=embargo_seconds, y_floor=y_floor,
            actor=ctx.actor, timeout_seconds=timeout_seconds,
        )
    except (KeyError, ValidationError, ValueError, ResearchError) as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"experiment_id": experiment_id})
    ok = report["status"] == "completed"
    diagnostics = []
    if not ok:
        diagnostics.append({
            "code": report.get("error_code") or "CHILD_ERROR",
            "level": "ERROR", "count": 1,
            "message": str(report.get("error"))[:500],
        })
    return _finish(ctx, make_envelope(
        tool, ok=ok, id=experiment_id,
        status=report["status"].upper(),
        inputs={"experiment_id": experiment_id,
                "random_seed": report.get("random_seed")},
        summary={
            **_metrics_summary(report),
            "error_code": report.get("error_code"),
            "conclusion": report.get("conclusion"),
        },
        diagnostics=diagnostics,
        artifacts=_experiment_artifacts(ctx, experiment_id),
    ))


def tf_experiment_get(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    """实验结果查询（Ledger 状态 + report.json 摘要）。"""
    tool = "tf_experiment_get"
    try:
        entity = ctx.ledger.get(experiment_id)
    except (KeyError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"experiment_id": experiment_id})
    report_path = (ctx.research_root / "experiments" / experiment_id
                   / "report.json")
    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": entity["status"],
        "refs": entity.get("refs", {}),
    }
    artifacts: list[dict[str, Any]] = []
    if report_path.exists():
        report = _read_json(report_path)
        summary.update(_metrics_summary(report))
        summary["error_code"] = report.get("error_code")
        summary["conclusion"] = report.get("conclusion")
        artifacts = _experiment_artifacts(ctx, experiment_id)
    return _finish(ctx, make_envelope(
        tool, ok=entity["status"] != "failed", id=experiment_id,
        status=str(entity["status"]).upper(),
        inputs={"experiment_id": experiment_id},
        summary=summary, artifacts=artifacts,
    ))


def _primary_surface_name(metrics: Mapping[str, Any]) -> str | None:
    """发布/比较口径：优先面 C，其次面 A，再次 validate（§4.3）。"""
    surfaces = (metrics or {}).get("surfaces") or {}
    for name in ("C", "A", "validate"):
        if (surfaces.get(name) or {}).get("n_samples"):
            return name
    return None


def tf_model_compare(
    ctx: ToolContext,
    experiment_ids: Sequence[str],
) -> dict[str, Any]:
    """模型比较：按主测试面（C→A→validate）CVRMSE 排名，附物理违规率。"""
    tool = "tf_model_compare"
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for exp_id in experiment_ids:
        report_path = (ctx.research_root / "experiments" / str(exp_id)
                       / "report.json")
        if not report_path.exists():
            skipped.append(str(exp_id))
            continue
        report = _read_json(report_path)
        if report.get("status") != "completed":
            skipped.append(str(exp_id))
            continue
        surface = _primary_surface_name(report.get("metrics") or {})
        surf = (report["metrics"]["surfaces"][surface] if surface else {})
        rows.append({
            "experiment_id": str(exp_id),
            "primary_surface": surface,
            "metrics": surf.get("metrics") or {},
            "physics_overall_rate": (report.get("physics") or {})
            .get("overall_rate"),
        })

    def _key(row: Mapping[str, Any]) -> float:
        v = row["metrics"].get("CVRMSE")
        return float(v) if v is not None else math.inf

    ranking = [r["experiment_id"] for r in sorted(rows, key=_key)]
    return _finish(ctx, make_envelope(
        tool, status="OK",
        inputs={"experiment_ids": [str(e) for e in experiment_ids]},
        summary={
            "experiments": rows,
            "ranking_by_cvrmse": ranking,
            "best": ranking[0] if ranking else None,
            "skipped": skipped,
        },
    ))


def tf_model_publish(
    ctx: ToolContext,
    experiment_id: str,
    *,
    model_id: str,
    version: str,
    description: str | None = None,
    default_out_of_range: str = "reject",
    run_smoke: bool = True,
) -> dict[str, Any]:
    """从已完成实验构建模型包并走发布门禁（model-package §8）。

    成功 → production；任一硬门禁失败 → ok=False + 对应 TFM-10xx 诊断，
    门禁明细可见（entry.last_gate_results）。
    """
    tool = "tf_model_publish"
    inputs = {"experiment_id": experiment_id, "model_id": model_id,
              "version": version}
    try:
        entity = ctx.ledger.get(experiment_id)
        report_path = (ctx.research_root / "experiments" / experiment_id
                       / "report.json")
        if not report_path.exists():
            raise ValueError(f"实验无 report.json（未执行？）: {experiment_id}")
        report = _read_json(report_path)
        if report.get("status") != "completed":
            raise ValueError(f"实验未完成，不得发布: {experiment_id}")
        goal_id = report["goal_id"]
        goal = ctx.ledger.get(goal_id)
        acceptance = (goal.get("definition") or {}).get("acceptance") or {}
        view_entity = ctx.ledger.get(report["dataset_view"])
        view_doc = dict(view_entity["definition"])
        features = [str(f) for f in view_doc["features"]]
        target = str(view_doc["target"])
        dataset_ref = str(view_doc["dataset"])
    except (KeyError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs=inputs)

    # 签名：property_code + 单位来自数据版本（两层命名，DD-11）
    variables = ctx.vault.load_variables(dataset_ref)
    unit_of = {v["property_code"]: v["unit"] for v in variables}

    def _unit(prop: str) -> str:
        if prop not in unit_of:
            raise VaultError("TFV-802", f"签名属性不在数据版本中: {prop}")
        return str(unit_of[prop])

    signature = {
        "model_id": model_id,
        "version": version,
        "object_model": (goal.get("definition") or {}).get("object_model"),
        "inputs": [{"property_code": f, "unit": _unit(f), "dtype": "float",
                    "required": True} for f in features],
        "outputs": [{"property_code": target, "unit": _unit(target),
                     "dtype": "float"}],
    }
    bounds = {
        v["property_code"]: (v.get("min_value"), v.get("max_value"))
        for v in variables
    }
    constraints = {
        "inputs": [{
            "property_code": f,
            "min_value": bounds.get(f, (None, None))[0],
            "max_value": bounds.get(f, (None, None))[1],
            "out_of_range": default_out_of_range,
        } for f in features],
    }
    # golden 输入：物化 View 的工况边界行（§8.2，覆盖边界而非随机抽样）
    mv = materialize_view(ctx.vault, view_doc, ctx.view_cache_root)
    golden_inputs = boundary_rows(mv.table.to_pandas(), features)
    exp_dir = ctx.research_root / "experiments" / experiment_id
    environment = _read_json(exp_dir / "environment.json")
    validation_doc = {
        "physics": report.get("physics"),
        "split": _read_json(exp_dir / "split.json"),
    }
    dataset_lineage = {
        "dataset": dataset_ref,
        "view_id": report["dataset_view"],
        "view_hash": (report.get("conclusion") or {}).get("view_hash"),
    }
    research_lineage = {
        "goal_id": goal_id,
        "hypothesis_id": report.get("hypothesis_id"),
        "experiment_id": experiment_id,
    }
    try:
        ctx.artifacts_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                dir=ctx.artifacts_root, prefix="pkg_") as staging:
            pkg = build_model_package(
                Path(staging) / "pkg",
                model_id=model_id, version=version, signature=signature,
                artifact_dir=exp_dir / "model",
                metrics=report.get("metrics") or {},
                validation=validation_doc,
                dataset_lineage=dataset_lineage,
                research_lineage=research_lineage,
                environment=environment,
                golden_inputs=golden_inputs,
                constraints=constraints,
                goal_id=goal_id, experiment_id=experiment_id,
                description=description,
            )
            entry = ctx.registry.register(pkg, actor=ctx.actor)
        result = ctx.registry.publish(
            model_id, version, actor=ctx.actor, acceptance=acceptance,
            tfom_registry=ctx.tfom_registry, run_smoke=run_smoke,
        )
    except (ModelRegistryError, VaultError, ValueError) as exc:
        env = _error_envelope(ctx, tool, exc, inputs=inputs)
        gates = ctx.registry.last_gate_results(model_id, version)
        if gates:
            env["summary"]["gates"] = gates
        return env

    ctx.ledger.register_model(
        model_id, experiment_id, actor=ctx.actor,
        metrics=(report.get("metrics") or {}).get("surfaces", {}),
        artifact_path=str(ctx.registry.package_dir(model_id, version)),
        reason="发布门禁全部通过",
    )
    ctx.ledger.create_decision(
        f"publish {model_id}@{version}", "approve",
        actor=ctx.actor,
        rationale="发布门禁全部通过（machine-checked，DD-13）",
        references=[experiment_id],
    )
    return _finish(ctx, make_envelope(
        tool, id=f"{model_id}@{version}", status="PRODUCTION",
        inputs=inputs,
        summary={
            "model_id": model_id, "version": version,
            "status": result["status"],
            "gates": result["gates"],
            "package_dir": str(ctx.registry.package_dir(model_id, version)),
            "golden_samples": len(golden_inputs),
        },
        artifacts=[_artifact(
            ctx.registry.package_dir(model_id, version) / "checksums.json",
            "checksums")],
    ))


# ---------------------------------------------------------------- 预处理工具（I-49）


def tf_preprocess_propose(
    ctx: ToolContext,
    ruleset: Mapping[str, Any],
) -> dict[str, Any]:
    """提交预处理规则集（Schema 校验 + 参数校验 + 版本化存储）。

    Agent 只能提出规则参数；执行的是规则库中预先注册的确定性变换
    （DD-02 方案 C）。新规则一律 status=proposed，需 human 审批。
    """
    tool = "tf_preprocess_propose"
    try:
        rs = RuleSet.model_validate(dict(ruleset))
        for rule in rs.rules:
            validate_rule_params(rule)  # 参数在 propose 时即校验
        # 工具层提交一律记为 proposed + 当前 actor（不信任输入的 status）
        rs = RuleSet.model_validate({
            **rs.model_dump(mode="json"),
            "rules": [
                {**r, "status": "proposed", "proposer": ctx.actor,
                 "approvals": []}
                for r in rs.model_dump(mode="json")["rules"]
            ],
        })
        ctx.preprocess_store.save(rs)
    except (ValidationError, PreprocessError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc,
                               inputs={"ruleset_id": ruleset.get("ruleset_id")})
    return _finish(ctx, make_envelope(
        tool, id=rs.ref, status="PROPOSED",
        inputs={"ruleset_id": rs.ruleset_id, "version": rs.version},
        summary={
            "ruleset": rs.ref,
            "content_hash": rs.content_hash(),
            "rules": [
                {"rule_id": r.rule_id, "rule_type": r.rule_type,
                 "sheet": r.sheet, "status": r.status}
                for r in rs.rules
            ],
        },
    ))


def tf_preprocess_approve(
    ctx: ToolContext,
    ruleset_id: str,
    *,
    version: int | None = None,
    rule_ids: Sequence[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """审批规则（actor 必须为 human，全部留痕）。"""
    tool = "tf_preprocess_approve"
    inputs = {"ruleset_id": ruleset_id, "version": version,
              "rule_ids": list(rule_ids) if rule_ids else None}
    if ctx.actor != "human":
        return _error_envelope(
            ctx, tool,
            PreprocessError(
                TFPP_APPROVAL_FORBIDDEN,
                f"审批 actor 必须为 human，当前: {ctx.actor!r}",
            ),
            inputs=inputs,
        )
    try:
        rs = ctx.preprocess_store.load(ruleset_id, version)
        targets = set(rule_ids) if rule_ids else {r.rule_id for r in rs.rules}
        unknown = targets - {r.rule_id for r in rs.rules}
        if unknown:
            raise PreprocessError("TFPP-003", f"规则不存在: {sorted(unknown)}")
        now = datetime.now(timezone.utc).isoformat()
        changed = []
        for rule in rs.rules:
            if rule.rule_id not in targets or rule.status == "approved":
                continue
            rule.status = "approved"
            rule.approvals.append(ApprovalRecord(
                actor=ctx.actor, action="approve", at=now, note=note,
            ))
            changed.append(rule.rule_id)
        ctx.preprocess_store.save(rs)  # 内容哈希不变，仅留痕元数据更新
    except (PreprocessError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs=inputs)
    return _finish(ctx, make_envelope(
        tool, id=rs.ref, status="APPROVED" if changed else "UNCHANGED",
        inputs=inputs,
        summary={
            "approved": changed,
            "rules": [
                {"rule_id": r.rule_id, "status": r.status,
                 "approvals": [a.model_dump(mode="json")
                               for a in r.approvals]}
                for r in rs.rules
            ],
        },
    ))


def tf_preprocess_apply(
    ctx: ToolContext,
    path: str | Path,
    ruleset_id: str,
    *,
    version: int | None = None,
    import_into_vault: bool = False,
    write_workbook: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行规则集：原始工作簿 + 规则集 → 内存表模型（可选落 xlsx / 落 vault）。

    审批门禁：`import_into_vault=True`（产生 vault revision 的正式导入）
    要求全部规则 approved；预览执行（不落 vault）允许 proposed 规则。
    逐规则记录 lineage（rule_id、参数、输入/输出指纹）。
    """
    tool = "tf_preprocess_apply"
    inputs = {"path": str(path), "ruleset_id": ruleset_id, "version": version,
              "import_into_vault": import_into_vault}
    try:
        rs = ctx.preprocess_store.load(ruleset_id, version)
        sheets = load_workbook_model(path)
        executions = apply_ruleset(
            sheets, rs, require_approved=import_into_vault,
        )
    except (PreprocessError, ValueError) as exc:
        return _error_envelope(ctx, tool, exc, inputs=inputs)

    run_dir = ctx.research_root / "preprocess" / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    lineage_doc = {
        "ruleset": rs.ref,
        "content_hash": rs.content_hash(),
        "source_path": str(path),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "actor": ctx.actor,
        "rules": [e.to_dict() for e in executions],
    }
    artifacts: list[dict[str, Any]] = []
    lineage_path = run_dir / "lineage.json"
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(lineage_doc, fp, ensure_ascii=False, sort_keys=True, indent=2)
        fp.write("\n")
    os.replace(tmp, lineage_path)
    artifacts.append(_artifact(lineage_path, "preprocess_lineage"))

    dataset_ref = None
    diagnostics: list[dict[str, Any]] = []
    if write_workbook:
        out = Path(output_path) if output_path else (
            run_dir / "processed.xlsx"
        )
        _write_workbook_model(sheets, out)
        artifacts.append(_artifact(out, "processed_workbook"))
    if import_into_vault:
        registry = ctx.tfom_registry or default_registry()
        conv = convert_legacy_tables(
            sheets, registry, source_name=Path(path).name,
        )
        result = import_parsed(
            conv.dataset, conv.timestamps, conv.columns,
            registry=registry,
            pre_diagnostics=conv.pre_diagnostics,
            degradations=conv.degradations,
            source_path=path,
        )
        diagnostics = diagnostic_dicts(result.diagnostics)
        if not result.ok:
            return _finish(ctx, make_envelope(
                tool, ok=False, status="FAILED", inputs=inputs,
                summary={"error": "导入存在 ERROR 级诊断，未写入 vault",
                         "ruleset": rs.ref},
                diagnostics=diagnostics,
                artifacts=artifacts,
            ))
        lineage = conv.lineage(path)
        lineage["preprocess"] = lineage_doc
        try:
            dataset_ref = ctx.vault.store(result, source_path=path,
                                          lineage=lineage)
        except VaultError as exc:
            return _error_envelope(ctx, tool, exc, inputs=inputs)
    return _finish(ctx, make_envelope(
        tool, ok=True, id=dataset_ref or rs.ref,
        status="IMPORTED" if dataset_ref else "APPLIED",
        inputs=inputs,
        summary={
            "ruleset": rs.ref,
            "content_hash": rs.content_hash(),
            "dataset_ref": dataset_ref,
            "rules": [
                {"rule_id": e.rule_id, "rule_type": e.rule_type,
                 "skipped": e.skipped,
                 "input_sha256": (e.input_sha256 or "")[:16],
                 "output_sha256": (e.output_sha256 or "")[:16],
                 "detail": e.detail}
                for e in executions
            ],
        },
        diagnostics=diagnostics,
        artifacts=artifacts,
    ))


def _write_workbook_model(sheets: Mapping[str, Any], path: Path) -> None:
    """内存表模型 → xlsx（openpyxl 写模式；仅 apply 显式要求时调用）。"""
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for name, sheet in sheets.items():
        ws = wb.create_sheet(name)
        for row in [*sheet.header_rows, *sheet.data_rows]:
            ws.append(list(row))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".xlsx")
    os.close(fd)
    wb.save(tmp)
    os.replace(tmp, path)


def tf_preprocess_list(ctx: ToolContext) -> dict[str, Any]:
    """列出全部规则集及规则审批状态。"""
    rulesets = ctx.preprocess_store.list_rulesets()
    return _finish(ctx, make_envelope(
        "tf_preprocess_list", status="OK",
        summary={"rulesets": rulesets, "count": len(rulesets)},
    ))


# architecture §6 工具清单 → 入口（与 pi/tools.json 保持一致）
TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "tf_dataset_import": tf_dataset_import,
    "tf_dataset_list": tf_dataset_list,
    "tf_dataset_get": tf_dataset_get,
    "tf_dataset_schema": tf_dataset_schema,
    "tf_dataset_profile": tf_dataset_profile,
    "tf_dataset_query": tf_dataset_query,
    "tf_dataset_sample": tf_dataset_sample,
    "tf_dataset_materialize": tf_dataset_materialize,
    "tf_dataset_compare": tf_dataset_compare,
    "tf_dataset_modelability": tf_dataset_modelability,
    "tf_goal_create": tf_goal_create,
    "tf_research_status": tf_research_status,
    "tf_hypothesis_create": tf_hypothesis_create,
    "tf_experiment_plan": tf_experiment_plan,
    "tf_experiment_run": tf_experiment_run,
    "tf_experiment_get": tf_experiment_get,
    "tf_model_compare": tf_model_compare,
    "tf_model_publish": tf_model_publish,
    "tf_preprocess_propose": tf_preprocess_propose,
    "tf_preprocess_approve": tf_preprocess_approve,
    "tf_preprocess_apply": tf_preprocess_apply,
    "tf_preprocess_list": tf_preprocess_list,
}
