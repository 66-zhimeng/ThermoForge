"""V2 研究工具边界：冻结评价、登记来源、隔离轨迹、预留实验额度。

Codex 只获得本模块声明的工具。旧实验内核及其完整留出制品留在服务端；
研究反馈采用白名单投影，不能把旧工具的 artifact 路径直接交给候选。
模型实验室仍为自治五连检；现有 AST 检查不是操作系统沙箱。
"""

from __future__ import annotations

import copy
import hashlib
import io
import ipaddress
import json
import math
import re
import socket
import subprocess
import threading
import urllib.parse
import urllib.request
from contextlib import nullcontext
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping

from thermoforge_core.contracts.experiment import Experiment, ModelSpec
from thermoforge_core.contracts.research_goal import ResearchGoal
from thermoforge_core.timeutil import parse_time_resolution, parse_timestamp
from thermoforge_data.views import materialize_view, validate_view_definition
from thermoforge_research.model_catalog import (
    DATA_ESTIMATORS, HYBRID_RESIDUALS, MODEL_CATALOG_HINT, PHYSICS_MODELS,
)
from thermoforge_research.model_lab import NAME_PATTERN, parse_lab_ref
from thermoforge_research.modelability import build_modelability_report
from thermoforge_research.runner import current_environment_lock
from thermoforge_research.splits import temporal_split
from thermoforge_research.tools import (
    ToolContext, tf_experiment_plan, tf_experiment_run, tf_lab_submit,
    tf_literature_search,
)
from thermoforge_research.whitelist import (
    check_candidate_inputs, check_view_within_whitelist,
)
from thermoforge_v2.experiment_identity import experiment_fingerprint

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_CHARS = 120_000
SOURCE_EXCERPT_CHARS = 6000
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, maximum: int = 12000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空文本")
    if len(value) > maximum:
        raise ValueError(f"{name} 超过 {maximum} 字符")
    return value.strip()


def _lock_for(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def prepare_protocol(ctx: ToolContext, config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """在启动前校验并冻结目标、数据视图和唯一评价口径。"""
    cfg = copy.deepcopy(dict(config_dict))
    goal_id = _text(cfg.get("goal_id"), "goal_id")
    dataset_ref = _text(cfg.get("dataset_ref"), "dataset_ref")
    entity = ctx.ledger.get(goal_id)
    goal = ResearchGoal(**entity["definition"])
    revision = ctx.vault.resolve(dataset_ref)
    variables = ctx.vault.load_variables(dataset_ref)
    violations = check_candidate_inputs(goal.candidate_inputs, target=goal.target,
                                        variables=variables)
    if violations:
        raise ValueError("DD-16: " + "; ".join(violations))
    if cfg.get("view_id"):
        view = copy.deepcopy(ctx.ledger.get(str(cfg["view_id"]))["definition"])
    else:
        scoped = {e.split(".", 1)[0] for e in goal.candidate_inputs if "." in e}
        objects = [str(o["object_id"]) for o in ctx.vault.load_objects(dataset_ref)
                   if o["object_model_id"] == goal.object_model
                   and (not scoped or o["object_id"] in scoped)]
        view = {"dataset": dataset_ref, "scope": {"object_model": goal.object_model},
                "objects": objects,
                "features": list(dict.fromkeys(e.split(".", 1)[-1]
                                                for e in goal.candidate_inputs)),
                "target": goal.target}
    validate_view_definition(view)
    if view["dataset"] != dataset_ref:
        raise ValueError("view_id 的数据版本与 dataset_ref 不一致")
    if not view.get("objects"):
        view["objects"] = [str(o["object_id"]) for o in ctx.vault.load_objects(dataset_ref)
                           if o["object_model_id"] == goal.object_model]
    violations = check_view_within_whitelist(
        features=view["features"], view_target=view.get("target"),
        view_objects=view["objects"], candidate_inputs=goal.candidate_inputs,
        goal_target=goal.target)
    # 旧白名单工具按对象集合合并；V2 多对象长表逐对象再检查，避免借另一对象放行。
    for obj in view["objects"]:
        violations.extend(check_view_within_whitelist(
            features=view["features"], view_target=view.get("target"),
            view_objects=[obj], candidate_inputs=goal.candidate_inputs,
            goal_target=goal.target))
    if violations:
        raise ValueError("DD-16: " + "; ".join(dict.fromkeys(violations)))
    validation = cfg.get("validation") or {
        "temporal_split": {"train": 0.70, "validate": 0.15, "test": 0.15}}
    environment_lock = current_environment_lock()[0]
    runtime = cfg.get("runtime") or {"random_seed": cfg.get("seed", cfg.get("random_seed", 42))}
    runtime = {**runtime, "environment_lock": environment_lock}
    exp = Experiment(
        experiment_id="EXP-0001", goal_id=goal_id, hypothesis_id="H-0001",
        dataset_view="VIEW-0001", model={"category": "data", "estimator": "ridge"},
        target=goal.target, validation=validation,
        metrics=cfg.get("metrics") or ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2"],
        physics_tests=cfg.get("physics_tests") or {"enabled": True}, runtime=runtime)
    if exp.validation.temporal_split.validate_ <= 0:
        raise ValueError("V2 搜索必须保留非空 validate 比例")
    if exp.validation.rolling_cv.enabled:
        raise ValueError("V2 搜索暂不允许旧 rolling_cv：其折可能训练最终留出数据")
    holdout = exp.validation.equipment_holdout
    if holdout.enabled and (not holdout.holdout_objects or
                            not set(holdout.holdout_objects) < set(view["objects"])):
        raise ValueError("设备留出必须是视图对象的非空真子集")
    for key, default in (("purge_seconds", 2700.0), ("embargo_seconds", 2700.0)):
        value = float(cfg.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须为有限非负数")
        cfg[key] = value
    y_floor = cfg.get("y_floor", 1e-6)
    if y_floor is not None and (not math.isfinite(float(y_floor)) or float(y_floor) <= 0):
        raise ValueError("y_floor 必须为有限正数")
    # 在统一服务预物化，启动前发现缺变量/空对象，而非六条轨迹各自猜测配置。
    materialized = materialize_view(ctx.vault, view, ctx.view_cache_root)
    if not materialized.table.num_rows:
        raise ValueError("冻结视图无数据")
    with open(revision.path / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    resolution = parse_time_resolution(view.get("resolution") or manifest["time_resolution"])
    timestamps = sorted(set(materialized.table.column("timestamp").to_pylist()))
    split = temporal_split(timestamps, resolution,
                           train=exp.validation.temporal_split.train,
                           validate=exp.validation.temporal_split.validate_,
                           test=exp.validation.temporal_split.test,
                           purge_seconds=cfg["purge_seconds"], embargo_seconds=cfg["embargo_seconds"])
    if not split.train_idx or not split.validate_idx or not split.test_idx:
        raise ValueError("时间切分及 purge/embargo 后训练、验证和留出都必须非空")
    # 语义可建模性门只看训练段及非留出设备，避免准入诊断先读取留出标签。
    held_objects = set(holdout.holdout_objects) if holdout.enabled else set()
    train_objects = [o for o in view["objects"] if o not in held_objects]

    class TrainingVault:
        def resolve(self, ref):
            return ctx.vault.resolve(ref)

        def load_data(self, ref):
            import pyarrow as pa
            start, end = map(parse_timestamp, split.boundaries["train_range"])
            # 还原旧可建模性检查接受的宽表，但数据必须来自已经过滤/重采样的
            # 冻结 View，不能拿未过滤原始数据替代实际训练样本做准入判定。
            frame = materialized.table.to_pandas()
            frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)
                          & frame["object_id"].isin(train_objects)]
            props = list(dict.fromkeys(view["features"] + [view["target"]]))
            wide = frame.pivot(index="timestamp", columns="object_id", values=props)
            wide.columns = [f"{obj}.{prop}" for prop, obj in wide.columns]
            return pa.Table.from_pandas(wide.reset_index(), preserve_index=False)

        def load_objects(self, ref):
            return [o for o in ctx.vault.load_objects(ref)
                    if o["object_id"] in train_objects]

        def load_variables(self, ref):
            return ctx.vault.load_variables(ref)

    modelability = build_modelability_report(
        TrainingVault(), dataset_ref, target=goal.target,
        candidate_inputs=[e for e in goal.candidate_inputs
                          if e.split(".", 1)[-1] in view["features"]
                          and ("." not in e or e.split(".", 1)[0] in train_objects)],
        object_model=goal.object_model,
        registry=ctx.tfom_registry)
    if modelability["verdict"] != "PASS":
        raise ValueError("训练数据可建模性门未通过: " + "; ".join(
            c["summary"] for c in modelability["checks"]
            if c["level"] == "blocker" and not c["passed"]))
    protocol = {
        "version": 2, "goal_id": goal_id, "dataset_ref": dataset_ref,
        "dataset_content_sha256": revision.content_sha256,
        "goal_definition": goal.model_dump(mode="json"),
        "view_definition": view, "view_hash": materialized.view_hash,
        "validation": exp.validation.model_dump(by_alias=True, mode="json"),
        "metrics": exp.metrics, "runtime": exp.runtime.model_dump(mode="json"),
        "physics_tests": exp.physics_tests.model_dump(mode="json"),
        "purge_seconds": cfg["purge_seconds"], "embargo_seconds": cfg["embargo_seconds"],
        "y_floor": float(y_floor) if y_floor is not None else None,
        "feedback_surface": "validate", "holdout_surfaces": ["A", "B", "C"],
        "environment_lock": environment_lock,
        "split_boundaries": split.boundaries,
        "modelability": {"verdict": modelability["verdict"], "evaluated_on": "train",
                         "warnings": modelability["warnings"]},
    }
    protocol["fingerprint"] = _digest(protocol)
    return protocol


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _public_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("资料地址只接受公开 HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("资料 URL 不得包含凭据或片段；请提供稳定文献地址")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or
                                  (443 if parsed.scheme == "https" else 80))
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("资料读取不允许本机、私网或保留地址")
    return url


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any,
                         headers: Any, newurl: str) -> Any:
        return super().redirect_request(req, fp, code, msg, headers, _public_url(newurl))


def read_public_source(url: str) -> dict[str, Any]:
    """有限读取公开资料；转跳逐次检查，不能借 URL 工具读取本机服务。"""
    request = urllib.request.Request(_public_url(url), headers={
        "User-Agent": "ThermoForge/2.0 (research-source-reader)"})
    opener = urllib.request.build_opener(_PublicRedirects())
    with opener.open(request, timeout=20) as response:
        final_url = _public_url(response.geturl())
        payload = response.read(MAX_SOURCE_BYTES + 1)
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
    if len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("资料超过 2 MiB 读取上限，请提供较小的 HTML/文本版本")
    if content_type == "application/pdf" or payload.startswith(b"%PDF-"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF 阅读依赖 pypdf 未安装；可使用论文 HTML 版本") from exc
        reader = PdfReader(io.BytesIO(payload))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif content_type.startswith("text/") or content_type in {"application/xml", "application/xhtml+xml"}:
        text = payload.decode(charset, errors="replace")
        if "html" in content_type:
            parser = _TextExtractor()
            parser.feed(text)
            text = "\n".join(parser.parts)
    else:
        raise ValueError(f"暂不支持资料类型 {content_type}")
    if not text.strip():
        raise ValueError("资料未提取出文本，不能登记为已阅读")
    truncated = len(text) > MAX_SOURCE_CHARS
    return {"url": final_url, "content": text[:MAX_SOURCE_CHARS],
            "read_scope": "excerpt" if truncated else "fulltext",
            "content_truncated": truncated, "content_type": content_type,
            "retrieved_at": _now(), "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "content_sha256": hashlib.sha256(text[:MAX_SOURCE_CHARS].encode("utf-8")).hexdigest(),
            "verification": "retrieved"}


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required or [], "additionalProperties": False}


_STRING = {"type": "string"}
_STRINGS = {"type": "array", "items": _STRING}


class ResearchTools:
    """一个身份绑定的一组研究工具；共享 store 负责原子预算和作业幂等。"""

    def __init__(self, store: Any, ctx: ToolContext, run_id: str, track_id: str,
                 experiment_semaphore: Any = None):
        self.store, self.base_ctx = store, ctx
        self.run_id, self.track_id = run_id, track_id
        self.experiment_semaphore = experiment_semaphore
        track = store.get_track(run_id, track_id)
        self.ctx = ToolContext(vault_root=ctx.vault_root,
                               research_root=track["research_root"],
                               models_root=Path(track["research_root"]) / "models",
                               tfom_registry=ctx.tfom_registry,
                               actor=f"v2:{run_id}:{track_id}")
        self.lock = _lock_for(str(self.ctx.research_root))

    @property
    def protocol(self) -> dict[str, Any]:
        return self.store.get_run(self.run_id)["protocol"]

    @property
    def autonomous(self) -> bool:
        # 未记录该字段的旧运行保持原验收行为，不能在恢复时更换研究协议。
        return self.store.get_run(self.run_id)["config"].get("research_mode", "acceptance") == "autonomous"

    def _records(self, kind: str) -> list[dict[str, Any]]:
        return self.store.records(self.run_id, kind, track_id=self.track_id)

    def _record(self, kind: str, record_id: str, *, allow_shared: bool = False) -> dict[str, Any]:
        for item in self._records(kind):
            if item["id"] == record_id:
                return item
        if allow_shared:
            run = self.store.get_run(self.run_id)
            track = self.store.get_track(self.run_id, self.track_id)
            foreign = next((item for item in self.store.records(self.run_id, kind)
                            if item["id"] == record_id), None)
            if foreign is not None and track.get("role") == "main":
                # 首轮提案也对协调者保持隔离；共同任务与证据必须先确定，
                # 防止主智能体根据先到的候选想法继续引导尚未提交的候选。
                if (not self.autonomous or run.get("final_report_phase")
                        or self.store.autonomy_state(self.run_id)["sharing_ready"]):
                    return foreign
            messages = [m for m in self.store.records(self.run_id, "messages")
                        if m.get("from") == "main" and m.get("to") in {self.track_id, "all"}
                        and record_id in (m.get("evidence_ids") or [])]
            if foreign is not None and foreign.get("track_id") == "main" and any(
                    m.get("initial_task") is True for m in messages):
                # 初始共同证据与后期跨候选分享不同；只有服务标明初始任务、
                # 且主智能体本身登记的显式引用能在独立模式/首轮使用。
                return foreign
            # 自主研究按已完成的研究阶段开放分享，不能用对话次数绕过首轮隔离。
            sharing_ready = (self.store.autonomy_state(self.run_id)["sharing_ready"]
                             if self.autonomous else int(track.get("turns", 0)) >= 2)
            if self.autonomous and track.get("role") == "candidate":
                # 失败候选曾被阶段屏障豁免，恢复后仍须先完成自己的独立首轮。
                # 全队已经进入 sharing 不能替代当前候选的首次提案/实验。
                proposals = {p["id"] for p in self._records("proposals")
                             if p.get("status") == "committed"}
                independent_done = any(j.get("proposal_id") in proposals and j.get("status") in {
                    "completed", "failed", "cancelled", "interrupted"} for j in self._records("jobs"))
                sharing_ready = sharing_ready and independent_done
            if run["config"]["strategy"] != "independent" and sharing_ready:
                if messages and foreign is not None:
                    return foreign
        raise ValueError(f"{kind} 引用不存在或不属于当前轨迹: {record_id}")

    def _refs(self, kind: str, values: Any, *, allow_shared: bool = False) -> list[str]:
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{kind} 引用必须为 ID 列表")
        for value in values:
            self._record(kind, value, allow_shared=allow_shared)
        return list(dict.fromkeys(values))

    def _add(self, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.store.add_record(self.run_id, kind, dict(record), track_id=self.track_id)

    def tool_specs(self) -> list[dict[str, Any]]:
        specs = [
            ("research_protocol", "读取冻结目标、允许特征、模型接口与评价协议；只反馈 validate。", {}, []),
            ("research_data_summary", "查看训练与验证段的固定分位数和少量样本；不读取最终留出。", {}, []),
            ("research_literature_search", "检索论文并持久登记实际返回的标题/摘要来源。", {
                "query": _STRING, "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "sources": _STRINGS, "year_from": {"type": "integer"}}, ["query"]),
            ("research_source_read", "读取公开论文/资料 URL，保存内容与实际阅读范围。", {
                "url": _STRING, "title": _STRING}, ["url", "title"]),
            ("research_source_excerpt", "按偏移继续阅读自己已登记资料，记录本次实际读取范围。", {
                "source_id": _STRING, "offset": {"type": "integer", "minimum": 0}}, ["source_id"]),
            ("research_source_register", "登记外部取得的资料摘录或历史证据；history必须用history_ids引用本轨迹真实job。用户任务、冻结协议和自主猜想不必登记成来源，可直接创建conjecture想法。资料明确标为提交者提供，不能伪称工具验证。", {
                "title": _STRING, "kind": {"enum": ["paper", "article", "history"]},
                "url": _STRING, "doi": _STRING, "content": _STRING,
                "read_scope": {"enum": ["metadata", "abstract", "excerpt", "history"]},
                "history_ids": _STRINGS}, ["title", "kind", "read_scope"]),
            ("research_idea_create", "实验前登记想法来源、研究理由、预测与反证条件；自主猜想可以没有论文。", {
                "statement": _STRING, "reason": _STRING, "prediction": _STRING,
                "falsification": _STRING, "origin": {"enum": ["literature", "history", "conjecture", "mixed"]},
                "source_ids": _STRINGS, "parent_idea_ids": _STRINGS,
                "parent_job_ids": _STRINGS, "strategy": _STRING},
             ["statement", "reason", "prediction", "falsification", "origin"]),
            ("research_lab_submit", "自治提交单文件模型源码，五连检通过返回固定 name@vN 引用，无需人工审批。", {
                "name": _STRING, "source": _STRING, "description": _STRING}, ["name", "source"]),
            ("research_proposal_commit", "自主研究在实验前冻结方案。首轮各候选分别提交后才开放实验；后续方案必须引用自己的真实实验反馈及分析发现。", {
                "idea_id": _STRING, "model": ModelSpec.model_json_schema(),
                "purpose": {"enum": ["explore", "refine", "replicate"]},
                "expected_cost": _STRING, "idempotency_key": _STRING},
             ["idea_id", "model", "purpose", "idempotency_key"]),
            ("research_experiment_run", "预留预算并运行自己的模型实验，重复幂等键不会再次训练；只返回验证反馈。", {
                "idea_id": _STRING, "model": ModelSpec.model_json_schema(),
                "proposal_id": _STRING, "idempotency_key": _STRING}, ["idea_id", "model", "idempotency_key"]),
            ("research_history", "读取当前轨迹来源、想法、实验反馈、发现与报告；不读取其他候选或最终留出。", {
                "kind": {"enum": ["sources", "ideas", "proposals", "jobs", "findings", "reports", "stops"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, []),
            ("research_finding_create", "根据自己的实验登记发现、解释和局限，也记录失败原因。", {
                "statement": _STRING, "interpretation": _STRING, "limitations": _STRING,
                "job_ids": _STRINGS, "idea_ids": _STRINGS,
                "failure_category": {"enum": ["hypothesis", "implementation", "data", "budget", "none"]}},
             ["statement", "interpretation", "limitations", "job_ids"]),
            ("research_report_submit", "提交独立研究报告，引用想法、实验、发现，说明负结果与适用条件。", {
                "title": _STRING, "summary": _STRING, "body": _STRING,
                "idea_ids": _STRINGS, "job_ids": _STRINGS, "finding_ids": _STRINGS,
                "limitations": _STRING},
             ["title", "summary", "body", "idea_ids", "job_ids", "limitations"]),
            ("research_stop", "有依据地结束自己的自主研究。必须先提交覆盖全部已结算实验的报告，引用停止依据，且没有执行中的实验。不会停止其他候选。", {
                "reason": _STRING, "evidence_ids": _STRINGS}, ["reason", "evidence_ids"]),
        ]
        return [{"name": name, "description": description,
                 "inputSchema": _schema(props, required + (["proposal_id"] if (
                     name == "research_experiment_run" and self.autonomous) else []))}
                for name, description, props, required in specs]

    def call(self, name: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            spec = next((s for s in self.tool_specs() if s["name"] == name), None)
            if spec is None:
                raise ValueError(f"当前研究身份不允许工具: {name}")
            if not isinstance(args or {}, Mapping):
                raise ValueError("工具参数必须为 object")
            values = dict(args or {})
            schema = spec["inputSchema"]
            unknown = set(values) - set(schema["properties"])
            missing = set(schema["required"]) - set(values)
            if unknown or missing:
                raise ValueError(f"工具参数不匹配: unknown={sorted(unknown)}, missing={sorted(missing)}")
            result = getattr(self, "_" + name.removeprefix("research_"))(**values)
            response = {"ok": True, "tool": name, "id": result.get("id"), "summary": result}
            if len(json.dumps(response, ensure_ascii=False).encode("utf-8")) > 32 * 1024:
                # 已保存事实不丢失，响应只做摘要；不能以全量工件路径绕过身份边界。
                def compact(value):
                    if isinstance(value, str):
                        return value[:1200] + ("…（响应截断）" if len(value) > 1200 else "")
                    if isinstance(value, list):
                        return [compact(v) for v in value[:3]]
                    if isinstance(value, dict):
                        return {k: compact(v) for k, v in value.items()}
                    return value
                response["summary"] = compact(result)
                response["truncated"] = True
                response["next"] = "使用 research_history 减小 limit，或 research_source_excerpt 按 offset 阅读资料。"
                if len(json.dumps(response, ensure_ascii=False).encode("utf-8")) > 32 * 1024:
                    response["summary"] = {"id": result.get("id"), "message": "已保存，结果超过响应上限，请缩小查询范围"}
            return response
        except Exception as exc:
            return {"ok": False, "tool": name, "error": str(exc)[:2000],
                    "summary": {"error": str(exc)[:2000]}}

    def _protocol(self) -> dict[str, Any]:
        p = self.protocol
        response = {"goal": p["goal_definition"], "view": p["view_definition"],
                "validation": p["validation"], "metrics": p["metrics"],
                "fingerprint": p["fingerprint"], "feedback_surface": "validate",
                "random_seed": p["runtime"]["random_seed"],
                "split_boundaries": p.get("split_boundaries"),
                "evaluation_note": "最终留出 A/B/C、物理留出检查和完整原始制品不向研究会话反馈。",
                "model_catalog": {
                    "data_estimators": list(DATA_ESTIMATORS),
                    "physics_models": list(PHYSICS_MODELS),
                    "hybrid_residuals": list(HYBRID_RESIDUALS),
                    "note": MODEL_CATALOG_HINT.replace("tf_lab_submit", "research_lab_submit"),
                },
                "model_interface": {
                    "module": "MODEL_FORMAT='thermoforge.lab.<name>.v1'; INPUT_ROLES=[] 或角色名列表; build_model(hyperparameters, seed); load_model(directory)",
                    "model": "fit(df,y)->self; predict(df)->数值序列; save(directory) 写 model.json，format=MODEL_FORMAT，禁止pickle",
                    "data": "df 只含冻结features和object_id/timestamp，目标不在df；五连检通用特征名f1..f4，勿硬编码真实列名。角色映射可用 hyperparameters.inputs='role=column;...'。",
                    "validation": "静态扫描+接口、fit/predict、保存、重载、同种子确定性五连检；无人工批准。",
                }}
        response["research_mode"] = "autonomous" if self.autonomous else "acceptance"
        if self.autonomous:
            state = self.store.autonomy_state(self.run_id)
            # 只返回阶段，不公开其他候选尚未共享的方案、结果及重复实验信息。
            response["research_stage"] = {key: state[key] for key in (
                "stage", "proposal_barrier_open", "sharing_ready")}
            response["research_budget"] = {key: state.get(key) for key in (
                "closing_for_tokens", "report_token_reserve", "tokens_remaining")}
            response["research_workflow"] = (
                "独立登记想法→research_proposal_commit 冻结方案→全部候选提交后运行实验"
                "→登记实际反馈的 finding→引用自己的最新实验修订方案。"
                "达到目标或有依据停止时先提交完整报告，再调用 research_stop。"
                "不同候选可得到相同结论，不强迫制造差异。")
        return response

    def _data_summary(self) -> dict[str, Any]:
        p = self.protocol
        with self.lock:
            frame = materialize_view(self.ctx.vault, p["view_definition"],
                                     self.ctx.view_cache_root).table.to_pandas()
        boundaries = p["split_boundaries"]
        holdout = p["validation"]["equipment_holdout"]
        if holdout["enabled"]:
            frame = frame[~frame["object_id"].isin(holdout["holdout_objects"])]
        columns = p["view_definition"]["features"] + [p["view_definition"]["target"]]
        sections = {}
        for split_name in ("train", "validate"):
            lo, hi = map(parse_timestamp, boundaries[split_name + "_range"])
            selected = frame[(frame["timestamp"] >= lo) & (frame["timestamp"] < hi)]
            stats = {}
            for name in columns:
                values = selected[name].dropna()
                stats[name] = {"count": len(values), "missing": int(selected[name].isna().sum()),
                               "quantiles": {str(q): float(v) if math.isfinite(float(v)) else None
                                             for q, v in values.quantile([0, .25, .5, .75, 1]).items()}}
            sections[split_name] = {"n_rows": len(selected), "variables": stats,
                                    "samples": json.loads(selected.head(12).to_json(orient="records", date_format="iso"))}
        return {"protocol_fingerprint": p["fingerprint"], "sections": sections}

    def _literature_search(self, query: str, limit: int = 5,
                           sources: list[str] | None = None,
                           year_from: int | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"limit": max(1, min(10, int(limit))), "year_from": year_from}
        if sources is not None:
            kwargs["sources"] = sources
        env = tf_literature_search(self.ctx, _text(query, "query", 2000), **kwargs)
        if not env["ok"]:
            raise ValueError(str(env["summary"]))
        records = []
        for item in env["summary"]["results"]:
            content = item.get("abstract") or ""
            records.append(self._add("sources", {
                **item, "kind": "paper", "content": content,
                "read_scope": "abstract" if content else "metadata",
                "content_sha256": _digest({"title": item["title"], "content": content}),
                "retrieved_at": _now(), "verification": "retrieved", "query": query}))
        return {"sources": records, "diagnostics": env["diagnostics"]}

    def _source_read(self, url: str, title: str) -> dict[str, Any]:
        retrieved = read_public_source(_text(url, "url", 4096))
        record = self._add("sources", {**retrieved,
                                       "retrieved_scope": retrieved["read_scope"],
                                       "read_scope": "excerpt" if len(retrieved["content"]) > SOURCE_EXCERPT_CHARS else retrieved["read_scope"],
                                       "title": _text(title, "title", 1000), "kind": "paper"})
        # 完整读取工件持久保存；模型本次只收到摘录，阅读范围分别记录。
        return {**record, "content": record["content"][:SOURCE_EXCERPT_CHARS],
                "delivered_chars": min(len(record["content"]), SOURCE_EXCERPT_CHARS),
                "delivered_scope": "excerpt" if len(record["content"]) > SOURCE_EXCERPT_CHARS else record["read_scope"]}

    def _source_excerpt(self, source_id: str, offset: int = 0) -> dict[str, Any]:
        source = self._record("sources", source_id, allow_shared=True)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset 必须为非负整数")
        content = source.get("content") or ""
        excerpt = content[offset:offset + SOURCE_EXCERPT_CHARS]
        self.store.event(self.run_id, "source.read", {
            "source_id": source_id, "offset": offset, "chars": len(excerpt)}, track_id=self.track_id)
        return {"id": source_id, "title": source["title"], "url": source.get("url"),
                "content": excerpt, "offset": offset, "next_offset": offset + len(excerpt),
                "has_more": offset + len(excerpt) < len(content), "delivered_scope": "excerpt"}

    def _source_register(self, title: str, kind: str, read_scope: str,
                         content: str = "", url: str | None = None,
                         doi: str | None = None, history_ids: list[str] | None = None) -> dict[str, Any]:
        if kind not in {"paper", "article", "history"}:
            raise ValueError("无效来源类型")
        if read_scope not in {"metadata", "abstract", "excerpt", "history"}:
            raise ValueError("来源阅读范围不可伪称为工具已读全文")
        if kind == "history":
            if not history_ids:
                raise ValueError("历史来源必须引用当前轨迹的 job ID")
            self._refs("jobs", history_ids)
        elif not url and not doi:
            raise ValueError("文献/文章来源必须给稳定 URL 或 DOI")
        if read_scope != "metadata":
            _text(content, "content", MAX_SOURCE_CHARS)
        if url:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("来源 URL 必须为不含凭据的 HTTP(S) 地址")
        return self._add("sources", {"title": _text(title, "title", 1000), "kind": kind,
                                     "url": url, "doi": doi, "content": content,
                                     "read_scope": read_scope, "history_ids": history_ids or [],
                                     "retrieved_at": _now(), "verification": "agent_supplied",
                                     "content_sha256": _digest(content)})

    def _idea_create(self, statement: str, reason: str, prediction: str,
                     falsification: str, origin: str, source_ids: list[str] | None = None,
                     parent_idea_ids: list[str] | None = None,
                     parent_job_ids: list[str] | None = None,
                     strategy: str | None = None) -> dict[str, Any]:
        if origin not in {"literature", "history", "conjecture", "mixed"}:
            raise ValueError("无效想法来源类型")
        sources = self._refs("sources", source_ids or [], allow_shared=True)
        parents = self._refs("ideas", parent_idea_ids or [], allow_shared=True)
        jobs = self._refs("jobs", parent_job_ids or [], allow_shared=True)
        for parent_id in parents:
            parent = self._record("ideas", parent_id, allow_shared=True)
            if parent.get("track_id") != self.track_id:
                inherited_sources = set(parent.get("source_ids") or [])
                if not inherited_sources.issubset(sources):
                    raise ValueError("引用其他轨迹的想法时，必须同时显式引用其已授权共享的 source_ids")
        if origin in {"literature", "mixed"} and not sources:
            raise ValueError("文献或混合来源必须引用已登记资料")
        if origin == "history" and not jobs and not any(
                self._record("sources", s, allow_shared=True).get("kind") == "history" for s in sources):
            raise ValueError("历史来源必须引用自己的实验或历史资料")
        return self._add("ideas", {
            "statement": _text(statement, "statement"), "reason": _text(reason, "reason"),
            "prediction": _text(prediction, "prediction"),
            "falsification": _text(falsification, "falsification"), "origin": origin,
            "source_ids": sources, "parent_idea_ids": parents, "parent_job_ids": jobs,
            "strategy": strategy, "protocol_fingerprint": self.protocol["fingerprint"]})

    def _lab_submit(self, name: str, source: str, description: str | None = None) -> dict[str, Any]:
        if not NAME_PATTERN.fullmatch(name):
            raise ValueError("模型名称只允许小写字母开头的字母、数字与下划线，最多64字符")
        with self.lock:
            env = tf_lab_submit(self.ctx, name, source, description=description)
        if not env["ok"]:
            raise ValueError(json.dumps({"summary": env["summary"], "diagnostics": env["diagnostics"]}, ensure_ascii=False))
        return {"id": env["id"], **env["summary"]}

    def _model_request(self, model: Mapping[str, Any]) -> tuple[ModelSpec, dict[str, Any]]:
        p = self.protocol
        spec = ModelSpec(**dict(model))
        if spec.category != "lab" and not p["goal_definition"]["model_types"].get(spec.category, False):
            raise ValueError("目标不允许该模型类别")
        lab_hash = None
        if spec.category == "lab":
            name, version = parse_lab_ref(str(spec.hyperparameters["lab"]))
            if version is None:
                raise ValueError("实验必须使用不可漂移的模型引用 name@vN")
            self.ctx.lab_store.require_runnable(name, version)
            lab_hash = self.ctx.lab_store.get(name, version)["content_hash"]
        request = {"model": spec.model_dump(mode="json"), "protocol_fingerprint": p["fingerprint"]}
        if self.autonomous:
            request["experiment_fingerprint"] = experiment_fingerprint(
                p["fingerprint"], request["model"], lab_content_hash=lab_hash)
            if lab_hash is not None:
                request["lab_content_hash"] = lab_hash
        return spec, request

    def _proposal_commit(self, idea_id: str, model: Mapping[str, Any], purpose: str,
                         idempotency_key: str, expected_cost: str | None = None) -> dict[str, Any]:
        if not self.autonomous:
            raise ValueError("只有自主研究模式需要提交冻结方案")
        if purpose not in {"explore", "refine", "replicate"}:
            raise ValueError("方案目的必须为 explore、refine 或 replicate")
        idea = self._record("ideas", idea_id)
        if idea.get("protocol_fingerprint") != self.protocol["fingerprint"]:
            raise ValueError("想法不属于当前冻结评价协议")
        # 重新验证引用的可见性，避免把未获共享授权的跨轨迹证据写入冻结方案。
        self._refs("sources", idea.get("source_ids") or [], allow_shared=True)
        self._refs("ideas", idea.get("parent_idea_ids") or [], allow_shared=True)
        parents = self._refs("jobs", idea.get("parent_job_ids") or [], allow_shared=True)
        _, request = self._model_request(model)
        findings = [finding["id"] for finding in self._records("findings")
                    if set(finding.get("job_ids") or []) & set(parents)]
        details = {"purpose": purpose,
                   "expected_cost": _text(expected_cost, "expected_cost", 2000) if expected_cost is not None else None,
                   "parent_job_ids": parents, "finding_ids": findings,
                   "experiment_fingerprint": request["experiment_fingerprint"],
                   "lab_content_hash": request.get("lab_content_hash")}
        return self.store.commit_proposal(
            self.run_id, self.track_id, idea_id, request["model"], details,
            _text(idempotency_key, "idempotency_key", 200))

    def _stop(self, reason: str, evidence_ids: list[str]) -> dict[str, Any]:
        if not self.autonomous:
            raise ValueError("只有自主研究模式支持有依据的研究停止请求")
        if not isinstance(evidence_ids, list) or not evidence_ids or any(
                not isinstance(value, str) or not value.strip() for value in evidence_ids):
            raise ValueError("停止请求必须引用非空证据 ID 列表")
        return self.store.request_research_stop(
            self.run_id, self.track_id, _text(reason, "reason"),
            list(dict.fromkeys(evidence_ids)))

    @staticmethod
    def _job_response(job: Mapping[str, Any]) -> dict[str, Any]:
        response = {"id": job["id"], "status": job["status"], "result": job.get("result")}
        if "executed" in job:
            response["executed"] = job["executed"]
        if job.get("reused_from_job_id"):
            response.update(reused=True, reused_from_job_id=job["reused_from_job_id"],
                            reuse_note="复用已共享阶段的同协议、同模型既有结果；本作业未重新训练。")
        return response

    def _experiment_run(self, idea_id: str, model: Mapping[str, Any],
                        idempotency_key: str, proposal_id: str | None = None) -> dict[str, Any]:
        idea = self._record("ideas", idea_id)
        p = self.protocol
        if idea.get("protocol_fingerprint") != p["fingerprint"]:
            raise ValueError("想法不属于当前冻结评价协议")
        spec, request = self._model_request(model)
        if self.autonomous:
            # store 在预算预留事务内核验方案/模型/协议及首轮提交屏障。
            request["proposal_id"] = _text(proposal_id, "proposal_id", 200)
        elif proposal_id is not None:
            raise ValueError("验收模式不接受自主研究方案引用")
        job = self.store.reserve_job(self.run_id, self.track_id, idea_id, request,
                                     _text(idempotency_key, "idempotency_key", 200))
        if not job.get("fresh", False):
            return self._job_response(job) | {"duplicate": True}
        with self.experiment_semaphore if self.experiment_semaphore is not None else nullcontext():
            started = self.store.start_job(job["id"])
            if not started.get("started"):
                return self._job_response(started)
            run = self.store.get_run(self.run_id)
            try:
                with self.lock:
                    # V1 ledger 的无证据规则以 goal 为界；V2 的真实科学谱系由 ideas
                    # 持有，每个想法用一份同协议目标适配旧内核，不伪造 EXP/F 依据。
                    local_goal_doc = copy.deepcopy(p["goal_definition"])
                    local_goal_doc["goal_id"] = self.ctx.ledger.allocator.allocate("RG-")
                    local_goal = self.ctx.ledger.create_goal(
                        local_goal_doc["name"], actor=self.ctx.actor,
                        definition=local_goal_doc, entity_id=local_goal_doc["goal_id"],
                        reason=f"V2 {idea_id} 冻结协议适配")
                    hypothesis = self.ctx.ledger.create_hypothesis(
                        local_goal["id"], idea["statement"], actor=self.ctx.actor,
                        reason=idea["reason"])
                    view = self.ctx.ledger.register_view(p["view_definition"], actor=self.ctx.actor)
                    definition = {"goal_id": local_goal["id"], "hypothesis_id": hypothesis["id"],
                                  "dataset_view": view["id"], "model": request["model"],
                                  "target": p["goal_definition"]["target"],
                                  "validation": p["validation"], "metrics": p["metrics"],
                                  "physics_tests": p["physics_tests"], "runtime": p["runtime"],
                                  "description": f"V2 {self.run_id}/{self.track_id}/{idea_id}",
                                  "author": self.ctx.actor}
                    plan = tf_experiment_plan(self.ctx, definition)
                    if not plan["ok"]:
                        raise ValueError(str(plan["summary"]))
                    exp_id = plan["id"]
                    self.store.update_record(self.run_id, "jobs", job["id"], {"experiment_id": exp_id})
                    timeout = float(run["config"].get("experiment_timeout_seconds", 600))
                    if not math.isfinite(timeout) or timeout <= 0:
                        raise ValueError("experiment_timeout_seconds 必须为有限正数")
                    env = tf_experiment_run(self.ctx, exp_id, purge_seconds=p["purge_seconds"],
                                            embargo_seconds=p["embargo_seconds"], y_floor=p["y_floor"],
                                            timeout_seconds=timeout)
                surface = (env.get("summary", {}).get("surfaces") or {}).get("validate") or {}
                feedback = {"experiment_id": exp_id, "protocol_fingerprint": p["fingerprint"],
                            "feedback_surface": "validate", "metrics": surface.get("metrics") or {},
                            "n_samples": surface.get("n_samples"), "model": {"spec": request["model"]},
                            "duration_seconds": env.get("summary", {}).get("duration_seconds")}
                if spec.category == "lab":
                    name, version = parse_lab_ref(str(spec.hyperparameters["lab"]))
                    lab = self.ctx.lab_store.get(name, version)
                    feedback["model"].update(lab_ref=spec.hyperparameters["lab"], content_hash=lab["content_hash"])
                success = bool(env["ok"])
                if not success:
                    feedback.update(failure_category="implementation", error_code=env.get("summary", {}).get("error_code"),
                                    error="; ".join(d.get("message", "") for d in env.get("diagnostics", []))[:1500])
                settled = self.store.settle_job(job["id"], "completed" if success else "failed", feedback)
            except Exception as exc:
                feedback = {"protocol_fingerprint": p["fingerprint"], "error": str(exc)[:1500],
                            "failure_category": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "implementation"}
                settled = self.store.settle_job(job["id"], "failed", feedback)
        return self._job_response(settled)

    def _history(self, kind: str = "jobs", limit: int = 20) -> dict[str, Any]:
        if kind not in {"sources", "ideas", "proposals", "jobs", "findings", "reports", "stops"}:
            raise ValueError("不允许读取该类记录")
        records = self._records(kind)[-max(1, min(50, int(limit))):]
        if kind == "sources":
            records = [{**r, "content": (r.get("content") or "")[:SOURCE_EXCERPT_CHARS],
                        "delivered_scope": "excerpt" if len(r.get("content") or "") > SOURCE_EXCERPT_CHARS
                        else r.get("read_scope")} for r in records]
        return {"kind": kind, "records": records}

    def _finding_create(self, statement: str, interpretation: str, limitations: str,
                        job_ids: list[str], idea_ids: list[str] | None = None,
                        failure_category: str = "none") -> dict[str, Any]:
        jobs = self._refs("jobs", job_ids)
        if not jobs or any(self._record("jobs", j)["status"] not in {"completed", "failed", "cancelled", "interrupted"} for j in jobs):
            raise ValueError("发现必须引用已结束的实验作业")
        if failure_category not in {"hypothesis", "implementation", "data", "budget", "none"}:
            raise ValueError("无效失败分类")
        return self._add("findings", {
            "statement": _text(statement, "statement"), "interpretation": _text(interpretation, "interpretation"),
            "limitations": _text(limitations, "limitations"), "job_ids": jobs,
            "idea_ids": self._refs("ideas", idea_ids or []), "failure_category": failure_category,
            "protocol_fingerprint": self.protocol["fingerprint"]})

    def _report_submit(self, title: str, summary: str, body: str, idea_ids: list[str],
                       job_ids: list[str], limitations: str,
                       finding_ids: list[str] | None = None) -> dict[str, Any]:
        ideas, jobs = self._refs("ideas", idea_ids), self._refs("jobs", job_ids)
        if not ideas:
            raise ValueError("报告必须引用想法，以便追溯方向来源")
        return self._add("reports", {
            "kind": "track", "title": _text(title, "title", 1000),
            "summary": _text(summary, "summary"), "body": _text(body, "body", 100_000),
            "idea_ids": ideas, "job_ids": jobs, "finding_ids": self._refs("findings", finding_ids or []),
            "limitations": _text(limitations, "limitations"),
            "protocol_fingerprint": self.protocol["fingerprint"]})
