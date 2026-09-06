"""外部操作副驾驶的任务级 MCP；研究生命周期属于独立服务。"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
import uuid

from .client import V2Client


MAX_RESPONSE_BYTES = 32000
_EVIDENCE_FIELDS = ("source_ids", "idea_ids", "job_ids", "finding_ids", "evidence_ids",
                    "parent_idea_ids", "parent_job_ids")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _clip(value, size=320):
    """按 UTF-8 字节保留原文片段；不让截断成为无标记的改写。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    value = str(value)
    raw = value.encode("utf-8")
    marker = "…[截断]"
    if len(raw) <= size:
        return value
    return raw[:max(0, size - len(marker.encode("utf-8")))].decode("utf-8", errors="ignore") + marker


def _bounded(value, *, size=320, depth=3):
    if isinstance(value, dict):
        if depth <= 0:
            return "[嵌套内容省略，见完整工件]"
        result = {_clip(key, 100): _bounded(item, size=size, depth=depth-1)
                  for key, item in list(value.items())[:8]}
        if len(value) > 8:
            result["_omitted_fields"] = len(value) - 8
        while len(_encode(result).encode("utf-8")) > max(512, size * 4) and len(result) > 1:
            removable = [key for key in result if key != "_omitted_fields"]
            result.pop(removable[-1])
            result["_omitted_fields"] = result.get("_omitted_fields", 0) + 1
        return result
    if isinstance(value, (list, tuple)):
        if depth <= 0:
            return "[列表内容省略，见完整工件]"
        result = [_bounded(item, size=size, depth=depth-1) for item in value[:8]]
        if len(value) > 8:
            result.append(f"[其余 {len(value)-8} 项省略]")
        return result
    return _clip(value, size)


def _pick(record, fields, *, size=320):
    return {key: _bounded(record.get(key), size=size) for key in fields}


def _records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _latest_per_track(records):
    # store.snapshot 的记录按写入顺序提供；每条轨迹最后登记的记录优先。
    latest = {}
    for item in records:
        latest[item.get("track_id")] = item
    return list(latest.values())


def _compact_envelope(name, result, path):
    """仅抽取已存事实，轮流填充各类记录，防止长原文挤掉来源或比较依据。"""
    data = result if isinstance(result, dict) else {}
    run = data.get("run") if isinstance(data.get("run"), dict) else data
    collections = {key: _records(data.get(key)) for key in
                   ("tracks", "reports", "sources", "ideas", "jobs", "events")}
    counts = {}
    supplied_counts = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    for key in ("tracks", "ideas", "jobs", "sources"):
        counts[key] = supplied_counts.get(key) if key in supplied_counts else (
            len(data[key]) if isinstance(data.get(key), list) else None)
    comparison = data.get("comparability") or {}
    comparison = comparison if isinstance(comparison, dict) else {}
    excluded = comparison.get("excluded")
    groups = _records(comparison.get("groups"))
    config = run.get("config") if isinstance(run.get("config"), dict) else {}
    summary = {
        **_pick(run, ("run_id", "status", "version", "updated_at", "error")),
        "title": _clip(data.get("title")), "summary": _bounded(counts),
        "budget": _pick(config, ("token_budget", "max_experiments", "max_experiments_per_track", "max_turns")),
        "progress": _pick(run, ("tokens_used", "experiments_reserved", "experiments_settled")),
        "usage_totals": _bounded((data.get("usage") or {}).get("totals")) if isinstance(data.get("usage"), dict) else None,
        "comparison": {"note": _clip(comparison.get("note"), 640),
                       "excluded_count": len(excluded) if isinstance(excluded, list) else None,
                       "group_count": len(groups) if isinstance(comparison.get("groups"), list) else None,
                       "groups": []},
        "message": "以下为原始记录的有界摘录，未生成新结论。列表按 sampling 抽样；字段截断有标记。"
                   "未返回的内容不等于不存在；null 表示缺失或不可得，不按零计算。完整结果见工件。",
    }
    if "cursor" in data:
        summary["cursor"] = _bounded(data["cursor"])
    latest_reports = _latest_per_track(collections["reports"])
    # 先覆盖不同轨迹，再列最近想法；来源优先匹配这些想法的引用。
    ideas = _latest_per_track(collections["ideas"])
    ideas += [item for item in reversed(collections["ideas"]) if item not in ideas]
    referenced = {sid for idea in ideas[:32] for sid in (idea.get("source_ids") or []) if isinstance(sid, str)}
    sources = sorted(collections["sources"], key=lambda item: item.get("id") not in referenced)
    errors = []
    for record in reversed(collections["jobs"]):
        result_record = record.get("result") if isinstance(record.get("result"), dict) else {}
        error = result_record.get("error") or record.get("error")
        if error is not None:
            errors.append({**_pick(record, ("id", "track_id", "status", "created_at")),
                           "error": _bounded(error), "failure_category": _bounded(result_record.get("failure_category"))})
    pools = {
        "latest_reports": [_pick(r, ("id", "track_id", "kind", "title", "summary", "limitations", *_EVIDENCE_FIELDS), size=480)
                           for r in latest_reports],
        "tracks": [{**_pick(r, ("track_id", "role", "status", "phase", "turns", "pid", "session_id", "error")),
                    "usage": _bounded(r.get("usage"), size=80)} for r in collections["tracks"]],
        "sources": [_pick(r, ("id", "track_id", "title", "url", "doi", "read_scope", "verification")) for r in sources[:32]],
        "ideas": [_pick(r, ("id", "track_id", "statement", "reason", "origin", *_EVIDENCE_FIELDS)) for r in ideas[:32]],
        "comparison_groups": [{"protocol_fingerprint": _clip(g.get("protocol_fingerprint")),
                               "compared_count": len(g["rows"]) if isinstance(g.get("rows"), list) else None,
                               "best_per_metric": _bounded(g.get("best_per_metric"), size=100, depth=4)} for g in groups[:8]],
        "recent_errors": errors[:8],
        "recent_events": [_pick(e, ("seq", "track_id", "kind", "at", "payload"), size=160) for e in collections["events"][-8:]],
    }
    # prepare/list 也可能超限；保留真实候选与缺项，仍不能虚构发现结果。
    if isinstance(result, list):
        pools["runs"] = [_pick(r, ("run_id", "status", "version", "updated_at", "error")) for r in _records(result)[:32]]
    for key in ("goals", "datasets"):
        if key in data:
            pools[key] = [_pick(r, ("id", "goal_id", "ref", "dataset_ref", "name", "title")) for r in _records(data[key])[:16]]
    if "available" in data:
        summary.update(_pick(data, ("available", "ready", "missing", "errors", "defaults")))
    if "final_evaluation" in data:
        final = data.get("final_evaluation")
        summary["final_evaluation"] = _pick(final, ("status", "job_id", "experiment_id", "selection_reason", "selection_metric",
                                                    "feedback_to_agents", "surfaces")) if isinstance(final, dict) else None
    summary["progress"]["jobs_by_status"] = dict(Counter(str(j.get("status") or "unknown")
        for j in collections["jobs"])) if isinstance(data.get("jobs"), list) else None
    totals = {**{key: len(value) if isinstance(data.get(key), list) else None for key, value in collections.items()},
              "latest_reports": len(latest_reports) if isinstance(data.get("reports"), list) else None,
              "comparison_groups": len(groups) if isinstance(comparison.get("groups"), list) else None,
              "recent_errors": len(errors) if isinstance(data.get("jobs"), list) else None,
              "recent_events": len(collections["events"]) if isinstance(data.get("events"), list) else None}
    summary["sampling"] = {key: {"available": totals.get(key, len(data[key]) if isinstance(data.get(key), list) else
                                len(result) if key == "runs" else None), "returned": 0}
                           for key in pools}
    report_artifacts = [_pick(a, ("format", "path"), size=2048) for a in _records(data.get("artifacts"))[:3]]
    envelope = {"ok": True, "tool": _clip(name, 120), "summary": summary, "truncated": True,
                "artifacts": [{"path": str(path), "kind": "full_result"}, *report_artifacts]}
    summary["sampling_note"] = "每轨迹最新报告与想法优先；来源优先匹配想法引用；错误/事件取最近记录。受总字节上限约束。"
    for key in pools:
        if key != "comparison_groups":
            summary[key] = []
    # 基础字段也有长度界限；极长操作系统路径等异常输入仍不能突破响应协议。
    if len(_encode(envelope).encode("utf-8")) > MAX_RESPONSE_BYTES:
        for depth in (5, 4, 3, 2, 1):
            fallback = _bounded(envelope, size=256, depth=depth)
            if len(_encode(fallback).encode("utf-8")) <= MAX_RESPONSE_BYTES:
                return fallback
    for index in range(max((len(pool) for pool in pools.values()), default=0)):
        for key, pool in pools.items():
            if index >= len(pool):
                continue
            target = summary["comparison"]["groups"] if key == "comparison_groups" else summary[key]
            target.append(pool[index])
            summary["sampling"][key]["returned"] += 1
            if len(_encode(envelope).encode("utf-8")) > MAX_RESPONSE_BYTES:
                target.pop()
                summary["sampling"][key]["returned"] -= 1
    return envelope


def _invoke(name, function, client=None):
    try:
        client = client or V2Client()
        result = function(client)
        envelope = {"ok": True, "tool": name, "summary": result}
        body = _encode(envelope)
        if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
            directory = client.root / "tool_artifacts"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{uuid.uuid4().hex}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(body, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
            body = _encode(_compact_envelope(name, result, path))
        return body
    except Exception as exc:
        return _encode({"ok": False, "tool": _clip(name, 120), "code": _clip(getattr(exc, "code", "TFV2-ERROR"), 120),
                        "error": _clip(str(exc), 16000)})


def tf_v2_prepare(config: dict | None = None) -> str:
    """查询可用研究目标、数据、Codex 能力与默认值；传配置可验证缺少的输入。"""
    return _invoke("tf_v2_prepare", lambda c: c.prepare(config))


def tf_v2_start(config: dict, idempotency_key: str) -> str:
    """按用户目标和预算后台启动主 Codex＋候选；相同请求重试必须复用幂等键。"""
    return _invoke("tf_v2_start", lambda c: c.start(config, idempotency_key))


def tf_v2_list() -> str:
    """列出已持久保存的 V2 研究运行。"""
    return _invoke("tf_v2_list", lambda c: c.list_runs())


def tf_v2_status(run_id: str) -> str:
    """读取后台研究与六条独立轨迹的真实状态、实验和预算。"""
    return _invoke("tf_v2_status", lambda c: c.get_run(run_id))


def tf_v2_events(run_id: str, after: int = 0, limit: int = 100) -> str:
    """按游标读取新事件；继续查询复用返回 cursor，避免重复读取历史。"""
    return _invoke("tf_v2_events", lambda c: c.events(run_id, after, limit))


def tf_v2_control(run_id: str, action: str, expected_version: int | None = None,
                  changes: dict | None = None) -> str:
    """pause/resume/cancel/update：统一控制研究。调整指导或预算用 changes，返回实际状态。"""
    return _invoke("tf_v2_control", lambda c: c.control(run_id, action, expected_version, changes))


def tf_v2_report(run_id: str, track_id: str | None = None) -> str:
    """生成带来源、实验、失败和比较依据的报告，返回 Markdown/HTML/JSON 工件。"""
    return _invoke("tf_v2_report", lambda c: c.get_report(run_id, track_id))


CONTROL_TOOLS = (tf_v2_prepare, tf_v2_start, tf_v2_list, tf_v2_status,
                 tf_v2_events, tf_v2_control, tf_v2_report)


def agent_tools():
    """网页副驾复用同一任务级控制面，根目录来自其真实 ToolContext。"""
    from thermoforge_agent.schema import tool_schema
    methods = {"tf_v2_prepare": "prepare", "tf_v2_start": "start", "tf_v2_list": "list_runs",
               "tf_v2_status": "get_run", "tf_v2_events": "events", "tf_v2_control": "control",
               "tf_v2_report": "get_report"}
    def bind(name):
        def dispatch(ctx, **arguments):
            client = V2Client(research_root=ctx.research_root, vault_root=ctx.vault_root,
                              models_root=ctx.models_root)
            return json.loads(_invoke(name, lambda c: getattr(c, methods[name])(**arguments), client))
        return dispatch
    return ([tool_schema(fn.__name__, fn) for fn in CONTROL_TOOLS],
            {fn.__name__: bind(fn.__name__) for fn in CONTROL_TOOLS})


def build_server():
    from mcp.server.mcpserver import MCPServer
    server = MCPServer(name="thermoforge-v2", instructions=(
        "你是 ThermoForge 外部操作副驾驶，内部主智能体与五候选由软件拥有。"
        "先 prepare 发现真实目标、数据与默认配置。用户已要求启动且必要输入齐备时直接 start；"
        "不要虚构ID、预算或来源。保存run_id，后续用status/events/control/report。"
        "软件自主推进，不需要逐轮发送继续。查询不是启动或修改授权。报告以工具事实为准。"))
    for tool in CONTROL_TOOLS:
        server.add_tool(tool, name=tool.__name__)
    return server


def main():
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
