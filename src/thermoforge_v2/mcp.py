"""外部操作副驾驶的任务级 MCP；研究生命周期属于独立服务。"""

from __future__ import annotations

import json
from pathlib import Path
import uuid

from .client import V2Client


def _invoke(name, function, client=None):
    try:
        client = client or V2Client()
        result = function(client)
        envelope = {"ok": True, "tool": name, "summary": result}
        body = json.dumps(envelope, ensure_ascii=False, default=str)
        if len(body.encode("utf-8")) > 32000:
            directory = client.root / "tool_artifacts"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{uuid.uuid4().hex}.json"
            path.write_text(body, encoding="utf-8", newline="\n")
            summary = {"message": "完整结果已保存为本地工件", "path": str(path)}
            if isinstance(result, dict):
                run = result.get("run") or result
                summary.update({k: run[k] for k in ("run_id", "status", "version") if k in run})
                if "tracks" in result:
                    summary["tracks"] = [{k: t.get(k) for k in ("track_id", "status", "phase", "turns", "error")}
                                          for t in result["tracks"]]
            body = json.dumps({"ok": True, "tool": name, "summary": summary, "truncated": True,
                               "artifacts": [{"path": str(path), "kind": "full_result"}]}, ensure_ascii=False)
        return body
    except Exception as exc:
        return json.dumps({"ok": False, "tool": name, "code": getattr(exc, "code", "TFV2-ERROR"),
                           "error": str(exc)}, ensure_ascii=False)


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
