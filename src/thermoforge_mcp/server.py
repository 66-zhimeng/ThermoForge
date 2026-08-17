"""MCP server 构建：TOOL_REGISTRY → MCP 工具清单。

工具函数第一个参数是 `ToolContext`，那是控制面的东西，不该出现在给外部
Agent 看的参数表里。这里为每个工具生成一个去掉 `ctx` 的包装函数，签名与
类型注解照抄原函数——MCP SDK 靠 `inspect.signature` 推导入参 Schema，
所以把 `__signature__` 和 `__annotations__` 补对了就够，不需要写死一份
Schema 再和代码对不上。
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, get_type_hints

from thermoforge_research.tools import TOOL_REGISTRY, ToolContext

SERVER_NAME = "thermoforge"

# 需要 actor=human 的工具不暴露（见模块 __init__ 的说明）。
# 只剩预处理审批：它改的是数据，必须人批；模型实验室改的是假设，全自治。
EXCLUDED_TOOLS = frozenset({"tf_preprocess_approve"})

# MCP 客户端一次读进上下文的量有限；这里与工具信封本身的 32KB 上限同调，
# 超出的部分工具层已经落成 artifact 并置 truncated，不需要再截一次。
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env_name: str, default: str) -> Path:
    value = os.environ.get(env_name)
    return Path(value) if value else _REPO_ROOT / default


def tool_context(actor: str = "mcp") -> ToolContext:
    """MCP 侧的上下文。actor 记成 `mcp`，Ledger 里能看出是外部 Agent 干的。"""
    return ToolContext(
        vault_root=_root("TF_VAULT_ROOT", "vault"),
        research_root=_root("TF_RESEARCH_ROOT", "research"),
        models_root=_root("TF_MODELS_ROOT", "models"),
        actor=actor,
    )


def _describe(fn: Callable) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.split("\n\n", 1)[0].replace("\n", " ").strip()


def make_wrapper(name: str, fn: Callable) -> Callable[..., str]:
    """去掉 `ctx` 的包装函数，返回信封 JSON 字符串。

    返回字符串而不是 dict：MCP 工具结果最终要变成文本给模型看，自己
    序列化能保证 `ensure_ascii=False`——否则中文诊断信息会变成一串
    `\\uXXXX`，模型读得懂但人看日志时抓瞎。
    """
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # 前向引用解析不了时退回不带注解，Schema 仍可用
        hints = {}
    # 工具模块开了 `from __future__ import annotations`，签名里的注解是**字符串**。
    # 直接把这份签名交出去，pydantic 会尝试在本模块的命名空间里按名字解析
    # `Mapping` / `Sequence`，解析不到就报 "is not fully defined"。所以这里
    # 把每个参数的注解换成 get_type_hints 解析好的真实类型对象。
    parameters = [
        param.replace(annotation=hints.get(key, param.annotation))
        for key, param in signature.parameters.items() if key != "ctx"
    ]

    def wrapper(**kwargs: Any) -> str:
        envelope = fn(tool_context(), **kwargs)
        return json.dumps(envelope, ensure_ascii=False, default=str)

    wrapper.__name__ = name
    wrapper.__doc__ = _describe(fn)
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=str)
    wrapper.__annotations__ = {
        key: hints[key] for key in (p.name for p in parameters)
        if key in hints
    } | {"return": str}
    return wrapper


def tf_status() -> str:
    """ThermoForge 状态面板：研究目标进展、最近实验、生产模型、待审批规则。"""
    from thermoforge_cli.status import build_status

    return json.dumps(build_status(tool_context(), recent=8),
                      ensure_ascii=False, default=str)


def tf_experiment_report(experiment_id: str) -> str:
    """读取一次实验的完整报告（各评估面指标、物理检查、切分、可复现信息）。

    比 `tf_experiment_get` 多给出按面拆开的指标明细与补齐的 R²，适合
    「解释这次实验为什么好/为什么差」这类问题。
    """
    from thermoforge_webui.services.experiments import load_detail

    detail = load_detail(experiment_id)
    if detail is None:
        return json.dumps({"ok": False, "error": f"找不到实验 {experiment_id}"},
                          ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "experiment_id": detail.experiment_id,
        "status": detail.status,
        "model": detail.model_label,
        "target": detail.target,
        "surfaces": detail.surfaces,
        "physics": detail.physics,
        "split": detail.split,
        "r2_backfilled": detail.r2_backfilled,
        "report": detail.report,
    }, ensure_ascii=False, default=str)


EXTRA_TOOLS: tuple[Callable[..., str], ...] = (tf_status, tf_experiment_report)

# harness/prompts/mcp.md 缺失时的兜底（真正生效的是那个文件）
_FALLBACK_INSTRUCTIONS = (
    "ThermoForge 是数据中心暖通的物理-数据混合建模研究系统。"
    "所有能力都经这些工具，每个工具返回统一信封："
    "`ok` 为 false 表示工具级失败（不会抛异常），必须读 `ok` 而不是"
    "看有没有报错。有副作用的工具返回稳定 ID（dataset@rev_NNNN / "
    "RG- / H- / EXP- / VIEW- / model@version）。\n"
    "先调 tf_status 看当前进展，再决定下一步。\n"
    "注意：候选输入白名单是硬门禁——派生量（由公式算出来的列）不能"
    "用来预测它的原料，否则是循环论证。预处理规则的审批需要人在"
    "网页控制台完成，这里没有审批工具。"
)


def build_server():
    """装配 MCPServer。"""
    from mcp.server.mcpserver import MCPServer

    from thermoforge_agent import prompts

    server = MCPServer(
        name=SERVER_NAME,
        # 说明书走 prompts 层：外部 agent 拿到的不只是「工具怎么调」，还有
        # harness/skills 里的方法论（取证 + 系统辨识阶梯）。没有它，外部
        # agent 会重复副驾踩过的坑——猜不存在的 estimator、跳过朴素基线。
        instructions=prompts.load("mcp", fallback=_FALLBACK_INSTRUCTIONS),
    )
    for name, fn in sorted(TOOL_REGISTRY.items()):
        if name in EXCLUDED_TOOLS:
            continue
        server.add_tool(make_wrapper(name, fn), name=name,
                        description=_describe(fn))
    for extra in EXTRA_TOOLS:
        server.add_tool(extra, name=extra.__name__,
                        description=_describe(extra))
    return server


def main() -> int:
    """stdio 传输启动（MCP 客户端以子进程方式拉起本进程）。"""
    build_server().run(transport="stdio")
    return 0
