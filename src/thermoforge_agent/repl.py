"""`tf agent` 终端 REPL：多轮对话、工具调用一行摘要、审批弹确认。"""

from __future__ import annotations

import sys
from typing import Any, Mapping

from .agent import deepseek_harness

BANNER = (
    "ThermoForge 内置研发 Agent（输入 /exit 退出，/tools 查看可用工具）"
)


def make_approval_handler(input_fn=input):
    """REPL 审批：向用户弹 y/n 确认。"""

    def handler(tool: str, arguments: Mapping[str, Any],
                reason: str) -> bool:
        print(f"\n[审批请求] 工具 {tool}", file=sys.stderr)
        print(f"  理由: {reason}", file=sys.stderr)
        print(f"  参数: {arguments}", file=sys.stderr)
        answer = input_fn("确认以 human 身份执行？[y/N] ")
        return answer.strip().lower() in ("y", "yes")

    return handler


def run_repl(agent: deepseek_harness, *, input_fn=input) -> None:
    def on_tool_call(name: str, ok: bool) -> None:
        print(f"  → 调用 {name} → {'ok' if ok else 'FAILED'}",
              file=sys.stderr)

    agent.on_tool_call = on_tool_call
    if agent.approval_handler is None:
        agent.approval_handler = make_approval_handler(input_fn)
    print(BANNER, file=sys.stderr)
    while True:
        try:
            text = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/tools":
            print("\n".join(sorted(agent.dispatch)), file=sys.stderr)
            continue
        try:
            answer = agent.ask(text)
        except Exception as exc:
            print(f"agent: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not (agent.config.stream and agent.on_content_delta):
            print(answer)
        else:
            print()  # 流式已逐段输出，补换行
