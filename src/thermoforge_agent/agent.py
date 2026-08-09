"""PiAgent 对话循环：多轮上下文 + function calling + 审批拦截 + 会话留痕。

- 模型请求工具 → 本地执行（TOOL_REGISTRY 薄封装）→ 信封 JSON 原样回传
  （32KB 截断约定天然适配上下文窗口）。
- human-only 工具（默认 tf_preprocess_approve）不在工具清单内；模型经
  `tf_human_approval` 发起审批 → `approval_handler` 弹确认 → 用户同意后
  以 actor=human 的 ToolContext 执行（I-49）。
- 每轮 user/assistant/tool 事件写 `research/agent_sessions/<ts>.jsonl`，
  供谱系追溯。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from thermoforge_research.tools import TOOL_REGISTRY, ToolContext

from .client import ChatClient, ChatResult, ToolCallRequest
from .config import AgentConfig
from .schema import HUMAN_APPROVAL_TOOL, build_tool_schemas

ApprovalHandler = Callable[[str, Mapping[str, Any], str], bool]
"""（tool, arguments, reason）→ 用户是否同意。"""

ToolCallObserver = Callable[[str, bool], None]
"""（tool, ok）→ REPL 打印一行摘要。"""

DECLINED_ENVELOPE = {
    "ok": False,
    "tool": HUMAN_APPROVAL_TOOL,
    "id": None,
    "status": "DECLINED_BY_USER",
    "inputs": {},
    "summary": {"error": "用户拒绝了该审批请求，动作未执行"},
    "diagnostics": [],
    "artifacts": [],
    "truncated": False,
}


class PiAgent:
    """内置研发 Agent。

    ::

        agent = PiAgent(config, ctx)
        answer = agent.ask("列出所有数据集")
    """

    def __init__(
        self,
        config: AgentConfig,
        ctx: ToolContext,
        *,
        client: Any | None = None,  # 测试注入 stub；缺省走真实 ChatClient
        system_prompt: str | None = None,
        approval_handler: ApprovalHandler | None = None,
        on_tool_call: ToolCallObserver | None = None,
        on_content_delta: Callable[[str], None] | None = None,
        session_dir: str | Path | None = None,
    ):
        self.config = config
        self.ctx = ctx
        self.client = client or ChatClient(config)
        self.system_prompt = system_prompt or _default_system_prompt()
        self.approval_handler = approval_handler
        self.on_tool_call = on_tool_call
        self.on_content_delta = on_content_delta
        self.schemas, self.dispatch = build_tool_schemas(config.tools_exclude)
        self._human_ctx = ToolContext(
            vault_root=ctx.vault_root, research_root=ctx.research_root,
            models_root=ctx.models_root, tfom_registry=ctx.tfom_registry,
            actor="human",
        )
        self.session_dir = Path(
            session_dir or ctx.research_root / "agent_sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self._session_path = self.session_dir / f"{stamp}.jsonl"
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    # ---------------------------------------------------------------- 对话

    def ask(self, text: str) -> str:
        """一轮用户提问；内部按需要执行多轮工具调用，返回最终回答。"""
        self._append({"role": "user", "content": text})
        for _ in range(self.config.max_tool_rounds):
            result = self.client.chat(
                self.messages, self.schemas,
                on_content_delta=self.on_content_delta,
            )
            if not isinstance(result, ChatResult):
                raise TypeError("client.chat 必须返回 ChatResult")
            self._append(result.raw_message
                         or {"role": "assistant", "content": result.content})
            if not result.tool_calls:
                return result.content or ""
            for call in result.tool_calls:
                envelope = self._execute(call)
                self._append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(envelope, ensure_ascii=False),
                })
        raise RuntimeError(
            f"工具调用超过 {self.config.max_tool_rounds} 轮仍未收敛，已中止"
        )

    # ---------------------------------------------------------------- 工具执行

    def _execute(self, call: ToolCallRequest) -> dict[str, Any]:
        if call.name == HUMAN_APPROVAL_TOOL:
            return self._execute_approval(call)
        fn = self.dispatch.get(call.name)
        if fn is None:
            envelope = self._error_envelope(
                call.name, f"工具未暴露给 Agent: {call.name}")
        else:
            try:
                envelope = fn(self.ctx, **call.arguments)
            except TypeError as exc:  # 参数不匹配：回给模型修正
                envelope = self._error_envelope(call.name, str(exc))
        self._log({"type": "tool_result", "tool": call.name,
                   "ok": envelope.get("ok"), "id": envelope.get("id"),
                   "status": envelope.get("status")})
        if self.on_tool_call:
            self.on_tool_call(call.name, bool(envelope.get("ok")))
        return envelope

    def _execute_approval(self, call: ToolCallRequest) -> dict[str, Any]:
        """审批元工具：弹确认 → 同意后以 actor=human 执行目标工具。"""
        target = str(call.arguments.get("tool") or "")
        arguments = call.arguments.get("arguments") or {}
        reason = str(call.arguments.get("reason") or "")
        fn = TOOL_REGISTRY.get(target)
        approved = False
        if fn is not None and self.approval_handler is not None:
            approved = bool(self.approval_handler(target, arguments, reason))
        self._log({"type": "approval", "tool": target,
                   "arguments": dict(arguments), "reason": reason,
                   "approved": approved})
        if fn is None:
            return self._error_envelope(target, f"未知工具: {target}")
        if not approved:
            if self.on_tool_call:
                self.on_tool_call(target, False)
            return {**DECLINED_ENVELOPE, "tool": target,
                    "inputs": dict(arguments)}
        try:
            envelope = fn(self._human_ctx, **arguments)
        except TypeError as exc:
            envelope = self._error_envelope(target, str(exc))
        if self.on_tool_call:
            self.on_tool_call(target, bool(envelope.get("ok")))
        return envelope

    @staticmethod
    def _error_envelope(tool: str, message: str) -> dict[str, Any]:
        return {
            "ok": False, "tool": tool, "id": None, "status": "FAILED",
            "inputs": {}, "summary": {"error": message},
            "diagnostics": [], "artifacts": [], "truncated": False,
        }

    # ---------------------------------------------------------------- 留痕

    def _append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._log({"type": "message", "role": message.get("role"),
                   "content": _summarize_content(message)})

    def _log(self, event: Mapping[str, Any]) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(), **event}
        with open(self._session_path, "a", encoding="utf-8",
                  newline="\n") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summarize_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, str) and len(content) > 2000:
        return content[:2000] + "…(截断)"
    return content


def _default_system_prompt() -> str:
    path = (Path(__file__).resolve().parents[2]
            / "pi" / "prompts" / "system.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "你是 ThermoForge 内置研发 Agent，使用提供的工具完成 HVAC 建模研究任务。"
