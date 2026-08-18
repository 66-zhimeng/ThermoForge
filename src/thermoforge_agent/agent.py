"""HarnessAgent 对话循环：多轮上下文 + function calling + 审批拦截 + 会话留痕。

- 模型请求工具 → 本地执行（TOOL_REGISTRY 薄封装）→ 信封 JSON 原样回传
  （32KB 截断约定天然适配上下文窗口）。
- human-only 工具（默认 tf_preprocess_approve）不在工具清单内；模型经
  `tf_human_approval` 发起审批 → `approval_handler` 弹确认 → 用户同意后
  以 actor=human 的 ToolContext 执行（I-49）。
- 每轮 user/assistant/tool 事件写 `research/agent_sessions/<ts>.jsonl`，
  供谱系追溯；assistant 事件另记思维链（reasoning，端点返回时）、
  工具调用参数（arguments）与 token 用量/估算费用（usage/cost），
  可还原完整探索路径。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from thermoforge_research.tools import TOOL_REGISTRY, ToolContext

from . import prompts
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

TOOL_LIMIT_FINALIZE_PROMPT = """本轮工具调用预算已经用完。不要再调用任何工具。
请只基于上面已经返回的工具结果，直接给用户一个诚实、可执行的中文答复：
说明已经完成什么、还缺什么；如果任务尚未完成，明确建议用户继续追问以从当前上下文续做。
不要声称未执行的动作已经完成。"""


class HarnessAgent:
    """内置研发 Agent。

    ::

        agent = HarnessAgent(config, ctx)
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
        # 每次模型调用的用量/思维链（与会话 JSONL 同源；界面轮询用）
        self.usage_log: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    # ---------------------------------------------------------------- 对话

    def ask(self, text: str) -> str:
        """一轮用户提问；内部按需要执行多轮工具调用，返回最终回答。"""
        self._repair_tool_history()
        self._append({"role": "user", "content": text})
        for _ in range(self.config.max_tool_rounds):
            result = self.client.chat(
                self.messages, self.schemas,
                on_content_delta=self.on_content_delta,
            )
            if not isinstance(result, ChatResult):
                raise TypeError("client.chat 必须返回 ChatResult")
            self._record_usage(result, kind="chat")
            self._append(_assistant_message(result),
                         reasoning=result.reasoning,
                         usage=result.usage, cost=result.cost)
            if not result.tool_calls:
                return result.content or ""
            for call in result.tool_calls:
                try:
                    envelope = self._execute(call)
                except Exception as exc:
                    # Last-resort protocol guard: logging, approval callbacks,
                    # and UI observers are also outside the tool's control.
                    envelope = self._error_envelope(
                        call.name, f"{type(exc).__name__}: {exc}")
                self._append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _serialize_tool_result(call, envelope),
                })
        return self._finalize_after_tool_limit()

    def _finalize_after_tool_limit(self) -> str:
        """工具预算耗尽后关闭工具，再给模型一次总结机会。

        `max_tool_rounds` 是防失控安全阀，不应把已经成功执行的整轮工作变成
        UI 错误。最后一次 completion 不暴露工具，只允许基于现有结果收尾。
        少数兼容端点即使没有 tools 仍可能返回 tool_calls；此时也要逐个补上
        LIMIT_REACHED 结果，保持 Chat Completions 历史合法，再返回确定性提示。
        """
        limit = self.config.max_tool_rounds
        self._log({"type": "tool_round_limit", "limit": limit})
        final_messages = [
            *self.messages,
            {"role": "system", "content": TOOL_LIMIT_FINALIZE_PROMPT},
        ]
        try:
            result = self.client.chat(
                final_messages, None,
                on_content_delta=self.on_content_delta,
            )
            if not isinstance(result, ChatResult):
                raise TypeError("client.chat 必须返回 ChatResult")
            self._record_usage(result, kind="finalize")
        except Exception as exc:
            self._log({
                "type": "tool_limit_finalize_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            return self._append_tool_limit_fallback()

        if not result.tool_calls:
            answer = (result.content or "").strip()
            if answer:
                self._append(_assistant_message(result),
                             reasoning=result.reasoning,
                             usage=result.usage, cost=result.cost)
                return answer
            return self._append_tool_limit_fallback()

        self._append(_assistant_message(result), reasoning=result.reasoning,
                     usage=result.usage, cost=result.cost)
        for call in result.tool_calls:
            envelope = self._error_envelope(
                call.name,
                f"本轮已达到工具调用上限 {limit}，该调用未执行；"
                "请在下一轮对话中继续。",
            )
            envelope["status"] = "LIMIT_REACHED"
            self._append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": _serialize_tool_result(call, envelope),
            })
        return self._append_tool_limit_fallback()

    def _append_tool_limit_fallback(self) -> str:
        answer = (
            f"本轮已完成 {self.config.max_tool_rounds} 轮工具调用，"
            "但任务还需要更多步骤。已完成的结果和上下文都已保留；"
            "请继续追问，我会从当前进度续做。"
        )
        self._append({"role": "assistant", "content": answer})
        return answer

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
                if not isinstance(envelope, Mapping):
                    raise TypeError(
                        f"工具返回值必须是 Mapping，实际为 "
                        f"{type(envelope).__name__}")
                envelope = dict(envelope)
            except Exception as exc:  # 所有工具异常都必须闭合 tool_call_id
                envelope = self._error_envelope(
                    call.name, f"{type(exc).__name__}: {exc}")
        self._log({"type": "tool_result", "tool": call.name,
                   "ok": envelope.get("ok"), "id": envelope.get("id"),
                   "status": envelope.get("status")})
        self._notify_tool_call(call.name, bool(envelope.get("ok")))
        return envelope

    def _execute_approval(self, call: ToolCallRequest) -> dict[str, Any]:
        """审批元工具：弹确认 → 同意后以 actor=human 执行目标工具。"""
        target = str(call.arguments.get("tool") or "")
        arguments = call.arguments.get("arguments") or {}
        reason = str(call.arguments.get("reason") or "")
        fn = TOOL_REGISTRY.get(target)
        approved = False
        if fn is not None and self.approval_handler is not None:
            try:
                approved = bool(
                    self.approval_handler(target, arguments, reason))
            except Exception as exc:
                return self._error_envelope(
                    target, f"{type(exc).__name__}: {exc}")
        self._log({"type": "approval", "tool": target,
                   "arguments": dict(arguments), "reason": reason,
                   "approved": approved})
        if fn is None:
            return self._error_envelope(target, f"未知工具: {target}")
        if not approved:
            self._notify_tool_call(target, False)
            return {**DECLINED_ENVELOPE, "tool": target,
                    "inputs": dict(arguments)}
        try:
            envelope = fn(self._human_ctx, **arguments)
            if not isinstance(envelope, Mapping):
                raise TypeError(
                    f"工具返回值必须是 Mapping，实际为 "
                    f"{type(envelope).__name__}")
            envelope = dict(envelope)
        except Exception as exc:
            envelope = self._error_envelope(
                target, f"{type(exc).__name__}: {exc}")
        self._notify_tool_call(target, bool(envelope.get("ok")))
        return envelope

    def _notify_tool_call(self, tool: str, ok: bool) -> None:
        """UI observers must never break the assistant/tool message protocol."""
        if self.on_tool_call is None:
            return
        try:
            self.on_tool_call(tool, ok)
        except Exception as exc:
            self._log({
                "type": "observer_error",
                "tool": tool,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def _repair_tool_history(self) -> None:
        """Close orphaned tool calls left by an interrupted previous turn.

        Chat Completions requires every assistant ``tool_calls`` item to be
        followed immediately by one matching tool message.  Repair happens
        before appending the next user message so an already-failed Copilot
        session can recover without forcing the user to reset the chat.
        """
        repaired: list[dict[str, Any]] = []
        inserted: list[str] = []
        dropped: list[str] = []
        index = 0
        while index < len(self.messages):
            message = self.messages[index]
            if message.get("role") == "tool":
                dropped.append(str(message.get("tool_call_id") or ""))
                index += 1
                continue

            normalized = dict(message)
            calls = (normalized.get("tool_calls") or []) \
                if normalized.get("role") == "assistant" else []
            if not calls:
                repaired.append(normalized)
                index += 1
                continue

            expected: list[tuple[str, str]] = []
            normalized_calls = []
            for call_index, raw_call in enumerate(calls):
                call = dict(raw_call)
                call_id = str(call.get("id") or
                              f"recovered-{index}-{call_index}")
                call["id"] = call_id
                function = call.get("function") or {}
                expected.append((call_id, str(function.get("name") or "")))
                normalized_calls.append(call)
            normalized["tool_calls"] = normalized_calls
            repaired.append(normalized)
            index += 1

            expected_ids = {call_id for call_id, _ in expected}
            seen: set[str] = set()
            while (index < len(self.messages)
                   and self.messages[index].get("role") == "tool"):
                tool_message = dict(self.messages[index])
                call_id = str(tool_message.get("tool_call_id") or "")
                if call_id in expected_ids and call_id not in seen:
                    repaired.append(tool_message)
                    seen.add(call_id)
                else:
                    dropped.append(call_id)
                index += 1

            names = dict(expected)
            for call_id, _ in expected:
                if call_id in seen:
                    continue
                envelope = self._error_envelope(
                    names[call_id],
                    "工具调用在上次执行中中断，副驾已自动补全失败结果；"
                    "请基于当前状态继续。",
                )
                repaired.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(envelope, ensure_ascii=False),
                })
                inserted.append(call_id)

        if inserted or dropped:
            self.messages = repaired
            self._log({
                "type": "history_repaired",
                "inserted_tool_call_ids": inserted,
                "dropped_tool_call_ids": dropped,
            })

    @staticmethod
    def _error_envelope(tool: str, message: str) -> dict[str, Any]:
        return {
            "ok": False, "tool": tool, "id": None, "status": "FAILED",
            "inputs": {}, "summary": {"error": message},
            "diagnostics": [], "artifacts": [], "truncated": False,
        }

    # ---------------------------------------------------------------- 留痕

    def _record_usage(self, result: ChatResult, *, kind: str) -> None:
        """每次模型调用记一条（含思维链）；端点不返回用量时 usage 为 None，
        但条目照记——调用次数本身就是界面要展示的信息。"""
        self.usage_log.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "usage": result.usage,
            "cost": result.cost,
            "reasoning": result.reasoning,
        })

    def usage_snapshot(self) -> list[dict[str, Any]]:
        """逐次模型调用的用量/思维链副本（界面轮询用，后台线程写这里读）。"""
        return list(self.usage_log)

    def _append(self, message: dict[str, Any], *,
                reasoning: str | None = None,
                usage: Mapping[str, Any] | None = None,
                cost: float | None = None) -> None:
        event = {"type": "message", "role": message.get("role"),
                 "content": _summarize_content(message)}
        if reasoning:
            event["reasoning"] = reasoning
        if usage:
            event["usage"] = dict(usage)
        if cost is not None:
            event["cost"] = cost
        if message.get("tool_calls"):
            event["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": _summarize_arguments(
                        (call.get("function") or {}).get("arguments")),
                }
                for call in message["tool_calls"]
            ]
        if message.get("role") == "tool":
            event["tool_call_id"] = message.get("tool_call_id")
        self._log(event)
        self.messages.append(message)

    def _log(self, event: Mapping[str, Any]) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(), **event}
        with open(self._session_path, "a", encoding="utf-8",
                  newline="\n") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _assistant_message(result: ChatResult) -> dict[str, Any]:
    """Build one protocol-complete assistant message from a normalized result."""
    message = dict(result.raw_message or {})
    message["role"] = "assistant"
    message["content"] = result.content
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments, ensure_ascii=False, default=str),
                },
            }
            for call in result.tool_calls
        ]
    else:
        message.pop("tool_calls", None)
    return message


def _serialize_tool_result(
    call: ToolCallRequest,
    envelope: Mapping[str, Any],
) -> str:
    """Always produce a tool content string, even for a broken tool envelope."""
    try:
        return json.dumps(envelope, ensure_ascii=False)
    except Exception as exc:
        fallback = HarnessAgent._error_envelope(
            call.name, f"工具结果序列化失败: {type(exc).__name__}: {exc}")
        return json.dumps(fallback, ensure_ascii=False)


def _summarize_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, str) and len(content) > 2000:
        return content[:2000] + "…(截断)"
    return content


def _summarize_arguments(arguments: Any) -> Any:
    """工具调用参数留痕：保留模型产出的原始 JSON 字符串，超长截断。"""
    if isinstance(arguments, str) and len(arguments) > 2000:
        return arguments[:2000] + "…(截断)"
    return arguments


def _default_system_prompt() -> str:
    """CLI 位点的提示词：走 prompts.load 才会带上绑定的技能。

    以前这里直接读 system.md 原文，绕过了 prompts 层——结果
    `harness/skills/*.md` 写得再细也到不了模型手上（技能只在测试里被拼过）。
    """
    return prompts.load(
        "cli",
        fallback="你是 ThermoForge 内置研发 Agent，使用提供的工具完成 "
                 "HVAC 建模研究任务。",
    )
