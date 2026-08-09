"""OpenAI 兼容 chat completions 客户端（function calling + 可选流式）。

`ChatClient.chat()` 返回协议无关的 `ChatResult`，对话循环（agent.py）
不依赖 SDK 类型——测试用 stub client 注入，不触网。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .config import AgentConfig


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    """一次 chat completion 的归一化结果。"""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)  # 回显进上下文


class ChatClient:
    """openai SDK 薄封装（OpenAI 兼容端点：Moonshot/Kimi/OpenAI/DeepSeek…）。"""

    def __init__(self, config: AgentConfig):
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(api_key=config.api_key,
                              base_url=config.base_url)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        on_content_delta: Callable[[str], None] | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if self.config.stream and on_content_delta is not None:
            return self._chat_stream(kwargs, on_content_delta)
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return _from_sdk_message(message)

    def _chat_stream(self, kwargs: dict[str, Any],
                     on_content_delta: Callable[[str], None]) -> ChatResult:
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        stream = self._client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                on_content_delta(delta.content)
            for tc in delta.tool_calls or []:
                acc = tool_acc.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments
        content = "".join(content_parts) or None
        raw: dict[str, Any] = {"role": "assistant", "content": content}
        calls: list[ToolCallRequest] = []
        raw_calls = []
        for index in sorted(tool_acc):
            acc = tool_acc[index]
            calls.append(_make_call(acc["id"], acc["name"], acc["arguments"]))
            raw_calls.append({
                "id": acc["id"], "type": "function",
                "function": {"name": acc["name"],
                             "arguments": acc["arguments"]},
            })
        if raw_calls:
            raw["tool_calls"] = raw_calls
        return ChatResult(content=content, tool_calls=calls,
                          raw_message=raw)

    def check(self) -> dict[str, Any]:
        """最小连通性验证（`tf agent --check`）。"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return {
            "ok": True,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "response_id": getattr(response, "id", None),
        }


def _make_call(call_id: str, name: str, arguments: str) -> ToolCallRequest:
    try:
        args = json.loads(arguments) if arguments else {}
    except ValueError:
        args = {"_raw": arguments}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCallRequest(id=call_id, name=name, arguments=args)


def _from_sdk_message(message: Any) -> ChatResult:
    calls: list[ToolCallRequest] = []
    raw_calls = []
    for tc in message.tool_calls or []:
        calls.append(_make_call(tc.id, tc.function.name,
                                tc.function.arguments or ""))
        raw_calls.append({
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name,
                         "arguments": tc.function.arguments or ""},
        })
    raw: dict[str, Any] = {"role": "assistant",
                           "content": message.content}
    if raw_calls:
        raw["tool_calls"] = raw_calls
    return ChatResult(content=message.content, tool_calls=calls,
                      raw_message=raw)
