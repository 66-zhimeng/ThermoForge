"""OpenAI 兼容 chat completions 客户端（function calling + 可选流式）。

`ChatClient.chat()` 返回协议无关的 `ChatResult`，对话循环（agent.py）
不依赖 SDK 类型——测试用 stub client 注入，不触网。

推理模型返回的思维链（DeepSeek `reasoning_content`、部分端点的
`reasoning`）归一化进 `ChatResult.reasoning`，仅供会话留痕，
不回显进后续请求上下文（多数端点不接受该字段回传）。
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
    reasoning: str | None = None  # 思维链：仅留痕，不进 raw_message
    usage: dict[str, Any] | None = None  # 端点返回的 token 用量（原样字段）
    cost: float | None = None  # 按 [pricing] 单价估算的费用（元）；未配置为 None


class ChatClient:
    """openai SDK 薄封装（OpenAI 兼容端点：Moonshot/Kimi/OpenAI/DeepSeek…）。"""

    def __init__(self, config: AgentConfig):
        import httpx
        from openai import OpenAI

        self.config = config
        # SDK 默认 connect=5s。实测首次 TLS 握手可能就要 5s（冷启动），
        # 正好卡在边界上，表现为「时通时不通」——单独放宽连接阶段。
        timeout = httpx.Timeout(
            config.timeout_seconds,
            connect=max(20.0, min(config.timeout_seconds, 30.0)),
        )
        kwargs: dict[str, Any] = {}
        if config.proxy:  # 受限网络：显式代理优先于 httpx 的环境变量行为
            kwargs["http_client"] = httpx.Client(proxy=config.proxy,
                                                 timeout=timeout)
        elif _is_local_endpoint(config.base_url):
            # 本地端点（Ollama/vLLM/测试用 mock）必须直连：Windows 上
            # httpx 走 urllib.getproxies()，会读到注册表里的系统代理，而
            # 注册表的 ProxyOverride（localhost;127.*）httpx 并不认，于是
            # 本机请求被塞给代理，代理拒绝转发到 127.0.0.1 → 502。
            kwargs["http_client"] = httpx.Client(timeout=timeout,
                                                 trust_env=False)
        self._client = OpenAI(api_key=config.api_key,
                              base_url=config.base_url,
                              timeout=timeout,
                              max_retries=config.max_retries,
                              **kwargs)

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
        usage = _usage_dict(getattr(response, "usage", None))
        return _from_sdk_message(message, usage=usage,
                                 cost=_usage_cost(usage, self.config))

    def _chat_stream(self, kwargs: dict[str, Any],
                     on_content_delta: Callable[[str], None]) -> ChatResult:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        # include_usage：用量只在最后一个空 choices 的 chunk 里返回
        stream = self._client.chat.completions.create(
            stream=True, stream_options={"include_usage": True}, **kwargs)
        for chunk in stream:
            if not chunk.choices:
                chunk_usage = _usage_dict(getattr(chunk, "usage", None))
                if chunk_usage:
                    usage = chunk_usage
                continue
            delta = chunk.choices[0].delta
            reasoning_piece = _extract_reasoning(delta)
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
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
                          raw_message=raw,
                          reasoning="".join(reasoning_parts) or None,
                          usage=usage,
                          cost=_usage_cost(usage, self.config))

    def check(self, timeout: float = 25.0) -> dict[str, Any]:
        """最小连通性验证（`tf agent --check`）。

        排障用途，超时比正常对话更短——等 60 秒才知道「端点不通」对
        使用者没有价值。连接类失败自动重试一次：首次 TLS 握手慢是常态，
        第一次点「测试」就报连不上会让人以为配错了。鉴权/模型名一类的
        错误立即抛出，不做无谓重试。
        """
        import httpx

        client = self._client.with_options(
            timeout=httpx.Timeout(timeout, connect=timeout), max_retries=0)
        last: Exception | None = None
        for _attempt in (1, 2):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                break
            except Exception as exc:
                if type(exc).__name__ not in ("APIConnectionError",
                                              "APITimeoutError"):
                    raise
                last = exc
        else:
            raise last  # type: ignore[misc]
        return {
            "ok": True,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "response_id": getattr(response, "id", None),
        }


def _is_local_endpoint(base_url: str) -> bool:
    """base_url 是否指向本机（回环地址或 localhost）。"""
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(base_url).hostname or "").strip("[]")
    if not host:
        return False
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _make_call(call_id: str, name: str, arguments: str) -> ToolCallRequest:
    try:
        args = json.loads(arguments) if arguments else {}
    except ValueError:
        args = {"_raw": arguments}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCallRequest(id=call_id, name=name, arguments=args)


def _extract_reasoning(obj: Any) -> str | None:
    """从 SDK message/delta 上取思维链字段（DeepSeek `reasoning_content`，
    部分端点用 `reasoning`）；SDK 把未知字段挂在 extra 上，getattr 即可。"""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    """SDK 的 CompletionUsage → 普通 dict（保留 cache hit 等扩展字段）。"""
    if usage is None:
        return None
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return {k: v for k, v in model_dump(exclude_none=True).items()
                if v is not None}
    if isinstance(usage, Mapping):
        return dict(usage)
    return None


def _usage_cost(usage: Mapping[str, Any] | None, config: Any) -> float | None:
    """按 [pricing] 单价把 usage 折算成元；未配置单价返回 None（界面只显示
    token）。有缓存命中字段时分开计价，命中单价缺省按输入原价算。"""
    if not usage:
        return None
    in_price = float(getattr(config, "price_input_per_mtok", 0.0) or 0.0)
    out_price = float(getattr(config, "price_output_per_mtok", 0.0) or 0.0)
    if not in_price and not out_price:
        return None
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        input_tokens = usage.get("prompt_tokens") or 0
        input_cost = input_tokens * in_price
    else:
        hit_price = float(
            getattr(config, "price_input_cache_hit_per_mtok", 0.0)
            or 0.0) or in_price
        input_cost = (hit or 0) * hit_price + (miss or 0) * in_price
    completion_tokens = usage.get("completion_tokens") or 0
    return (input_cost + completion_tokens * out_price) / 1e6


def _from_sdk_message(message: Any, *, usage: dict[str, Any] | None = None,
                      cost: float | None = None) -> ChatResult:
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
                      raw_message=raw, reasoning=_extract_reasoning(message),
                      usage=usage, cost=cost)
