"""Agent 接口配置与分层网络自检。

从旧控制台迁移，逻辑口径不变：

- 环境变量优先于 `pi/agent.toml`；密钥读回一律掩码，完整值不出界面。
- 写文件走临时文件 + `os.replace`，与项目其它写路径同调。
- 自检逐层做 DNS → TCP → TLS → HTTP：「连不上」的成因差别极大，
  笼统一句话没法排查，逐层测能直接指出断点。
"""

from __future__ import annotations

import os
import socket
import ssl
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .context import CONFIG_PATH

# 服务商目录：只是把常见端点的地址填对，**不构成限制**——base_url 与模型
# 名始终可以手填，任何 OpenAI 兼容的 chat completions + function calling
# 端点都能用（含 vLLM / one-api / new-api 这类自建网关）。模型名随服务商
# 更新，这里只作候选提示，以各家文档为准。
PROVIDERS: list[dict[str, Any]] = [
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com",
     "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
     "note": "OpenAI 兼容格式，接口地址不要加 /v1。"},
    {"id": "openrouter", "name": "OpenRouter（通用路由）",
     "base_url": "https://openrouter.ai/api/v1",
     "models": ["deepseek/deepseek-chat", "anthropic/claude-sonnet-4",
                "openai/gpt-4o", "google/gemini-2.0-flash-001"],
     "note": "一个 key 转发到几百个模型，模型名写成 服务商/模型 的形式。"},
    {"id": "moonshot", "name": "Moonshot / Kimi",
     "base_url": "https://api.moonshot.cn/v1",
     "models": ["kimi-k2-0905-preview", "moonshot-v1-128k"], "note": ""},
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1",
     "models": ["gpt-4o", "gpt-4o-mini"], "note": ""},
    {"id": "dashscope", "name": "阿里云百炼 / 通义千问",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
     "note": "用「兼容模式」地址，不是原生 DashScope 地址。"},
    {"id": "zhipu", "name": "智谱 GLM",
     "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "models": ["glm-4-plus", "glm-4-flash"], "note": ""},
    {"id": "siliconflow", "name": "硅基流动 SiliconFlow",
     "base_url": "https://api.siliconflow.cn/v1",
     "models": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"],
     "note": ""},
    {"id": "volcengine", "name": "火山引擎 / 豆包",
     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
     "models": ["doubao-pro-32k"],
     "note": "模型名填推理接入点 ID（ep-... 开头）也可以。"},
    {"id": "ollama", "name": "Ollama（本机模型）",
     "base_url": "http://localhost:11434/v1",
     "models": ["qwen2.5", "llama3.1"],
     "note": "本机跑的模型，API 密钥随便填一个非空值即可。"},
    {"id": "custom", "name": "自定义 / 自建网关", "base_url": "", "models": [],
     "note": "任何 OpenAI 兼容端点：one-api、new-api、vLLM、LM Studio…"},
]

_TCP_TIMEOUT = 15.0
_TLS_TIMEOUT = 20.0
_SLOW_STEP_MS = 4000  # 超过这个耗时就提示「首次连接慢，偶发超时源于此」


@dataclass(frozen=True)
class ConfigView:
    """当前生效配置（密钥已掩码，完整值不出现在这里）。"""

    configured: bool
    api_key_masked: str
    from_env: bool
    base_url: str
    model: str
    proxy: str
    config_path: str


def mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 10:
        return secret[:2] + "…"
    return f"{secret[:6]}…{secret[-4:]}"


def _load_doc() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "rb") as fp:
        return tomllib.load(fp)


def read_config() -> ConfigView:
    """当前生效配置。环境变量优先于配置文件。"""
    from thermoforge_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL

    doc = _load_doc()
    env_key = os.environ.get("TF_AGENT_API_KEY")
    key = env_key or str(doc.get("api_key") or "")
    return ConfigView(
        configured=bool(key),
        api_key_masked=mask(key),
        from_env=bool(env_key),
        base_url=(os.environ.get("TF_AGENT_BASE_URL")
                  or doc.get("base_url") or DEFAULT_BASE_URL),
        model=(os.environ.get("TF_AGENT_MODEL")
               or doc.get("model") or DEFAULT_MODEL),
        proxy=str(os.environ.get("TF_AGENT_PROXY") or doc.get("proxy") or ""),
        config_path=str(CONFIG_PATH),
    )


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_config(api_key: str, base_url: str, model: str,
                 proxy: str = "") -> None:
    """写 `pi/agent.toml`（gitignore 内）。空密钥表示保留原值。"""
    current = _load_doc()
    key = api_key or str(current.get("api_key") or "")
    if not key:
        raise ValueError("API 密钥不能为空")
    if not base_url:
        raise ValueError("接口地址不能为空")
    if not model:
        raise ValueError("模型名不能为空")
    lines = [
        "# ThermoForge Agent 配置（本文件已 gitignore，密钥不会入库）",
        f'api_key = "{_toml_escape(key)}"',
        f'base_url = "{_toml_escape(base_url)}"',
        f'model = "{_toml_escape(model)}"',
    ]
    if proxy:
        lines.append(f'proxy = "{_toml_escape(proxy)}"')
    lines.append("")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = CONFIG_PATH.with_suffix(".toml.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(tmp, CONFIG_PATH)


def agent_config():
    """`AgentConfig` 或 None（未配置密钥）。"""
    from thermoforge_agent import AgentConfig

    return AgentConfig.load(config_path=CONFIG_PATH)


def explain_exception(exc: Exception, config: Any) -> str:
    """把 SDK 异常翻译成能照着做的话。

    「APIConnectionError: Connection error.」对使用者没有信息量——它到底
    是网络不通、key 不对，还是模型名写错，处理方式完全不同。
    """
    name = type(exc).__name__
    text = str(exc)
    if "Connection" in name or "Timeout" in name:
        return (f"连不上 {config.base_url}。这不是密钥的问题——请求根本没"
                f"到达服务器。检查：网络能否访问该域名、是否需要代理、"
                f"接口地址有没有写错（多写或少写 /v1 都会出问题）。")
    if "Authentication" in name or "401" in text:
        return "密钥被拒绝（401）。检查 key 是否复制完整、是否属于这个服务商。"
    if "NotFound" in name or "404" in text:
        hint = ""
        if "deepseek" in str(config.base_url).lower():
            hint = ("　DeepSeek 现在的模型名是 deepseek-v4-pro 或 "
                    "deepseek-v4-flash（deepseek-chat 是旧名字），"
                    "接口地址是 https://api.deepseek.com，不要加 /v1。")
        return (f"接口或模型不存在（404）。当前模型名 “{config.model}”，"
                f"接口地址 “{config.base_url}”。{hint}")
    if "PermissionDenied" in name or "403" in text:
        return "密钥没有访问该模型的权限（403）。"
    if "RateLimit" in name or "429" in text:
        return "被限流或余额不足（429）。"
    return f"{name}: {text[:300]}"


def check_endpoint() -> dict[str, Any]:
    """最小一次真实调用，确认密钥 + 模型名可用。"""
    from thermoforge_agent.client import ChatClient

    config = agent_config()
    if config is None:
        return {"ok": False, "error": "还没有配置 API 密钥"}
    try:
        return ChatClient(config).check()
    except Exception as exc:  # SDK 异常种类多，统一翻译成人话
        return {"ok": False, "error": explain_exception(exc, config),
                "detail": f"{type(exc).__name__}: {exc}"[:300]}


def _timed(steps: list[dict[str, Any]], name: str,
           fn: Callable[[], str]) -> bool:
    start = time.monotonic()
    try:
        detail = fn()
        ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"[:200]
        ok = False
    steps.append({"name": name, "ok": ok,
                  "ms": round((time.monotonic() - start) * 1000),
                  "detail": detail})
    return ok


def diagnose_network() -> dict[str, Any]:
    """分层自检：DNS → TCP → TLS → HTTP，断在哪层就说哪层的处理办法。"""
    config = agent_config()
    base_url = config.base_url if config else read_config().base_url
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    steps: list[dict[str, Any]] = []

    if not host:
        return {"ok": False, "host": base_url, "port": port,
                "steps": [{"name": "解析地址", "ok": False, "ms": 0,
                           "detail": f"接口地址填得不对：{base_url!r}"}],
                "verdict": "接口地址不是一个合法 URL，先去上面把它填对。"}

    def _dns() -> str:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        return "、".join(sorted({i[4][0] for i in infos})[:3])

    if not _timed(steps, f"DNS 解析 {host}", _dns):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "域名解析不了。检查网络连接或 DNS 设置；"
                           "如果接口地址是手填的，确认没有拼错。"}

    def _tcp() -> str:
        with socket.create_connection((host, port), timeout=_TCP_TIMEOUT) as sock:
            return f"已连到 {sock.getpeername()[0]}:{port}"

    if not _timed(steps, f"TCP 连接 {port} 端口", _tcp):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "域名能解析但连不上端口，通常是防火墙或需要代理。"
                           "如果你在公司网络里，去上面填代理地址。"}

    def _tls() -> str:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=_TLS_TIMEOUT) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                return f"{tls.version()}（证书签发给 {host}）"

    if parsed.scheme == "https" and not _timed(steps, "TLS 握手", _tls):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "TCP 通了但 TLS 握手失败，多半是中间设备（企业"
                           "网关、杀毒软件）在拦截或替换证书。"}

    result = check_endpoint()
    steps.append({"name": "调用模型接口", "ok": bool(result.get("ok")), "ms": None,
                  "detail": (f"模型 {result.get('model')} 响应正常"
                             if result.get("ok")
                             else str(result.get("error"))[:200])})
    if result.get("ok"):
        verdict = "全部通过，可以去「研究」页让 AI 开始跑了。"
        slow = [s for s in steps if s.get("ms") and s["ms"] > _SLOW_STEP_MS]
        if slow:
            verdict += (f"　注意：{slow[0]['name']} 用了 {slow[0]['ms']}ms，"
                        "首次连接偏慢属正常，但偶发超时也源于此。")
    else:
        verdict = ("网络这一层没问题（DNS/TCP/TLS 都通），问题出在接口本身"
                   "——看上面最后一行的说明，通常是密钥或模型名。")
    return {"ok": bool(result.get("ok")), "host": host, "port": port,
            "steps": steps, "verdict": verdict}
