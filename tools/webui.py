"""本地 Web 控制台：配置密钥 + 和 Pi Agent 对话 + 看状态。

面向不想用命令行的使用者。只用标准库 `http.server`，不引入任何 Web
框架依赖——项目依赖一直很克制，一个本地控制台不值得为它加一层。

启动::

    .venv/Scripts/python tools/webui.py        # 然后打开 http://127.0.0.1:8765

安全边界（本地工具，但涉及 API 密钥，仍按最小暴露处理）：

- **只绑 127.0.0.1**，不监听 0.0.0.0——否则同局域网的任何人都能打开
  这个能改密钥、能跑实验的界面。
- 密钥写入 `pi/agent.toml`（已 gitignore），读回时一律掩码，
  完整值不出现在任何响应里。
- 跨站请求：拒绝带外部 `Origin` 的写操作（浏览器里的其他页面无法
  借你的浏览器操作本机服务）。

Agent 的 human-only 工具（如预处理审批）在 Web 侧不自动放行：
`approval_handler` 一律返回 False 并把请求原样回传前端，由使用者
在界面上确认后以 `actor=human` 单独执行（保持 I-49 的留痕语义）。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CONFIG_PATH = REPO_ROOT / "pi" / "agent.toml"
MAX_BODY = 1 << 20  # 1MB，足够长对话；防止意外的大 body

# 服务商目录：只是把常见端点的地址填对，**不构成限制**——base_url 与
# 模型名始终可以手填，任何 OpenAI 兼容的 chat completions + function
# calling 端点都能用（含自建 vLLM / one-api / new-api 这类聚合网关）。
# 模型名随服务商更新，这里只作候选提示，以各家文档为准。
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
     "note": "本机跑的模型，API Key 随便填一个非空值即可。"},
    {"id": "custom", "name": "自定义 / 自建网关", "base_url": "", "models": [],
     "note": "任何 OpenAI 兼容端点：one-api、new-api、vLLM、LM Studio…"},
]


# ---------------------------------------------------------------- 状态


class Console:
    """进程内共享状态：Agent 会话与待审批请求。"""

    def __init__(self) -> None:
        self.agent: Any | None = None
        self.pending: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def ctx(self):
        from thermoforge_research.tools import ToolContext

        return ToolContext(
            vault_root=REPO_ROOT / "vault",
            research_root=REPO_ROOT / "research",
            models_root=REPO_ROOT / "models",
            actor="webui",
        )

    def config(self):
        from thermoforge_agent import AgentConfig

        return AgentConfig.load(config_path=CONFIG_PATH)

    def reset(self) -> None:
        with self.lock:
            self.agent = None
            self.pending.clear()

    def ensure_agent(self, calls: list[dict[str, Any]]):
        """取得（或新建）Agent 会话，并把本轮工具调用记进 calls。"""
        from thermoforge_agent import PiAgent

        config = self.config()
        if config is None:
            return None
        if self.agent is None:
            self.agent = PiAgent(
                config, self.ctx(),
                approval_handler=self._on_approval,
                on_tool_call=lambda tool, ok: calls.append(
                    {"tool": tool, "ok": bool(ok)}),
            )
        else:  # 复用会话上下文，只把观察器指到本轮的 calls
            self.agent.on_tool_call = lambda tool, ok: calls.append(
                {"tool": tool, "ok": bool(ok)})
        return self.agent

    def _on_approval(self, tool: str, arguments: dict[str, Any],
                     reason: str) -> bool:
        """Web 侧不自动放行：记下请求，让使用者显式确认。"""
        self.pending.append({"tool": tool, "arguments": arguments,
                             "reason": reason})
        return False


CONSOLE = Console()


# ---------------------------------------------------------------- 配置读写


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 10:
        return secret[:2] + "…"
    return f"{secret[:6]}…{secret[-4:]}"


def read_config() -> dict[str, Any]:
    """当前生效配置（密钥掩码）。环境变量优先于配置文件。"""
    import tomllib

    from thermoforge_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL

    doc: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as fp:
            doc = tomllib.load(fp)
    env_key = os.environ.get("TF_AGENT_API_KEY")
    key = env_key or str(doc.get("api_key") or "")
    return {
        "configured": bool(key),
        "api_key_masked": mask(key),
        "from_env": bool(env_key),
        "base_url": (os.environ.get("TF_AGENT_BASE_URL")
                     or doc.get("base_url") or DEFAULT_BASE_URL),
        "model": (os.environ.get("TF_AGENT_MODEL")
                  or doc.get("model") or DEFAULT_MODEL),
        "proxy": str(os.environ.get("TF_AGENT_PROXY")
                     or doc.get("proxy") or ""),
        "config_path": str(CONFIG_PATH),
        "providers": PROVIDERS,
    }


def write_config(api_key: str, base_url: str, model: str,
                 proxy: str = "") -> None:
    """写 pi/agent.toml（gitignore 内）。空密钥表示保留原值。"""
    import tomllib

    current: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as fp:
            current = tomllib.load(fp)
    key = api_key or str(current.get("api_key") or "")
    if not key:
        raise ValueError("API key 不能为空")
    if not base_url:
        raise ValueError("接口地址不能为空")
    if not model:
        raise ValueError("模型名不能为空")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ThermoForge Agent 配置（本文件已 gitignore，密钥不会入库）",
        f'api_key = "{_toml_escape(key)}"',
        f'base_url = "{_toml_escape(base_url)}"',
        f'model = "{_toml_escape(model)}"',
    ]
    if proxy:
        lines.append(f'proxy = "{_toml_escape(proxy)}"')
    lines.append("")
    tmp = CONFIG_PATH.with_suffix(".toml.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(tmp, CONFIG_PATH)
    CONSOLE.reset()  # 配置变了，旧会话作废


# ---------------------------------------------------------------- API


def api_status() -> dict[str, Any]:
    from thermoforge_cli.status import build_status

    return build_status(CONSOLE.ctx(), recent=8)


def api_datasets() -> dict[str, Any]:
    ctx = CONSOLE.ctx()
    out = []
    for d in ctx.vault.list_datasets():
        ref = d["revisions"][-1]["ref"]
        variables = ctx.vault.load_variables(ref)
        out.append({
            "dataset_id": d["dataset_id"],
            "ref": ref,
            "variable_count": len(variables),
            "objects": sorted({v["object_id"] for v in variables}),
        })
    return {"datasets": out}


def diagnose(exc: Exception, config: Any) -> str:
    """把 SDK 异常翻译成能照着做的话。

    「APIConnectionError: Connection error.」对使用者没有信息量——它
    到底是网络不通、key 不对，还是模型名写错，处理方式完全不同。
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


def api_check() -> dict[str, Any]:
    from thermoforge_agent.client import ChatClient

    config = CONSOLE.config()
    if config is None:
        return {"ok": False, "error": "还没有配置 API key"}
    try:
        return ChatClient(config).check()
    except Exception as exc:
        return {"ok": False, "error": diagnose(exc, config),
                "detail": f"{type(exc).__name__}: {exc}"[:300]}


def api_diagnose() -> dict[str, Any]:
    """分层网络自检：DNS → TCP → TLS → HTTP。

    「连不上」有太多种原因，笼统一句话没法排查。逐层测能直接指出断点：
    DNS 失败是解析问题，TCP 失败多为防火墙，TLS 慢/失败常是中间设备，
    HTTP 通了则说明网络没问题、该看密钥和模型名。
    """
    import socket
    import ssl
    import time
    from urllib.parse import urlparse

    config = CONSOLE.config()
    base_url = (config.base_url if config
                else read_config()["base_url"])
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    steps: list[dict[str, Any]] = []

    def step(name: str, fn) -> bool:
        t0 = time.monotonic()
        try:
            detail = fn()
            steps.append({"name": name, "ok": True,
                          "ms": round((time.monotonic() - t0) * 1000),
                          "detail": detail})
            return True
        except Exception as exc:
            steps.append({"name": name, "ok": False,
                          "ms": round((time.monotonic() - t0) * 1000),
                          "detail": f"{type(exc).__name__}: {exc}"[:200]})
            return False

    if not host:
        return {"ok": False, "host": base_url,
                "steps": [{"name": "解析地址", "ok": False, "ms": 0,
                           "detail": f"接口地址填得不对：{base_url!r}"}]}

    addrs: list[str] = []

    def _dns() -> str:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        addrs.extend(sorted({i[4][0] for i in infos}))
        return "、".join(addrs[:3])

    if not step(f"DNS 解析 {host}", _dns):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "域名解析不了。检查网络连接或 DNS 设置；"
                           "如果接口地址是手填的，确认没有拼错。"}

    def _tcp() -> str:
        with socket.create_connection((host, port), timeout=15) as sock:
            return f"已连到 {sock.getpeername()[0]}:{port}"

    if not step(f"TCP 连接 {port} 端口", _tcp):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "域名能解析但连不上端口，通常是防火墙或需要代理。"
                           "如果你在公司网络里，去下面填代理地址。"}

    def _tls() -> str:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=20) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                return f"{tls.version()}（证书签发给 {host}）"

    if parsed.scheme == "https" and not step("TLS 握手", _tls):
        return {"ok": False, "host": host, "port": port, "steps": steps,
                "verdict": "TCP 通了但 TLS 握手失败，多半是中间设备（企业"
                           "网关、杀毒软件）在拦截或替换证书。"}

    result = api_check()
    steps.append({"name": "调用模型接口", "ok": bool(result.get("ok")),
                  "ms": None,
                  "detail": (f"模型 {result.get('model')} 响应正常"
                             if result.get("ok")
                             else str(result.get("error"))[:200])})
    if result.get("ok"):
        verdict = "全部通过，可以去「对话」页开始用了。"
    else:
        verdict = ("网络这一层没问题（DNS/TCP/TLS 都通），问题出在接口"
                   "本身——看上面最后一行的说明，通常是密钥或模型名。")
    slow = [s for s in steps if s.get("ms") and s["ms"] > 4000]
    if slow and result.get("ok"):
        verdict += f"　注意：{slow[0]['name']} 用了 {slow[0]['ms']}ms，" \
                   "首次连接偏慢属正常，但偶发超时也源于此。"
    return {"ok": bool(result.get("ok")), "host": host, "port": port,
            "steps": steps, "verdict": verdict}


def api_chat(message: str) -> dict[str, Any]:
    if not message.strip():
        return {"ok": False, "error": "消息不能为空"}
    calls: list[dict[str, Any]] = []
    with CONSOLE.lock:
        CONSOLE.pending.clear()
        agent = CONSOLE.ensure_agent(calls)
        if agent is None:
            return {"ok": False, "error": "还没有配置 API key",
                    "need_config": True}
        try:
            reply = agent.ask(message)
        except Exception as exc:
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        pending = list(CONSOLE.pending)
    return {"ok": True, "reply": reply, "tool_calls": calls,
            "pending_approvals": pending}


def api_approve(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """以 actor=human 执行一个待审批工具（留痕语义与 CLI 一致）。"""
    from thermoforge_research.tools import TOOL_REGISTRY, ToolContext

    fn = TOOL_REGISTRY.get(tool)
    if fn is None:
        return {"ok": False, "error": f"未知工具: {tool}"}
    human_ctx = ToolContext(
        vault_root=REPO_ROOT / "vault", research_root=REPO_ROOT / "research",
        models_root=REPO_ROOT / "models", actor="human")
    try:
        envelope = fn(human_ctx, **arguments)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with CONSOLE.lock:
        CONSOLE.pending = [p for p in CONSOLE.pending if p["tool"] != tool]
    return {"ok": True, "envelope": envelope}


# ---------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "ThermoForgeConsole/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默访问日志
        pass

    # ---- 工具

    def _send(self, payload: Any, status: int = 200,
              content_type: str = "application/json") -> None:
        if content_type.startswith("application/json"):
            body = json.dumps(payload, ensure_ascii=False,
                              default=str).encode("utf-8")
        else:
            body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        raw = self.rfile.read(length)
        try:
            doc = json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}
        return doc if isinstance(doc, dict) else {}

    def _same_origin(self) -> bool:
        """写操作拒绝跨站：浏览器里的别的页面不能借道操作本机服务。"""
        origin = self.headers.get("Origin")
        if not origin:
            return True  # 非浏览器发起（curl 等）不带 Origin
        allowed = {f"http://{HOST}:{self.server.server_address[1]}",
                   f"http://localhost:{self.server.server_address[1]}"}
        return origin in allowed

    # ---- 路由

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send(PAGE, content_type="text/html")
        elif route == "/api/config":
            self._send(read_config())
        elif route == "/api/status":
            self._send(api_status())
        elif route == "/api/datasets":
            self._send(api_datasets())
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if not self._same_origin():
            self._send({"error": "跨站请求已拒绝"}, status=403)
            return
        route = self.path.split("?", 1)[0]
        doc = self._body()
        try:
            if route == "/api/config":
                write_config(str(doc.get("api_key") or "").strip(),
                             str(doc.get("base_url") or "").strip(),
                             str(doc.get("model") or "").strip(),
                             str(doc.get("proxy") or "").strip())
                self._send({"ok": True, **read_config()})
            elif route == "/api/check":
                self._send(api_check())
            elif route == "/api/diagnose":
                self._send(api_diagnose())
            elif route == "/api/chat":
                self._send(api_chat(str(doc.get("message") or "")))
            elif route == "/api/approve":
                self._send(api_approve(str(doc.get("tool") or ""),
                                       dict(doc.get("arguments") or {})))
            elif route == "/api/reset":
                CONSOLE.reset()
                self._send({"ok": True})
            else:
                self._send({"error": "not found"}, status=404)
        except Exception as exc:
            self._send({"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}, status=500)


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThermoForge 控制台</title>
<style>
:root{
  --bg:#f4f6f8; --panel:#ffffff; --sunk:#eef1f5; --line:#dfe3e9;
  --fg:#12161c; --muted:#66707d; --accent:#2f6feb; --accent-fg:#ffffff;
  --ok:#0b7a48; --ok-bg:#e6f5ed; --err:#b3261e; --err-bg:#fdecea;
  --warn:#8a5a00; --warn-bg:#fdf3e2; --radius:10px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#101318; --panel:#181c23; --sunk:#12151b; --line:#2a2f39;
  --fg:#e8eaee; --muted:#98a2b0; --accent:#5b8cf7; --accent-fg:#0b0e13;
  --ok:#5fd39b; --ok-bg:#12291f; --err:#f2837c; --err-bg:#2c1614;
  --warn:#e0b256; --warn-bg:#2b2213;
}}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.app{display:grid;grid-template-columns:212px 1fr;min-height:100vh}

/* ── 侧栏 ───────────────────────────────── */
.side{background:var(--panel);border-right:1px solid var(--line);
  display:flex;flex-direction:column;padding:18px 14px;gap:6px}
.brand{font-weight:680;font-size:15px;padding:0 8px 14px;letter-spacing:.2px}
.brand small{display:block;font-weight:400;font-size:12px;color:var(--muted);
  letter-spacing:0}
.nav{display:flex;flex-direction:column;gap:2px}
.nav button{display:flex;align-items:center;gap:9px;width:100%;text-align:left;
  padding:9px 10px;border:0;border-radius:8px;background:transparent;
  color:var(--fg);font:inherit;font-size:14px;cursor:pointer}
.nav button:hover{background:var(--sunk)}
.nav button.on{background:var(--accent);color:var(--accent-fg);font-weight:560}
.nav .ico{width:18px;text-align:center;opacity:.9}
.side .foot{margin-top:auto;padding:10px 8px 0;border-top:1px solid var(--line);
  font-size:12px;color:var(--muted);line-height:1.5}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--muted);margin-right:6px;vertical-align:middle}
.dot.ok{background:var(--ok)} .dot.no{background:var(--err)}

/* ── 内容 ───────────────────────────────── */
.main{padding:26px 30px;overflow:auto}
.wrap{max-width:820px;margin:0 auto}
h2.title{font-size:19px;margin:0 0 4px;font-weight:640}
p.sub{margin:0 0 20px;color:var(--muted);font-size:14px}
.card{background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:20px;margin-bottom:16px}
.card h3{font-size:14px;margin:0 0 4px;font-weight:620}
.card .desc{font-size:13px;color:var(--muted);margin:0 0 14px}
label{display:block;font-size:13px;color:var(--muted);margin:14px 0 5px}
label:first-of-type{margin-top:0}
input,select,textarea{width:100%;padding:10px 12px;border:1px solid var(--line);
  border-radius:8px;background:var(--sunk);color:var(--fg);font:inherit}
input:focus,select:focus,textarea:focus{outline:2px solid var(--accent);
  outline-offset:-1px;background:var(--panel)}
textarea{resize:vertical;min-height:78px}
.hint{font-size:12.5px;color:var(--muted);margin-top:6px}
.btns{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}
button.b{padding:9px 17px;border-radius:8px;font:inherit;cursor:pointer;
  border:1px solid var(--line);background:var(--panel);color:var(--fg)}
button.b:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
button.b.primary{background:var(--accent);color:var(--accent-fg);
  border-color:var(--accent)}
button.b.primary:hover:not(:disabled){opacity:.9;color:var(--accent-fg)}
button.b:disabled{opacity:.55;cursor:default}
.note{margin-top:14px;padding:11px 13px;border-radius:8px;font-size:13.5px;
  display:none;border:1px solid var(--line)}
.note.show{display:block}
.note.ok{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}
.note.bad{background:var(--err-bg);border-color:var(--err);color:var(--err)}
.note.busy{background:var(--sunk);color:var(--muted)}
.note .mono{margin-top:7px;font:12px/1.5 ui-monospace,Consolas,monospace;
  opacity:.8;word-break:break-all}

/* 分层自检 */
.steps{margin-top:14px;display:none}
.steps.show{display:block}
.step{display:flex;align-items:baseline;gap:10px;padding:8px 11px;
  border:1px solid var(--line);border-radius:8px;margin-bottom:6px;
  background:var(--sunk);font-size:13.5px}
.step .mk{font-weight:700;width:16px;flex:none}
.step.ok .mk{color:var(--ok)} .step.bad .mk{color:var(--err)}
.step .nm{font-weight:560;flex:none;min-width:150px}
.step .dt{color:var(--muted);word-break:break-all;flex:1}
.step .ms{color:var(--muted);font-size:12px;flex:none}
.verdict{margin-top:10px;padding:11px 13px;border-radius:8px;font-size:13.5px;
  background:var(--warn-bg);border:1px solid var(--warn);color:var(--warn)}

/* ── 对话 ───────────────────────────────── */
.chatwrap{display:flex;flex-direction:column;height:calc(100vh - 52px);
  max-width:820px;margin:0 auto}
#log{flex:1;overflow-y:auto;padding:4px 2px 12px}
.msg{margin-bottom:18px;display:flex;flex-direction:column}
.msg .who{font-size:12px;color:var(--muted);margin-bottom:4px}
.msg .bubble{padding:11px 14px;border-radius:11px;white-space:pre-wrap;
  word-break:break-word;max-width:86%}
.msg.me{align-items:flex-end}
.msg.me .bubble{background:var(--accent);color:var(--accent-fg)}
.msg.ai .bubble{background:var(--panel);border:1px solid var(--line);
  max-width:100%;white-space:normal}
.msg.sys .bubble{background:var(--sunk);color:var(--muted);font-size:13.5px;
  max-width:100%}

/* Agent 回复的 Markdown 渲染 */
.md>*:first-child{margin-top:0} .md>*:last-child{margin-bottom:0}
.md h1,.md h2,.md h3,.md h4{margin:18px 0 8px;font-weight:640;line-height:1.35}
.md h1{font-size:17px} .md h2{font-size:16px}
.md h3{font-size:15px} .md h4{font-size:14px}
.md p{margin:0 0 10px}
.md ul,.md ol{margin:0 0 10px;padding-left:22px}
.md li{margin:3px 0}
.md li>ul,.md li>ol{margin:3px 0}
.md code{background:var(--sunk);border:1px solid var(--line);padding:1px 5px;
  border-radius:5px;font:12.5px/1.5 ui-monospace,Consolas,monospace}
.md pre{background:var(--sunk);border:1px solid var(--line);padding:12px 14px;
  border-radius:8px;overflow-x:auto;margin:0 0 10px}
.md pre code{background:none;border:0;padding:0;font-size:12.5px}
.md hr{border:0;border-top:1px solid var(--line);margin:16px 0}
.md blockquote{margin:0 0 10px;padding:2px 0 2px 12px;
  border-left:3px solid var(--line);color:var(--muted)}
.md a{color:var(--accent)}
.md strong{font-weight:640}
.tablebox{overflow-x:auto;margin:0 0 10px}
.md table{border-collapse:collapse;font-size:13.5px;min-width:100%}
.md th,.md td{border:1px solid var(--line);padding:7px 11px;text-align:left;
  white-space:nowrap}
.md th{background:var(--sunk);font-weight:620}
.md tbody tr:nth-child(even){background:color-mix(in srgb,var(--sunk) 45%,transparent)}
.tools{margin-top:7px;font:12px/1.7 ui-monospace,Consolas,monospace;
  color:var(--muted)}
.tools span{display:inline-block;padding:1px 7px;border:1px solid var(--line);
  border-radius:99px;margin:0 5px 4px 0}
.tools span.no{color:var(--err);border-color:var(--err)}
.composer{border-top:1px solid var(--line);padding-top:12px;background:var(--bg)}
.composer .row{display:flex;gap:10px;align-items:flex-end}
.composer textarea{flex:1;min-height:52px;max-height:180px}
.approve{border:1px solid var(--warn);background:var(--warn-bg);color:var(--warn);
  border-radius:9px;padding:12px;margin:0 0 16px}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}
.chips button{padding:5px 11px;font-size:12.5px;border:1px dashed var(--line);
  background:transparent;color:var(--muted);border-radius:99px;cursor:pointer}
.chips button:hover{border-style:solid;border-color:var(--accent);
  color:var(--accent)}
pre.data{font:12.5px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;
  word-break:break-all;margin:0}

@media (max-width:760px){
  .app{grid-template-columns:1fr}
  .side{flex-direction:row;align-items:center;overflow-x:auto;
    border-right:0;border-bottom:1px solid var(--line);padding:10px}
  .brand,.side .foot{display:none}
  .nav{flex-direction:row}
  .main{padding:18px 14px}
  .chatwrap{height:auto;min-height:60vh}
}
</style></head><body>
<div class="app">
  <aside class="side">
    <div class="brand">ThermoForge<small>自主建模研究系统</small></div>
    <nav class="nav">
      <button data-v="chat" class="on"><span class="ico">💬</span>对话</button>
      <button data-v="setup"><span class="ico">🔑</span>模型设置</button>
      <button data-v="data"><span class="ico">📊</span>数据与状态</button>
    </nav>
    <div class="foot"><span id="dot" class="dot"></span><span id="dotText">检查中…</span></div>
  </aside>

  <main class="main">
    <!-- 对话 -->
    <section id="v-chat">
      <div class="chatwrap">
        <div id="log"></div>
        <div class="composer">
          <div class="chips">
            <button data-q="现在有哪些数据集？每个的质量怎么样？">看看有哪些数据</button>
            <button data-q="用 WX_SIX_PARAM 数据集，目标 total_power，输入用两侧供回水温、两侧流量和运行台数，先做可建模性检查再跑一组实验">跑一组实验</button>
            <button data-q="把最近几次实验的结果对比一下，告诉我哪个最好、为什么">对比实验结果</button>
          </div>
          <div class="row">
            <textarea id="msg" placeholder="用大白话说要做什么。Ctrl+Enter 发送。"></textarea>
            <button class="b primary" id="send">发送</button>
          </div>
          <div class="hint">Agent 会自己调用系统工具（导入、派生、建目标、跑实验、发布），每步都有记录。
            <a href="#" id="clear" style="color:var(--muted)">清空会话</a></div>
        </div>
      </div>
    </section>

    <!-- 设置 -->
    <section id="v-setup" hidden>
      <div class="wrap">
        <h2 class="title">模型设置</h2>
        <p class="sub">任何 OpenAI 兼容的接口都能用。下拉里是常见服务商的地址，选完仍可以随意改。</p>

        <div class="card">
          <h3>接口配置</h3>
          <p class="desc">改完记得点保存，然后用「网络自检」确认能通。</p>

          <label for="prov">服务商</label>
          <select id="prov"></select>
          <div class="hint" id="provNote"></div>

          <label for="key">API Key</label>
          <input id="key" type="password" placeholder="留空 = 不修改已保存的密钥" autocomplete="off">

          <label for="baseurl">接口地址 Base URL</label>
          <input id="baseurl" placeholder="https://api.example.com/v1" spellcheck="false">

          <label for="model">模型名</label>
          <input id="model" list="modelList" placeholder="选一个或直接填" spellcheck="false">
          <datalist id="modelList"></datalist>
          <div class="hint">下拉里的模型名只是候选，服务商随时会更新，以对方文档为准。</div>

          <label for="proxy">代理（可选，公司网络常需要）</label>
          <input id="proxy" placeholder="http://127.0.0.1:7890　留空表示不用代理" spellcheck="false">

          <div class="btns">
            <button class="b primary" id="save">保存</button>
            <button class="b" id="test">测试连通性</button>
            <button class="b" id="diag">网络自检</button>
          </div>
          <div class="note" id="note"></div>
          <div class="steps" id="steps"></div>
        </div>

        <div class="card">
          <h3>密钥存在哪</h3>
          <p class="desc" id="cfgPath"></p>
          <div class="hint">该文件已在 .gitignore 中，不会被提交。环境变量
            <code>TF_AGENT_API_KEY</code> 优先级更高，设了就会覆盖这里的值。</div>
        </div>
      </div>
    </section>

    <!-- 数据 -->
    <section id="v-data" hidden>
      <div class="wrap">
        <h2 class="title">数据与状态</h2>
        <p class="sub">当前工作区里的数据集、研究目标、实验和已发布的模型。</p>
        <div class="card"><h3>数据集</h3><pre class="data" id="datasets">载入中…</pre></div>
        <div class="card"><h3>研究状态</h3><pre class="data" id="status">载入中…</pre></div>
        <div class="btns"><button class="b" id="reload">刷新</button></div>
      </div>
    </section>
  </main>
</div>

<script>
const $=s=>document.querySelector(s);
const jget=p=>fetch(p).then(r=>r.json());
const jpost=(p,b)=>fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b||{})}).then(r=>r.json());
let PROVIDERS=[];

/* ── 导航 ── */
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  ['chat','setup','data'].forEach(v=>$('#v-'+v).hidden = v!==b.dataset.v);
  if(b.dataset.v==='data') loadData();
});

/* ── 通用：按钮忙碌态 + 错误一定可见 ── */
function note(kind,text,mono){
  const n=$('#note'); n.className='note show '+kind; n.textContent=text;
  if(mono){const d=document.createElement('div');d.className='mono';
    d.textContent=mono;n.appendChild(d);}
}
async function busy(btn,label,fn){
  const old=btn.textContent; btn.disabled=true; btn.textContent=label;
  try{ await fn(); }
  catch(e){ note('bad','请求没送到本机服务：'+e+
    '　启动窗口可能被关了，重新双击 启动网页版.bat。'); }
  finally{ btn.disabled=false; btn.textContent=old; }
}

/* ── 设置 ── */
function fillProviders(cfg){
  const sel=$('#prov'); sel.innerHTML='';
  PROVIDERS.forEach(p=>{const o=document.createElement('option');
    o.value=p.id; o.textContent=p.name; sel.appendChild(o);});
  const hit=PROVIDERS.find(p=>p.base_url&&p.base_url===cfg.base_url);
  sel.value=hit?hit.id:'custom';
  applyProvider(false);
}
function applyProvider(overwrite){
  const p=PROVIDERS.find(x=>x.id===$('#prov').value);
  if(!p) return;
  $('#provNote').textContent=p.note||'';
  const dl=$('#modelList'); dl.innerHTML='';
  (p.models||[]).forEach(m=>{const o=document.createElement('option');
    o.value=m; dl.appendChild(o);});
  if(overwrite&&p.id!=='custom'){
    $('#baseurl').value=p.base_url;
    if(p.models&&p.models.length) $('#model').value=p.models[0];
  }
}
$('#prov').onchange=()=>applyProvider(true);

$('#save').onclick=()=>busy($('#save'),'保存中…',async()=>{
  const r=await jpost('/api/config',{api_key:$('#key').value,
    base_url:$('#baseurl').value.trim(),model:$('#model').value.trim(),
    proxy:$('#proxy').value.trim()});
  if(r.ok){ $('#key').value=''; note('ok','已保存。接着点「测试连通性」确认能用。');
    loadConfig(); }
  else note('bad','保存失败：'+(r.error||'未知原因'));
});

$('#test').onclick=()=>busy($('#test'),'测试中…',async()=>{
  $('#steps').className='steps'; note('busy','正在调用模型接口…');
  const r=await jpost('/api/check');
  if(r.ok) note('ok','连通正常，模型 '+r.model+'。可以去「对话」页了。');
  else note('bad',r.error||'连不上',r.detail);
});

$('#diag').onclick=()=>busy($('#diag'),'自检中…',async()=>{
  note('busy','正在逐层测试 DNS → TCP → TLS → 接口…');
  const r=await jpost('/api/diagnose');
  const box=$('#steps'); box.className='steps show'; box.innerHTML='';
  (r.steps||[]).forEach(s=>{
    const d=document.createElement('div');
    d.className='step '+(s.ok?'ok':'bad');
    d.innerHTML='<span class="mk">'+(s.ok?'✓':'✗')+'</span>'+
      '<span class="nm"></span><span class="dt"></span><span class="ms"></span>';
    d.querySelector('.nm').textContent=s.name;
    d.querySelector('.dt').textContent=s.detail||'';
    d.querySelector('.ms').textContent=s.ms==null?'':s.ms+'ms';
    box.appendChild(d);
  });
  if(r.verdict){const v=document.createElement('div');v.className='verdict';
    v.textContent=r.verdict;box.appendChild(v);}
  note(r.ok?'ok':'bad',r.ok?'自检通过。':'自检发现问题，看下面每一层的结果。');
});

async function loadConfig(){
  try{
    const c=await jget('/api/config');
    PROVIDERS=c.providers||[];
    $('#baseurl').value=c.base_url; $('#model').value=c.model;
    $('#proxy').value=c.proxy||'';
    $('#cfgPath').textContent=c.config_path;
    fillProviders(c);
    $('#dot').className='dot '+(c.configured?'ok':'no');
    $('#dotText').textContent=c.configured
      ? (c.from_env?'密钥来自环境变量':'密钥已配置 '+c.api_key_masked)
      : '未配置密钥';
    if(c.from_env) $('#provNote').textContent=
      '注意：环境变量 TF_AGENT_API_KEY 已设置，优先级高于这里保存的值。';
  }catch(e){
    $('#dot').className='dot no'; $('#dotText').textContent='连不上本机服务';
  }
}

/* ── Markdown 渲染 ────────────────────────
   Agent 的回复是 Markdown，直接当文本显示满屏都是 ### 和 |---|。
   自己写而不引 CDN：这个页面要能离线用，CSP 也不该为渲染开口子。
   安全前提：**先整体转义 HTML**，之后只插入自己生成的标签，
   工具返回的内容永远不会被当作 HTML 执行。 */
function esc(s){
  return String(s).replace(/[&<>"']/g,c=>(
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function inlineMd(s){
  return s
    .replace(/`([^`]+)`/g,(m,c)=>'<code>'+c+'</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g,'$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g,'<del>$1</del>')
    // 只放行 http/https，避免 javascript: 之类的伪协议
    .replace(/\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}
function mdToHtml(src){
  const lines=esc(src).replace(/\r\n?/g,'\n').split('\n');
  const out=[]; let i=0;
  const listStack=[];               // 每层记 {tag, indent}
  function closeLists(toIndent){
    while(listStack.length &&
          listStack[listStack.length-1].indent>=toIndent){
      out.push('</'+listStack.pop().tag+'>');
    }
  }
  function closeAll(){ while(listStack.length) out.push('</'+listStack.pop().tag+'>'); }

  while(i<lines.length){
    const raw=lines[i], line=raw.trim();

    if(/^```/.test(line)){                                   // 代码块
      closeAll(); const buf=[]; i++;
      while(i<lines.length && !/^```/.test(lines[i].trim())) buf.push(lines[i++]);
      i++; out.push('<pre><code>'+buf.join('\n')+'</code></pre>'); continue;
    }
    if(!line){ closeAll(); i++; continue; }                   // 空行
    if(/^(-{3,}|\*{3,}|_{3,})$/.test(line)){                  // 分隔线
      closeAll(); out.push('<hr>'); i++; continue;
    }
    const h=line.match(/^(#{1,6})\s+(.*)$/);                  // 标题
    if(h){ closeAll();
      const lv=Math.min(h[1].length,4);
      out.push('<h'+lv+'>'+inlineMd(h[2])+'</h'+lv+'>'); i++; continue; }

    // 表格：当前行像表格行，且下一行是 |---|---| 分隔行
    if(line.startsWith('|') && i+1<lines.length &&
       /^\|[\s:|-]+\|$/.test(lines[i+1].trim())){
      closeAll();
      const cells=r=>r.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
      const head=cells(line); i+=2;
      const body=[];
      while(i<lines.length && lines[i].trim().startsWith('|')) body.push(cells(lines[i++]));
      out.push('<div class="tablebox"><table><thead><tr>'+
        head.map(c=>'<th>'+inlineMd(c)+'</th>').join('')+'</tr></thead><tbody>'+
        body.map(r=>'<tr>'+r.map(c=>'<td>'+inlineMd(c)+'</td>').join('')+'</tr>').join('')+
        '</tbody></table></div>');
      continue;
    }
    // 引用：此时 > 已被 esc() 变成 &gt;，必须按转义后的形态匹配
    if(/^&gt;\s?/.test(line)){
      closeAll(); const buf=[];
      while(i<lines.length && /^&gt;\s?/.test(lines[i].trim()))
        buf.push(lines[i++].trim().replace(/^&gt;\s?/,''));
      out.push('<blockquote>'+inlineMd(buf.join(' '))+'</blockquote>'); continue;
    }

    const li=raw.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);      // 列表（含嵌套）
    if(li){
      const indent=li[1].length, ordered=/\d/.test(li[2]);
      const tag=ordered?'ol':'ul';
      closeLists(indent+1);
      const top=listStack[listStack.length-1];
      if(!top || top.indent<indent){
        listStack.push({tag,indent}); out.push('<'+tag+'>');
      }
      out.push('<li>'+inlineMd(li[3])+'</li>'); i++; continue;
    }

    closeAll();                                               // 普通段落
    const buf=[lines[i++]];
    while(i<lines.length){
      const nxt=lines[i].trim();
      if(!nxt || /^(#{1,6}\s|```|&gt;|\||-{3,})/.test(nxt) ||
         /^(\s*)([-*+]|\d+[.)])\s+/.test(lines[i])) break;
      buf.push(lines[i++]);
    }
    out.push('<p>'+inlineMd(buf.join(' '))+'</p>');
  }
  closeAll();
  return out.join('');
}

/* ── 对话 ── */
function bubble(who,text,tools){
  const d=document.createElement('div'); d.className='msg '+who;
  const label={me:'你',ai:'Agent',sys:'系统'}[who];
  d.innerHTML='<div class="who">'+label+'</div><div class="bubble"></div>';
  const body=d.querySelector('.bubble');
  if(who==='ai'){ body.className='bubble md'; body.innerHTML=mdToHtml(text); }
  else body.textContent=text;
  if(tools&&tools.length){
    const t=document.createElement('div'); t.className='tools';
    tools.forEach(x=>{const s=document.createElement('span');
      s.textContent=x.tool+(x.ok?' ✓':' ✗'); if(!x.ok)s.className='no';
      t.appendChild(s);});
    d.appendChild(t);
  }
  $('#log').appendChild(d); $('#log').scrollTop=$('#log').scrollHeight;
  return d;
}
function approval(p){
  const d=document.createElement('div'); d.className='approve';
  d.innerHTML='<b>需要你确认</b><div style="font-size:13px;margin-top:5px">Agent 想执行 <code>'
    +p.tool+'</code>：'+(p.reason||'')+'</div>';
  const b=document.createElement('button'); b.className='b'; b.textContent='批准并执行';
  b.style.marginTop='10px';
  b.onclick=async()=>{b.disabled=true;
    const r=await jpost('/api/approve',{tool:p.tool,arguments:p.arguments});
    const s=document.createElement('div'); s.style.marginTop='8px';
    s.style.fontSize='13px';
    s.textContent=r.ok?'已以 human 身份执行完成。':'失败：'+r.error;
    d.appendChild(s);};
  d.appendChild(b); $('#log').appendChild(d);
  $('#log').scrollTop=$('#log').scrollHeight;
}
async function send(){
  const t=$('#msg').value.trim(); if(!t) return;
  $('#msg').value=''; $('#send').disabled=true;
  bubble('me',t);
  const wait=bubble('sys','思考中…（会自己调用工具，复杂任务要等一会）');
  try{
    const r=await jpost('/api/chat',{message:t});
    wait.remove();
    if(!r.ok){
      bubble('sys','出错了：'+r.error);
      if(r.need_config) bubble('sys','去左边「模型设置」填好 API Key，再回来。');
    }else{
      bubble('ai',r.reply,r.tool_calls);
      (r.pending_approvals||[]).forEach(approval);
    }
  }catch(e){ wait.remove(); bubble('sys','请求失败：'+e); }
  $('#send').disabled=false; $('#msg').focus();
}
$('#send').onclick=send;
$('#msg').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();send();}});
document.querySelectorAll('.chips button').forEach(b=>b.onclick=()=>{
  $('#msg').value=b.dataset.q; $('#msg').focus();});
$('#clear').onclick=async e=>{e.preventDefault(); await jpost('/api/reset');
  $('#log').innerHTML=''; bubble('sys','会话已清空。');};

/* ── 数据 ── */
async function loadData(){
  try{
    const d=await jget('/api/datasets');
    $('#datasets').textContent=(d.datasets||[]).map(x=>
      x.ref+'\n    '+x.variable_count+' 个变量 · 对象 '+x.objects.length+' 个：'
      +x.objects.slice(0,8).join(', ')+(x.objects.length>8?' …':'')
    ).join('\n\n')||'还没有数据集。';
    const s=await jget('/api/status');
    $('#status').textContent=JSON.stringify(s,null,2);
  }catch(e){
    $('#datasets').textContent='连不上本机服务：'+e;
    $('#status').textContent='重新双击 启动网页版.bat 再试。';
  }
}
$('#reload').onclick=loadData;

loadConfig();
bubble('sys','先在左边「模型设置」里配好接口，然后回来直接说要做什么。');
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    port = DEFAULT_PORT
    if argv and argv[0].isdigit():
        port = int(argv[0])
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    server = ThreadingHTTPServer((HOST, port), partial(Handler))
    url = f"http://{HOST}:{port}"
    print(f"ThermoForge 控制台已启动：{url}")
    print("在浏览器里打开上面这个地址。按 Ctrl+C 关闭。")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
