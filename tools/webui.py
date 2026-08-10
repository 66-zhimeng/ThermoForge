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
        "config_path": str(CONFIG_PATH),
    }


def write_config(api_key: str, base_url: str, model: str) -> None:
    """写 pi/agent.toml（gitignore 内）。空密钥表示保留原值。"""
    import tomllib

    current: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as fp:
            current = tomllib.load(fp)
    key = api_key or str(current.get("api_key") or "")
    if not key:
        raise ValueError("API key 不能为空")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ThermoForge Agent 配置（本文件已 gitignore，密钥不会入库）",
        f'api_key = "{_toml_escape(key)}"',
        f'base_url = "{_toml_escape(base_url)}"',
        f'model = "{_toml_escape(model)}"',
        "",
    ]
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


def api_check() -> dict[str, Any]:
    from thermoforge_agent.client import ChatClient

    config = CONSOLE.config()
    if config is None:
        return {"ok": False, "error": "还没有配置 API key"}
    try:
        return ChatClient(config).check()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


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
                             str(doc.get("model") or "").strip())
                self._send({"ok": True, **read_config()})
            elif route == "/api/check":
                self._send(api_check())
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
:root{--bg:#f6f7f9;--card:#fff;--line:#e2e5ea;--fg:#1b1f24;--muted:#6b7280;
--accent:#2563eb;--ok:#0f9d58;--warn:#b45309;--err:#b91c1c;}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2027;
--line:#2c313a;--fg:#e6e8eb;--muted:#9aa3af;--accent:#60a5fa;}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;
background:var(--bg);color:var(--fg)}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card);
display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:650}
.tabs{display:flex;gap:4px;margin-left:auto}
.tabs button{padding:6px 14px;border:1px solid var(--line);background:transparent;
color:var(--fg);border-radius:7px;cursor:pointer;font-size:14px}
.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:960px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px;margin-bottom:16px}
.card h2{font-size:15px;margin:0 0 12px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
input,textarea,select{width:100%;padding:9px 11px;border:1px solid var(--line);
border-radius:7px;background:var(--bg);color:var(--fg);font:inherit}
textarea{resize:vertical;min-height:76px}
button.act{margin-top:12px;padding:9px 18px;border:0;border-radius:7px;
background:var(--accent);color:#fff;font:inherit;cursor:pointer}
button.act:disabled{opacity:.5;cursor:default}
.hint{font-size:13px;color:var(--muted);margin-top:8px}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;
border:1px solid var(--line)}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.no{color:var(--err);border-color:var(--err)}
#log{height:52vh;overflow-y:auto;border:1px solid var(--line);border-radius:8px;
padding:14px;background:var(--bg)}
.msg{margin-bottom:14px}
.msg .who{font-size:12px;color:var(--muted);margin-bottom:3px}
.msg .body{white-space:pre-wrap;word-break:break-word}
.msg.me .body{background:var(--accent);color:#fff;padding:9px 12px;
border-radius:9px;display:inline-block;max-width:88%}
.msg.ai .body{background:var(--card);border:1px solid var(--line);
padding:9px 12px;border-radius:9px}
.calls{font-size:12px;color:var(--muted);margin-top:5px;font-family:ui-monospace,
Consolas,monospace}
pre{white-space:pre-wrap;word-break:break-word;font:13px/1.5 ui-monospace,
Consolas,monospace;margin:0}
.row{display:flex;gap:10px;align-items:flex-end}
.row textarea{flex:1}
.approve{border:1px solid var(--warn);border-radius:8px;padding:10px;margin-top:10px}
</style></head><body>
<header>
  <h1>ThermoForge 控制台</h1>
  <span id="cfgpill" class="pill">检查中…</span>
  <div class="tabs">
    <button data-tab="chat" class="on">对话</button>
    <button data-tab="setup">设置</button>
    <button data-tab="status">状态</button>
  </div>
</header>
<main>
  <section id="tab-chat">
    <div class="card">
      <div id="log"></div>
      <div class="row" style="margin-top:12px">
        <textarea id="msg" placeholder="用大白话说要做什么，例如：把 WX_2025_HVAC 里两路总管流量求和、温度取一，派生一个新数据集"></textarea>
      </div>
      <button class="act" id="send">发送</button>
      <button class="act" id="clear" style="background:transparent;color:var(--muted);border:1px solid var(--line)">清空会话</button>
      <div class="hint">Agent 会自己调用系统工具（导入、派生、建目标、跑实验、发布），每一步都有记录。</div>
    </div>
  </section>

  <section id="tab-setup" hidden>
    <div class="card">
      <h2>API 密钥</h2>
      <div class="hint" id="cfgnote"></div>
      <label>API Key（留空表示不修改）</label>
      <input id="key" type="password" placeholder="sk-...">
      <label>接口地址 Base URL</label>
      <input id="baseurl">
      <label>模型名</label>
      <input id="model">
      <button class="act" id="save">保存</button>
      <button class="act" id="test" style="background:transparent;color:var(--fg);border:1px solid var(--line)">测试连通性</button>
      <div class="hint" id="testout"></div>
      <div class="hint">密钥写入 <code>pi/agent.toml</code>，该文件已在 .gitignore 中，不会提交。
      任何 OpenAI 兼容端点都可用（Moonshot/Kimi、DeepSeek、OpenAI…）。</div>
    </div>
  </section>

  <section id="tab-status" hidden>
    <div class="card"><h2>数据集</h2><pre id="datasets">载入中…</pre></div>
    <div class="card"><h2>研究状态</h2><pre id="status">载入中…</pre></div>
  </section>
</main>
<script>
const $=s=>document.querySelector(s);
const api=(p,o)=>fetch(p,o).then(r=>r.json());
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b||{})});

document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  ['chat','setup','status'].forEach(t=>$('#tab-'+t).hidden = t!==b.dataset.tab);
  if(b.dataset.tab==='status') loadStatus();
});

function addMsg(who,text,calls){
  const d=document.createElement('div');
  d.className='msg '+(who==='me'?'me':'ai');
  const label=who==='me'?'你':'Agent';
  d.innerHTML='<div class="who">'+label+'</div><div class="body"></div>';
  d.querySelector('.body').textContent=text;
  if(calls&&calls.length){
    const c=document.createElement('div');c.className='calls';
    c.textContent='调用：'+calls.map(x=>x.tool+(x.ok?' ✓':' ✗')).join('  ');
    d.appendChild(c);
  }
  $('#log').appendChild(d); $('#log').scrollTop=$('#log').scrollHeight;
  return d;
}

function addApproval(p){
  const d=document.createElement('div'); d.className='approve';
  d.innerHTML='<b>需要你确认</b><div class="hint">Agent 请求执行 <code>'+p.tool+
    '</code>：'+(p.reason||'')+'</div>';
  const b=document.createElement('button'); b.className='act'; b.textContent='批准并执行';
  b.onclick=async()=>{b.disabled=true;
    const r=await post('/api/approve',{tool:p.tool,arguments:p.arguments});
    d.appendChild(Object.assign(document.createElement('div'),
      {className:'hint',textContent:r.ok?'已以 human 身份执行完成':'失败：'+r.error}));};
  d.appendChild(b); $('#log').appendChild(d);
}

async function send(){
  const t=$('#msg').value.trim(); if(!t) return;
  $('#msg').value=''; $('#send').disabled=true;
  addMsg('me',t);
  const thinking=addMsg('ai','思考中…');
  try{
    const r=await post('/api/chat',{message:t});
    thinking.remove();
    if(!r.ok){ addMsg('ai','出错了：'+r.error);
      if(r.need_config) addMsg('ai','请先到「设置」页填 API key。'); }
    else{ addMsg('ai',r.reply,r.tool_calls);
      (r.pending_approvals||[]).forEach(addApproval); }
  }catch(e){ thinking.remove(); addMsg('ai','请求失败：'+e); }
  $('#send').disabled=false; $('#msg').focus();
}
$('#send').onclick=send;
$('#msg').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)) send();});
$('#clear').onclick=async()=>{await post('/api/reset');$('#log').innerHTML='';
  addMsg('ai','会话已清空。');};

async function loadConfig(){
  const c=await api('/api/config');
  $('#baseurl').value=c.base_url; $('#model').value=c.model;
  const pill=$('#cfgpill');
  pill.textContent=c.configured?('密钥已配置 '+c.api_key_masked):'未配置密钥';
  pill.className='pill '+(c.configured?'ok':'no');
  $('#cfgnote').textContent=c.from_env
    ? '当前生效的密钥来自环境变量 TF_AGENT_API_KEY，优先级高于此处保存的值。'
    : '密钥保存在 '+c.config_path;
}
$('#save').onclick=async()=>{
  const r=await post('/api/config',{api_key:$('#key').value,
    base_url:$('#baseurl').value,model:$('#model').value});
  $('#key').value='';
  $('#testout').textContent=r.ok?'已保存。建议点一下「测试连通性」。':('保存失败：'+r.error);
  loadConfig();
};
$('#test').onclick=async()=>{
  $('#testout').textContent='测试中…';
  const r=await post('/api/check');
  $('#testout').textContent=r.ok?('连通正常，模型 '+r.model):('连不上：'+r.error);
};

async function loadStatus(){
  const d=await api('/api/datasets');
  $('#datasets').textContent=d.datasets.map(x=>
    x.ref+'   '+x.variable_count+' 个变量   对象：'+x.objects.slice(0,6).join(', ')
    +(x.objects.length>6?' …共'+x.objects.length+' 个':'')).join('\n')||'还没有数据集';
  const s=await api('/api/status');
  $('#status').textContent=JSON.stringify(s,null,2);
}

loadConfig();
addMsg('ai','你好。先在「设置」里填好 API key，然后就可以直接说要做什么了。');
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
