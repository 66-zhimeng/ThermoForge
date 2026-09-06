"""V2 Codex 后台实例：独占 stdio App Server，协议基线 0.153.4。

参见 plan/v2-implementation-plan.md 与 https://learn.chatgpt.com/docs/app-server。
研究工具通过 dynamicTools 回调中央服务；不依赖 Codex 原生子智能体。
权限配置约束 Codex 工具，不是将整个 App Server 放入独立 OS 沙箱。
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Awaitable, Callable, Sequence
import weakref

Json = dict[str, Any]
ToolHandler = Callable[[str, Json], Awaitable[Any]]
EventHandler = Callable[[Json], Awaitable[None]]

VALIDATED_CODEX_VERSION = "0.153.4"
SUPPORTED_CODEX_VERSIONS = frozenset({VALIDATED_CODEX_VERSION})
PERMISSION_PROFILE = "thermoforge_research"
DISABLED_FEATURES = (
    "multi_agent", "multi_agent_v2", "shell_tool", "unified_exec", "plugins",
    "apps", "hooks", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "in_app_browser", "computer_use", "remote_plugin", "recommended_plugins",
    "view_image", "image_generation", "code_mode",
    "workspace_dependencies", "skill_search", "skill_mcp_dependency_install",
    "memories", "goals", "request_permissions_tool", "shell_snapshot",
)
_INITIALIZATION_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _initialization_lock():
    # Windows 权限 helper 同时配置多个实例时可能竞争。只序列化会话初始化，
    # 进程仍各自独立，后续六个研究 turn 不持有此锁。
    if os.name != "nt":
        return nullcontext()
    loop = asyncio.get_running_loop()
    if loop not in _INITIALIZATION_LOCKS:
        _INITIALIZATION_LOCKS[loop] = asyncio.Lock()
    return _INITIALIZATION_LOCKS[loop]


class CodexError(RuntimeError):
    """传输、协议或实例生命周期错误；失败不会悄悄创建替代会话。"""


@dataclass(frozen=True)
class CodexConfig:
    cwd: Path
    model: str | None = None
    effort: str | None = None
    command: Sequence[str] | None = None
    expected_version: str | None = VALIDATED_CODEX_VERSION
    request_timeout: float = 30.0
    turn_timeout: float = 1800.0
    tool_timeout: float = 1200.0
    shutdown_timeout: float = 5.0
    developer_instructions: str = ""
    # 仅传递启动所需环境，不将认证文件复制到候选工作区。
    env: dict[str, str] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class TurnResult:
    status: str
    text: str
    thread_id: str
    turn_id: str
    usage: Json | None = None
    error: Json | None = None


def _runtime_version(executable: Path) -> str | None:
    """只读版本探测，不启动模型，也不修改安装。"""
    try:
        result = subprocess.run(
            [str(executable), "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            encoding="utf-8", timeout=5, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        found = re.search(r"codex-cli\s+(\d+\.\d+\.\d+)(?:\s|$)", result.stdout)
        return found.group(1) if result.returncode == 0 and found else None
    except (OSError, subprocess.SubprocessError):
        return None


def _desktop_codex_command() -> list[str] | None:
    local = os.environ.get("LOCALAPPDATA")
    if os.name != "nt" or not local:
        return None
    directory = Path(local) / "OpenAI" / "Codex" / "bin"
    try:
        candidates = sorted(directory.glob("*/codex.exe"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for candidate in candidates:
        if _runtime_version(candidate) in SUPPORTED_CODEX_VERSIONS:
            return [str(candidate)]
    return None


def resolve_codex_command() -> list[str]:
    """优先直接启动原生可执行文件；Windows npm 包不经 shell 插值。"""
    override = os.environ.get("TF_CODEX_BIN")
    if not override:
        desktop = _desktop_codex_command()
        if desktop:
            return desktop
    found = override or shutil.which("codex")
    if not found:
        raise CodexError("未找到 Codex CLI；请安装并登录 Codex，或设置 TF_CODEX_BIN。")
    path = Path(found)
    if path.suffix.lower() in {".cmd", ".ps1"}:
        package = path.parent / "node_modules" / "@openai" / "codex"
        # npm 的架构包内含真正的 Codex 进程，避免 Windows cmd 窗口与注入。
        candidates = sorted(package.glob("node_modules/@openai/codex-win32-*/vendor/*/bin/codex.exe"))
        if len(candidates) == 1:
            return [str(candidates[0])]
        node = shutil.which("node")
        script = package / "bin" / "codex.js"
        if node and script.is_file():
            return [node, str(script)]
        raise CodexError("无法解析 npm Codex 可执行文件，请将 TF_CODEX_BIN 指向 codex.exe。")
    return [str(path)]


def restricted_config() -> Json:
    """命令行最高优先级配置；不修改用户的 Codex 设置。"""
    result: Json = {f"features.{name}": False for name in DISABLED_FEATURES}
    result.update({
        # Astra 经此执行宿主转发动态工具。禁用它也会禁用 ThermoForge 工具。
        # 这里不启用 shell、原生子智能体、插件或额外 MCP。
        "features.code_mode_host": True,
        "features.skip_host_skill_discovery": True,
        "web_search": "disabled",
        "skills.include_instructions": False,
        "default_permissions": PERMISSION_PROFILE,
        f"permissions.{PERMISSION_PROFILE}.filesystem": {
            ":root": "deny", ":minimal": "read", ":workspace_roots": {".": "read"},
        },
        f"permissions.{PERMISSION_PROFILE}.network.enabled": False,
        "approval_policy": "never",
    })
    return result


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k)}={_toml(v)}" for k, v in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False)


def _clean_event(value: Any) -> Any:
    """保留可观察事件与推理摘要，不把原始内部推理块写入研究证据。"""
    if isinstance(value, dict):
        return {k: _clean_event(v) for k, v in value.items()
                if not (value.get("type") == "reasoning" and k == "content")}
    if isinstance(value, list):
        return [_clean_event(v) for v in value]
    return value


class CodexSession:
    """一个对象拥有一个隐藏进程和一个持久会话。所有回调均为 async。

    event_handler 按序执行，不能在回调内等待当前 turn 完成。
    tool_handler 独立运行，不阻塞协议读取/取消；它必须自行实施来源、预算和幂等性。
    """

    def __init__(self, config: CodexConfig, tool_specs: Sequence[Json],
                 tool_handler: ToolHandler, event_handler: EventHandler | None = None):
        self.config = config
        self.tool_specs = [dict(spec, type="function") for spec in tool_specs]
        names = [s.get("name") for s in self.tool_specs]
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
            raise ValueError("动态工具名称必须唯一并符合 Codex 命名规则。")
        self._tool_names = set(names)
        self._tool_handler = tool_handler
        self._event_handler = event_handler
        self.process: asyncio.subprocess.Process | None = None
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.state = "new"
        self.info: Json = {}
        self.usage: Json | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._turn_done: asyncio.Future | None = None
        self._messages: dict[str, Json] = {}
        self._tasks: set[asyncio.Task] = set()
        self._reader: asyncio.Task | None = None
        self._stderr_reader: asyncio.Task | None = None
        self._event_worker_task: asyncio.Task | None = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._stderr: deque[str] = deque(maxlen=20)
        self._tool_cache: dict[str, tuple[str, asyncio.Future]] = {}
        self._closing = False

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    async def start(self, thread_id: str | None = None) -> Json:
        if self.state != "new":
            raise CodexError("实例已启动；恢复时请关闭旧实例，再使用原 thread_id 建立新实例。")
        cwd = Path(self.config.cwd).resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        command = list(self.config.command) if self.config.command else resolve_codex_command()
        command += ["app-server", "--stdio"]
        for key, value in restricted_config().items():
            command += ["-c", f"{key}={_toml(value)}"]
        self.state = "starting"
        self._event_worker_task = asyncio.create_task(self._event_worker())
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.config.env or {})},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                limit=4 * 1024 * 1024,
            )
            self._reader = asyncio.create_task(self._read_loop())
            self._stderr_reader = asyncio.create_task(self._read_stderr())
            initialized = await self._request("initialize", {
                "clientInfo": {"name": "thermoforge", "title": "ThermoForge", "version": "2.0.0"},
                "capabilities": {"experimentalApi": True},
            })
            agent = initialized.get("userAgent", "")
            expected = self.config.expected_version
            if expected and not re.search(rf"/{re.escape(expected)}(?:\s|$)", agent):
                raise CodexError(
                    f"Codex 协议版本不匹配，要求 {expected}；请更新 Codex，或将 TF_CODEX_BIN 指向已验证运行时。"
                    "旧版 0.151.0 虽可建立连接，但不能运行当前 gpt-6-astra 模型。")
            await self._send({"method": "initialized", "params": {}})
            account = await self._request("account/read", {"refreshToken": False})
            if account.get("requiresOpenaiAuth") and not account.get("account"):
                raise CodexError("Codex 尚未登录；请先通过 Codex 完成认证。")
            cfg = (await self._request("config/read", {"includeLayers": False, "cwd": str(cwd)})).get("config", {})
            actual_features = cfg.get("features", {})
            if any(actual_features.get(name) is not False for name in DISABLED_FEATURES):
                raise CodexError("Codex 未应用研究实例的工具限制，拒绝启动研究。")
            if actual_features.get("code_mode_host") is not True:
                raise CodexError("Codex 动态工具执行宿主未启用，无法执行研究工具。")
            profile_config = cfg.get("permissions", {}).get(PERMISSION_PROFILE, {})
            filesystem = {k: v for k, v in profile_config.get("filesystem", {}).items() if v is not None}
            expected_filesystem = restricted_config()[f"permissions.{PERMISSION_PROFILE}.filesystem"]
            if (filesystem != expected_filesystem or profile_config.get("extends")
                    or profile_config.get("workspace_roots")
                    or profile_config.get("network", {}).get("enabled") is not False):
                raise CodexError("研究权限配置被继承规则扩宽或未生效，拒绝启动研究。")
            # 空对象会与已有配置深合并，必须逐个禁用继承的 MCP。
            server_names = list(cfg.get("mcp_servers", {}))
            if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in server_names):
                raise CodexError("存在不能可靠覆盖的 MCP 配置名称，拒绝继承外部工具。")
            overrides = {f"mcp_servers.{name}.enabled": False for name in server_names}
            params: Json = {
                "cwd": str(cwd), "permissions": PERMISSION_PROFILE,
                "runtimeWorkspaceRoots": [str(cwd)], "approvalPolicy": "never",
                "config": overrides,
                "developerInstructions": self.config.developer_instructions,
            }
            if self.config.model:
                params["model"] = self.config.model
            async with _initialization_lock():
                if thread_id:
                    params["threadId"] = thread_id
                    result = await self._request("thread/resume", params)
                else:
                    params.update({"dynamicTools": self.tool_specs, "ephemeral": False,
                                   "allowProviderModelFallback": False})
                    result = await self._request("thread/start", params)
            self.thread_id = result.get("thread", {}).get("id")
            if not self.thread_id or (thread_id and self.thread_id != thread_id):
                raise CodexError("Codex 未返回预期的持久会话 ID。")
            profile = result.get("activePermissionProfile")
            if not profile or profile.get("id") != PERMISSION_PROFILE:
                raise CodexError("Codex 未启用研究专用读访问配置，拒绝继续。")
            self.state = "idle"
            self.info = {"pid": self.pid, "thread_id": self.thread_id, "state": self.state,
                         "model": result.get("model"), "model_provider": result.get("modelProvider"),
                         "reasoning_effort": self.config.effort or result.get("reasoningEffort")
                         or cfg.get("model_reasoning_effort"),
                         "server": agent, "permission_profile": profile,
                         "auth_type": (account.get("account") or {}).get("type"),
                         "dynamic_tools": sorted(self._tool_names),
                         "native_shell": False, "native_subagents": False,
                         "isolation": "codex_permission_profile_and_tool_gates"}
            self._emit("session/ready", self.info)
            return dict(self.info)
        except BaseException:
            self.state = "failed"
            await self.close()
            raise

    async def turn(self, prompt: str) -> TurnResult:
        if self.state != "idle" or self._turn_lock.locked() or not self.thread_id:
            raise CodexError(f"实例不能开始新轮次：{self.state}")
        if not prompt.strip():
            raise ValueError("研究指令不能为空。")
        async with self._turn_lock:
            self.state = "running"
            self.turn_id = None
            self.usage = None
            self._messages.clear()
            self._turn_done = asyncio.get_running_loop().create_future()
            params: Json = {"threadId": self.thread_id, "input": [{"type": "text", "text": prompt}]}
            if self.config.effort:
                params["effort"] = self.config.effort
            try:
                response = await self._request("turn/start", params)
                self.turn_id = response["turn"]["id"]
                completed = await asyncio.wait_for(asyncio.shield(self._turn_done), self.config.turn_timeout)
                messages = list(self._messages.values())
                finals = [m for m in messages if m.get("phase") == "final_answer"]
                text = "\n\n".join(m.get("text", "") for m in (finals or messages))
                return TurnResult(completed["status"], text, self.thread_id, self.turn_id,
                                  self.usage, completed.get("error"))
            except (asyncio.CancelledError, TimeoutError):
                try:
                    await self.interrupt()
                except (CodexError, TimeoutError):
                    await self.close()
                # 取消应停止真实生成；未收到 terminal 事件则关闭进程，不能假装已暂停。
                if self._turn_done and not self._turn_done.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(self._turn_done), self.config.shutdown_timeout)
                    except (TimeoutError, CodexError):
                        await self.close()
                raise
            except Exception:
                # turn/start 失败/断线的执行状态可能不确定，不能将它标为可再次提交。
                await self.close()
                raise
            finally:
                if self.state not in {"closed", "failed"}:
                    self.state = "idle"
                if self._turn_done and not self._turn_done.done():
                    self._turn_done.cancel()
                elif self._turn_done and not self._turn_done.cancelled():
                    self._turn_done.exception()
                self._turn_done = None

    async def interrupt(self) -> None:
        if self.state not in {"running", "interrupting"}:
            return
        if self.thread_id and self.turn_id:
            self.state = "interrupting"
            await self._request("turn/interrupt", {"threadId": self.thread_id, "turnId": self.turn_id})

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        proc = self.process
        if proc and proc.returncode is None:
            if proc.stdin:
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), self.config.shutdown_timeout)
            except TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), self.config.shutdown_timeout)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
        self._fail_pending(CodexError("Codex 实例已关闭。"))
        for task in [self._reader, self._stderr_reader, *self._tasks]:
            if task and task is not asyncio.current_task():
                task.cancel()
        await asyncio.gather(*(t for t in [self._reader, self._stderr_reader, *self._tasks]
                               if t and t is not asyncio.current_task()), return_exceptions=True)
        self.state = "closed"
        self._emit("session/closed", {"pid": self.pid, "thread_id": self.thread_id})
        self._events.put_nowait(None)
        if self._event_worker_task and self._event_worker_task is not asyncio.current_task():
            try:
                await asyncio.wait_for(self._event_worker_task, self.config.shutdown_timeout)
            except TimeoutError:
                self._event_worker_task.cancel()

    async def _send(self, message: Json) -> None:
        async with self._write_lock:
            if not self.process or not self.process.stdin or self.process.returncode is not None:
                raise CodexError("Codex 进程不在线。")
            try:
                self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionError) as exc:
                raise CodexError("Codex stdio 连接已断开。") from exc

    async def _request(self, method: str, params: Json) -> Json:
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, self.config.request_timeout)
        finally:
            self._pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        assert self.process and self.process.stdout
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise CodexError("Codex 返回非对象协议消息。")
                method = message.get("method")
                if method and "id" in message:
                    task = asyncio.create_task(self._server_request(message))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                elif method:
                    self._notification(method, message.get("params", {}))
                elif "id" in message:
                    future = self._pending.get(message["id"])
                    if future and not future.done():
                        if "error" in message:
                            future.set_exception(CodexError(str(message["error"].get("message", "协议请求失败"))))
                        else:
                            future.set_result(message.get("result", {}))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._fail_pending(CodexError(f"Codex 协议读取失败：{exc}"))
        finally:
            if not self._closing:
                self.state = "failed"
                self._fail_pending(CodexError("Codex 进程已断开，研究需要从原会话恢复。"))
                self._emit("session/disconnected", {"pid": self.pid, "thread_id": self.thread_id})

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            # 不向事件/报告直接转发外部进程日志（可能包含配置或认证信息）。
            self._stderr.append(line.decode("utf-8", errors="replace")[:2000])

    def _notification(self, method: str, params: Json) -> None:
        if self.thread_id and params.get("threadId") not in {None, self.thread_id}:
            return
        if method == "turn/started":
            self.turn_id = params.get("turn", {}).get("id", self.turn_id)
        if method == "thread/tokenUsage/updated":
            self.usage = params.get("tokenUsage")
        if method == "item/completed" and params.get("item", {}).get("type") == "agentMessage":
            item = params["item"]
            self._messages[item["id"]] = item
        if method == "turn/completed" and self._turn_done and not self._turn_done.done():
            turn = params.get("turn", {})
            if self.turn_id is None or turn.get("id") == self.turn_id:
                for item in turn.get("items", []):
                    if item.get("type") == "agentMessage":
                        self._messages[item["id"]] = item
                self._turn_done.set_result(turn)
        if "reasoning/textDelta" not in method:
            self._emit(method, params)

    async def _server_request(self, message: Json) -> None:
        request_id, method, params = message["id"], message["method"], message.get("params", {})
        try:
            if method == "item/tool/call":
                result = await self._call_tool(params)
            elif method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
                result = {"decision": "decline"}
                self._emit("session/needs_input", {"reason": "unexpected_native_approval", "method": method})
            elif method == "item/permissions/requestApproval":
                result = {"permissions": {}, "scope": "turn"}
                self._emit("session/needs_input", {"reason": "permission_request_denied"})
            elif method == "tool/requestUserInput":
                result = {"answers": {}}
                self._emit("session/needs_input", {"reason": "user_input", "request": params})
            elif method == "mcpServer/elicitation/request":
                result = {"action": "decline", "content": None}
                self._emit("session/needs_input", {"reason": "unexpected_mcp_elicitation"})
            else:
                await self._send({"id": request_id, "error": {"code": -32601, "message": "Unsupported server request"}})
                return
            await self._send({"id": request_id, "result": result})
        except asyncio.CancelledError:
            raise
        except (CodexError, BrokenPipeError):
            return

    async def _call_tool(self, params: Json) -> Json:
        tool, args = params.get("tool"), params.get("arguments")
        if (tool not in self._tool_names or not isinstance(args, dict)
                or params.get("threadId") != self.thread_id
                or params.get("turnId") != self.turn_id):
            return self._tool_output({"ok": False, "error": "工具、参数或会话不匹配。"})
        call_id = params.get("callId")
        if not isinstance(call_id, str) or not call_id:
            return self._tool_output({"ok": False, "error": "缺少工具调用 ID。"})
        signature = json.dumps([tool, args], sort_keys=True, ensure_ascii=False)
        if call_id in self._tool_cache:
            previous, future = self._tool_cache[call_id]
            if previous != signature:
                return self._tool_output({"ok": False, "error": "重复调用 ID 的参数发生变化。"})
            return await asyncio.shield(future)
        future = asyncio.get_running_loop().create_future()
        self._tool_cache[call_id] = signature, future
        try:
            result = await asyncio.wait_for(self._tool_handler(tool, args), self.config.tool_timeout)
            output = self._tool_output(result)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            output = self._tool_output({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        future.set_result(output)
        return output

    @staticmethod
    def _tool_output(result: Any) -> Json:
        return {"contentItems": [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False, default=str)}],
                "success": not isinstance(result, dict) or result.get("ok") is not False}

    def _fail_pending(self, error: CodexError) -> None:
        for future in [*self._pending.values(), self._turn_done]:
            if future is not None and not future.done():
                future.set_exception(error)

    def _emit(self, method: str, params: Json) -> None:
        self._events.put_nowait({"method": method, "params": _clean_event(params)})

    async def _event_worker(self) -> None:
        while (event := await self._events.get()) is not None:
            if self._event_handler:
                try:
                    await self._event_handler(event)
                except Exception:
                    # 观察回调失败必须显式中止，不能在失去持久记录后继续产生实验。
                    self.state = "failed"
                    self._fail_pending(CodexError("Codex 事件记录回调失败。"))
                    if self.process and self.process.returncode is None:
                        self.process.terminate()
                    return
