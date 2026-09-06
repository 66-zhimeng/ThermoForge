"""独立后台服务：GUI、CLI 与 MCP 共用运行状态，客户端退出不终止研究。"""

from __future__ import annotations

import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import threading
from typing import Any

from thermoforge_research.tools import ToolContext

from .contracts import RunConfig, V2Error
from .engine import ResearchEngine
from .research_tools import prepare_protocol
from .store import RunStore, encode


class ServiceLock:
    """整个独立服务持有操作系统锁，退出即释放，PID 重用不影响唯一归属。"""

    def __init__(self, path):
        self.path, self.fp = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.fp.write(b"0")
            self.fp.flush()
        self.fp.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fp.close()
            raise V2Error("TFV2-SERVICE-OWNER", "该研究目录已有后台服务")
        return self

    def __exit__(self, *args):
        if self.fp:
            self.fp.close()


def _windows_child_lifetime():
    """服务退出时由 Windows 回收其研究子进程，避免崩溃留下无主生成/训练。"""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
            "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "无法设置后台子进程生命周期")
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "无法归属后台子进程生命周期")
    # 不主动关闭该句柄（会连本服务一起终止）；OS 在服务退出时自动回收。
    return handle


class ResearchService:
    def __init__(self, ctx, *, engine_factory=ResearchEngine):
        self.ctx = ctx
        self.store = RunStore(ctx.research_root / "v2")
        self.loop = asyncio.new_event_loop()
        self.engine = engine_factory(self.store, ctx)
        self.thread = threading.Thread(target=self.loop.run_forever, name="v2-scheduler", daemon=True)
        self._prepare_lock = threading.Lock()
        self._report_lock = threading.Lock()

    def start(self):
        self.store.clear_leases()
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.engine.recover(), self.loop).result(60)

    def close(self):
        if self.thread.is_alive():
            asyncio.run_coroutine_threadsafe(self.engine.close(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)
        self.loop.close()

    def fresh_context(self):
        return ToolContext(vault_root=self.ctx.vault_root, research_root=self.ctx.research_root,
                           models_root=self.ctx.models_root, tfom_registry=self.ctx.tfom_registry, actor="v2-service")

    def prepare(self, config=None):
        from .codex import resolve_codex_command, VALIDATED_CODEX_VERSION
        ctx = self.fresh_context()
        errors = []
        try:
            command = resolve_codex_command()
            backend = {"available": True, "command": command, "required_version": VALIDATED_CODEX_VERSION,
                       "authentication": "validated_when_starting_instance"}
        except Exception as exc:
            backend = {"available": False, "error": str(exc)}
            errors.append(str(exc))
        goals = ctx.ledger.list_goals()
        datasets = []
        for dataset in ctx.vault.list_datasets():
            datasets.extend({"dataset_id": dataset["dataset_id"], **r} for r in dataset["revisions"])
        result = {"available": backend["available"], "backend": backend, "goals": goals,
                  "datasets": datasets, "errors": errors, "defaults": {
                      k: field.default for k, field in RunConfig.model_fields.items()
                      if not field.is_required()}, "missing": []}
        if config is not None:
            result["missing"] = [k for k in ("goal_id", "dataset_ref") if not config.get(k)]
            if not result["missing"]:
                try:
                    parsed = RunConfig.model_validate(config).model_dump(mode="json")
                    protocol = prepare_protocol(ctx, parsed)
                    result.update(config=parsed, protocol=protocol)
                except Exception as exc:
                    result["errors"].append(str(exc))
        result["ready"] = not result["errors"] and not result["missing"] and config is not None
        return result

    def start_run(self, config, idempotency_key):
        parsed = RunConfig.model_validate(config).model_dump(mode="json")
        with self._prepare_lock:
            run = self.store.find_request(parsed, idempotency_key)
            if run is None:
                protocol = prepare_protocol(self.fresh_context(), parsed)
                run = self.store.create_run(parsed, protocol, idempotency_key)
        if run["status"] == "queued":
            self.loop.call_soon_threadsafe(self.engine.launch, run["run_id"])
        return run

    def control(self, run_id, action, expected_version=None, changes=None):
        result = self.store.control(run_id, action, expected_version, changes)
        if result["status"] == "queued":
            self.loop.call_soon_threadsafe(self.engine.launch, run_id)
        return result

    def get_report(self, run_id, track_id=None):
        from .reports import build_report, render_html, render_markdown
        snapshot = self.store.snapshot(run_id)
        # 外部报告入口可以读取最终留出，内部六个 Codex 永远不调用本入口。
        if snapshot["run"]["status"] in {"completed", "cancelled"}:
            with self._report_lock:
                snapshot["final_evaluation"] = self._final_evaluation(snapshot)
        report = build_report(snapshot, track_id=track_id)
        directory = self.store.root / "runs" / run_id / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        stem = "team" if track_id is None else track_id
        if track_id is not None and track_id not in {t["track_id"] for t in snapshot["tracks"]}:
            raise V2Error("TFV2-NOT-FOUND", "找不到报告轨迹")
        for ext, content in (("json", encode(report)), ("md", render_markdown(report)), ("html", render_html(report))):
            target = directory / f"{stem}.{ext}"
            temporary = target.with_name(target.name + "." + secrets.token_hex(4) + ".tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            os.replace(temporary, target)
        report["artifacts"] = [{"format": ext, "path": str(directory / f"{stem}.{ext}")} for ext in ("json", "md", "html")]
        return report

    def _final_evaluation(self, snapshot):
        from .strategy import compare_jobs
        rid = snapshot["run"]["run_id"]
        existing = self.store.records(rid, "finalizations")
        if existing:
            return existing[0]
        ranked = compare_jobs(snapshot["jobs"], snapshot["run"]["protocol"]["fingerprint"], 1)["top_k"]
        if not ranked:
            return {"status": "unavailable", "reason": "没有可据验证指标选定的已完成实验"}
        selected = next(j for j in snapshot["jobs"] if j["id"] == ranked[0]["job_id"])
        track = next(t for t in snapshot["tracks"] if t["track_id"] == selected["track_id"])
        exp_id = selected.get("experiment_id") or (selected.get("result") or {}).get("experiment_id")
        if not exp_id or not exp_id.startswith("EXP-") or not exp_id[4:].isdigit():
            return {"status": "unavailable", "reason": "所选实验工件缺失"}
        path = Path(track["research_root"]) / "experiments" / exp_id / "report.json"
        if not path.is_file():
            return {"status": "unavailable", "reason": "所选实验完整报告不存在"}
        full = json.loads(path.read_text(encoding="utf-8"))
        return self.store.add_record(rid, "finalizations", {
            "status": "evaluated", "job_id": selected["id"], "track_id_selected": selected["track_id"],
            "experiment_id": exp_id, "selection_reason": "先按冻结协议的 validate CVRMSE 选定，再读取留出结果。",
            "selection_metric": ranked[0]["CVRMSE"], "protocol_fingerprint": snapshot["run"]["protocol"]["fingerprint"],
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "surfaces": {k: v for k, v in ((full.get("metrics") or {}).get("surfaces") or {}).items() if k in {"A", "B", "C"}},
            "physics": full.get("physics"), "feedback_to_agents": False})

    def dispatch(self, method, params):
        if method == "health":
            return {"service": "thermoforge-v2", "pid": os.getpid(), "version": "2.0.0",
                    "roots": {"research": str(self.ctx.research_root), "vault": str(self.ctx.vault_root), "models": str(self.ctx.models_root)}}
        if method == "prepare":
            return self.prepare(**params)
        if method == "start":
            return self.start_run(**params)
        if method == "list":
            return self.store.list_runs()
        if method == "status":
            return self.store.snapshot(**params)
        if method == "events":
            return self.store.events(**params)
        if method == "control":
            return self.control(**params)
        if method == "report":
            return self.get_report(**params)
        raise V2Error("TFV2-INPUT", "未知服务方法")


def serve(ctx):
    root = ctx.research_root / "v2"
    root.mkdir(parents=True, exist_ok=True)
    with ServiceLock(root / "service.lock"):
        job_handle = _windows_child_lifetime()
        token = secrets.token_urlsafe(32)
        service = ResearchService(ctx)
        service.start()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # 协议日志只记录业务事件，不打印本地认证 token。

            def do_POST(self):
                if self.path != "/rpc" or self.headers.get("Origin") or not secrets.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + token):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024 * 1024:
                        raise V2Error("TFV2-INPUT", "请求大小不合法")
                    request = json.loads(self.rfile.read(length))
                    result = service.dispatch(request["method"], request.get("params") or {})
                    body = {"ok": True, "result": result}
                except Exception as exc:
                    body = {"ok": False, "code": getattr(exc, "code", "TFV2-REQUEST"), "error": str(exc)[:8000]}
                encoded = encode(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        descriptor = {"pid": os.getpid(), "port": server.server_address[1], "token": token}
        path = root / "service.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encode(descriptor), encoding="utf-8", newline="\n")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        def shutdown(*args):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            service.close()
            server.server_close()
            path.unlink(missing_ok=True)
        return job_handle
