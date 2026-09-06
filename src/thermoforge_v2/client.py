"""统一任务客户端，按需启动独立本机服务；不在 MCP/网页进程中跑研究。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .contracts import V2Error

REPO_ROOT = Path(__file__).resolve().parents[2]


class V2Client:
    def __init__(self, research_root=None, vault_root=None, models_root=None, *, autostart=True):
        self.research_root = Path(research_root or os.environ.get("TF_RESEARCH_ROOT") or REPO_ROOT / "research").resolve()
        self.vault_root = Path(vault_root or os.environ.get("TF_VAULT_ROOT") or REPO_ROOT / "vault").resolve()
        self.models_root = Path(models_root or os.environ.get("TF_MODELS_ROOT") or REPO_ROOT / "models").resolve()
        self.root = self.research_root / "v2"
        self.autostart = autostart
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _descriptor(self):
        return json.loads((self.root / "service.json").read_text(encoding="utf-8"))

    def _request(self, descriptor, method, params, timeout=60):
        # 描述文件不可把本地 token 重定向到远端；地址始终固定 loopback。
        port = int(descriptor["port"])
        if not 1 <= port <= 65535:
            raise ValueError("无效的本机服务端口")
        request = urllib.request.Request(f"http://127.0.0.1:{port}/rpc",
            data=json.dumps({"method": method, "params": params}, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + descriptor["token"], "Content-Type": "application/json"}, method="POST")
        with self.opener.open(request, timeout=timeout) as response:
            body = json.loads(response.read())
        if not body.get("ok"):
            raise V2Error(body.get("code", "TFV2-SERVICE"), body.get("error", "后台服务错误"))
        return body["result"]

    def _alive(self):
        try:
            descriptor = self._descriptor()
            health = self._request(descriptor, "health", {}, timeout=1)
            if health.get("service") == "thermoforge-v2" and health.get("pid") == descriptor["pid"]:
                expected = {"research": str(self.research_root), "vault": str(self.vault_root), "models": str(self.models_root)}
                if health.get("roots") != expected:
                    raise V2Error("TFV2-ROOTS", "已有服务使用不同的 Vault/模型目录，请使用相同目录配置")
                return descriptor
        except V2Error:
            raise
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            pass
        return None

    def _ensure(self):
        descriptor = self._alive()
        if descriptor:
            return descriptor
        if not self.autostart:
            raise V2Error("TFV2-OFFLINE", "ThermoForge V2 后台服务未运行")
        self.root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "thermoforge_v2", "serve", "--research-root", str(self.research_root),
               "--vault-root", str(self.vault_root), "--models-root", str(self.models_root)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS} if os.name == "nt" else {"start_new_session": True}
        with (self.root / "service.log").open("ab") as log:
            subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             env=env, close_fds=True, **options)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            descriptor = self._alive()
            if descriptor:
                return descriptor
            time.sleep(0.2)
        raise V2Error("TFV2-START", f"后台服务未就绪，查看 {self.root / 'service.log'}")

    def _call(self, method, **params):
        return self._request(self._ensure(), method, params)

    def prepare(self, config=None):
        return self._call("prepare", config=config)

    def list_runs(self):
        return self._call("list")

    def get_run(self, run_id):
        return self._call("status", run_id=run_id)

    def events(self, run_id, after=0, limit=100):
        return self._call("events", run_id=run_id, after=after, limit=limit)

    def start(self, config, idempotency_key):
        return self._call("start", config=config, idempotency_key=idempotency_key)

    def control(self, run_id, action, expected_version=None, changes=None):
        return self._call("control", run_id=run_id, action=action, expected_version=expected_version, changes=changes)

    def get_report(self, run_id, track_id=None):
        return self._call("report", run_id=run_id, track_id=track_id)
