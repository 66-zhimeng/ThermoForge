"""真实隐藏后台服务的连接、生命周期与 MCP/CLI 接口（不调用模型）。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from thermoforge_v2.client import V2Client
from thermoforge_v2.contracts import V2Error


@pytest.fixture
def running_client(tmp_path):
    client = V2Client(research_root=tmp_path / "research", vault_root=tmp_path / "vault", models_root=tmp_path / "models")
    assert client.list_runs() == []
    descriptor = client._descriptor()
    yield client
    # 仅终止此测试创建且经认证健康检查匹配的服务，不扫描或终止用户进程。
    current = client._alive()
    if current and current["pid"] == descriptor["pid"]:
        os.kill(descriptor["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and client._alive():
            time.sleep(0.1)


def test_client_exit_and_cli_reconnect_same_backend(running_client):
    client = running_client
    pid = client._descriptor()["pid"]
    result = subprocess.run([sys.executable, "-m", "thermoforge_v2", "list",
        "--research-root", str(client.research_root), "--vault-root", str(client.vault_root),
        "--models-root", str(client.models_root)], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"] == []
    assert client._alive()["pid"] == pid
    assert V2Client(client.research_root, client.vault_root, client.models_root).list_runs() == []
    assert client._descriptor()["pid"] == pid


def test_service_requires_local_token_and_rejects_browser_origin(running_client):
    client = running_client
    descriptor = client._descriptor()
    url = f"http://127.0.0.1:{descriptor['port']}/rpc"
    body = b'{"method":"health"}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for headers in ({}, {"Authorization": "Bearer " + descriptor["token"], "Origin": "https://example.com"}):
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(urllib.request.Request(url, data=body, headers=headers), timeout=5)
        assert error.value.code == 403


def test_wrong_roots_do_not_silently_use_another_vault(running_client, tmp_path):
    client = running_client
    with pytest.raises(V2Error, match="不同"):
        V2Client(client.research_root, tmp_path / "other-vault", client.models_root).list_runs()


def test_service_reports_domain_error_without_losing_connection(running_client):
    client = running_client
    with pytest.raises(V2Error, match="找不到"):
        client.get_run("RUN-does-not-exist")
    assert client.list_runs() == []


def test_dedicated_and_existing_mcp_expose_same_control_tools():
    from thermoforge_v2.mcp import build_server, CONTROL_TOOLS
    from thermoforge_mcp.server import build_server as existing_server
    expected = {fn.__name__ for fn in CONTROL_TOOLS}
    assert len(expected) == 7
    assert {tool.name for tool in asyncio.run(build_server().list_tools())} == expected
    assert expected <= {tool.name for tool in asyncio.run(existing_server().list_tools())}


def test_cli_preserves_ok_false_contract(monkeypatch, capsys):
    from thermoforge_cli.main import main
    monkeypatch.setattr(V2Client, "list_runs", lambda self: (_ for _ in ()).throw(V2Error("test", "offline")))
    assert main(["v2", "list"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_tf_v2_help_reaches_v2_command_parser(capsys):
    from thermoforge_cli.main import main
    with pytest.raises(SystemExit) as result:
        main(["v2", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "tf v2" in output and "prepare" in output and "control" in output


def test_service_crash_releases_owner_lock(running_client):
    client = running_client
    old = client._descriptor()["pid"]
    os.kill(old, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and client._alive():
        time.sleep(0.1)
    assert client.list_runs() == []
    assert client._descriptor()["pid"] != old
    # fixture 原进程已退出，单独收尾本次恢复创建的测试进程。
    recovered = client._alive()
    os.kill(recovered["pid"], signal.SIGTERM)
