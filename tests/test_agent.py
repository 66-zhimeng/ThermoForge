"""PiAgent（内置研发 Agent）测试——不依赖真实 API。

- StubClient：脚本化 ChatResult（function calling 循环 / 多轮 / 错误透传）；
- mock HTTP server：`--check` 走真实 openai SDK 路径；
- 配置优先级、缺 key 指引、Schema 生成与白名单、审批 y/n 交互。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from thermoforge_agent.agent import PiAgent
from thermoforge_agent.client import ChatResult, ToolCallRequest
from thermoforge_agent.config import AgentConfig, config_guidance
from thermoforge_agent.schema import (
    HUMAN_APPROVAL_TOOL,
    build_tool_schemas,
    tool_schema,
)
from thermoforge_cli.main import main as cli_main
from thermoforge_research.tools import TOOL_REGISTRY, tf_preprocess_propose

from phase34_helpers import make_ctx
from test_preprocess import _ruleset_doc


def _config(**kw):
    return AgentConfig(api_key="sk-test", base_url="http://127.0.0.1:1/v1",
                       model="test-model", **kw)


class StubClient:
    """脚本化 client：每次 chat() 弹出下一个 ChatResult。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, on_content_delta=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "tools": tools})
        assert self.script, "stub 脚本耗尽"
        return self.script.pop(0)


def tool_call(name, arguments, call_id="call-1"):
    return ChatResult(
        content=None,
        tool_calls=[ToolCallRequest(call_id, name, arguments)],
        raw_message={"role": "assistant", "content": None,
                     "tool_calls": [{
                         "id": call_id, "type": "function",
                         "function": {"name": name,
                                      "arguments": json.dumps(arguments)}}]},
    )


def final(text):
    return ChatResult(content=text,
                      raw_message={"role": "assistant", "content": text})


def _tool_messages(agent):
    return [m for m in agent.messages if m.get("role") == "tool"]


# ---------------------------------------------------------------- 对话循环


def test_function_calling_loop(tmp_path):
    ctx, ref = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([
        tool_call("tf_dataset_list", {}),
        final("当前共有 1 个数据集：WX 合成数据。"),
    ])
    agent = PiAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")
    answer = agent.ask("列出所有数据集")
    assert "1 个数据集" in answer
    tools = _tool_messages(agent)
    assert len(tools) == 1
    envelope = json.loads(tools[0]["content"])
    assert envelope["ok"] is True and envelope["tool"] == "tf_dataset_list"
    assert envelope["summary"]["count"] == 1
    # 第二轮 chat 的上下文里带着工具结果（信封原样回传）
    second = stub.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second)


def test_multi_turn_history(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([final("回答一"), final("回答二")])
    agent = PiAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")
    agent.ask("第一个问题")
    agent.ask("第二个问题")
    second_call = stub.calls[1]["messages"]
    user_texts = [m["content"] for m in second_call if m["role"] == "user"]
    assert user_texts == ["第一个问题", "第二个问题"]


def test_error_envelope_passthrough(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([
        tool_call("tf_dataset_get", {"ref": "NOPE@rev_9999"}),
        final("数据版本不存在（TFV-703）。"),
    ])
    agent = PiAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")
    answer = agent.ask("查看 NOPE@rev_9999")
    envelope = json.loads(_tool_messages(agent)[0]["content"])
    assert envelope["ok"] is False
    assert envelope["diagnostics"][0]["code"] == "TFV-703"
    assert "TFV-703" in answer  # 模型拿到错误码后可如实转述


def test_session_log_jsonl(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([tool_call("tf_dataset_list", {}), final("done")])
    agent = PiAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research" / "agent_sessions")
    agent.ask("hi")
    logs = list((tmp_path / "research" / "agent_sessions").glob("*.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in
              logs[0].read_text(encoding="utf-8").splitlines()]
    types = [e["type"] for e in events]
    assert "message" in types and "tool_result" in types
    tool_event = next(e for e in events if e["type"] == "tool_result")
    assert tool_event["tool"] == "tf_dataset_list" and tool_event["ok"] is True


# ---------------------------------------------------------------- Schema 生成


def test_schema_generation_and_whitelist():
    schemas, dispatch = build_tool_schemas()  # 默认排除 approve
    names = {s["function"]["name"] for s in schemas}
    assert "tf_preprocess_approve" not in names  # human-only 不直接暴露
    assert HUMAN_APPROVAL_TOOL in names          # 审批元工具在
    assert names - {HUMAN_APPROVAL_TOOL} == set(TOOL_REGISTRY) - {
        "tf_preprocess_approve"}
    assert "tf_preprocess_approve" not in dispatch

    sample = tool_schema("tf_dataset_sample",
                         TOOL_REGISTRY["tf_dataset_sample"])
    params = sample["function"]["parameters"]
    assert params["required"] == ["ref"]
    assert params["properties"]["n"] == {"type": "integer"}
    assert params["properties"]["variable_ids"]["type"] == "array"
    assert sample["function"]["description"]  # docstring 首段


# ---------------------------------------------------------------- 审批交互


def test_approval_yes_executes_as_human(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    tf_preprocess_propose(ctx, _ruleset_doc())  # 先有 proposed 规则集
    stub = StubClient([
        tool_call(HUMAN_APPROVAL_TOOL, {
            "tool": "tf_preprocess_approve",
            "arguments": {"ruleset_id": "WX_FIX"},
            "reason": "规则已复核"}),
        final("已审批。"),
    ])
    approvals = []
    agent = PiAgent(_config(), ctx, client=stub,
                    approval_handler=lambda t, a, r: approvals.append(t) or True,
                    session_dir=tmp_path / "research")
    agent.ask("帮我审批 WX_FIX")
    assert approvals == ["tf_preprocess_approve"]
    envelope = json.loads(_tool_messages(agent)[0]["content"])
    assert envelope["ok"] is True and envelope["status"] == "APPROVED"
    # 以 actor=human 执行并留痕
    rule = envelope["summary"]["rules"][0]
    assert rule["approvals"][0]["actor"] == "human"


def test_approval_no_declines(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    tf_preprocess_propose(ctx, _ruleset_doc())
    stub = StubClient([
        tool_call(HUMAN_APPROVAL_TOOL, {
            "tool": "tf_preprocess_approve",
            "arguments": {"ruleset_id": "WX_FIX"},
            "reason": "试试"}),
        final("用户未批准，未执行。"),
    ])
    agent = PiAgent(_config(), ctx, client=stub,
                    approval_handler=lambda t, a, r: False,
                    session_dir=tmp_path / "research")
    agent.ask("审批 WX_FIX")
    envelope = json.loads(_tool_messages(agent)[0]["content"])
    assert envelope["ok"] is False
    assert envelope["status"] == "DECLINED_BY_USER"
    # 未执行：规则仍为 proposed
    rs = ctx.preprocess_store.load("WX_FIX")
    assert rs.rules[0].status == "proposed"


# ---------------------------------------------------------------- 配置


def test_config_env_beats_file(tmp_path, monkeypatch):
    cfg_file = tmp_path / "agent.toml"
    cfg_file.write_text(
        'api_key = "sk-file"\nbase_url = "http://file/v1"\nmodel = "m-file"\n',
        encoding="utf-8", newline="\n")
    monkeypatch.setenv("TF_AGENT_API_KEY", "sk-env")
    monkeypatch.setenv("TF_AGENT_BASE_URL", "http://env/v1")
    config = AgentConfig.load(config_path=cfg_file)
    assert config.api_key == "sk-env"          # 环境变量 > 配置文件
    assert config.base_url == "http://env/v1"
    assert config.model == "m-file"            # 文件兜底
    # CLI 显式参数最高
    config = AgentConfig.load(config_path=cfg_file, model="m-cli")
    assert config.model == "m-cli"


def test_config_api_key_env_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moonshot")
    monkeypatch.delenv("TF_AGENT_API_KEY", raising=False)
    config = AgentConfig.load(api_key_env="MOONSHOT_API_KEY",
                              config_path=tmp_path / "none.toml")
    assert config.api_key == "sk-moonshot"


def test_config_missing_key_guidance(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TF_AGENT_API_KEY", raising=False)
    # CLI 分支不传 config_path，会落到模块默认的 pi/agent.toml；使用者一旦
    # 真的配了密钥，这个「无 key」用例就会被真实配置污染。改指临时路径。
    monkeypatch.setattr("thermoforge_agent.config.CONFIG_PATH",
                        tmp_path / "none.toml")
    config = AgentConfig.load(config_path=tmp_path / "none.toml")
    assert config is None
    text = config_guidance()
    assert "platform.moonshot.cn" in text and "pi/agent.toml" in text
    code = cli_main(["--vault-root", str(tmp_path / "v"),
                     "--research-root", str(tmp_path / "r"),
                     "--models-root", str(tmp_path / "m"),
                     "agent"], )
    assert code == 2
    err = capsys.readouterr().err
    assert "未找到 API key" in err and "agent --check" in err


# ---------------------------------------------------------------- --check（mock HTTP）


class _MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps({
            "id": "chatcmpl-mock", "object": "chat.completion",
            "created": 0, "model": "mock-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "pong"}}],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_agent_check_ok(tmp_path, monkeypatch, capsys, mock_server):
    monkeypatch.setenv("TF_AGENT_API_KEY", "sk-test")
    monkeypatch.setenv("TF_AGENT_BASE_URL", mock_server)
    code = cli_main(["--vault-root", str(tmp_path / "v"),
                     "--research-root", str(tmp_path / "r"),
                     "--models-root", str(tmp_path / "m"),
                     "agent", "--check"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["response_id"] == "chatcmpl-mock"


def test_agent_check_connection_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TF_AGENT_API_KEY", "sk-test")
    monkeypatch.setenv("TF_AGENT_BASE_URL", "http://127.0.0.1:9/v1")
    code = cli_main(["--vault-root", str(tmp_path / "v"),
                     "--research-root", str(tmp_path / "r"),
                     "--models-root", str(tmp_path / "m"),
                     "agent", "--check"])
    assert code == 2
    assert "连通性检查失败" in capsys.readouterr().err


def test_agent_single_message_with_stub(tmp_path, monkeypatch, capsys):
    """CLI 单轮路径：monkeypatch ChatClient → stub（验证接线）。"""
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([final("单轮回答")])
    monkeypatch.setenv("TF_AGENT_API_KEY", "sk-test")
    monkeypatch.setattr("thermoforge_agent.agent.ChatClient",
                        lambda config: stub)
    code = cli_main(["--vault-root", str(tmp_path / "vault"),
                     "--research-root", str(tmp_path / "research"),
                     "--models-root", str(tmp_path / "models"),
                     "agent", "-m", "你好"])
    assert code == 0
    assert "单轮回答" in capsys.readouterr().out
