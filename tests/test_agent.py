"""HarnessAgent（内置研发 Agent）测试——不依赖真实 API。

- StubClient：脚本化 ChatResult（function calling 循环 / 多轮 / 错误透传）；
- mock HTTP server：`--check` 走真实 openai SDK 路径；
- 配置优先级、缺 key 指引、Schema 生成与白名单、审批 y/n 交互。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from thermoforge_agent.agent import HarnessAgent
from thermoforge_agent.client import (
    ChatResult,
    ToolCallRequest,
    _is_local_endpoint,
)
from thermoforge_agent.config import AgentConfig, config_guidance
from thermoforge_agent.schema import (
    HUMAN_APPROVAL_TOOL,
    build_tool_schemas,
    tool_schema,
)
from thermoforge_cli.main import main as cli_main
from thermoforge_research.model_catalog import (
    DATA_ESTIMATORS,
    HYBRID_RESIDUALS,
    PHYSICS_MODELS,
)
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
    agent = HarnessAgent(_config(), ctx, client=stub,
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


def test_tool_round_limit_disables_tools_and_requests_final_answer(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([
        tool_call("tf_dataset_list", {}, call_id="call-1"),
        tool_call("tf_dataset_list", {}, call_id="call-2"),
        final("工具预算已用完；根据现有结果，当前有一个数据集。"),
    ])
    agent = HarnessAgent(_config(max_tool_rounds=2), ctx, client=stub,
                    session_dir=tmp_path / "research")

    answer = agent.ask("查清楚数据集")

    assert "当前有一个数据集" in answer
    assert len(_tool_messages(agent)) == 2
    assert stub.calls[2]["tools"] is None
    assert stub.calls[2]["messages"][-1]["role"] == "system"
    assert "不要再调用任何工具" in stub.calls[2]["messages"][-1]["content"]


def test_tool_round_limit_closes_unexpected_finalizer_tool_call(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([
        tool_call("tf_dataset_list", {}, call_id="call-1"),
        tool_call("tf_experiment_run", {"experiment_id": "EXP-9999"},
                  call_id="call-2"),
    ])
    agent = HarnessAgent(_config(max_tool_rounds=1), ctx, client=stub,
                    session_dir=tmp_path / "research")

    answer = agent.ask("一直调用工具")

    assert "请继续追问" in answer
    limit_result = json.loads(_tool_messages(agent)[-1]["content"])
    assert limit_result["status"] == "LIMIT_REACHED"
    assert _tool_messages(agent)[-1]["tool_call_id"] == "call-2"
    # Even a non-compliant endpoint leaves protocol-complete history.
    assert [message["role"] for message in agent.messages[-3:]] == [
        "assistant", "tool", "assistant",
    ]


def test_multiple_tool_calls_all_receive_matching_results(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    calls = [
        ToolCallRequest("call-1", "tf_dataset_list", {}),
        ToolCallRequest("call-2", "tf_dataset_list", {}),
    ]
    stub = StubClient([
        # Deliberately omit raw tool_calls: HarnessAgent must reconstruct them from
        # the normalized requests before the next Chat Completions call.
        ChatResult(content=None, tool_calls=calls,
                   raw_message={"role": "assistant", "content": None}),
        final("done"),
    ])
    agent = HarnessAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")

    assert agent.ask("list twice") == "done"
    messages = stub.calls[1]["messages"]
    assistant_index = next(
        i for i, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls"))
    assistant = messages[assistant_index]
    tool_messages = messages[assistant_index + 1:assistant_index + 3]
    assert [call["id"] for call in assistant["tool_calls"]] == [
        "call-1", "call-2",
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-1", "call-2",
    ]


def test_unexpected_tool_exception_is_returned_as_tool_result(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([tool_call("explode", {}), final("recovered")])
    agent = HarnessAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")

    def explode(_ctx):
        raise RuntimeError("boom")

    agent.dispatch["explode"] = explode
    assert agent.ask("trigger") == "recovered"
    tool_message = _tool_messages(agent)[0]
    envelope = json.loads(tool_message["content"])
    assert envelope["ok"] is False
    assert "RuntimeError: boom" in envelope["summary"]["error"]
    assert tool_message["tool_call_id"] == "call-1"


def test_next_question_repairs_orphaned_tool_call_history(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([final("history repaired")])
    agent = HarnessAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")
    agent.messages.extend([
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"id": "call-1", "type": "function",
                 "function": {"name": "one", "arguments": "{}"}},
                {"id": "call-2", "type": "function",
                 "function": {"name": "two", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        # This user message was appended by the turn that received HTTP 400.
        {"role": "user", "content": "previous retry"},
    ])

    assert agent.ask("retry now") == "history repaired"
    messages = stub.calls[0]["messages"]
    assistant_index = next(
        i for i, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls"))
    repaired_tools = messages[assistant_index + 1:assistant_index + 3]
    assert [message["tool_call_id"] for message in repaired_tools] == [
        "call-1", "call-2",
    ]
    repaired_envelope = json.loads(repaired_tools[1]["content"])
    assert repaired_envelope["ok"] is False
    assert messages[assistant_index + 3]["content"] == "previous retry"


def test_multi_turn_history(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([final("回答一"), final("回答二")])
    agent = HarnessAgent(_config(), ctx, client=stub,
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
    agent = HarnessAgent(_config(), ctx, client=stub,
                    session_dir=tmp_path / "research")
    answer = agent.ask("查看 NOPE@rev_9999")
    envelope = json.loads(_tool_messages(agent)[0]["content"])
    assert envelope["ok"] is False
    assert envelope["diagnostics"][0]["code"] == "TFV-703"
    assert "TFV-703" in answer  # 模型拿到错误码后可如实转述


def test_session_log_jsonl(tmp_path):
    ctx, _ = make_ctx(tmp_path, n_steps=100)
    stub = StubClient([tool_call("tf_dataset_list", {}), final("done")])
    agent = HarnessAgent(_config(), ctx, client=stub,
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


def test_experiment_plan_schema_exposes_contract_and_model_catalog():
    schema = tool_schema(
        "tf_experiment_plan", TOOL_REGISTRY["tf_experiment_plan"])
    definition = schema["function"]["parameters"]["properties"]["definition"]

    assert "experiment_id" not in definition["properties"]
    assert set(definition["required"]) >= {
        "goal_id", "hypothesis_id", "dataset_view", "model", "target",
        "validation", "metrics", "runtime",
    }
    model = definition["properties"]["model"]
    # 闭集来自 model_catalog（唯一真源），这里不再重抄一份名单 ——
    # 抄一份就等于允许它与执行侧分叉，那正是本测试要防的事
    assert model["properties"]["estimator"]["enum"] == list(DATA_ESTIMATORS)
    assert model["properties"]["physics"]["enum"] == list(PHYSICS_MODELS)
    assert model["properties"]["residual"]["enum"] == list(HYBRID_RESIDUALS)
    temporal = definition["properties"]["validation"]["properties"][
        "temporal_split"]
    assert temporal["required"] == ["train", "validate", "test"]


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
    agent = HarnessAgent(_config(), ctx, client=stub,
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
    agent = HarnessAgent(_config(), ctx, client=stub,
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
    # CLI 分支不传 config_path，会落到模块默认的 harness/agent.toml；使用者一旦
    # 真的配了密钥，这个「无 key」用例就会被真实配置污染。改指临时路径。
    monkeypatch.setattr("thermoforge_agent.config.CONFIG_PATH",
                        tmp_path / "none.toml")
    config = AgentConfig.load(config_path=tmp_path / "none.toml")
    assert config is None
    text = config_guidance()
    assert "platform.moonshot.cn" in text and "harness/agent.toml" in text
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


@pytest.mark.parametrize("url, expected", [
    ("http://127.0.0.1:8000/v1", True),
    ("http://localhost:8000/v1", True),
    ("http://[::1]:8000/v1", True),
    ("https://api.deepseek.com", False),
    ("http://192.168.1.10:8000/v1", False),
    ("", False),
])
def test_is_local_endpoint(url, expected):
    assert _is_local_endpoint(url) is expected


def test_agent_check_ignores_proxy_for_local_endpoint(
        tmp_path, monkeypatch, capsys, mock_server):
    """本地端点必须直连：系统/环境代理不得劫持 127.0.0.1（否则代理回 502）。"""
    monkeypatch.setenv("TF_AGENT_API_KEY", "sk-test")
    monkeypatch.setenv("TF_AGENT_BASE_URL", mock_server)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9/")   # 必定不可用
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9/")
    code = cli_main(["--vault-root", str(tmp_path / "v"),
                     "--research-root", str(tmp_path / "r"),
                     "--models-root", str(tmp_path / "m"),
                     "agent", "--check"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


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


# ---------------------------------------------------------------- 技能装载

def test_skills_are_bound_and_fingerprinted():
    """技能装到研究位点、不装到副驾，且计入指纹。"""
    from thermoforge_agent import prompts

    expected = ["measurement-forensics", "system-identification"]
    assert set(expected) <= set(prompts.available_skills())

    bound = prompts.skill_registry()
    assert bound["cli"] == expected          # 取证在前、建模在后
    assert bound["planner"] == expected
    assert bound["mcp"] == expected
    assert bound["copilot"] == []      # 副驾负责导航，不指挥建模升级

    cli = prompts.load("cli")
    assert "系统辨识" in cli
    assert "现场数据取证" in cli        # 两份技能都拼进去了
    assert "gordon_ng" in cli
    assert "朴素持续" in cli            # §1 的核心教训在位
    # 副驾没装技能：直接比文件原文，不用「某个词不出现」当代理判据
    # （提示词里正常提到「系统辨识族」就会误伤那种写法）
    copilot_file = prompts.normalize(
        prompts.prompt_path("copilot").read_text(encoding="utf-8"))
    assert prompts.load("copilot") == copilot_file

    # 指纹覆盖技能内容：位点摘要含技能，故两位点摘要不同
    assert prompts.digest("cli") != prompts.digest("copilot")
    assert len(prompts.fingerprint()) == 64


def test_missing_skill_degrades_without_raising(monkeypatch):
    """技能文件不存在时退化成「没装这项本事」，不炸。"""
    from thermoforge_agent import prompts

    monkeypatch.setitem(prompts.SKILL_BINDINGS, "cli", ("no-such-skill",))
    text = prompts.load("cli")
    assert text                        # 仍返回位点提示词本体
    assert "no-such-skill" not in text
