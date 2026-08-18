"""副驾与 MCP server 的测试。

副驾用 stub client 注入，不触网：验证的是「模型说要跳到某页」之后，
界面这一侧有没有正确记下意图、非法页面会不会被挡、需要人批的动作会不会
真的把后台线程卡住等人点。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from thermoforge_agent.client import ChatResult, ToolCallRequest
from thermoforge_agent.config import AgentConfig
from thermoforge_webui.services.copilot import UI_GOTO_TOOL, CopilotSession


class StubClient:
    """按脚本逐轮返回结果的假模型。"""

    def __init__(self, script: list[ChatResult]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []
        self.schemas: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, on_content_delta=None) -> ChatResult:
        self.calls.append(list(messages))
        self.schemas = list(tools or [])
        return self.script.pop(0)


def _config() -> AgentConfig:
    return AgentConfig(api_key="sk-test", base_url="http://localhost/v1",
                       model="stub")


def _wait_idle(session: CopilotSession, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while session.busy and time.time() < deadline:
        time.sleep(0.02)


def _tool_call(name: str, arguments: dict[str, Any]) -> ChatResult:
    return ChatResult(
        content=None,
        tool_calls=[ToolCallRequest(id="c1", name=name, arguments=arguments)],
        raw_message={"role": "assistant", "content": None})


# ---------------------------------------------------------------- 界面工具


def test_ui_goto_records_navigation_intent() -> None:
    session = CopilotSession(client=StubClient([
        _tool_call(UI_GOTO_TOOL, {"page": "results", "note": "看这次",
                                  "experiment_id": "EXP-0013"}),
        ChatResult(content="EXP-0013 最好，已经带你过去了。"),
    ]))
    session.ask("哪次实验最好", _config())
    _wait_idle(session)

    assert session.nav_intent is not None
    assert session.nav_intent["page"] == "results"
    assert session.nav_intent["selections"]["experiment_id"] == "EXP-0013"
    messages, _, state = session.snapshot()
    assert state == "idle"
    assert messages[-1].role == "assistant"
    assert "EXP-0013" in messages[-1].content


def test_ui_goto_rejects_unknown_page() -> None:
    """模型会编页面名。编了就得挡住，并把可选项回给它。"""
    session = CopilotSession(client=StubClient([
        _tool_call(UI_GOTO_TOOL, {"page": "dashboard", "note": "x"}),
        ChatResult(content="抱歉，换一页。"),
    ]))
    session.ask("随便问问", _config())
    _wait_idle(session)

    assert session.nav_intent is None
    # 失败信封回给了模型，里面带可选页面清单
    tool_message = session._agent.messages[-2]
    envelope = json.loads(tool_message["content"])
    assert envelope["ok"] is False
    assert "results" in envelope["summary"]["error"]


def test_ui_goto_schema_is_exposed_to_the_model() -> None:
    stub = StubClient([ChatResult(content="好的")])
    session = CopilotSession(client=stub)
    session.ask("你好", _config())
    _wait_idle(session)
    names = [schema["function"]["name"] for schema in stub.schemas]
    assert UI_GOTO_TOOL in names
    # 数据工具也还在——副驾既能查也能跳
    assert "tf_research_status" in names


def test_tool_calls_are_recorded_as_events() -> None:
    session = CopilotSession(client=StubClient([
        _tool_call("tf_dataset_list", {}),
        ChatResult(content="共 3 个数据集。"),
    ]))
    session.ask("有哪些数据集", _config())
    _wait_idle(session)
    _, events, _ = session.snapshot()
    kinds = [event.kind for event in events]
    assert "tool" in kinds and "answer" in kinds
    assert any(event.text == "tf_dataset_list" for event in events)


def test_turn_records_reasoning_and_per_call_usage() -> None:
    """一轮回答要带上思维链与逐次调用用量，供界面展开/画图。"""
    usage_a = {"prompt_tokens": 100, "completion_tokens": 10}
    usage_b = {"prompt_tokens": 150, "completion_tokens": 30}
    session = CopilotSession(client=StubClient([
        ChatResult(content=None,
                   tool_calls=[ToolCallRequest(id="c1", name="tf_dataset_list",
                                               arguments={})],
                   raw_message={"role": "assistant", "content": None},
                   reasoning="先查数据集", usage=usage_a, cost=0.001),
        ChatResult(content="共 3 个数据集。", reasoning="可以回答了",
                   usage=usage_b, cost=0.002),
    ]))
    session.ask("有哪些数据集", _config())
    _wait_idle(session)

    messages, _, _ = session.snapshot()
    answer = messages[-1]
    assert answer.role == "assistant"
    assert answer.reasoning == "可以回答了"  # 最终答复附带的思维链
    assert [u["usage"] for u in answer.usages] == [usage_a, usage_b]
    assert [u["cost"] for u in answer.usages] == [0.001, 0.002]
    # 轮询接口：本轮两次调用；reset 后清零
    assert len(session.current_usage()) == 2
    session.reset()
    assert session.current_usage() == []


def test_error_is_surfaced_not_swallowed() -> None:
    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("接口挂了")

    session = CopilotSession(client=Boom())
    session.ask("在吗", _config())
    _wait_idle(session)
    assert session.state == "error"
    assert "接口挂了" in (session.error or "")


def test_tool_round_limit_finishes_copilot_without_error() -> None:
    stub = StubClient([
        _tool_call("tf_dataset_list", {}),
        ChatResult(content="工具预算用完，先根据现有数据给出结论。"),
    ])
    config = AgentConfig(api_key="sk-test", base_url="http://localhost/v1",
                         model="stub", max_tool_rounds=1)
    session = CopilotSession(client=stub)

    session.ask("做一个很长的调查", config)
    _wait_idle(session)

    assert session.state == "idle"
    assert session.error is None
    assert "给出结论" in session.messages[-1].content


def test_tool_exception_does_not_poison_copilot_history() -> None:
    stub = StubClient([
        _tool_call(UI_GOTO_TOOL, {"page": "results"}),
        ChatResult(content="工具失败了，但对话还能继续。"),
    ])
    session = CopilotSession(client=stub)

    def broken_ui_tool(*args, **kwargs):
        raise RuntimeError("navigation failed")

    session._ui_goto = broken_ui_tool
    session.ask("带我去结果页", _config())
    _wait_idle(session)

    assert session.state == "idle"
    tool_message = session._agent.messages[-2]
    envelope = json.loads(tool_message["content"])
    assert envelope["ok"] is False
    assert "RuntimeError: navigation failed" in envelope["summary"]["error"]
    second_request = stub.calls[1]
    assistant_index = next(
        i for i, message in enumerate(second_request)
        if message.get("role") == "assistant" and message.get("tool_calls"))
    assert second_request[assistant_index + 1]["tool_call_id"] == "c1"


# ---------------------------------------------------------------- 审批闸门


def test_human_approval_blocks_until_decided() -> None:
    """需要人批的动作必须真的把那一轮卡住，而不是先跑了再问。"""
    session = CopilotSession()
    approved: list[bool] = []

    worker = threading.Thread(target=lambda: approved.append(
        session._on_approval("tf_preprocess_approve", {"ruleset_id": "X"},
                             "需要批准清洗规则")))
    worker.start()
    for _ in range(200):
        if session.pending_approval is not None:
            break
        time.sleep(0.01)
    assert session.pending_approval is not None
    assert session.state == "awaiting_approval"

    session.decide_approval(True)
    worker.join(timeout=5)
    assert approved == [True]
    assert session.pending_approval is None


def test_human_approval_rejection_returns_false() -> None:
    session = CopilotSession()
    decided: list[bool] = []
    worker = threading.Thread(target=lambda: decided.append(
        session._on_approval("tf_preprocess_approve", {}, "理由")))
    worker.start()
    for _ in range(200):
        if session.pending_approval is not None:
            break
        time.sleep(0.01)
    session.decide_approval(False)
    worker.join(timeout=5)
    assert decided == [False]


# ---------------------------------------------------------------- MCP


def test_mcp_exposes_registry_without_approval_tool() -> None:
    import asyncio

    from thermoforge_mcp.server import EXCLUDED_TOOLS, build_server

    tools = asyncio.run(build_server().list_tools())
    names = {tool.name for tool in tools}
    assert "tf_dataset_list" in names
    assert "tf_status" in names and "tf_experiment_report" in names
    # 审批要 actor=human，经 MCP 调用会让留痕失真，所以不暴露
    assert EXCLUDED_TOOLS.isdisjoint(names)


def test_mcp_tool_schema_hides_ctx_and_keeps_params() -> None:
    import asyncio

    from thermoforge_mcp.server import build_server

    tools = asyncio.run(build_server().list_tools())
    tool = next(t for t in tools if t.name == "tf_dataset_modelability")
    properties = tool.input_schema["properties"]
    assert "ctx" not in properties          # 控制面参数不给外部 Agent
    assert "ref" in properties              # 业务参数保留
    assert tool.input_schema["required"] == ["ref"]
    assert tool.description                 # 中文说明来自 docstring 首段


def test_mcp_wrapper_returns_envelope_json() -> None:
    from thermoforge_mcp.server import make_wrapper

    def fake_tool(ctx, ref: str) -> dict[str, Any]:
        """假工具。"""
        return {"ok": True, "tool": "fake", "summary": {"ref": ref,
                                                        "note": "中文不转义"}}

    wrapper = make_wrapper("fake_tool", fake_tool)
    payload = json.loads(wrapper(ref="D@rev_0001"))
    assert payload["summary"]["ref"] == "D@rev_0001"
    assert "中文不转义" in payload["summary"]["note"]


@pytest.mark.parametrize("key", ["overview", "data", "quality", "research",
                                 "results", "models", "report", "settings"])
def test_navigation_catalog_covers_every_page(key: str) -> None:
    """副驾靠这份清单决定往哪跳，漏一个它就永远跳不到那一页。"""
    from thermoforge_webui.navigation import PAGE_BY_KEY

    assert key in PAGE_BY_KEY
    assert PAGE_BY_KEY[key].purpose
