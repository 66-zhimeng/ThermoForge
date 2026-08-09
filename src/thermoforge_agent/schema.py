"""从 TOOL_REGISTRY 自动生成 function calling 的 JSON Schema。

参数名/类型来自函数签名（`typing.get_type_hints` 解析），说明来自
docstring 首段。`ctx`（ToolContext）参数不暴露给模型。
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from thermoforge_research.tools import TOOL_REGISTRY

# 审批元工具：human-only 工具不直接暴露，模型通过它发起审批请求，
# REPL 弹确认后以 actor=human 执行（I-49 审批链路的对话层落地）
HUMAN_APPROVAL_TOOL = "tf_human_approval"

HUMAN_APPROVAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": HUMAN_APPROVAL_TOOL,
        "description": (
            "请求人工审批执行一个受限工具（如 tf_preprocess_approve）。"
            "当任务需要审批类动作时调用本工具；系统会向用户弹确认，"
            "用户同意后才以 human 身份执行，结果照常以信封返回。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {"type": "string",
                         "description": "目标工具名，如 tf_preprocess_approve"},
                "arguments": {"type": "object",
                              "description": "目标工具的关键字参数"},
                "reason": {"type": "string",
                           "description": "为什么需要该审批（展示给用户）"},
            },
            "required": ["tool", "arguments", "reason"],
        },
    },
}


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _json_type(non_none[0]) if non_none else {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if origin in (list, Sequence) or annotation is list:
        item_args = get_args(annotation)
        item = _json_type(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, Mapping) or annotation is dict:
        return {"type": "object"}
    return {"type": "string"}  # str / Path / 其他标量


def tool_schema(name: str, fn: Callable) -> dict[str, Any]:
    """单个工具函数 → OpenAI function schema。"""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name == "ctx":
            continue
        properties[param_name] = _json_type(
            hints.get(param_name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object",
                           "properties": properties,
                           "required": required},
        },
    }


def build_tool_schemas(
    exclude: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Callable]]:
    """(schemas, dispatch)。排除表内的工具不暴露；附审批元工具 schema。

    `exclude=None` 时用默认排除表（DEFAULT_TOOLS_EXCLUDE）。
    """
    from .config import DEFAULT_TOOLS_EXCLUDE

    excluded = set(DEFAULT_TOOLS_EXCLUDE if exclude is None else exclude)
    schemas: list[dict[str, Any]] = []
    dispatch: dict[str, Callable] = {}
    for name, fn in sorted(TOOL_REGISTRY.items()):
        if name in excluded:
            continue
        schemas.append(tool_schema(name, fn))
        dispatch[name] = fn
    if excluded:
        schemas.append(HUMAN_APPROVAL_SCHEMA)
    return schemas, dispatch
