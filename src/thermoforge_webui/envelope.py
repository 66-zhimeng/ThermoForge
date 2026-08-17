"""工具信封在页面侧的读取助手。

信封响应体上限 32KB，超出会截断：列表被裁短，并在末尾追加一条形如
`... (truncated, total=241)` 的字符串标记，完整内容落 artifact
（见 `thermoforge_research.envelope`）。页面渲染必须容忍这条标记——
241 变量的数据集必然触发截断，直接 `.get` 会 AttributeError。
"""

from __future__ import annotations

from typing import Any


def entries(value: Any) -> list[dict[str, Any]]:
    """取列表里的记录项，丢掉截断标记等非记录元素。"""
    return [item for item in (value or []) if isinstance(item, dict)]


def is_truncated(value: Any) -> bool:
    """列表是否被截断（末尾带标记）。"""
    return bool(value) and any(isinstance(item, str) for item in value)
