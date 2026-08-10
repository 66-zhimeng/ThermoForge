"""ThermoForge MCP server：把工具注册表暴露给外部 Agent。

装了之后，Claude Code / Claude Desktop / 任何 MCP 客户端都能直接驱动
ThermoForge——查数据、做体检、跑实验、比模型，用的是**同一套 22 个工具与
同一份工件**，和网页副驾、`tf` CLI 完全一致，不存在第二条数据通路。

注册到 Claude Code（仓库根目录执行一次）::

    claude mcp add thermoforge -- <仓库>/.venv/Scripts/python -m thermoforge_mcp

之后在 Claude Code 里直接问「ThermoForge 最新实验怎么样」即可。

**为什么不暴露审批类工具**：`tf_preprocess_approve` 要求 `actor=human`
（I-49 的留痕语义——规则是谁批的必须查得到）。经 MCP 调用时 actor 只能记成
机器身份，那条留痕就失真了。所以审批一律留在网页控制台或
`tf --actor human preprocess approve`。
"""

from __future__ import annotations

from .server import build_server, main

__all__ = ["build_server", "main"]
