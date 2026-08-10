"""`python -m thermoforge_mcp` 入口。

MCP 客户端把本进程当子进程拉起，**stdout 是协议通道**：任何多余的打印都会
破坏 JSON-RPC 帧。所以这里不打印任何东西，日志一律走 stderr。
"""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    # stderr 用 UTF-8：Windows 默认 cp1252，中文诊断会在日志里抛编码错误
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
