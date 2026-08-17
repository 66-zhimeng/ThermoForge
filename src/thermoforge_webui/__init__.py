"""ThermoForge 本地 Web 控制台（工程师控制台 + 演示/报告产出）。

面向两类使用：日常在界面上看数据质量、发起 AI 深度研究、读实验结果；
以及把某次研究导出成能直接发给别人的报告（HTML / Markdown / PDF）。

启动（仓库根目录）::

    .venv/Scripts/python tools/webui.py       # 等价于 streamlit run 本包 app

设计边界（沿用旧控制台的安全口径）：

- **只绑 127.0.0.1**：这个界面能改密钥、能跑实验，不能让同网段的人打开。
- 密钥写入 `harness/agent.toml`（已 gitignore），读回一律掩码。
- **Agent 不直接读原始数据**：所有能力经 `TOOL_REGISTRY` 的信封；界面
  自己读 parquet 只用于「把已完成实验的结果画出来」，不喂给模型。
- human-only 工具（预处理审批）在界面上以 `actor=human` 单独执行，
  保持 I-49 的留痕语义。
"""

from __future__ import annotations

__all__ = ["APP_TITLE", "DEFAULT_PORT"]

APP_TITLE = "ThermoForge 控制台"
DEFAULT_PORT = 8765
