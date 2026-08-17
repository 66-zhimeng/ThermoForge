"""ThermoForge 内置研发 Agent（HarnessAgent）。

- `config.AgentConfig`：环境变量 > harness/agent.toml 的配置加载与指引。
- `client.ChatClient`：OpenAI 兼容 chat completions + function calling。
- `agent.HarnessAgent`：对话循环（工具执行、审批拦截、会话留痕）。
- `repl.run_repl`：终端 REPL（`tf agent`）。
"""

from .agent import HarnessAgent
from .config import AgentConfig, config_guidance

__all__ = ["AgentConfig", "HarnessAgent", "config_guidance"]
