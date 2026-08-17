"""ThermoForge 内置研发 Agent（deepseek_harness）。

- `config.AgentConfig`：环境变量 > pi/agent.toml 的配置加载与指引。
- `client.ChatClient`：OpenAI 兼容 chat completions + function calling。
- `agent.deepseek_harness`：对话循环（工具执行、审批拦截、会话留痕）。
- `repl.run_repl`：终端 REPL（`tf agent`）。
"""

from .agent import deepseek_harness
from .config import AgentConfig, config_guidance

__all__ = ["AgentConfig", "deepseek_harness", "config_guidance"]
