"""PiAgent 配置：环境变量 > 配置文件（pi/agent.toml，不入库）。

环境变量：

- `TF_AGENT_API_KEY`：API 密钥（也可用 `--api-key-env` 指定其他变量名，
  如 `MOONSHOT_API_KEY`）。
- `TF_AGENT_BASE_URL`：OpenAI 兼容端点（Moonshot/Kimi、OpenAI、DeepSeek
  等均可）。
- `TF_AGENT_MODEL`：模型名。

配置文件 `pi/agent.toml`（已 gitignore；模板见 pi/agent.example.toml）。
CLI 显式参数（--model/--base-url）优先级最高。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2-0905-preview"

# human-only 工具：不直接暴露给模型，经 tf_human_approval 弹确认执行
DEFAULT_TOOLS_EXCLUDE = ("tf_preprocess_approve",)

CONFIG_PATH = Path("pi/agent.toml")


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    stream: bool = False
    tools_exclude: tuple[str, ...] = DEFAULT_TOOLS_EXCLUDE
    max_tool_rounds: int = 16  # 单轮提问允许的最大工具调用轮数（防失控循环）
    # openai SDK 默认超时 600s、重试 2 次：端点不通时要等十分钟才报错，
    # 界面上和「卡死」无法区分。收紧到可感知的量级。
    timeout_seconds: float = 60.0
    max_retries: int = 1

    @classmethod
    def load(
        cls,
        *,
        api_key_env: str = "TF_AGENT_API_KEY",
        base_url: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        config_path: str | Path | None = None,
    ) -> "AgentConfig | None":
        """加载配置；无 API key 返回 None（调用方给配置指引）。"""
        path = Path(config_path) if config_path else CONFIG_PATH
        file_doc: dict = {}
        if path.exists():
            with open(path, "rb") as fp:
                file_doc = tomllib.load(fp)
        tools_doc = file_doc.get("tools") or {}
        exclude = tuple(tools_doc.get("exclude") or DEFAULT_TOOLS_EXCLUDE)
        api_key = os.environ.get(api_key_env) or file_doc.get("api_key")
        if not api_key:
            return None
        return cls(
            api_key=str(api_key),
            base_url=(base_url or os.environ.get("TF_AGENT_BASE_URL")
                      or file_doc.get("base_url") or DEFAULT_BASE_URL),
            model=(model or os.environ.get("TF_AGENT_MODEL")
                   or file_doc.get("model") or DEFAULT_MODEL),
            stream=bool(stream) if stream is not None else False,
            tools_exclude=exclude,
        )


def config_guidance(api_key_env: str = "TF_AGENT_API_KEY") -> str:
    """缺 key 时的配置指引（不给报错堆栈）。"""
    return f"""\
未找到 API key，无法启动内置 Agent。任选其一配置（优先级：环境变量 > 配置文件）：

1. 环境变量（二选一指定变量名，默认 {api_key_env}）：
     export {api_key_env}=sk-...
     export TF_AGENT_BASE_URL=https://api.moonshot.cn/v1   # 可选，OpenAI 兼容端点
     export TF_AGENT_MODEL=kimi-k2-0905-preview            # 可选

2. 配置文件：cp pi/agent.example.toml pi/agent.toml，填入 api_key
   （pi/agent.toml 已 gitignore，密钥不会入库）。

密钥申请：Moonshot/Kimi → https://platform.moonshot.cn/console/api-keys
          OpenAI → https://platform.openai.com/api-keys
          DeepSeek → https://platform.deepseek.com/api_keys
任何「OpenAI 兼容 chat completions + function calling」的端点均可。

配置完成后用 `.venv/Scripts/tf agent --check` 验证连通性。"""
