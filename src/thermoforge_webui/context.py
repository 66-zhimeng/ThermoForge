"""运行上下文：仓库根、工件根、ToolContext 构造。

界面里有两种 actor，必须分开，不能图省事共用一个：

- `webui`：界面代表使用者做只读查询与常规工具调用。
- `human`：预处理审批这类 human-only 动作，Ledger 里要看得出是人批的
  （I-49），Agent 自己永远拿不到这个身份。
"""

from __future__ import annotations

import os
from pathlib import Path

from thermoforge_research.tools import ToolContext

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "pi" / "agent.toml"

ACTOR_UI = "webui"
ACTOR_HUMAN = "human"


def _root(env_name: str, default: str) -> Path:
    """工件根：环境变量可覆盖，默认落在仓库下（与 CLI 默认一致）。"""
    value = os.environ.get(env_name)
    return Path(value) if value else REPO_ROOT / default


VAULT_ROOT = _root("TF_VAULT_ROOT", "vault")
RESEARCH_ROOT = _root("TF_RESEARCH_ROOT", "research")
MODELS_ROOT = _root("TF_MODELS_ROOT", "models")


def tool_context(actor: str = ACTOR_UI) -> ToolContext:
    """构造 ToolContext。

    每次调用都新建：ToolContext 内部缓存了 DuckDB 连接与文件句柄，而
    Windows 上句柄不及时释放会让 `os.replace` 失败（implementation-notes
    §10.3）。Streamlit 每次交互都会重跑脚本，短命上下文正合适。
    """
    return ToolContext(
        vault_root=VAULT_ROOT,
        research_root=RESEARCH_ROOT,
        models_root=MODELS_ROOT,
        actor=actor,
    )


def human_context() -> ToolContext:
    """human-only 动作专用上下文（预处理审批等）。"""
    return tool_context(ACTOR_HUMAN)


def experiments_root() -> Path:
    return RESEARCH_ROOT / "experiments"


def roots_summary() -> dict[str, str]:
    """给界面显示「现在在读哪几个目录」，排查串数据时很关键。"""
    return {
        "仓库": str(REPO_ROOT),
        "数据 Vault": str(VAULT_ROOT),
        "研究工件": str(RESEARCH_ROOT),
        "模型注册表": str(MODELS_ROOT),
    }
