"""提示词加载与指纹（控制面「决策规则」的唯一来源）。

四个 agent 位点（命令行 REPL、网页副驾、研究规划器、MCP server）各有一套
系统提示词。以前它们散在各自的代码里，改 `pi/prompts/system.md` 只有命令行
会变——同一句「技能」在不同入口行为不一致，这本身就是缺陷。

现在一律从 `pi/prompts/<name>.md` 读：**装技能 = 改文件**，四处行为一致。

## 为什么要指纹

实验产物记了 `code_version` 与 `environment_lock`，唯独没记「当时是哪套
决策规则在指挥」。提示词一改，规划器提的假设就变，跑出来的实验也就变了，
但 Ledger 里两批实验长得一模一样——这对一个把可追溯性当立身之本的系统
是个真实缺口（issues.md I-55）。

`fingerprint()` 给出一个覆盖全部提示词文件的稳定十六进制串，随实验落盘。
换行一律规范化成 `\\n` 再算：Windows 检出是 CRLF，不规范化的话同一份内容
在两台机器上会得到两个指纹。
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "pi" / "prompts"

# 位点 → 文件名。改这里等于改「有哪些可装技能的位点」。
PROMPT_FILES = {
    "cli": "system.md",        # tf agent 命令行 REPL
    "copilot": "copilot.md",   # 网页副驾
    "planner": "planner.md",   # AI 研究的规划器
    "mcp": "mcp.md",           # MCP server 给外部 agent 的说明
}

_MISSING = "（缺少提示词文件：{path}）"


def prompt_path(name: str) -> Path:
    filename = PROMPT_FILES.get(name)
    if filename is None:
        raise KeyError(f"未登记的提示词位点: {name!r}（已登记 "
                       f"{sorted(PROMPT_FILES)}）")
    return PROMPTS_DIR / filename


def normalize(text: str) -> str:
    """换行规范化。跨平台指纹稳定性的前提，与项目其它文本处理同调。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load(name: str, fallback: str = "") -> str:
    """读某个位点的提示词。文件缺失时退回 `fallback`，不抛异常。

    不抛是刻意的：提示词文件没了，agent 应该退化成「能用但没那么懂行」，
    而不是整个界面打不开。指纹会如实反映这一点（缺失记为空内容）。
    """
    path = prompt_path(name)
    if not path.is_file():
        return fallback or _MISSING.format(path=path)
    return normalize(path.read_text(encoding="utf-8"))


def digest(name: str) -> str:
    """单个位点的内容指纹（sha256 前 16 位）。文件缺失记为全 0。"""
    path = prompt_path(name)
    if not path.is_file():
        return "0" * 16
    payload = normalize(path.read_text(encoding="utf-8")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def registry() -> dict[str, str]:
    """全部位点的指纹表，界面上用来展示「当前生效的技能版本」。"""
    return {name: digest(name) for name in sorted(PROMPT_FILES)}


@lru_cache(maxsize=1)
def _cached_fingerprint(stamp: tuple[tuple[str, float], ...]) -> str:
    del stamp  # 仅用于缓存失效，值本身不参与计算
    lines = [f"{name}:{digest(name)}" for name in sorted(PROMPT_FILES)]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint() -> str:
    """整套提示词的指纹（64 位十六进制），写进实验产物。

    以文件 mtime 作为缓存键：长跑的研究循环不必每轮重读四个文件，但
    改了文件立刻生效——否则「改完技能要重启界面」又是一个反直觉行为。
    """
    stamp = tuple(
        (name, prompt_path(name).stat().st_mtime
         if prompt_path(name).is_file() else 0.0)
        for name in sorted(PROMPT_FILES)
    )
    return _cached_fingerprint(stamp)
