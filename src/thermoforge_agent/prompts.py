"""提示词加载与指纹（控制面「决策规则」的唯一来源）。

四个 agent 位点（命令行 REPL、网页副驾、研究规划器、MCP server）各有一套
系统提示词。以前它们散在各自的代码里，改 `harness/prompts/system.md` 只有命令行
会变——同一句「技能」在不同入口行为不一致，这本身就是缺陷。

现在一律从 `harness/prompts/<name>.md` 读：**装技能 = 改文件**，四处行为一致。

## 技能

`harness/skills/*.md` 是可复用的方法论（怎么做系统辨识、怎么校验），与位点提示词
（这个 agent 是谁、能调什么工具）分开：提示词描述**身份与接口**，技能描述
**做法**。技能按 `SKILL_BINDINGS` 绑定到位点，加载时追加在提示词之后，
并一并计入指纹 —— 换了技能，实验产物里的 `prompt_fingerprint` 就会变。

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
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "harness" / "prompts"
SKILLS_DIR = REPO_ROOT / "harness" / "skills"

# 位点 → 文件名。改这里等于改「有哪些可装技能的位点」。
PROMPT_FILES = {
    "cli": "system.md",        # tf agent 命令行 REPL
    "copilot": "copilot.md",   # 网页副驾
    "planner": "planner.md",   # AI 研究的规划器
    "mcp": "mcp.md",           # MCP server 给外部 agent 的说明
}

# 位点 → 装载的技能（`harness/skills/<name>.md`，不含扩展名）。
# 只装到真正做研究的位点：网页副驾负责导航与解读，不直接指挥建模升级。
# 顺序即装载顺序：取证在前、建模在后，与实际工作顺序一致。
SKILL_BINDINGS = {
    "cli": ("measurement-forensics", "system-identification"),
    "planner": ("measurement-forensics", "system-identification"),
    "mcp": ("measurement-forensics", "system-identification"),
    "copilot": (),
}

_MISSING = "（缺少提示词文件：{path}）"


def prompt_path(name: str) -> Path:
    filename = PROMPT_FILES.get(name)
    if filename is None:
        raise KeyError(f"未登记的提示词位点: {name!r}（已登记 "
                       f"{sorted(PROMPT_FILES)}）")
    return PROMPTS_DIR / filename


def skill_path(skill: str) -> Path:
    return SKILLS_DIR / f"{skill}.md"


def available_skills() -> list[str]:
    """磁盘上实际存在的技能（按名排序）。"""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.stem for p in SKILLS_DIR.glob("*.md"))


def skills_for(name: str) -> tuple[str, ...]:
    """某位点绑定的技能名。未登记的位点视为不装技能。"""
    return tuple(SKILL_BINDINGS.get(name, ()))


def normalize(text: str) -> str:
    """换行规范化。跨平台指纹稳定性的前提，与项目其它文本处理同调。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load(name: str, fallback: str = "", **placeholders: str) -> str:
    """读某个位点的提示词。文件缺失时退回 `fallback`，不抛异常。

    不抛是刻意的：提示词文件没了，agent 应该退化成「能用但没那么懂行」，
    而不是整个界面打不开。指纹会如实反映这一点（缺失记为空内容）。

    `placeholders` 填充 `$name` 占位符（`string.Template.safe_substitute`：
    填不上的原样留着，不抛）。占位符只用于**代码派生的事实**（页面清单、
    工具名这类随代码走的东西），决策规则一律写死在文件里——否则「改文件
    = 改行为」这条就不成立了。占位符在替换**前**参与指纹计算，于是
    指纹只反映决策规则本身，不随页面清单抖动。
    """
    path = prompt_path(name)
    base = (normalize(path.read_text(encoding="utf-8")) if path.is_file()
            else (fallback or _MISSING.format(path=path)))
    parts = [base]
    for skill in skills_for(name):
        sp = skill_path(skill)
        if sp.is_file():          # 技能缺失同样不抛：退化成「没装这项本事」
            parts.append(normalize(sp.read_text(encoding="utf-8")))
    text = "\n\n---\n\n".join(parts)
    return Template(text).safe_substitute(placeholders) if placeholders \
        else text


def digest(name: str) -> str:
    """单个位点的内容指纹（sha256 前 16 位），**含其绑定的技能**。

    含技能是刻意的：同一份 system.md 配不同技能，agent 的行为不同，
    实验产物必须能区分这两种情况。文件缺失记为全 0 参与计算。
    """
    payload = load(name, fallback="").encode("utf-8")
    if not prompt_path(name).is_file() and not skills_for(name):
        return "0" * 16
    return hashlib.sha256(payload).hexdigest()[:16]


def registry() -> dict[str, str]:
    """全部位点的指纹表，界面上用来展示「当前生效的技能版本」。"""
    return {name: digest(name) for name in sorted(PROMPT_FILES)}


def skill_registry() -> dict[str, list[str]]:
    """位点 → 实际装上的技能（磁盘存在的那些）。界面与报告用。"""
    have = set(available_skills())
    return {name: [s for s in skills_for(name) if s in have]
            for name in sorted(PROMPT_FILES)}


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
        [(name, prompt_path(name).stat().st_mtime
          if prompt_path(name).is_file() else 0.0)
         for name in sorted(PROMPT_FILES)]
        + [(f"skill:{s}", skill_path(s).stat().st_mtime)
           for s in available_skills()]
    )
    return _cached_fingerprint(stamp)
