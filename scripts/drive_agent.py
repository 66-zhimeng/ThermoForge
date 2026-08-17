"""非交互地驱动内置研发 Agent 跑一轮系统辨识。

`tf agent` 是交互式 REPL，不便在自动化环境里用。这个脚本给同一个 Agent
喂一串任务、把每轮回答打出来，会话完整落 `research/agent_sessions/*.jsonl`。

用法::

    python scripts/drive_agent.py                    # 跑默认任务串
    python scripts/drive_agent.py --task "..."       # 单条任务
    python scripts/drive_agent.py --list-tasks       # 只看会问什么
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thermoforge_agent.agent import HarnessAgent
from thermoforge_agent.config import AgentConfig
from thermoforge_research.tools import ToolContext

# 一轮系统辨识的任务串：先摸清现状，再逐级升级，每级读校验再决定下一步。
# 任务本身不指定具体模型 —— 让 Agent 依据装载的 system-identification
# 技能自己选路线，这才是在检验技能是否真的生效。
DEFAULT_TASKS = [
    "用 tf_goal_get 读 RG-0007（冷机功耗）与 RG-0008（板换换热量）的完整定义，"
    "确认白名单原文、写法（裸 property 还是带对象）与验收门槛。再看两个数据集的 schema。"
    "简要说明各自标签是什么、哪些列是实测哪些是算出来的。",

    "为 RG-0007 物化视图并执行第一级实验（category=data, estimator=ridge）。"
    "features 必须严格取自刚读到的白名单；target 用 property_code。"
    "切分要覆盖多季节（rolling_cv），purge/embargo 必设。"
    "metrics 至少含 CVRMSE、NMBE、R2。执行后报告指标。",

    "升级到第二级：Gordon-Ng 熵产模型（category=physics, physics=gordon_ng）。"
    "hyperparameters.inputs 把 cooling_load / t_evap_out / t_cond_in 映射到视图列名。"
    "执行后与第一级对比，并做参数体检：三个参数是否非负、量级是否合理、有没有贴边界。",

    "为 RG-0008 物化视图并执行 ε-NTU 实验（category=physics, physics=eps_ntu）。"
    "inputs 映射 t_hot_in/t_cold_in/f_hot/f_cold 到视图列名。执行后报告指标。",

    "横向比较全部已执行实验，给出：① 离验收门槛还差多少 ② 下一步最该做什么 "
    "③ 哪些是数据本身的边界、堆模型也解决不了。只讲跑出来的数，不要推测。",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", action="append", default=None,
                    help="自定义任务（可多次给出）；不给则跑默认任务串")
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--vault-root", type=Path, default=Path("vault"))
    ap.add_argument("--research-root", type=Path, default=Path("research"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    args = ap.parse_args(argv)

    tasks = args.task or DEFAULT_TASKS
    if args.list_tasks:
        for i, t in enumerate(tasks, 1):
            print(f"[{i}] {t}\n")
        return 0

    ctx = ToolContext(vault_root=args.vault_root,
                      research_root=args.research_root,
                      models_root=args.models_root, actor="agent")
    config = AgentConfig.load()
    if config is None:
        raise SystemExit("未找到 agent 配置（harness/agent.toml 或 TF_AGENT_* 环境变量）")
    agent = HarnessAgent(
        config, ctx,
        on_tool_call=lambda name, ok: print(f"    → {name} {'ok' if ok else 'FAILED'}",
                                            flush=True),
    )
    print(f"会话日志: {agent._session_path}")
    print(f"模型: {config.model}")
    print(f"可用工具: {len(agent.schemas)}\n", flush=True)

    for i, task in enumerate(tasks, 1):
        print("=" * 78)
        print(f"【任务 {i}/{len(tasks)}】{task}")
        print("=" * 78)
        try:
            print(agent.ask(task))
        except Exception as exc:                      # noqa: BLE001
            print(f"!! 本轮失败: {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
