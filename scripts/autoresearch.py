"""无人值守研究驱动器：让主智能体在一组 Research Goal 上一直跑到跑不动为止。

与网页版「AI 深度研究」用的是同一套东西 —— 同一个 `ResearchOrchestrator`、
同一个 LLM 规划器（`harness/prompts/planner.md` + 绑定技能）、同一批工具。
区别只有三个：

1. **没有轮次上限**：网页版默认 5~6 轮，这里 `--max-rounds` 默认 10000，
   实际停在编排器的停止条件上（验收达标 / 连续无信息增益 / 数据覆盖不足 /
   缺必需变量 / 预算耗尽），而不是停在一个人为的计数器上。
2. **多目标轮转**：一个目标停下后换下一个；一整轮所有目标都没产出新的
   最优指标，才判定「没法跑了」退出。中间新增的证据会进下一圈的上下文，
   所以第二圈不是简单重跑。
3. **能被外部观测**：每轮都把状态写进 `--state-file`（JSON），日志写
   `--log-file`。想知道现在跑到哪了，读这两个文件即可，不必进程内观察。

额度/鉴权类错误单独识别（`_is_fatal_api_error`）：这类错误重试没有意义，
直接写 `fatal` 状态退出，让外部值守流程去处理（通知人、换 key）。

用法::

    python scripts/autoresearch.py --goals RG-0016,RG-0017
    python scripts/autoresearch.py --goals RG-0017 --no-gain-rounds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from thermoforge_research.orchestrator import ResearchOrchestrator
from thermoforge_research.tools import ToolContext

# 额度耗尽 / 鉴权失效的特征串：命中即认为重试无意义
FATAL_API_MARKERS = (
    "insufficient", "balance", "quota", "billing", "credit", "arrears",
    "payment", "expired", "invalid api key", "unauthorized", "forbidden",
    "余额", "额度", "欠费", "认证失败", "401", "402", "403",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_fatal_api_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in FATAL_API_MARKERS)


class Journal:
    """状态文件 + 日志文件。两者都用原子写，外部随时可读。"""

    def __init__(self, state_file: Path, log_file: Path) -> None:
        self.state_file = state_file
        self.log_file = log_file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "started_at": _now(), "updated_at": _now(), "status": "running",
            "cycle": 0, "goals": {}, "events": [], "fatal": None,
        }

    def log(self, message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8", newline="\n") as fp:
            fp.write(line + "\n")

    def event(self, kind: str, **payload: Any) -> None:
        self.state["events"].append({"at": _now(), "kind": kind, **payload})
        self.state["events"] = self.state["events"][-200:]
        self.flush()

    def flush(self) -> None:
        self.state["updated_at"] = _now()
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, self.state_file)


def _usable_views(ctx: ToolContext, goal_def: Mapping[str, Any],
                  dataset_ref: str | None = None) -> list[dict]:
    """目标能用的已登记视图：数据集一致 + 目标列一致 + 特征 ⊆ 白名单。

    直接读 `research/views/*.yaml`（与网页版 `cache.views()` 同源）——
    Ledger 没有公开的按 kind 列举接口，不去碰它的私有方法。

    **`dataset_ref` 过滤是必需的，不是可选优化**：同一个目标列可能同时存在
    于多个数据集（实测 `approach` 同时在全年的 `WX_CT_MODEL` 和制冷季的
    `WX_CT_SEASON` 里）。不过滤的话，规划器会在两个口径之间来回跳，
    指标互不可比、也悄悄绕开了限定口径的初衷（实测 RG-0019：本该只用
    制冷季，却跑出了两轮全年视图的实验）。
    """
    import yaml

    whitelist = set(goal_def.get("candidate_inputs") or [])
    target = goal_def.get("target")
    directory = ctx.research_root / "views"
    out: list[dict] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("VIEW-*.yaml")):
        entity = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        definition = entity.get("definition") or {}
        if definition.get("target") != target:
            continue
        if dataset_ref and definition.get("dataset") != dataset_ref:
            continue
        if not set(definition.get("features") or []) <= whitelist:
            continue
        out.append({"id": entity.get("id") or path.stem,
                    "definition": definition})
    return out


def _dataset_ref(views: list[dict]) -> str | None:
    for view in views:
        ref = (view.get("definition") or {}).get("dataset")
        if ref:
            return str(ref)
    return None


def _seed_history(ctx: ToolContext, goal_id: str, context: Any) -> int:
    """把该目标**历史上已经跑过**的实验塞进规划上下文，供重复计划拦截使用。

    规划器的去重是拿 `context.traces` 比对的，而 traces 是进程内的 ——
    进程一重启就空了，于是重启后第一轮很容易原样重交上一次跑过的组合
    （实测 EXP-0146/0147 就是这么来的：同视图同模块同超参，指标逐位相同）。
    这里从实验目录读回历史规格，伪造成 trace 补进去，让拦截跨进程也有效。
    """
    from thermoforge_webui.services.planner import PlannerTrace

    seeded = 0
    directory = ctx.research_root / "experiments"
    if not directory.is_dir():
        return 0
    for path in sorted(directory.glob("EXP-*/spec.json")):
        try:
            exp = json.loads(path.read_text(encoding="utf-8"))["experiment"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if exp.get("goal_id") != goal_id:
            continue
        trace = PlannerTrace(round_index=-1, prompt_summary="历史实验（跨进程去重）")
        trace.plan = {"view_id": exp.get("dataset_view"),
                      "model": exp.get("model") or {}}
        context.traces.append(trace)
        seeded += 1
    return seeded


def _goal_best(ctx: ToolContext, goal_id: str,
               evaluated_on: str) -> dict[str, Any] | None:
    """扫该目标**历史上所有**实验，找判据面上最优的一个。

    编排器的 `best_cvrmse` 只统计本次运行。重启一次就归零，简报里写的
    「最优」于是比真实最优还差——8-19 RG-0026 状态文件写 4.94%，而全目标
    最优是上一次运行的 EXP-0154 的 4.00%。这里按目标而不是按进程统计。
    """
    directory = ctx.research_root / "experiments"
    if not directory.is_dir():
        return None
    best: dict[str, Any] | None = None
    for path in sorted(directory.glob("EXP-*/spec.json")):
        try:
            exp = json.loads(path.read_text(encoding="utf-8"))["experiment"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if exp.get("goal_id") != goal_id:
            continue
        try:
            metrics = json.loads(
                (path.parent / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue          # 还在跑或已崩，没有指标
        if evaluated_on == "rolling_cv":
            block = (metrics.get("rolling_cv") or {}).get("metrics") or {}
        else:
            block = (((metrics.get("surfaces") or {})
                      .get(evaluated_on) or {}).get("metrics") or {})
        cvrmse = block.get("CVRMSE")
        if not isinstance(cvrmse, (int, float)):
            continue
        if best is None or cvrmse < best["cvrmse"]:
            best = {"experiment_id": exp.get("experiment_id") or path.parent.name,
                    "cvrmse": float(cvrmse), "nmbe": block.get("NMBE"),
                    "view_id": exp.get("dataset_view"),
                    "model": ((exp.get("model") or {}).get("hyperparameters")
                              or {}).get("lab")}
    return best


def _run_goal(ctx: ToolContext, goal_id: str, journal: Journal,
              *, max_rounds: int, no_gain_rounds: int, guidance: str,
              same_origin_corr: float | None = None,
              rolling_horizon_seconds: float | None = None,
              dataset_ref: str | None = None) -> dict[str, Any]:
    """跑一个目标到它自己的停止条件。返回 {stop_reason, rounds, best_cvrmse}。"""
    from thermoforge_webui.config import agent_config
    from thermoforge_webui.services.planner import PlannerContext, make_planner
    from thermoforge_webui.services.research import make_ask

    entity = ctx.ledger.get(goal_id)
    definition = dict(entity.get("definition") or {})
    definition.setdefault("id", goal_id)
    views = _usable_views(ctx, definition, dataset_ref)
    if not views:
        return {"stop_reason": "no_usable_view", "rounds": 0}
    ref = dataset_ref or _dataset_ref(views)
    lab_modules = [m for m in ctx.lab_store.list() if m.get("runnable")]

    config = agent_config()
    if config is None:
        raise RuntimeError("harness/agent.toml 未配置 api_key（unauthorized）")
    context = PlannerContext(goal=definition, views=views, dataset_ref=ref,
                             extra_guidance=guidance, lab_modules=lab_modules)
    _seed_history(ctx, goal_id, context)
    planner = make_planner(make_ask(config), context)

    orchestrator = ResearchOrchestrator(
        ctx, goal_id, planner, dataset_ref=ref,
        max_rounds=max_rounds, no_gain_rounds=no_gain_rounds,
        same_origin_corr=same_origin_corr,
        rolling_horizon_seconds=rolling_horizon_seconds)
    outcome = orchestrator.run()
    summary = outcome.get("summary") or {}
    stop = summary.get("stop") or {}
    rounds = summary.get("rounds") or []
    # 失败轮的明细必须落盘。此前这里只留下 experiments 里的一个 null，事后
    # 想知道「9 轮里 5 轮为什么废了」只能去翻 experiments/*/stderr.log——而
    # 失败轮恰恰是最该看的（8-19 RG-0028 七轮里三轮崩在列名、三轮是空转）。
    failures = [{"round": r.get("round"),
                 "experiment_id": r.get("experiment_id"),
                 "detail": str(r.get("detail") or "")[:600]}
                for r in rounds if r.get("status") == "failed"]
    return {
        "stop_reason": stop.get("reason"),
        "stop_detail": stop.get("detail") or stop.get("message"),
        "rounds": len(rounds),
        "rounds_failed": len(failures),
        "failures": failures,
        "best_cvrmse": summary.get("best_cvrmse"),
        "goal_best": _goal_best(
            ctx, goal_id,
            str((definition.get("acceptance") or {}).get("evaluated_on")
                or "rolling_cv")),
        "experiments": [r.get("experiment_id") for r in rounds],
        "last_metrics": (rounds[-1].get("primary_metrics")
                         if rounds else None),
        "acceptance_unmet": (rounds[-1].get("acceptance_unmet")
                             if rounds else None),
    }


# 目标已经走到终局，再跑一遍只是重复烧钱
TERMINAL_GOAL_STATES = {"PUBLISH", "STOPPED"}

# 这些停止原因是「环境不满足」而非「研究做完了」：环境改了就该能重开
REOPENABLE_STOPS = {"modelability_failed", "missing_required_variables",
                    "insufficient_data_coverage", "no_usable_view"}


def _reopen(ctx: ToolContext, goal_id: str, journal: Journal) -> bool:
    """把因环境问题停掉的目标转回 active。转换连原因一起进账本，可追溯。"""
    entity = ctx.ledger.get(goal_id)
    if entity.get("status") != "STOPPED":
        return False
    last = [t for t in entity.get("transitions", []) if t.get("to") == "STOPPED"]
    reason = str((last[-1] if last else {}).get("reason") or "")
    if not any(stop in reason for stop in REOPENABLE_STOPS):
        journal.log(f"{goal_id} 停在「{reason[:60]}」，不属于可重开类型，跳过")
        return False
    ctx.ledger.transition(
        goal_id, "active", actor=ctx.actor,
        reason=f"重开：上次停于环境问题（{reason[:80]}），该问题已处理")
    journal.log(f"{goal_id} 已重开（上次停因：{reason[:60]}）")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--goals", required=True, help="逗号分隔的 goal_id")
    ap.add_argument("--vault-root", type=Path, default=Path("vault"))
    ap.add_argument("--research-root", type=Path, default=Path("research"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    ap.add_argument("--max-rounds", type=int, default=10000,
                    help="单个目标的编排器硬上限；默认给到实际不生效")
    ap.add_argument("--no-gain-rounds", type=int, default=4,
                    help="连续多少轮无信息增益判停")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="最多轮转几圈；0=不限，直到一整圈都没进展")
    ap.add_argument("--guidance", default="",
                    help="给规划器的额外要求（进提示词）")
    ap.add_argument("--rolling-horizon-days", type=float, default=None,
                    help="覆盖滚动折的窗口/步长（天）。折太短会让折内 CV_y 趋近 0，"
                         "R² 折均被退化折支配 —— 此时它衡量的是那一折目标波动"
                         "大不大，不是模型好不好")
    ap.add_argument("--dataset-ref", default=None,
                    help="把目标钉在这个数据集修订版上（形如 X@rev_0001）。"
                         "同一目标列可能存在于多个数据集，不钉住规划器会在"
                         "不同口径之间来回跳，指标互不可比")
    ap.add_argument("--max-transient", type=int, default=5,
                    help="单个目标允许的连续瞬时错误（超时/网络）次数，"
                         "指数退避重试；超过就本进程内不再重试该目标")
    ap.add_argument("--reopen", action="store_true",
                    help="把因环境问题（门禁未过/缺变量/覆盖不足）停掉的目标"
                         "转回 active 重跑；研究做完了的（达标/无增益）不动")
    ap.add_argument("--same-origin-corr", type=float, default=None,
                    help="覆盖 G3 同源阻断阈值（默认 0.98）。只在确认高相关是"
                         "物理耦合而非同源时使用；用了什么值会写进报告 artifact")
    ap.add_argument("--state-file", type=Path, default=None,
                    help="默认按目标集自动命名，见下方并发说明")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="同 --state-file")
    args = ap.parse_args(argv)

    # 状态/日志文件按**目标集**自动命名。并发跑多个目标是正常需求（账本有
    # 文件锁、ID 分配互斥、实验目录各自独立，研究工件本来就是安全的），
    # 但两个驱动器写同一个状态文件会互相覆盖，值守读到的就是乱的
    # ——实测 RG-0026 与 RG-0028 并发时踩过。
    slug = "_".join(g.strip() for g in args.goals.split(",") if g.strip())[:60]
    if args.state_file is None:
        args.state_file = Path(f"research/autoresearch_state.{slug}.json")
    if args.log_file is None:
        args.log_file = Path(f"research/autoresearch.{slug}.log")

    goals = [g.strip() for g in args.goals.split(",") if g.strip()]
    journal = Journal(args.state_file, args.log_file)
    journal.state["queue"] = goals
    # 把自己的命令行记进状态文件：守护脚本（scripts/watchdog.py）靠它在
    # 进程意外死掉后按原样重新拉起，不必去猜当时带了哪些 guidance/阈值。
    journal.state["argv"] = [str(Path(__file__).resolve()),
                             *(argv if argv is not None else sys.argv[1:])]
    journal.log(f"启动：目标队列 {goals}，max_rounds={args.max_rounds}，"
                f"no_gain_rounds={args.no_gain_rounds}")

    ctx = ToolContext(vault_root=args.vault_root,
                      research_root=args.research_root,
                      models_root=args.models_root, actor="agent")

    cycle = 0
    transient: dict[str, int] = {}   # goal_id -> 连续瞬时错误次数
    while True:
        cycle += 1
        journal.state["cycle"] = cycle
        progressed = False
        retry_pending = False
        for goal_id in goals:
            status = ctx.ledger.get(goal_id).get("status")
            if status in TERMINAL_GOAL_STATES:
                if not (args.reopen and _reopen(ctx, goal_id, journal)):
                    journal.log(f"跳过 {goal_id}（已是终局状态 {status}）")
                    continue
                status = "active"
            journal.log(f"第 {cycle} 圈 · {goal_id} 开始（当前状态 {status}）")
            journal.event("goal_start", goal=goal_id, cycle=cycle)
            started = time.time()
            try:
                result = _run_goal(
                    ctx, goal_id, journal, max_rounds=args.max_rounds,
                    no_gain_rounds=args.no_gain_rounds, guidance=args.guidance,
                    same_origin_corr=args.same_origin_corr,
                    rolling_horizon_seconds=(args.rolling_horizon_days * 86400.0
                                             if args.rolling_horizon_days else None),
                    dataset_ref=args.dataset_ref)
            except BaseException as exc:  # noqa: BLE001 —— 值守进程不能因单点崩掉
                detail = traceback.format_exc(limit=6)
                fatal = _is_fatal_api_error(exc)
                journal.log(f"{goal_id} 异常（fatal={fatal}）: {exc}")
                journal.event("goal_error", goal=goal_id, fatal=fatal,
                              error=f"{type(exc).__name__}: {exc}",
                              traceback=detail)
                if fatal or isinstance(exc, KeyboardInterrupt):
                    journal.state["status"] = "fatal" if fatal else "interrupted"
                    journal.state["fatal"] = {
                        "at": _now(), "goal": goal_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "likely_cause": ("接口额度/鉴权（余额耗尽或 key 失效）"
                                         if fatal else "手动中断"),
                    }
                    journal.flush()
                    return 3 if fatal else 130

                # 超时/网络抖动不是「跑不动了」，是这一轮没连上。记一笔，
                # 退避后进下一圈重试；连续失败太多次才认输。
                transient[goal_id] = transient.get(goal_id, 0) + 1
                if transient[goal_id] >= args.max_transient:
                    journal.log(f"{goal_id} 连续 {transient[goal_id]} 次瞬时错误，"
                                f"本进程不再重试")
                    continue
                backoff = min(300, 30 * 2 ** (transient[goal_id] - 1))
                journal.log(f"{goal_id} 第 {transient[goal_id]} 次瞬时错误，"
                            f"{backoff}s 后重试")
                time.sleep(backoff)
                retry_pending = True
                continue

            transient.pop(goal_id, None)
            result["seconds"] = round(time.time() - started, 1)
            result["at"] = _now()
            journal.state["goals"][goal_id] = result
            journal.event("goal_done", goal=goal_id, cycle=cycle, **{
                k: result[k] for k in ("stop_reason", "rounds", "best_cvrmse")})
            journal.log(f"{goal_id} 停止：{result['stop_reason']}，"
                        f"跑了 {result['rounds']} 轮，"
                        f"最优 CVRMSE={result.get('best_cvrmse')}")
            if result["rounds"] > 0:
                progressed = True

        if not progressed and not retry_pending:
            journal.state["status"] = "exhausted"
            journal.log(f"第 {cycle} 圈没有任何目标产出新轮次 —— 跑不动了，退出。")
            journal.flush()
            return 0
        if args.max_cycles and cycle >= args.max_cycles:
            journal.state["status"] = "max_cycles"
            journal.log(f"达到 --max-cycles={args.max_cycles}，退出。")
            journal.flush()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
