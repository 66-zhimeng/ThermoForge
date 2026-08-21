"""AI 深度研究：后台跑编排循环，前台看实时进度。

编排器 `ResearchOrchestrator` 是同步阻塞的，一轮实验要几十秒到几分钟。
放在 Streamlit 的脚本线程里会把整个界面卡死，所以扔进后台线程，用事件
队列把进度推回来；界面只读事件列表。

两种模式（启动时选）：

- **自动**：一路跑到编排器的停止条件，中途只在需要人拍板时停下。
- **逐轮审批**：每轮规划完先展示假设与实验计划，等人点「批准」才真跑。
  规划器回调里用 `threading.Event` 阻塞——编排器是同步的，卡住 planner
  就等于卡住那一轮，不需要改编排器一行代码。

任何时候都能中断：`stop()` 置位后，规划器下一次被调用时直接返回 None，
编排器以 `no_information_gain` 正常收尾，工件保持完整（不会留下半个实验）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..context import tool_context
from .planner import PlannerContext, PlannerError, make_planner

# 逐轮审批模式下等待人操作的上限：超过就当作放弃，避免线程永久挂着
APPROVAL_TIMEOUT_SECONDS = 3600.0

MODE_AUTO = "auto"
MODE_STEP = "step"

STOP_REASON_LABELS = {
    "acceptance_met": "达到验收标准 🎉",
    "budget_exhausted": "预算耗尽",
    "no_information_gain": "连续多轮没有信息增益",
    "insufficient_data_coverage": "数据覆盖不足",
    "missing_required_variables": "缺少必需变量",
    "human_confirmation_required": "需要人工确认",
    "modelability_failed": "可建模性门禁未过（有阻断项）",
}


@dataclass
class ResearchEvent:
    """事件流的一条。`kind` 决定界面怎么画。"""

    kind: str  # started/planning/plan/approved/rejected/experiment/stop/error
    text: str
    at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


class ResearchSession:
    """一次研究运行的生命周期与状态。线程安全靠一把锁 + 两个 Event。"""

    def __init__(self, goal_id: str, dataset_ref: str, *,
                 mode: str = MODE_AUTO, max_rounds: int = 6,
                 guidance: str = "") -> None:
        self.goal_id = goal_id
        self.dataset_ref = dataset_ref
        self.mode = mode
        self.max_rounds = int(max_rounds)
        self.guidance = guidance

        self._lock = threading.Lock()
        self._events: list[ResearchEvent] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._decision = threading.Event()
        self._approved = False
        self._pending_plan: dict[str, Any] | None = None

        self.state = "idle"  # idle/running/awaiting/finished/error
        self.outcome: dict[str, Any] | None = None
        self.error: str | None = None
        self.planner_context: PlannerContext | None = None

    # ---- 事件

    def _emit(self, kind: str, text: str, **payload: Any) -> None:
        with self._lock:
            self._events.append(ResearchEvent(kind=kind, text=text,
                                              payload=payload))

    def events(self) -> list[ResearchEvent]:
        with self._lock:
            return list(self._events)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def pending_plan(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._pending_plan) if self._pending_plan else None

    # ---- 控制

    def start(self, ask: Callable[[str, str], str],
              planner_context: PlannerContext) -> None:
        if self.running:
            return
        self.planner_context = planner_context
        self.state = "running"
        self._stop.clear()
        self._emit("started",
                   f"开始研究 {self.goal_id}（"
                   f"{'自动' if self.mode == MODE_AUTO else '逐轮审批'}模式，"
                   f"最多 {self.max_rounds} 轮）")
        self._thread = threading.Thread(
            target=self._run, args=(ask, planner_context),
            name=f"research-{self.goal_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """请求停止。已经在跑的那一轮实验会跑完——中途杀掉会留下半个工件。"""
        self._stop.set()
        self._decision.set()  # 解开可能正在等审批的规划器
        self._emit("stop_requested", "已请求停止：当前这一轮跑完就收尾。")

    def decide(self, approved: bool, note: str = "") -> None:
        """逐轮审批模式下的人工决策。"""
        with self._lock:
            self._approved = approved
            self._pending_plan = None
        self._emit("approved" if approved else "rejected",
                   ("已批准，开始执行这一轮实验。" if approved
                    else f"已否决这一轮。{note}"))
        self._decision.set()

    # ---- 后台主体

    def _run(self, ask: Callable[[str, str], str],
             planner_context: PlannerContext) -> None:
        from thermoforge_research.orchestrator import ResearchOrchestrator

        base_planner = make_planner(ask, planner_context)

        def planner(round_index: int,
                    evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
            if self._stop.is_set():
                self._emit("stopped", "收到停止请求，不再规划新的假设。")
                return None
            self._emit("planning", f"第 {round_index + 1} 轮：正在规划假设…",
                       round_index=round_index)
            try:
                plan = base_planner(round_index, evidence)
            except PlannerError as exc:
                self._emit("error", f"规划失败：{exc}")
                raise
            # make_planner 把当轮 trace 挂在函数属性上；编排器只调本包装
            # 函数，透传给它，规划留痕才能随实验工件落盘
            planner.last_trace = getattr(base_planner, "last_trace", None)
            if plan is None:
                self._emit("plan_none", "模型认为没有值得再试的假设了。")
                return None
            if plan.get("needs_human"):
                self._emit("needs_human",
                           f"模型请求人工确认：{plan['needs_human']}")
                return plan
            self._emit("plan", plan.get("statement", ""), plan=dict(plan),
                       round_index=round_index,
                       reasoning=_latest_reasoning(planner_context),
                       cot=_latest_cot(planner_context),
                       usage=_latest_usage(planner_context),
                       cost=_latest_cost(planner_context))
            if self.mode == MODE_STEP and not self._await_approval(plan):
                return None
            return plan

        try:
            orchestrator = ResearchOrchestrator(
                tool_context(), self.goal_id, planner,
                dataset_ref=self.dataset_ref, max_rounds=self.max_rounds)
            outcome = orchestrator.run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
            self._emit("error", self.error)
            return

        self.outcome = outcome
        self.state = "finished"
        summary = outcome.get("summary") or {}
        stop = summary.get("stop") or {}
        reason = str(stop.get("reason") or "")
        self._emit("finished",
                   f"研究结束：{STOP_REASON_LABELS.get(reason, reason)}",
                   stop=stop, summary=summary)

    def _await_approval(self, plan: Mapping[str, Any]) -> bool:
        """挡住这一轮，等界面上点批准。返回是否继续。"""
        with self._lock:
            self._pending_plan = dict(plan)
        self.state = "awaiting"
        self._decision.clear()
        self._emit("awaiting", "等待人工批准这一轮实验…")
        got = self._decision.wait(timeout=APPROVAL_TIMEOUT_SECONDS)
        self.state = "running"
        if not got:
            self._emit("timeout", "等待审批超时，本轮放弃。")
            return False
        if self._stop.is_set():
            return False
        return self._approved


def _latest_reasoning(context: PlannerContext) -> str:
    return context.traces[-1].reasoning if context.traces else ""


def _latest_cot(context: PlannerContext) -> str:
    """端点返回的思维链（reasoning_content），与计划自带的 reasoning 分开。"""
    return context.traces[-1].cot if context.traces else ""


def _latest_usage(context: PlannerContext) -> dict[str, Any] | None:
    return context.traces[-1].usage if context.traces else None


def _latest_cost(context: PlannerContext) -> float | None:
    return context.traces[-1].cost if context.traces else None


def make_ask(config) -> Callable[[str, str], str]:
    """把 `ChatClient` 包成规划器要的 `ask(system, user) -> str`。

    用量挂在函数属性 `last_usage`/`last_cost` 上，端点思维链挂在
    `last_reasoning` 上：ask 协议只是 `str -> str`，测试塞的假模型不带
    这些属性，planner 读到 None 即可。
    """
    from thermoforge_agent.client import ChatClient

    client = ChatClient(config)

    def ask(system: str, user: str) -> str:
        result = client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        ask.last_usage = result.usage
        ask.last_cost = result.cost
        ask.last_reasoning = result.reasoning
        return result.content or ""

    ask.last_usage = None
    ask.last_cost = None
    ask.last_reasoning = None
    return ask


def summarize_outcome(outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    """把编排结果压成界面要的几个数。"""
    if not outcome:
        return {}
    summary = outcome.get("summary") or {}
    stop = summary.get("stop") or {}
    rounds = summary.get("rounds") or outcome.get("rounds") or []
    return {
        "rounds": len(rounds),
        "stop_reason": stop.get("reason"),
        "stop_label": STOP_REASON_LABELS.get(str(stop.get("reason") or ""),
                                             str(stop.get("reason") or "—")),
        "stop_detail": stop.get("detail") or stop.get("message"),
        "best_cvrmse": summary.get("best_cvrmse"),
        "experiments": [r.get("experiment_id") for r in rounds
                        if isinstance(r, Mapping) and r.get("experiment_id")],
    }
