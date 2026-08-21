"""Token 用量与思维链的归账聚合：按研究目标 / 假设 / 实验。

「这个实验/目标烧了多少 token、AI 当时是怎么想的」由这里回答。网页控制台
（screens/research.py、screens/results.py）与工具层（tf_usage_overview /
tf_goal_usage / tf_experiment_usage，经 MCP 暴露给外部 Agent）读的都是
这份聚合，保证两边数字一致。

数据都是**已落盘的只读工件**，本模块只做归因与聚合，不改任何写路径：

- `research/experiments/<EXP>/planner_trace.json`：研究循环的规划留痕
  （`orchestrator._persist_planner_trace` 落盘），与实验**精确绑定**；
- `research/planner_rounds/<RG>/round_*.json`：未产出实验的规划轮次
  （`orchestrator._persist_orphan_trace` 落盘），归到目标级；
- `research/agent_sessions/*.jsonl`：副驾 / CLI Agent 的会话留痕
  （`HarnessAgent._log`），按「轮」归因。

会话归因规则（刻意几句话能讲清）：一轮问答里，工具调用参数与工具返回 id
中出现的 RG-/H-/EXP- 编号按工具分两类——**写/动作工具**（建目标、建假设、
登记/运行实验、发布模型、人工审批）提到的对象才算「这轮在它身上花了钱」，
均摊归账；**只读查询**围绕唯一对象、或涉及多个对象但同属一个目标时才
归账（后者归到目标级）。都不满足的（跨目标总览/对比、闲聊）进「未归账」。
只认工具动作、不扫回答正文——对比类回答会提到一串实验 ID，扫正文会把
同一轮重复记给每个被提及者。

边界的诚实声明：经 MCP 由外部 Agent 驱动的研究，token 烧在外部进程里，
这里看不见（除非外部 Agent 自己也走 tf_* 留痕工具）。用量留痕是后加的
功能，早期会话只留调用次数、没有 token 数。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# 实体编号都有稳定前缀（ledger._require 保证）；只认这三类研究对象
_ENTITY_RE = re.compile(r"\b(?:RG|H|EXP)-\d+\b")

# 写/动作工具：被它们引用 = 这轮真的在这个对象上花了钱。其余工具一律算
# 只读查询——只读轮次只在围绕唯一对象/同一目标时才归账（见 attribute）。
WORK_TOOLS = frozenset({
    "tf_goal_create", "tf_hypothesis_create",
    "tf_experiment_plan", "tf_experiment_run",
    "tf_model_publish", "tf_human_approval",
})

# 界面列表里用户提问的截断长度（原文可能很长，留痕里已被截到 2000）
_QUESTION_PREVIEW = 80

# 未产出实验的规划轮次（orphan）的落盘原因 → 中文标签
ORPHAN_KIND_LABELS = {
    "planner_stop": "模型主动停止",
    "no_hypothesis": "无可行假设",
    "planner_error": "规划报错",
    "needs_human": "请求人工确认",
    "lab_gate_failed": "实验室门禁未过",
    "registration_failed": "假设/实验登记失败",
}


@dataclass
class TurnUsage:
    """会话里一轮问答的用量、思维链与它操作/查询过的实体。"""

    at: str
    question: str
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    cost: float | None = None  # 整轮都没定价时是 None（不是 0）
    calls: int = 0
    reasonings: list[str] = field(default_factory=list)
    work_entities: set[str] = field(default_factory=set)  # 写/动作工具引用的
    read_entities: set[str] = field(default_factory=set)  # 只读工具引用的

    @property
    def entities(self) -> set[str]:
        return self.work_entities | self.read_entities


@dataclass
class UsageEntry:
    """归到某个实体头上的一条用量 + 思维链（界面/工具按时间逐条展示）。"""

    at: str
    source: str  # 「规划」（研究循环）或「副驾」（问答会话）
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    cost: float | None = None
    calls: float = 0.0  # 均摊后可以是分数，展示时再取整
    share: float = 1.0  # 该条占整轮的几分之几（<1 说明与其它对象均摊）
    reasoning: list[str] = field(default_factory=list)
    reasoning_labels: list[str] = field(default_factory=list)  # 与 reasoning 对齐
    question: str = ""  # 副驾轮次的用户提问（预览截断）
    raw_reply: str = ""  # 规划留痕的原始回复
    detail: str = ""  # 补充说明，如「第 2 轮规划」「与 EXP-0002 均摊」


@dataclass
class EntityUsage:
    """一个实体（目标/假设/实验）名下的汇总与逐条留痕。"""

    entity_id: str
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    cost: float | None = None
    calls: float = 0.0
    entries: list[UsageEntry] = field(default_factory=list)

    @property
    def total_tokens(self) -> float:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageBook:
    """全量归账结果：实体 → 留痕条目，外加未归账桶与谱系映射。"""

    per_entity: dict[str, list[UsageEntry]] = field(default_factory=dict)
    unassigned: list[UsageEntry] = field(default_factory=list)
    experiment_refs: dict[str, tuple[str | None, str | None]] = \
        field(default_factory=dict)  # EXP → (goal_id, hypothesis_id)
    hypothesis_refs: dict[str, str | None] = field(default_factory=dict)
    hypothesis_statement: dict[str, str] = field(default_factory=dict)
    scanned_sessions: int = 0
    planner_traces: int = 0
    orphan_traces: int = 0  # 未产出实验的规划轮次（planner_rounds/）


@dataclass
class GoalUsage:
    """一个目标的归账视图：总量 + 按假设/实验的分解 + 直接关联条目。"""

    goal_id: str
    total: EntityUsage
    hypotheses: list[EntityUsage]  # 含下属实验上卷
    experiments: list[EntityUsage]  # 仅实验自己名下（planner trace + 直接提及）
    direct: EntityUsage  # 直接提到 RG- 编号的会话条目与孤儿规划轮次


# ---------------------------------------------------------------- 会话解析


def parse_session_file(path: Path) -> list[TurnUsage]:
    """把一个会话 JSONL 切成若干轮。坏行跳过，不让一份脏文件拖垮整页。"""
    turns: list[TurnUsage] = []
    current: TurnUsage | None = None
    try:
        with open(path, encoding="utf-8") as fp:
            lines = fp.readlines()
    except OSError:
        return turns
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "message" and event.get("role") == "user":
            current = TurnUsage(
                at=str(event.get("at") or ""),
                question=str(event.get("content") or "")[:_QUESTION_PREVIEW],
            )
            turns.append(current)
            continue
        if current is None:
            continue  # 会话开头的系统类事件，不属于任何一轮
        if event.get("type") == "message" and event.get("role") == "assistant":
            _absorb_assistant(current, event)
        elif event.get("type") == "tool_result":
            found = _ENTITY_RE.findall(str(event.get("id") or ""))
            _route_entities(current, str(event.get("tool") or ""), found)
    return turns


def _route_entities(turn: TurnUsage, tool: str, found: list[str]) -> None:
    if tool in WORK_TOOLS:
        turn.work_entities.update(found)
    else:
        turn.read_entities.update(found)


def _absorb_assistant(turn: TurnUsage, event: dict[str, Any]) -> None:
    turn.calls += 1  # 端点不返回用量时 usage 是 None，但调用次数照记
    usage = event.get("usage")
    if isinstance(usage, dict):
        turn.prompt_tokens += _num(usage.get("prompt_tokens"))
        turn.completion_tokens += _num(usage.get("completion_tokens"))
    cost = event.get("cost")
    if isinstance(cost, (int, float)):
        turn.cost = (turn.cost or 0.0) + float(cost)
    reasoning = str(event.get("reasoning") or "").strip()
    if reasoning:
        turn.reasonings.append(reasoning)
    for call in event.get("tool_calls") or []:
        call = call or {}
        found = _ENTITY_RE.findall(str(call.get("arguments") or ""))
        _route_entities(turn, str(call.get("name") or ""), found)


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


# ---------------------------------------------------------------- 归因


def attribute(turns: list[TurnUsage],
              per_entity: dict[str, list[UsageEntry]],
              unassigned: list[UsageEntry],
              experiment_refs: Mapping[str, tuple[str | None, str | None]]
              | None = None,
              hypothesis_refs: Mapping[str, str | None] | None = None
              ) -> None:
    """把会话轮次归进实体桶。总量守恒：各桶之和 + 未归账 = 全部轮次。

    归账目标的选择，按优先级：
    1. 写/动作工具引用的对象（这轮真的在它们身上做了事），多对象均摊；
    2. 只读轮次围绕唯一对象 → 全额归那个对象（「EXP-0001 为什么差」）；
    3. 只读轮次涉及多个对象、但同属一个目标 → 全额归那个目标
       （「读一下 RG-0015 和它的实验 0079~0082」）；
    4. 其余（跨目标总览/对比、闲聊）进未归账。
    """
    for turn in turns:
        targets = turn.work_entities
        note = ""
        if not targets:
            reads = turn.read_entities
            if len(reads) == 1:
                targets = reads
            elif reads:
                goals = {_goal_of(entity, experiment_refs or {},
                                  hypothesis_refs or {})
                         for entity in reads}
                if len(goals) == 1:
                    goal = next(iter(goals))
                    if goal is not None:
                        targets = {goal}
                        note = "同目标问答归并到目标级"
        if not targets:
            if turn.calls or turn.entities:
                unassigned.append(_turn_entry(turn, share=1.0))
            continue
        share = 1.0 / len(targets)
        for entity in sorted(targets):
            per_entity.setdefault(entity, []).append(
                _turn_entry(turn, share=share, peers=len(targets) - 1,
                            note=note))


def _goal_of(entity: str,
             experiment_refs: Mapping[str, tuple[str | None, str | None]],
             hypothesis_refs: Mapping[str, str | None]) -> str | None:
    """实体所属目标：EXP/H 查谱系映射，RG 自己就是目标；查不到为 None。"""
    if entity.startswith("EXP-"):
        return (experiment_refs.get(entity) or (None, None))[0]
    if entity.startswith("H-"):
        return hypothesis_refs.get(entity)
    return entity if entity.startswith("RG-") else None


def _turn_entry(turn: TurnUsage, *, share: float, peers: int = 0,
                note: str = "") -> UsageEntry:
    detail = "；".join(part for part in
                       (f"与另外 {peers} 个对象均摊" if share < 1.0 else "",
                        note) if part)
    return UsageEntry(
        at=turn.at,
        source="副驾",
        prompt_tokens=turn.prompt_tokens * share,
        completion_tokens=turn.completion_tokens * share,
        cost=turn.cost * share if turn.cost is not None else None,
        calls=turn.calls * share,
        share=share,
        reasoning=list(turn.reasonings),
        question=turn.question,
        detail=detail,
    )


# ---------------------------------------------------------------- 汇总


def build_book(research_root: Path,
               turns: list[TurnUsage]) -> UsageBook:
    """扫实验/假设/孤儿规划轮次工件 + 会话轮次，产出全量归账账本。"""
    book = UsageBook()
    _scan_experiments(research_root / "experiments", book)
    _scan_hypotheses(research_root / "hypotheses", book)
    _scan_orphan_rounds(research_root / "planner_rounds", book)
    attribute(turns, book.per_entity, book.unassigned,
              experiment_refs=book.experiment_refs,
              hypothesis_refs=book.hypothesis_refs)
    return book


def _scan_experiments(directory: Path, book: UsageBook) -> None:
    if not directory.is_dir():
        return
    for exp_dir in sorted(directory.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        report = _read_json(exp_dir / "report.json")
        book.experiment_refs[exp_id] = (
            str(report.get("goal_id")) if report.get("goal_id") else None,
            str(report.get("hypothesis_id"))
            if report.get("hypothesis_id") else None,
        )
        trace = _read_json(exp_dir / "planner_trace.json")
        if not trace:
            continue
        book.planner_traces += 1
        entry = _planner_entry(trace, _mtime_iso(exp_dir / "planner_trace.json"),
                               kind=None)
        book.per_entity.setdefault(exp_id, []).append(entry)


def _scan_orphan_rounds(directory: Path, book: UsageBook) -> None:
    """未产出实验的规划轮次：planner_rounds/<RG>/round_*.json → 归到目标级。"""
    if not directory.is_dir():
        return
    for goal_dir in sorted(directory.iterdir()):
        if not goal_dir.is_dir():
            continue
        goal_id = goal_dir.name
        for path in sorted(goal_dir.glob("round_*.json")):
            trace = _read_json(path)
            if not trace:
                continue
            book.orphan_traces += 1
            book.per_entity.setdefault(goal_id, []).append(
                _planner_entry(trace, _mtime_iso(path),
                               kind=str(trace.get("orphan_kind") or "")))


def _planner_entry(trace: Mapping[str, Any], at: str,
                   kind: str | None) -> UsageEntry:
    """规划留痕（无论是否产出实验）→ 统一条目。

    `cot` 是端点返回的思维链（reasoning_content），`reasoning` 是计划 JSON
    自带的「为什么这么选」字段——两回事，都保留并贴标签。
    """
    usage = trace.get("usage") if isinstance(trace.get("usage"), dict) else {}
    reasoning: list[str] = []
    labels: list[str] = []
    cot = str(trace.get("cot") or "").strip()
    if cot:
        reasoning.append(cot)
        labels.append("思维链（端点 CoT）")
    why = str(trace.get("reasoning") or "").strip()
    if why:
        reasoning.append(why)
        labels.append("规划理由")
    detail = f"第 {int(trace.get('round_index') or 0) + 1} 轮规划"
    if kind:
        detail += f"（未产出实验：{ORPHAN_KIND_LABELS.get(kind, kind)}）"
    elif trace.get("prompt_summary"):
        detail += f"（{trace['prompt_summary']}）"
    return UsageEntry(
        at=at,
        source="规划",
        prompt_tokens=_num(usage.get("prompt_tokens")),
        completion_tokens=_num(usage.get("completion_tokens")),
        cost=float(trace["cost"]) if isinstance(
            trace.get("cost"), (int, float)) else None,
        calls=float(trace.get("attempts") or 1),
        reasoning=reasoning,
        reasoning_labels=labels,
        raw_reply=str(trace.get("raw_reply") or ""),
        detail=detail,
    )


def _scan_hypotheses(directory: Path, book: UsageBook) -> None:
    if not directory.is_dir():
        return
    import yaml  # 与 ledger 工件同款：直接读 yaml，不走 ledger 锁

    for path in sorted(directory.glob("H-*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        hid = str(doc.get("id") or path.stem)
        refs = doc.get("refs") or {}
        book.hypothesis_refs[hid] = (
            str(refs.get("goal_id")) if refs.get("goal_id") else None)
        book.hypothesis_statement[hid] = str(doc.get("statement") or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fp:
            doc = json.load(fp)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _mtime_iso(path: Path) -> str:
    """planner_trace 里没写时间戳，用文件 mtime 顶上（落盘时刻即规划时刻）。"""
    from datetime import datetime

    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds")
    except OSError:
        return ""


# ---------------------------------------------------------------- 视图


def summarize(entries: list[UsageEntry], entity_id: str) -> EntityUsage:
    """把一组条目加成汇总。cost 全 None 则保持 None（= 未定价，不是 0）。"""
    out = EntityUsage(entity_id=entity_id,
                      entries=sorted(entries, key=lambda e: e.at))
    costs: list[float] = []
    for entry in out.entries:
        out.prompt_tokens += entry.prompt_tokens
        out.completion_tokens += entry.completion_tokens
        out.calls += entry.calls
        if entry.cost is not None:
            costs.append(entry.cost)
    out.cost = sum(costs) if costs else None
    return out


def experiment_view(book: UsageBook, experiment_id: str) -> EntityUsage:
    """单个实验：planner trace（精确归属）+ 会话里归到它的份额。"""
    return summarize(book.per_entity.get(experiment_id, []), experiment_id)


def goal_view(book: UsageBook, goal_id: str) -> GoalUsage:
    """目标视图：直接提及 + 各假设 + 各实验，互不重复（每实体只算一次）。"""
    hypotheses = sorted(h for h, g in book.hypothesis_refs.items()
                        if g == goal_id)
    experiments = sorted(e for e, (g, _) in book.experiment_refs.items()
                         if g == goal_id)

    hyp_views: list[EntityUsage] = []
    for hid in hypotheses:
        entries = list(book.per_entity.get(hid, []))
        for exp_id in experiments:
            if book.experiment_refs[exp_id][1] == hid:
                entries.extend(book.per_entity.get(exp_id, []))
        hyp_views.append(summarize(entries, hid))

    exp_views = [experiment_view(book, exp_id) for exp_id in experiments]
    direct = summarize(book.per_entity.get(goal_id, []), goal_id)

    total_entries = list(direct.entries)
    for view in hyp_views:
        total_entries.extend(view.entries)
    # 没挂在任何假设下的实验（旧工件 hypothesis_id 可能为空）也要进总账
    covered = {e.entity_id for e in exp_views
               if book.experiment_refs.get(e.entity_id, (None, None))[1]
               in hypotheses}
    for view in exp_views:
        if view.entity_id not in covered:
            total_entries.extend(view.entries)
    return GoalUsage(goal_id=goal_id,
                     total=summarize(total_entries, goal_id),
                     hypotheses=hyp_views, experiments=exp_views,
                     direct=direct)


def unassigned_view(book: UsageBook) -> EntityUsage:
    return summarize(book.unassigned, "（未归账）")
