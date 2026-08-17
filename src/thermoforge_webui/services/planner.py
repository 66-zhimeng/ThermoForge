"""LLM 规划器：把编排器的 planner 回调接到一个大模型上。

编排器（`ResearchOrchestrator`）本身不认识 LLM，它只要一个
`planner(round_index, evidence) -> plan | None`。这个模块负责：

1. 把「目标 + 可用视图 + 建模路线 + 历次证据」组织成提示词；
2. 让模型输出**严格 JSON**（不是函数调用——这里只要一个结构化决策，
   走 chat completions 更省一次往返，且不受服务商 tool 支持度影响）；
3. **在本地把返回值校验一遍再交给编排器**。模型会编不存在的视图 ID、
   会写没注册的 estimator、会漏掉 basis——这些如果直接下去，报错发生在
   编排器深处，人看到的是一句莫名其妙的契约异常。

模型返回 `null` / `{"stop": ...}` 表示没有可行假设，编排器会以
`no_information_gain` 收尾；返回 `needs_human` 则以
`human_confirmation_required` 停下等人。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from thermoforge_research.model_catalog import (
    PHYSICS_BALANCE,
    PHYSICS_IDENTIFICATION,
    PHYSICS_MODELS,
)

logger = logging.getLogger(__name__)

# 可用建模路线：与 `_child.py::_build_model` 的分派一一对应。写死在这里
# 是刻意的——模型只能在已实现的路线里选，不能发明新的。
MODEL_MENU = {
    "data": {
        "estimator": ["ridge", "linear"],
        "hyperparameters": {"alpha": "float，岭回归正则强度，默认 1.0"},
    },
    "physics": {
        "physics": list(PHYSICS_MODELS),
        "hyperparameters": {
            "rated_capacity_kw": f"float，额定制冷量（{'/'.join(PHYSICS_BALANCE)} 必填）",
            "rated_power_kw": "float，额定功率",
            "inputs": ("字符串映射，形如 chw_flow=chw_flow;chw_supply_temp=…；"
                       f"{'/'.join(PHYSICS_IDENTIFICATION)} 必填"),
        },
    },
    "hybrid": {
        "physics": list(PHYSICS_MODELS),
        "residual": ["xgboost"],
        "hyperparameters": {
            "n_estimators": "int", "max_depth": "int",
            "learning_rate": "float",
            "monotone_constraints": "字符串，形如 chw_flow:1",
        },
    },
    "lab": {
        "hyperparameters": {
            "lab": "已批准的模型实验室引用（name 或 name@vN，见 available_lab_modules）",
            "inputs": "字符串映射，同 physics（按模块声明的 INPUT_ROLES）",
        },
    },
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
MAX_REPAIR_ATTEMPTS = 2  # 一次原始尝试 + 一次带报错的重试


class PlannerError(RuntimeError):
    """规划器无法产出可用计划（已重试）。"""


@dataclass
class PlannerTrace:
    """一轮规划的留痕，界面上要能看到模型到底说了什么。"""

    round_index: int
    prompt_summary: str
    raw_reply: str = ""
    plan: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    reasoning: str = ""


@dataclass
class PlannerContext:
    """规划器需要知道的环境事实（都来自本地工件，不问模型）。"""

    goal: Mapping[str, Any]
    views: Sequence[Mapping[str, Any]]
    dataset_ref: str
    modelability_note: str = ""
    extra_guidance: str = ""
    lab_modules: Sequence[Mapping[str, Any]] = field(default_factory=list)
    traces: list[PlannerTrace] = field(default_factory=list)

    @property
    def view_ids(self) -> list[str]:
        return [str(v.get("id")) for v in self.views if v.get("id")]

    @property
    def lab_refs(self) -> list[str]:
        """已批准、可在 category=lab 实验里引用的模块（name 与 name@vN 皆可）。"""
        return [str(m.get("ref")) for m in self.lab_modules if m.get("ref")]


SYSTEM_PROMPT = """你是数据中心暖通领域的建模研究员，负责规划下一轮实验。

铁律：
1. 只能使用给定的 Dataset View（view_id 必须来自清单，不得编造）。
2. 只能使用给定的建模路线与超参名，不得发明新的 estimator 或方程版本。
   内置路线之外的全新函数形式，须先经 tf_lab_submit 提交模型实验室代码
   并通过人工审批，再以 category=lab + hyperparameters.lab 引用已批准模块。
3. 除第一轮外，basis 必须引用已有的实验或发现 ID，说明这一轮基于什么证据。
4. 候选输入白名单是硬约束：视图的特征必须是目标定义里的候选输入子集。
5. 宁可停下也不要凑数：没有信息增益时返回 {"stop": "理由"}。

只输出一个 JSON 对象，不要任何解释文字、不要 Markdown 代码围栏。
"""

PLAN_SCHEMA_HINT = """{
  "statement": "假设陈述，一句话说清这轮想验证什么",
  "reasoning": "为什么这么选（给人看，不进契约）",
  "basis": ["EXP-0003"],
  "view_id": "VIEW-0001",
  "model": {
    "category": "physics|data|hybrid",
    "physics": "cooling_balance_v2",
    "estimator": null,
    "residual": null,
    "hyperparameters": {}
  },
  "y_floor": null
}"""


def build_prompt(context: PlannerContext, round_index: int,
                 evidence: Mapping[str, Any]) -> str:
    """把环境事实和历史证据摊给模型。证据只给摘要，不给原始数据行。"""
    goal = context.goal
    views_desc = []
    for view in context.views:
        definition = view.get("definition") or {}
        views_desc.append({
            "view_id": view.get("id"),
            "dataset": definition.get("dataset"),
            "target": definition.get("target"),
            "features": definition.get("features"),
            "objects": definition.get("objects"),
            "filter": definition.get("filter"),
        })
    payload = {
        "round_index": round_index,
        "goal": {
            "goal_id": goal.get("id"),
            "name": goal.get("name"),
            "target": goal.get("target"),
            "candidate_inputs": goal.get("candidate_inputs"),
            "acceptance": goal.get("acceptance"),
            "model_types": goal.get("model_types"),
            "description": goal.get("description"),
        },
        "dataset_ref": context.dataset_ref,
        "available_views": views_desc,
        "available_lab_modules": list(context.lab_refs),
        "model_menu": MODEL_MENU,
        "evidence": _trim_evidence(evidence),
    }
    parts = [
        "以下是当前研究状态：",
        json.dumps(payload, ensure_ascii=False, indent=1),
    ]
    if context.modelability_note:
        parts.append(f"可建模性预检结论：{context.modelability_note}")
    if context.extra_guidance:
        parts.append(f"使用者的额外要求：{context.extra_guidance}")
    parts.append("按这个结构输出下一轮实验计划：\n" + PLAN_SCHEMA_HINT)
    parts.append('如果不该再做实验，输出 {"stop": "理由"}；'
                 '如果需要人来拍板，输出 {"needs_human": "要人确认什么"}。')
    return "\n\n".join(parts)


def _trim_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """证据里可能带整串轮次记录，截断到最近几轮，别把上下文撑爆。"""
    trimmed = dict(evidence)
    rounds = trimmed.get("rounds")
    if isinstance(rounds, list) and len(rounds) > 6:
        trimmed["rounds"] = rounds[-6:]
        trimmed["rounds_omitted"] = len(rounds) - 6
    return trimmed


def parse_plan(text: str) -> dict[str, Any]:
    """从模型回复里抠出 JSON。模型爱加代码围栏和寒暄，这里都容忍。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except ValueError:
        match = _JSON_BLOCK.search(stripped)
        if not match:
            raise PlannerError(f"回复里找不到 JSON：{text[:200]}") from None
        try:
            return json.loads(match.group(0))
        except ValueError as exc:
            raise PlannerError(f"JSON 解析失败：{exc}") from exc


def validate_plan(
    plan: Mapping[str, Any],
    context: PlannerContext,
    round_index: int,
    evidence: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """本地校验。返回 (可用计划, 错误说明)；两者必有其一为 None。

    校验的是「编排器和契约一定会拒绝的东西」——提前拒能给出人话报错，
    也能让模型拿着错误重试一次。
    """
    if "stop" in plan:
        return None, None  # 明确停止，不是错误
    if "needs_human" in plan:
        return {"needs_human": str(plan["needs_human"]),
                "statement": str(plan.get("statement") or "需要人工确认"),
                "basis": list(plan.get("basis") or [])}, None

    statement = str(plan.get("statement") or "").strip()
    if not statement:
        return None, "缺少 statement（假设陈述）"

    view_id = str(plan.get("view_id") or "").strip()
    if view_id not in context.view_ids:
        return None, (f"view_id={view_id!r} 不在可用清单里，"
                      f"可选：{context.view_ids}")

    model = plan.get("model")
    if not isinstance(model, Mapping):
        return None, "缺少 model 对象"
    error = _validate_model(model, context)
    if error:
        return None, error

    basis = [str(item) for item in (plan.get("basis") or [])]
    has_basis_catalog = (
        evidence is not None and "basis_candidates" in evidence)
    basis_candidates = {
        str(item.get("id")) if isinstance(item, Mapping) else str(item)
        for item in ((evidence or {}).get("basis_candidates") or [])
    }
    basis_required = round_index > 0 or bool(
        (evidence or {}).get("basis_required"))
    if basis_required and not basis:
        detail = (f"；可引用：{sorted(basis_candidates)}"
                  if basis_candidates else "")
        return None, (
            "非首轮假设必须给 basis（引用已有实验或发现 ID）" + detail)
    unknown_basis = sorted(set(basis) - basis_candidates)
    if has_basis_catalog and unknown_basis:
        return None, (
            f"basis 包含不可用 ID：{unknown_basis}；"
            f"可引用：{sorted(basis_candidates)}")

    result: dict[str, Any] = {
        "statement": statement,
        "basis": basis,
        "view_id": view_id,
        "model": dict(model),
    }
    if plan.get("y_floor") is not None:
        try:
            result["y_floor"] = float(plan["y_floor"])
        except (TypeError, ValueError):
            return None, "y_floor 必须是数字"
    if plan.get("validation"):
        result["validation"] = dict(plan["validation"])
    return result, None


def _validate_model(model: Mapping[str, Any],
                    context: PlannerContext | None = None) -> str | None:
    category = str(model.get("category") or "")
    if category not in MODEL_MENU:
        return f"model.category={category!r} 不可用，只能是 {list(MODEL_MENU)}"
    menu = MODEL_MENU[category]
    if category == "data":
        estimator = model.get("estimator") or model.get("residual")
        if estimator not in menu["estimator"]:
            return (f"data 路线的 estimator={estimator!r} 未实现，"
                    f"可选 {menu['estimator']}")
    if category in ("physics", "hybrid"):
        physics = model.get("physics")
        if physics not in menu["physics"]:
            return (f"{category} 路线的 physics={physics!r} 未实现，"
                    f"可选 {menu['physics']}")
    if category == "hybrid" and model.get("residual") not in menu["residual"]:
        return (f"hybrid 路线的 residual={model.get('residual')!r} 未实现，"
                f"可选 {menu['residual']}")
    if category == "lab":
        hp = model.get("hyperparameters") or {}
        lab_ref = str(hp.get("lab") or "").strip()
        if not lab_ref:
            return ("lab 路线必须在 hyperparameters.lab 给已批准模块引用"
                    "（name 或 name@vN）")
        if context is not None:
            name = lab_ref.partition("@v")[0]
            known = {r.partition("@v")[0] for r in context.lab_refs}
            if context.lab_refs and name not in known:
                return (f"lab 引用 {lab_ref!r} 不在已批准清单里，"
                        f"可选：{context.lab_refs}；新模块须先 tf_lab_submit "
                        "并经人工审批")
    return None


def make_planner(ask: Callable[[str, str], str], context: PlannerContext):
    """造一个符合编排器协议的 planner。

    `ask(system, user) -> str` 由调用方注入，方便测试时塞一个假模型。
    """

    def planner(round_index: int,
                evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
        prompt = build_prompt(context, round_index, evidence)
        trace = PlannerTrace(round_index=round_index,
                             prompt_summary=_summarize_prompt(evidence))
        context.traces.append(trace)
        feedback = ""
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            trace.attempts = attempt
            try:
                reply = ask(SYSTEM_PROMPT, prompt + feedback)
            except Exception as exc:  # 网络/鉴权等，交给上层显示
                trace.error = f"{type(exc).__name__}: {exc}"
                raise PlannerError(trace.error) from exc
            trace.raw_reply = reply
            try:
                raw = parse_plan(reply)
            except PlannerError as exc:
                feedback = f"\n\n上一次回复无法解析：{exc}。只输出 JSON 对象。"
                trace.error = str(exc)
                continue
            trace.reasoning = str(raw.get("reasoning") or "")
            plan, error = validate_plan(
                raw, context, round_index, evidence=evidence)
            if error is None:
                trace.plan = dict(plan) if plan else None
                trace.error = None
                return plan
            trace.error = error
            feedback = f"\n\n上一次的计划不合法：{error}。请修正后重新输出 JSON。"
        raise PlannerError(trace.error or "规划器多次尝试后仍未产出可用计划")

    return planner


def _summarize_prompt(evidence: Mapping[str, Any]) -> str:
    rounds = evidence.get("rounds")
    count = len(rounds) if isinstance(rounds, list) else 0
    best = evidence.get("best_cvrmse")
    return (f"已有 {count} 轮证据"
            + (f"，当前最优 CVRMSE {best:.4f}" if isinstance(best, (int, float))
               else "，尚无最优指标"))
