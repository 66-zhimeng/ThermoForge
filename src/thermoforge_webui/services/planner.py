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

from thermoforge_core.canonical import canonical_json
from thermoforge_models.lab import parse_inputs
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
            "lab": "模型实验室引用（name 或 name@vN，见 available_lab_modules）",
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
    """一轮规划的留痕，界面上要能看到模型到底说了什么、想了什么。"""

    round_index: int
    prompt_summary: str
    raw_reply: str = ""
    plan: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    reasoning: str = ""  # 计划 JSON 自带的「为什么这么选」字段
    cot: str = ""        # 端点返回的思维链（reasoning_content），与上一行是两回事
    usage: dict[str, Any] | None = None  # 本轮各次模型调用的用量合计
    cost: float | None = None            # 按 [pricing] 估算的费用（元）


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
        """可在 category=lab 实验里引用的模块（过校验且未停用；name@vN 皆可）。"""
        return [str(m.get("ref")) for m in self.lab_modules
                if m.get("ref") and m.get("runnable", True)]


# 文件缺失时的兜底（内容与 planner.md 保持同义；真正生效的是那个文件，
# 且规划器绑定了 harness/skills 的取证与系统辨识技能）
_FALLBACK_PROMPT = """你是数据中心暖通领域的建模研究员，负责规划下一轮实验。

铁律：
1. 只能使用给定的 Dataset View（view_id 必须来自清单，不得编造）。
2. 只能使用给定的建模路线与超参名，不得发明新的 estimator 或方程版本。
   内置路线之外的全新函数形式，走模型实验室：tf_lab_submit 提交模块代码，
   过结构校验后即可以 category=lab + hyperparameters.lab 引用（不需要审批）。
3. 除第一轮外，basis 必须引用已有的实验或发现 ID，说明这一轮基于什么证据。
4. 候选输入白名单是硬约束：视图的特征必须是目标定义里的候选输入子集。
5. 宁可停下也不要凑数：没有信息增益时返回 {"stop": "理由"}。

只输出一个 JSON 对象，不要任何解释文字、不要 Markdown 代码围栏。
"""


def system_prompt() -> str:
    """规划器提示词：以 `harness/prompts/planner.md` + 绑定技能为准。

    每轮现读，改文件立刻生效（同副驾）。技能书（现场数据取证、系统辨识
    阶梯）就是靠这条路进规划器的——绕开 prompts 层等于把方法论扔了。
    """
    from thermoforge_agent import prompts

    return prompts.load("planner", fallback=_FALLBACK_PROMPT)


PLAN_SCHEMA_HINT = """{
  "statement": "假设陈述，一句话说清这轮想验证什么",
  "reasoning": "为什么这么选（给人看，不进契约）",
  "basis": ["EXP-0003"],
  "view_id": "VIEW-0001",
  "model": {
    "category": "physics|data|hybrid|lab",
    "physics": "cooling_balance_v2",
    "estimator": null,
    "residual": null,
    "hyperparameters": {}
  },
  "y_floor": null
}

内置闭集表达不了的函数形式，在同一份 JSON 里加 lab_module，编排器会先提交
它过门禁（AST 扫描 + 子进程五连检），过了就在本轮直接引用，不需要人审批；
没过会把失败明细给你，下一轮改代码重交。此时 model.category 填 "lab"：

{
  ... 上面那些字段 ...,
  "model": {"category": "lab", "hyperparameters": {"lab": "模块名"}},
  "lab_module": {
    "name": "fan_law_static",
    "description": "一句话说明这个函数形式在建模什么",
    "source": "完整的 Python 模块源码字符串"
  }
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
        # 带上每个模块声明要吃的列。只给 ref 时规划器看不出「这个模块只读
        # frequency」，于是反复用换视图冒充换实验（RG-0028 四轮同指标）。
        "available_lab_modules": [
            {"ref": m.get("ref"), "reads_columns": m.get("input_roles") or [],
             "description": m.get("description")}
            for m in context.lab_modules if m.get("runnable", True)
        ] or list(context.lab_refs),
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
        # 明确停止，不是错误。**理由必须带出去**：模型判断「为什么做不到」
        # 往往是整轮研究最有价值的产出（比如「瓶颈是缺阀位测点」——那会
        # 直接变成给现场的加表建议）。此前这里返回 (None, None)，停止文本
        # 落在地上，编排器只记下一句通用的「planner 无可行假设」。
        return {"stop": str(plan["stop"]),
                "reasoning": str(plan.get("reasoning") or "")}, None
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
    # 同一轮里新交的模块还不在 context.lab_refs 里（那份清单是进本轮时的
    # 快照），要先认下它，否则「提交 + 立刻引用」永远过不了校验。
    pending = plan.get("lab_module")
    pending_lab = (str((pending or {}).get("name") or "").strip()
                   if isinstance(pending, Mapping) else "")
    error = _validate_model(model, context, pending_lab=pending_lab or None)
    if error:
        return None, error
    pending_source = (str((pending or {}).get("source") or "")
                      if isinstance(pending, Mapping) else "")
    error = _lab_view_error(context, view_id, model, pending_source or None)
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

    duplicate = _duplicate_of(context, view_id, model)
    if duplicate is not None:
        # round_index < 0 是补进来的历史实验（跨进程去重），不是本次运行的轮次
        where = (f"第 {duplicate} 轮" if duplicate >= 0 else "本目标此前跑过的某个实验")
        return None, (
            f"这个计划和{where}等价（模块真正会读的列 + 模型 + 超参都相同）："
            "确定性实验重跑只会得到逐位一样的指标，白烧一轮。**注意 lab 模块"
            "只读它 INPUT_ROLES 声明的列，视图里多出来的列它不看——所以光换"
            "视图不算换实验。**要换就换真的：改 INPUT_ROLES 让模块吃上新列"
            "（tf_lab_submit 提交新版本）、换模型类别、或换超参量级。"
            "若确实没有新招可试，回 {\"stop\": \"没有可试的新假设\"}。")

    result: dict[str, Any] = {
        "statement": statement,
        "basis": basis,
        "view_id": view_id,
        "model": dict(model),
    }
    lab_module = plan.get("lab_module")
    if lab_module is not None:
        if not isinstance(lab_module, Mapping):
            return None, "lab_module 必须是对象 {name, source, description?}"
        lab_name = str(lab_module.get("name") or "").strip()
        lab_source = str(lab_module.get("source") or "")
        if not lab_name or not lab_source.strip():
            return None, "lab_module 必须同时给 name 和 source（模块完整源码）"
        result["lab_module"] = {
            "name": lab_name, "source": lab_source,
            "description": str(lab_module.get("description") or "") or None,
        }
    if plan.get("y_floor") is not None:
        try:
            result["y_floor"] = float(plan["y_floor"])
        except (TypeError, ValueError):
            return None, "y_floor 必须是数字"
    if plan.get("validation"):
        result["validation"] = dict(plan["validation"])
    return result, None


def _normalize_hyperparameters(
    hyperparameters: Mapping[str, Any], lab_refs: Sequence[str] = (),
) -> dict[str, Any]:
    """把「写法不同但跑起来一样」的超参归一，否则去重形同虚设。

    两处坑，都是实测烧掉整轮才发现的（RG-0021 十一轮里六轮是重复）：

    - `inputs` 是角色→列名的映射字符串，`;` 与 `,` 等价、条目顺序无关
      （见 `thermoforge_models.lab.parse_inputs`）。`"a=a;b=b"` 与
      `"b=b,a=a"` 是同一件事，按原始字符串比对却是两件。
    - `lab` 引用可以是裸名也可以是 `name@vN`。裸名在 runner 侧解析成最新
      版本，所以 `"m"` 与 `"m@v3"`（当 v3 是最新时）跑的是同一份代码。
    """
    out: dict[str, Any] = {}
    latest = {}
    for ref in lab_refs:
        name, _, version = str(ref).partition("@v")
        if version.isdigit():
            latest[name] = max(latest.get(name, 0), int(version))
    for key in sorted(hyperparameters):
        value = hyperparameters[key]
        if key == "inputs":
            parsed = parse_inputs(value)
            value = sorted(f"{k}={v}" for k, v in (parsed or {}).items())
        elif key == "lab":
            name, sep, version = str(value).partition("@v")
            if not sep and name in latest:
                value = f"{name}@v{latest[name]}"
        out[str(key)] = value
    return out


#: 视图特征之外、runner 一定会带上的结构列（模块可以直接读）
_STRUCTURAL_COLUMNS = ("object_id", "timestamp")


def _lab_roles(context: PlannerContext | None, model: Mapping[str, Any],
               pending_source: str | None = None) -> list[str] | None:
    """模块声明要吃哪些角色列；查不到返回 None（不是空列表——空列表是
    「通用模型，不挑列」这个明确语义）。"""
    hp = model.get("hyperparameters") or {}
    lab_ref = str(hp.get("lab") or "").strip()
    if not lab_ref:
        return None
    name, _, version = lab_ref.partition("@v")
    if context is not None:
        matched = [m for m in context.lab_modules
                   if str(m.get("name")) == name
                   and (not version or str(m.get("version")) == version)]
        if matched:
            latest = max(matched, key=lambda m: int(m.get("version") or 0))
            roles = latest.get("input_roles")
            if roles is not None:
                return [str(r) for r in roles]
    if pending_source:
        return _roles_from_source(pending_source)
    return None


def _roles_from_source(source: str) -> list[str] | None:
    """从模块源码里读 INPUT_ROLES 字面量（同轮提交 + 立刻引用时用得上）。

    只认字面量：靠计算得出的声明这里读不到，返回 None 退回宽松处理。
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "INPUT_ROLES"
                   for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        # dict 时取键（同 `_lab_check.py` 的 `list(INPUT_ROLES)`）
        if isinstance(value, Mapping):
            return [str(k) for k in value]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return None
    return None


def _effective_columns(roles: Sequence[str],
                       hyperparameters: Mapping[str, Any]) -> list[str]:
    """角色名经 `inputs` 映射后，模块真正会去 df 里取的列名。"""
    mapping = parse_inputs(hyperparameters.get("inputs")) or {}
    return [str(mapping.get(role, role)) for role in roles]


def _view_of(context: PlannerContext, view_id: str) -> Mapping[str, Any]:
    for view in context.views:
        if str(view.get("id")) == view_id:
            return view.get("definition") or {}
    return {}


def _lab_view_error(context: PlannerContext, view_id: str,
                    model: Mapping[str, Any],
                    pending_source: str | None) -> str | None:
    """模块声明的列，视图里到底有没有。

    这道检查在实验之前，是因为**五连检结构上检不出列名写错**：
    `_lab_check.py` 用模块自己声明的 INPUT_ROLES 造合成数据，声明成
    `fan_freq_mean` 就喂 `fan_freq_mean`，五检全绿。等真跑到实数据上才
    `KeyError`——8-19 的 RG-0028 就这样连废三轮（EXP-0153/0155/0156，真
    实列名是 `tower_freq_mean` / `bank_fan_count`）。
    """
    if str(model.get("category") or "") != "lab":
        return None
    roles = _lab_roles(context, model, pending_source)
    if not roles:                      # 未知或通用模型，不设卡
        return None
    definition = _view_of(context, view_id)
    available = {str(c) for c in (definition.get("features") or [])}
    available.update(_STRUCTURAL_COLUMNS)
    if definition.get("target"):
        available.add(str(definition["target"]))
    columns = _effective_columns(roles, model.get("hyperparameters") or {})
    missing = [c for c in columns if c not in available]
    if not missing:
        return None
    return (
        f"模块声明要吃的列 {missing} 在 {view_id} 里不存在——真跑起来会 "
        f"KeyError，白烧一轮。该视图可用列：{sorted(available)}。"
        "要么换一个含这些列的视图，要么用 tf_lab_submit 提交把 INPUT_ROLES "
        "改成真实列名的新版本，要么用 hyperparameters.inputs 把角色映射到"
        "真实列名（形如 \"role=column;role2=column2\"）。")


def _fingerprint_scope(context: PlannerContext | None, view_id: str,
                       model: Mapping[str, Any]) -> dict[str, Any]:
    """决定「换了视图算不算换了实验」。

    对内置路线，视图 ID 就是范围。对 `category=lab` 不成立：模块把
    INPUT_ROLES 写死，视图里多出来的列它根本不读——换视图跑出来的指标
    逐位相同。8-19 的 RG-0028 EXP-0158/0163/0164/0165 就是同一个
    `ct_fan_tower_shared_b`（只声明 `frequency`）在三张视图上跑了四遍，
    CVRMSE 全是 0.0645，查重却因为 view_id 不同而放行。所以 lab 路线按
    **模块真正会读的列 + 目标**来算范围。
    """
    if str(model.get("category") or "") == "lab" and context is not None:
        roles = _lab_roles(context, model)
        if roles is not None:
            definition = _view_of(context, view_id)
            columns = _effective_columns(
                roles, model.get("hyperparameters") or {})
            return {"target": definition.get("target"),
                    "columns": sorted(set(columns))}
    return {"view_id": view_id}


def _plan_fingerprint(view_id: str, model: Mapping[str, Any],
                      lab_refs: Sequence[str] = (),
                      context: PlannerContext | None = None) -> str:
    """决定实验结果的三件事：数据范围 + 模型路线 + 超参。种子由编排器固定。"""
    payload = {
        "scope": _fingerprint_scope(context, view_id, model),
        "category": model.get("category"),
        "physics": model.get("physics"),
        "estimator": model.get("estimator"),
        "residual": model.get("residual"),
        "hyperparameters": _normalize_hyperparameters(
            model.get("hyperparameters") or {}, lab_refs),
    }
    return canonical_json(payload)


def _duplicate_of(context: PlannerContext, view_id: str,
                  model: Mapping[str, Any]) -> int | None:
    """这一计划是否与本次运行里已提交过的某一轮完全相同。

    实验是确定性的（固定种子 + 环境锁），同指纹重跑必然得到逐位一样的指标。
    没有这道拦截时，规划器会在停滞后反复重交同一个计划——实测 8-18 的
    EXP-0083/0084、8-19 的 EXP-0092~0096 都是这么烧掉的，编排器的无增益
    计数还得等满 N 轮才停。
    """
    lab_refs = list(context.lab_refs)
    target = _plan_fingerprint(view_id, model, lab_refs, context)
    for trace in context.traces:
        plan = trace.plan
        if not plan or not plan.get("view_id") or not plan.get("model"):
            continue
        if _plan_fingerprint(str(plan["view_id"]), plan["model"],
                             lab_refs, context) == target:
            return trace.round_index
    return None


def _validate_model(model: Mapping[str, Any],
                    context: PlannerContext | None = None,
                    *, pending_lab: str | None = None) -> str | None:
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
            return ("lab 路线必须在 hyperparameters.lab 给模块引用"
                    "（name 或 name@vN）")
        if context is not None:
            name = lab_ref.partition("@v")[0]
            known = {r.partition("@v")[0] for r in context.lab_refs}
            if pending_lab:
                known.add(pending_lab.partition("@v")[0])
            if context.lab_refs and name not in known:
                return (f"lab 引用 {lab_ref!r} 不在可用清单里，"
                        f"可选：{context.lab_refs}；新模块先用 tf_lab_submit "
                        "提交，过校验后即可引用")
    return None


def make_planner(ask: Callable[[str, str], str], context: PlannerContext):
    """造一个符合编排器协议的 planner。

    `ask(system, user) -> str` 由调用方注入，方便测试时塞一个假模型。

    当轮的 `PlannerTrace` 同步挂在函数属性 `last_trace` 上（同
    `make_ask` 的 `last_usage` 惯例）：编排器拿到 EXP-ID 后读它，
    把思维链/原始回复/用量随实验工件落盘。
    """

    def planner(round_index: int,
                evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
        prompt = build_prompt(context, round_index, evidence)
        trace = PlannerTrace(round_index=round_index,
                             prompt_summary=_summarize_prompt(evidence))
        context.traces.append(trace)
        planner.last_trace = trace
        feedback = ""
        usage_acc: dict[str, Any] = {}
        cost_acc = 0.0
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            trace.attempts = attempt
            try:
                reply = ask(system_prompt(), prompt + feedback)
            except Exception as exc:  # 网络/鉴权等，交给上层显示
                trace.error = f"{type(exc).__name__}: {exc}"
                raise PlannerError(trace.error) from exc
            # ask 闭包把每次调用的用量/思维链挂在函数属性上（见 make_ask）；
            # 修复重试也是真实调用，要累计而不是覆盖
            call_usage = getattr(ask, "last_usage", None) or {}
            for key, value in call_usage.items():
                if isinstance(value, (int, float)):
                    usage_acc[key] = usage_acc.get(key, 0) + value
            call_cost = getattr(ask, "last_cost", None)
            if call_cost:
                cost_acc += call_cost
            # 端点思维链（reasoning_content）：与 raw_reply 同规则，留最后一次
            # 尝试的——它和最终被采纳/最终失败的那次回复对应
            trace.cot = getattr(ask, "last_reasoning", None) or ""
            trace.usage = dict(usage_acc) or None
            trace.cost = cost_acc or None
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
