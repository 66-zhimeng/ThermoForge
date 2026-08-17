"""AI 研究页：建目标 → 建视图 → 让 AI 跑研究循环。

「训练」在这个系统里不是一个按钮，而是一条闭环：
目标 → 假设 → 实验 → 证据 → 结论 → 新假设。这一页把这条闭环摆出来，
AI 负责规划每一轮的假设与模型路线，编排器负责执行与留痕。

两种模式在启动时选：自动跑到停止条件，或每轮停下来等你批准。
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from .. import cache
from ..config import agent_config
from ..context import tool_context
from ..services import catalog
from ..services.experiments import CATEGORY_LABELS
from ..services.planner import PlannerContext
from ..services.research import (
    MODE_AUTO,
    MODE_STEP,
    ResearchSession,
    make_ask,
    summarize_outcome,
)
from ..ui import copilot_banner, empty_state, envelope_result, fmt_metric

SESSION_KEY = "research_session"

# 契约里 purpose 是自由字符串，但历史取值是这几个英文词。界面显示中文、
# 写进契约的仍是英文原值——翻译只发生在展示层，不动数据。
PURPOSE_LABELS = {
    "optimization": "优化（找更省的运行方式）",
    "diagnosis": "诊断（找异常与劣化）",
    "prediction": "预测（预估未来取值）",
    "control": "控制（给执行机构下指令）",
}

SESSION_STATE_LABELS = {
    "idle": "未开始",
    "running": "运行中",
    "awaiting": "等待审批",
    "finished": "已结束",
    "error": "出错",
}
EVENT_ICONS = {
    "started": "🚀", "planning": "🤔", "plan": "💡", "awaiting": "⏸",
    "approved": "✅", "rejected": "🚫", "needs_human": "🙋",
    "plan_none": "🛑", "stopped": "⏹", "stop_requested": "⏹",
    "finished": "🏁", "error": "❌", "timeout": "⌛",
}


def research_page() -> None:
    st.title("AI 研究")
    copilot_banner("research")
    session: ResearchSession | None = st.session_state.get(SESSION_KEY)
    if session is not None and (session.running or session.state != "idle"):
        _live(session)
        return
    _setup()


# ---------------------------------------------------------------- 准备


def _setup() -> None:
    config = agent_config()
    if config is None:
        st.warning("还没配置模型接口，AI 规划不了。先去「设置」页填 API 密钥。",
                   icon="🔑")

    revisions = cache.revisions()
    if not revisions:
        empty_state("Vault 里还没有数据集", "先去「数据」页导入数据。", icon="🤖")
        return

    tab_run, tab_goal, tab_view = st.tabs(["开始研究", "① 新建研究目标",
                                           "② 新建数据视图"])
    with tab_goal:
        _goal_form(revisions)
    with tab_view:
        _view_form(revisions)
    with tab_run:
        _run_form(config, revisions)


def _goal_form(revisions) -> None:
    st.caption("研究目标定义「要预测什么、允许用什么预测、达到什么算成功」。"
               "候选输入是**封闭白名单**：实验用到的特征必须是它的子集，"
               "而且不能是派生量——这条硬约束挡的是拿目标的换算值去预测目标。")
    labels = {row.ref: f"{row.dataset_id} · {row.revision}" for row in revisions}
    ref = str(st.selectbox("数据集修订版", options=list(labels),
                           format_func=lambda r: labels[r], key="goal_ref"))
    schema = cache.schema(ref)
    codes = catalog.property_codes(schema)
    objects = schema.get("objects") or []
    object_models = sorted({str(o.get("object_model_id")) for o in objects
                            if o.get("object_model_id")})
    measured = {vid.split(".", 1)[1] for vid
                in catalog.measured_variable_ids(schema) if "." in vid}

    with st.form("goal_form"):
        name = st.text_input("目标名称", placeholder="例如：运行冷机总功率模型")
        columns = st.columns(3)
        object_model = columns[0].selectbox("对象模型", object_models or ["—"])
        purpose = columns[1].selectbox(
            "用途", list(PURPOSE_LABELS),
            format_func=lambda key: PURPOSE_LABELS[key])
        target = columns[2].selectbox("目标变量", codes)
        candidates = [code for code in codes if code != target]
        default = [code for code in candidates if code in measured]
        inputs = st.multiselect(
            "候选输入（白名单）", options=candidates, default=default,
            help="默认只勾实测变量。派生变量在目标级校验时会被直接拒绝。", placeholder="请选择…")
        derived_picked = [code for code in inputs if code not in measured]
        if derived_picked:
            st.warning(f"这些是派生变量，提交会被工具层拒绝：{derived_picked}",
                       icon="⚠️")

        st.markdown("**验收标准**")
        acc = st.columns(4)
        cvrmse_max = acc[0].number_input("CVRMSE ≤", value=0.13, step=0.01,
                                         format="%.3f")
        nmbe_abs_max = acc[1].number_input("|NMBE| ≤", value=0.02, step=0.01,
                                           format="%.3f")
        latency_max = acc[2].number_input("推理延迟(ms) ≤", value=5.0, step=1.0)
        max_experiments = acc[3].number_input("最多实验数", value=10, step=1,
                                              min_value=1)
        description = st.text_area(
            "说明（写清禁用了什么、为什么）",
            placeholder="例如：禁用 load / current_percent，它们与目标同源。")

        if st.form_submit_button("创建目标", type="primary"):
            _create_goal(ref, {
                "name": name.strip(),
                "object_model": object_model,
                "purpose": purpose,
                "target": target,
                "candidate_inputs": inputs,
                "acceptance": {
                    "cvrmse_max": float(cvrmse_max),
                    "nmbe_abs_max": float(nmbe_abs_max),
                    "inference_latency_ms_max": float(latency_max),
                },
                "max_experiments": int(max_experiments),
                "description": description.strip() or None,
            })


def _create_goal(ref: str, definition: dict[str, Any]) -> None:
    from thermoforge_research.tools import tf_goal_create

    if not definition["name"]:
        st.error("目标名称不能为空。")
        return
    if not definition["candidate_inputs"]:
        st.error("候选输入不能为空。")
        return
    envelope = tf_goal_create(tool_context(), definition, dataset_ref=ref)
    if envelope_result(envelope, success="目标已创建"):
        cache.invalidate()


def _view_form(revisions) -> None:
    st.caption("Dataset View 是实验的取数口径：哪个数据集、哪些对象、哪些特征、"
               "什么过滤条件。实验计划必须引用一个已登记的 View——"
               "view_hash 里含数据修订版 ID，所以数据换版本不会错用旧缓存。")
    labels = {row.ref: f"{row.dataset_id} · {row.revision}" for row in revisions}
    ref = str(st.selectbox("数据集修订版", options=list(labels),
                           format_func=lambda r: labels[r], key="view_ref"))
    schema = cache.schema(ref)
    codes = catalog.property_codes(schema)
    objects = [str(o.get("object_id")) for o in schema.get("objects") or []]
    object_models = sorted({str(o.get("object_model_id")) for o
                            in schema.get("objects") or []
                            if o.get("object_model_id")})

    with st.form("view_form"):
        columns = st.columns(2)
        object_model = columns[0].selectbox("对象模型", object_models or ["—"])
        target = columns[1].selectbox("目标", codes, key="view_target")
        picked_objects = st.multiselect("对象", options=objects,
                                        default=objects, placeholder="请选择…")
        features = st.multiselect(
            "特征", options=[c for c in codes if c != target],
            default=[c for c in codes if c != target], placeholder="请选择…")
        filter_var = st.selectbox(
            "过滤条件（可选）", ["（不过滤）", *[c for c in codes]],
            help="常见做法：只保留运行工况，例如 any_running=true。"
                 "停机样本会把目标压成一片零，稀释掉真正要学的行为。")
        if st.form_submit_button("登记视图", type="primary"):
            definition: dict[str, Any] = {
                "dataset": ref,
                "scope": {"object_model": object_model},
                "objects": picked_objects,
                "features": features,
                "target": target,
            }
            if filter_var != "（不过滤）":
                definition["filter"] = {filter_var: True}
            _create_view(definition)


def _create_view(definition: dict[str, Any]) -> None:
    from thermoforge_research.tools import tf_dataset_materialize

    envelope = tf_dataset_materialize(tool_context(), definition)
    if envelope_result(envelope, success="视图已登记"):
        cache.invalidate()


def _run_form(config, revisions) -> None:
    goals = [g for g in cache.goals()
             if str(g.get("status")) not in ("PUBLISH", "STOPPED")]
    views = cache.views()
    if not goals:
        empty_state("还没有可用的研究目标",
                    "去「① 新建研究目标」页签建一个。已结束（PUBLISH/STOPPED）"
                    "的目标不会出现在这里。", icon="🎯")
        return
    if not views:
        empty_state("还没有登记 Dataset View",
                    "去「② 新建数据视图」页签建一个——实验计划必须引用它。",
                    icon="🔭")
        return

    labels = {str(g["id"]): f"{g['id']} · {g.get('name') or '—'}"
              for g in goals}
    goal_id = str(st.selectbox("研究目标", options=list(labels),
                               format_func=lambda k: labels[k]))
    goal = next(g for g in goals if str(g["id"]) == goal_id)
    definition = goal.get("definition") or {}

    ref_labels = {row.ref: f"{row.dataset_id} · {row.revision}"
                  for row in revisions}
    default_ref = _guess_ref(views, definition, list(ref_labels))
    dataset_ref = str(st.selectbox(
        "数据集修订版", options=list(ref_labels),
        index=list(ref_labels).index(default_ref) if default_ref else 0,
        format_func=lambda r: ref_labels[r], key="run_ref"))

    usable_views = [v for v in views
                    if (v.get("definition") or {}).get("dataset") == dataset_ref]
    if not usable_views:
        st.warning("这个修订版下没有已登记的视图，AI 无法规划实验。"
                   "去「② 新建数据视图」为它建一个。", icon="⚠️")
        return
    st.caption(f"AI 可选的视图：{'、'.join(str(v.get('id')) for v in usable_views)}")

    columns = st.columns([2, 2, 3])
    mode = columns[0].radio(
        "模式", [MODE_AUTO, MODE_STEP],
        format_func=lambda m: "自动跑" if m == MODE_AUTO else "逐轮审批",
        help="自动：一路跑到停止条件，只在需要人拍板时停。"
             "逐轮审批：每轮 AI 给出假设和实验计划，你批准后才真跑。")
    max_rounds = columns[1].number_input("最多轮数", min_value=1, max_value=30,
                                         value=5, step=1)
    guidance = columns[2].text_area(
        "给 AI 的额外要求（可选）", height=100,
        placeholder="例如：优先试物理模型；不要用 xgboost；重点看低负荷段。")

    disabled = config is None
    if st.button("开始研究", type="primary", disabled=disabled,
                 width="stretch"):
        _start(goal, goal_id, dataset_ref, usable_views, str(mode),
               int(max_rounds), guidance.strip(), config)
    if disabled:
        st.caption("需要先在「设置」页配置模型接口。")


def _guess_ref(views, definition, refs: list[str]) -> str | None:
    for view in views:
        ref = (view.get("definition") or {}).get("dataset")
        if ref in refs:
            return str(ref)
    return refs[0] if refs else None


def _start(goal: dict, goal_id: str, dataset_ref: str, views: list[dict],
           mode: str, max_rounds: int, guidance: str, config) -> None:
    definition = dict(goal.get("definition") or {})
    definition.setdefault("id", goal_id)
    # 已批准的实验室模块进规划上下文：模型只能引用看得见的（白名单同构）
    lab_modules = [m for m in tool_context().lab_store.list()
                   if m.get("status") == "approved"]
    context = PlannerContext(goal=definition, views=views,
                             dataset_ref=dataset_ref, extra_guidance=guidance,
                             lab_modules=lab_modules)
    session = ResearchSession(goal_id, dataset_ref, mode=mode,
                              max_rounds=max_rounds, guidance=guidance)
    session.start(make_ask(config), context)
    st.session_state[SESSION_KEY] = session
    st.rerun()


# ---------------------------------------------------------------- 运行中


def _live(session: ResearchSession) -> None:
    header = st.columns([3, 1, 1])
    header[0].markdown(
        f"### {session.goal_id}　"
        f"{'自动' if session.mode == MODE_AUTO else '逐轮审批'}模式")
    if session.running:
        header[1].button("停止", on_click=session.stop,
                         width="stretch")
    else:
        if header[1].button("返回", width="stretch"):
            st.session_state.pop(SESSION_KEY, None)
            cache.invalidate()
            st.rerun()
    header[2].caption("状态："
                      + SESSION_STATE_LABELS.get(session.state, session.state))

    if session.state == "awaiting":
        _approval(session)
    _events(session)
    if not session.running:
        _outcome(session)


@st.fragment(run_every=2)
def _events(session: ResearchSession) -> None:
    """事件流。2 秒轮询一次——后台线程不能碰 Streamlit，只能让前台来取。"""
    events = session.events()
    st.markdown("#### 进度")
    if not events:
        st.caption("正在启动…")
    for event in events:
        icon = EVENT_ICONS.get(event.kind, "•")
        stamp = time.strftime("%H:%M:%S", time.localtime(event.at))
        if event.kind == "plan":
            with st.container(border=True):
                st.markdown(f"{icon} `{stamp}`　**假设**：{event.text}")
                reasoning = event.payload.get("reasoning")
                if reasoning:
                    st.caption(f"AI 的理由：{reasoning}")
                plan = event.payload.get("plan") or {}
                category = str((plan.get("model") or {}).get("category") or "")
                st.caption(f"视图 `{plan.get('view_id')}`　路线 "
                           f"{CATEGORY_LABELS.get(category, category or '—')}"
                           f"　依据 {plan.get('basis') or '（首轮无）'}")
                with st.expander("完整计划"):
                    st.json(plan)
        elif event.kind == "error":
            st.error(f"`{stamp}`　{event.text}")
        elif event.kind == "finished":
            st.success(f"{icon} `{stamp}`　{event.text}")
        else:
            st.markdown(f"{icon} `{stamp}`　{event.text}")
    if session.running:
        st.caption("每 2 秒自动刷新。实验在子进程里跑，关掉浏览器不影响它跑完。")


def _approval(session: ResearchSession) -> None:
    plan = session.pending_plan
    if not plan:
        return
    st.markdown("#### 等你拍板")
    with st.container(border=True):
        st.markdown(f"**假设**：{plan.get('statement')}")
        st.caption(f"视图 `{plan.get('view_id')}`　依据 {plan.get('basis')}")
        st.json(plan.get("model") or {})
        columns = st.columns(2)
        if columns[0].button("批准并执行", type="primary",
                             width="stretch"):
            session.decide(True)
            st.rerun()
        if columns[1].button("否决这一轮", width="stretch"):
            session.decide(False, "研究会以「无信息增益」收尾。")
            st.rerun()


def _outcome(session: ResearchSession) -> None:
    if session.error:
        st.error(f"研究异常终止：{session.error}")
        return
    summary = summarize_outcome(session.outcome)
    if not summary:
        return
    st.markdown("#### 结果")
    columns = st.columns(3)
    columns[0].metric("完成轮数", summary.get("rounds", 0))
    columns[1].metric("最优 CVRMSE",
                      fmt_metric("CVRMSE", summary.get("best_cvrmse")))
    columns[2].metric("停止原因", summary.get("stop_label") or "—")
    if summary.get("stop_detail"):
        st.caption(summary["stop_detail"])
    if summary.get("experiments"):
        st.caption("产生的实验：" + "、".join(summary["experiments"])
                   + "　去「实验结果」页看图和指标。")
    if session.planner_context and session.planner_context.traces:
        with st.expander("AI 的规划留痕（每轮的原始回复）"):
            for trace in session.planner_context.traces:
                st.markdown(f"**第 {trace.round_index + 1} 轮**　"
                            f"尝试 {trace.attempts} 次　{trace.prompt_summary}")
                if trace.error:
                    st.caption(f"最后一次错误：{trace.error}")
                st.code(trace.raw_reply[:2000] or "（空）", language="json")
