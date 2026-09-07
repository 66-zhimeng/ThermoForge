"""V2 研究运行页：控制独立后台服务，不在 Streamlit 会话中运行智能体。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import streamlit as st

from thermoforge_v2.profile import CODEX_EFFORT, CODEX_MODEL
from thermoforge_v2.reports import build_report, render_html, render_markdown
from thermoforge_webui.context import tool_context
from thermoforge_webui.ui import copilot_banner


_RUN_KEY = "v2_run_id"
_LABELS = {"created": "已创建", "queued": "排队中", "pending": "待启动", "starting": "启动中",
           "running": "执行中", "paused": "已暂停", "pausing": "暂停中",
           "waiting": "等待反馈", "waiting_experiment": "等待实验", "idle": "空闲",
           "disconnected": "连接中断", "interrupted": "已中断", "failed": "失败",
           "completed": "已完成", "cancelled": "已取消", "cancelling": "取消中",
           "budget_exhausted": "预算耗尽", "needs_input": "等待必要输入",
           "committed": "已冻结", "independent_proposals": "独立提交方案",
           "independent_experiments": "独立实验与反馈", "sharing": "共享证据与调整",
           "connect": "连接实例", "reasoning": "研究中", "tool": "调用研究工具",
           "awaiting_independent_proposals": "等待首轮方案齐备",
           "awaiting_coordination": "等待主智能体比较", "research_stopped": "已说明停止依据",
           "reported": "已提交报告", "reported_with_limits": "已报告预算限制",
           "report_due": "待补齐报告", "final_report_due": "待生成综合报告", "stopped": "已停止"}


def _client():
    """与现有页面使用同一工件根；客户端只连接/控制独立服务。"""
    from thermoforge_v2.client import V2Client

    context = tool_context()
    return V2Client(research_root=context.research_root, vault_root=context.vault_root,
                    models_root=context.models_root)


def _label(status: Any) -> str:
    return _LABELS.get(str(status), str(status or "未记录"))


def _display(value: Any) -> str:
    if value is None:
        return "不可得"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, OverflowError, OSError):
            pass
    return _display(value)


def _options(items: Any, *, dataset: bool = False) -> dict[str, str]:
    """只列 prepare 实际发现的对象，不为缺失数据拼造修订号。"""
    options = {}
    for item in items or []:
        if isinstance(item, str):
            options[item] = item
        elif isinstance(item, dict):
            key = (item.get("ref") or item.get("dataset_ref") or item.get("revision_id")) if dataset else item.get("id") or item.get("goal_id")
            if key:
                options[str(key)] = f"{key} · {item.get('name') or item.get('title') or ''}".rstrip(" ·")
    return options


def research_v2_page() -> None:
    st.title("V2 研究运行")
    copilot_banner("research_v2")
    st.caption("ThermoForge 在后台管理主 Codex 与五个候选 Codex；关闭页面后仍由后台服务继续研究。")
    try:
        client = _client()
        preparation = client.prepare()
        runs = client.list_runs()
    except Exception as exc:
        st.error(f"无法连接 V2 研究服务：{exc}")
        st.caption("检查服务启动状态、Codex 安装和认证后，重新打开或刷新此页。")
        return

    create, monitor = st.tabs(["新建研究", "运行与报告"])
    with create:
        _create(client, preparation)
    with monitor:
        options = {str(r.get("run_id") or (r.get("run") or {}).get("run_id")):
                   str(r.get("run_id") or (r.get("run") or {}).get("run_id"))
                   for r in runs if isinstance(r, dict) and (r.get("run_id") or (r.get("run") or {}).get("run_id"))}
        if not options:
            st.info("还没有 V2 研究运行。可从「新建研究」启动，或通过副驾驶启动后回到此处查看。")
            return
        if st.session_state.get(_RUN_KEY) not in options:
            st.session_state[_RUN_KEY] = next(iter(options))
        run_id = st.selectbox("研究运行", list(options), format_func=options.get, key=_RUN_KEY)
        if st.button("刷新状态", key="v2_refresh"):
            st.rerun()
        _monitor(client, str(run_id))


def _create(client, preparation: dict[str, Any]) -> None:
    available = bool(preparation.get("available"))
    for error in preparation.get("errors") or []:
        st.warning(_display(error))
    if not available:
        st.info("全 Codex 研究当前不可启动。下方保留配置入口；请根据服务诊断完成准备。")
    goals = _options(preparation.get("goals"))
    datasets = _options(preparation.get("datasets"), dataset=True)
    if not goals or not datasets:
        st.info("需要已有研究目标与数据修订版。请先在「AI 研究」和「数据」页准备，或让副驾驶完成准备。")
        return
    defaults = preparation.get("defaults") or {}
    st.caption("自主研究：统一目标与预算，由各候选独立选择方法。先提交方案，再实验、分析反馈并修订。")
    strategy = st.selectbox("研究策略", ["independent", "top_k", "adaptive"],
                            key="v2_create_strategy",
                            format_func=lambda s: {"independent": "独立探索", "top_k": "保留优质路线", "adaptive": "根据进展调整策略"}[s])
    if strategy == "independent" and st.session_state.get("v2_create_reuse"):
        st.session_state["v2_create_reuse"] = False
    reuse = st.checkbox("共享后复用相同实验结果", key="v2_create_reuse",
                        disabled=strategy == "independent",
                        help="仅共享策略可用。首轮仍各自执行；复用结果会明确标注，不计为独立复现。")
    if strategy == "independent":
        st.caption("独立探索全程保留各自的研究路线，结题后统一比较。")
    else:
        st.caption("先独立完成首轮方案和实验反馈，再由主智能体分享有依据的结果。")
    with st.form("v2_create_run"):
        left, right = st.columns(2)
        goal_id = left.selectbox("研究目标", list(goals), format_func=goals.get, key="v2_create_goal")
        dataset_ref = right.selectbox("数据修订版", list(datasets), format_func=datasets.get, key="v2_create_dataset")
        view_id = left.text_input("数据视图 ID（可选）", value=str(defaults.get("view_id") or ""))
        model = right.text_input("Codex 模型", value=CODEX_MODEL, disabled=True)
        reasoning = right.text_input("思考强度（最高）", value=CODEX_EFFORT, disabled=True)
        right.caption("主智能体与所有候选统一使用 GPT-6 Astra、最高思考强度；每轮使用普通速度，关闭 Fast。")
        candidates = left.selectbox("研究配置", [5, 0], index=0,
                                    format_func=lambda n: "主 Codex ＋ 5 个候选" if n else "单 Codex 对照基线")
        max_experiments = left.number_input("全局实验上限", min_value=1, value=int(defaults.get("max_experiments", 20)), step=1)
        per_track = right.number_input("每条轨迹实验上限", min_value=1, value=int(defaults.get("max_experiments_per_track", 4)), step=1)
        max_turns = left.number_input("每条轨迹最多执行片段", min_value=3, value=max(3, int(defaults.get("max_turns", 8))), step=1,
                                     help="至少保留方案、实验与结题三个执行片段。")
        token_budget = right.number_input("全局 token 预算", min_value=1000, value=int(defaults.get("token_budget", 1000000)), step=1000)
        workers = left.number_input("实验并发数（当前固定串行）", min_value=1, max_value=1, value=1, step=1, disabled=True)
        timeout = right.number_input("单次实验超时（秒）", min_value=10, max_value=7200, value=int(defaults.get("experiment_timeout_seconds", 300)), step=30)
        guidance = st.text_area("研究目标与额外要求", value=str(defaults.get("guidance") or ""),
                                placeholder="例如：优先验证物理约束；分析失败原因；每项想法登记真实来源。")
        st.caption("六个 Codex 研究实例独立并行；训练实验在后台服务中全局串行排队。用量受后端可观测能力限制，未知费用不显示为零。")
        st.caption("token 预算是累计输入与输出的软上限，包含缓存输入；在途请求可能超出，不自动补充额度。")
        submitted = st.form_submit_button("启动后台研究", type="primary", disabled=not available)
    if not submitted:
        return
    config = {"goal_id": goal_id, "dataset_ref": dataset_ref, "candidates": candidates,
              "research_mode": "autonomous", "reuse_experiments": bool(reuse and strategy != "independent"),
              "model": model, "reasoning_effort": reasoning,
              "max_experiments": int(max_experiments),
              "max_experiments_per_track": int(per_track), "max_turns": int(max_turns),
              "token_budget": int(token_budget), "strategy": strategy,
              "experiment_workers": int(workers), "experiment_timeout_seconds": int(timeout),
              "guidance": guidance.strip()}
    if view_id.strip():
        config["view_id"] = view_id.strip()
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    keys = st.session_state.setdefault("v2_start_keys", {})
    request_key = keys.setdefault(digest, str(uuid4()))
    try:
        result = client.start(config, idempotency_key=request_key)
        run = result.get("run") or result
        run_id = run.get("run_id")
        if not run_id:
            st.error("服务未返回运行 ID，启动结果尚未确认；重试会复用同一请求标识。")
            return
        st.session_state[_RUN_KEY] = str(run_id)
        keys.pop(digest, None)
        st.success(f"已提交 {run_id}，实际状态：{_label(run.get('status'))}。在「运行与报告」查看。")
        st.rerun()
    except Exception as exc:
        st.error(f"启动失败：{exc}")


@st.fragment(run_every=3)
def _monitor(client, run_id: str) -> None:
    try:
        snapshot = client.get_run(run_id)
    except Exception as exc:
        st.error(f"读取运行失败：{exc}")
        return
    run = snapshot.get("run") or {}
    tracks = snapshot.get("tracks") or []
    jobs = snapshot.get("jobs") or []
    proposals = snapshot.get("proposals") or []
    stops = snapshot.get("stops") or []
    autonomous = (run.get("config") or {}).get("research_mode", "acceptance") == "autonomous"
    status = run.get("status")
    columns = st.columns(4)
    columns[0].metric("运行状态", _label(status))
    columns[1].metric("实例记录", len(tracks))
    columns[2].metric("执行中的轨迹", sum(t.get("status") == "running" for t in tracks))
    columns[3].metric("实验请求", len(jobs))
    if autonomous:
        progress = st.columns(3)
        stage = "生成综合报告" if run.get("final_report_phase") else _label(run.get("research_stage"))
        progress[0].metric("研究阶段", stage)
        progress[1].metric("已冻结方案", len(proposals))
        progress[2].metric("已登记停止决定", len(stops))
        if run.get("research_closure"):
            st.info("已进入 token 预算收尾：暂停新增方案与实验，优先整理发现、报告和停止依据。"
                    "预留量是估算，正在执行的请求仍可能超出；不会自动追加额度。")
    else:
        st.caption("这是功能验收运行，按其原有流程展示；新建研究使用自主探索。")
    st.caption(f"运行 {run_id} · 版本 {run.get('version', '不可得')} · 最近更新 {_timestamp(run.get('updated_at'))}")
    if run.get("error"):
        st.error(_display(run["error"]))
    controls = st.columns(3)
    terminal = status in {"completed", "cancelled"}
    for column, action, text, disabled in (
            (controls[0], "pause", "暂停研究", status not in {"running", "queued"}),
            (controls[1], "resume", "恢复研究", status not in {"paused", "interrupted", "needs_input", "budget_exhausted"}),
            (controls[2], "cancel", "取消研究", terminal or status == "cancelling")):
        if column.button(text, disabled=disabled, key=f"v2_{action}_{run_id}", width="stretch"):
            _control(client, run, action)

    st.markdown("#### 后台实例与研究轨迹")
    st.caption("进程 ID 仅用于核对实例；是否推进研究以阶段、最近事件和实验反馈为准。页面每 3 秒刷新。")
    if tracks:
        latest_proposals = {p.get("track_id"): p for p in proposals}
        latest_stops = {s.get("track_id"): s for s in stops}
        st.dataframe([{"轨迹": t.get("track_id"), "角色": "主智能体" if t.get("role") == "main" else t.get("role"),
                       "状态": _label(t.get("status")), "阶段": _label(t.get("phase")), "进程 ID": t.get("pid"),
                       "方案版本": latest_proposals.get(t.get("track_id"), {}).get("version"),
                       "停止依据": latest_stops.get(t.get("track_id"), {}).get("reason"),
                       "实例": t.get("instance_id"), "会话": t.get("session_id") or t.get("thread_id"),
                       "执行片段": t.get("turns"), "最近事件": _timestamp(t.get("last_event_at") or t.get("updated_at")),
                       "用量": _display(t.get("usage")), "错误": _display(t.get("error"))}
                      for t in tracks], hide_index=True, width="stretch")
    else:
        st.info("服务尚未登记研究实例。")
    if autonomous or proposals or stops:
        with st.expander("研究方案与停止依据", expanded=bool(proposals or stops)):
            if proposals:
                st.dataframe([{"方案": p.get("id"), "轨迹": p.get("track_id"), "版本": p.get("version"),
                               "状态": _label(p.get("status")), "想法": p.get("idea_id"),
                               "目的": {"explore": "探索", "refine": "根据反馈修订", "replicate": "复现"}.get(p.get("purpose"), p.get("purpose")),
                               "模型": _display(p.get("model")), "依据实验": _display(p.get("parent_job_ids") or [])}
                              for p in proposals], hide_index=True, width="stretch")
            else:
                st.caption("各候选正在独立准备首轮方案。方案齐备后，软件继续调度实验。")
            if stops:
                st.dataframe([{"轨迹": s.get("track_id"), "停止理由": s.get("reason"),
                               "依据": _display(s.get("evidence_ids") or []), "时间": _timestamp(s.get("created_at"))}
                              for s in stops], hide_index=True, width="stretch")
            else:
                st.caption("尚无研究停止决定。预算限制或中断状态仍以运行与轨迹记录为准。")
    with st.expander("实验队列", expanded=bool(jobs)):
        if jobs:
            st.dataframe([{"请求": j.get("id"), "轨迹": j.get("track_id"), "想法": j.get("idea_id"),
                           "方案": j.get("proposal_id"),
                           "状态": _label(j.get("status")), "实验": (j.get("result") or {}).get("experiment_id"),
                           "执行方式": "复用既有结果" if j.get("reused_from_job_id") else "已执行" if j.get("executed") is True else "未记录" if "executed" not in j else "尚未执行",
                           "指标": _display((j.get("result") or {}).get("metrics")),
                           "错误": _display((j.get("result") or {}).get("error") or j.get("error"))}
                          for j in jobs], hide_index=True, width="stretch")
        else:
            st.caption("尚无实验请求。")
    with st.expander("调整后续研究要求与预算"):
        _adjust(client, run)
    with st.expander("最近事件", expanded=True):
        _events(client, run_id)
    with st.expander("来源、方案与综合报告"):
        _report(client, run_id, tracks)


def _control(client, run: dict[str, Any], action: str, changes=None) -> None:
    try:
        result = client.control(str(run["run_id"]), action,
                                expected_version=run.get("version"), changes=changes)
        actual = result.get("run") or result
        st.success(f"服务已接收操作，实际状态：{_label(actual.get('status'))}。")
        # 控制也可能来自整页执行；全页刷新同时更新运行列表与版本。
        st.rerun()
    except Exception as exc:
        st.error(f"控制请求未完成：{exc}")


def _adjust(client, run: dict[str, Any]) -> None:
    config = run.get("config") or {}
    with st.form(f"v2_adjust_{run['run_id']}"):
        guidance = st.text_area("后续研究要求", value=str(config.get("guidance") or ""))
        experiments = st.number_input("调整全局实验上限", min_value=1, value=int(config.get("max_experiments", 20)), step=1)
        tokens = st.number_input("调整全局 token 预算", min_value=1000, value=int(config.get("token_budget", 1000000)), step=1000)
        st.caption("服务校验已消耗与已预留额度，并记录变更；新的要求用于后续研究动作。")
        submitted = st.form_submit_button("保存后续配置", disabled=run.get("status") in {"completed", "cancelled"})
    if submitted:
        _control(client, run, "update", {"guidance": guidance.strip(),
                 "max_experiments": int(experiments), "token_budget": int(tokens)})


def _events(client, run_id: str) -> None:
    state_key = f"v2_events_{run_id}"
    state = st.session_state.setdefault(state_key, {"cursor": 0, "events": []})
    try:
        result = client.events(run_id, after=state["cursor"], limit=100)
        new_events = result.get("events") or []
        # 新事件按服务游标读取；有限历史避免浏览器会话无限增长。
        state["events"] = (state["events"] + new_events)[-100:]
        state["cursor"] = result.get("cursor", state["cursor"])
    except Exception as exc:
        st.warning(f"事件暂时不可读：{exc}")
    if not state["events"]:
        st.caption("暂无事件。")
    for event in state["events"][-20:]:
        stamp = _timestamp(event.get("created_at") or event.get("at"))
        kind = event.get("kind") or event.get("type") or "事件"
        payload = event.get("payload") or {}
        description = event.get("message") or event.get("text") or event.get("summary")
        if not description and isinstance(payload, dict):
            description = " · ".join(f"{key}: {_display(payload[key])}" for key in
                                      ("message", "text", "action", "status", "id", "pid", "error")
                                      if payload.get(key) is not None)
        st.text(f"{stamp} · {event.get('track_id') or '运行'} · {kind} · "
                f"{description or ''}")


def _report(client, run_id: str, tracks: list[dict[str, Any]]) -> None:
    choices = [None] + [t["track_id"] for t in tracks if t.get("track_id")]
    chosen = st.selectbox("报告范围", choices, format_func=lambda t: "全体轨迹综合报告" if t is None else str(t),
                          key=f"v2_report_scope_{run_id}")
    key = f"v2_report_{run_id}_{chosen}"
    if st.button("生成最新事实报告", key=f"v2_report_generate_{run_id}"):
        try:
            report = client.get_report(run_id, track_id=chosen)
            if "sections" not in report and "run" in report:
                report = build_report(report, track_id=chosen)
            st.session_state[key] = report
        except Exception as exc:
            st.error(f"生成报告失败：{exc}")
    report = st.session_state.get(key)
    if not report:
        st.caption("报告包含来源、实验前预测、实测结果、负结果和保留／淘汰理由；支持下载 JSON、Markdown 与 HTML。")
        return
    st.caption(f"报告快照生成于 {report.get('generated_at', '未知')}；再次生成可更新事实。")
    name = f"thermoforge_{run_id}_{chosen or 'all'}"
    columns = st.columns(3)
    columns[0].download_button("下载 Markdown", render_markdown(report), f"{name}.md", "text/markdown", key=f"{key}_md")
    columns[1].download_button("下载 HTML", render_html(report), f"{name}.html", "text/html", key=f"{key}_html")
    columns[2].download_button("下载 JSON", json.dumps(report, ensure_ascii=False, indent=2, default=str),
                               f"{name}.json", "application/json", key=f"{key}_json")
    for section in report.get("sections") or []:
        st.markdown(f"**{section['title']}**")
        for paragraph in section.get("paragraphs") or []:
            st.write(str(paragraph))
        if section.get("rows"):
            st.dataframe(section["rows"], hide_index=True, width="stretch")
