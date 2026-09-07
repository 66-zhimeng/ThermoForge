"""Phase-driven autonomous research; a report alone does not terminate exploration."""
from __future__ import annotations

import asyncio
import json

from .objectives import evaluate_objective
from .turn_context import visible_objective_evidence


async def run_candidate(engine, rid, tid):
    store = engine.store
    while True:
        track, run = store.get_track(rid, tid), store.get_run(rid)
        jobs = store.records(rid, "jobs", tid)
        proposals = store.records(rid, "proposals", tid)
        stops = store.records(rid, "stops", tid)
        report = engine._current_track_report(rid, tid)
        visible_jobs, visible_findings = visible_objective_evidence(store, run, tid)
        assessment = evaluate_objective(run, visible_jobs, visible_findings, track_id=tid)
        ready_to_conclude = (assessment.get("objective_mode") in {"target", "optimize"}
                             and assessment.get("recommended_action") == "conclude")
        if stops and report:
            store.update_track(rid, tid, {"status": "completed", "phase": "research_stopped",
                                         "stop_reason": stops[-1]["reason"]})
            store.autonomy_state(rid)
            return
        budget_left = (len(jobs) < run["config"]["max_experiments_per_track"]
                       and run["experiments_reserved"] < run["config"]["max_experiments"])
        if not budget_left and report and jobs and run["status"] == "running":
            try:
                store.request_research_stop(rid, tid, "已用完本轨迹或全局实验请求额度；根据已保存结果结题，不宣称达到未知目标。",
                                            [j["id"] for j in jobs])
            except Exception:
                if store.get_run(rid)["status"] != "running":
                    break
                raise
            continue
        if not engine._can_continue(rid, tid):
            break
        state = store.autonomy_state(rid)
        closing_for_tokens = state.get("closing_for_tokens", False)
        if proposals and not state["proposal_barrier_open"] and not closing_for_tokens:
            store.update_track(rid, tid, {"status": "waiting", "phase": "awaiting_independent_proposals"})
            await asyncio.sleep(0.2)
            continue
        remaining_turns = run["config"]["max_turns"] - track["turns"]
        if not proposals and budget_left and remaining_turns > 1 and not closing_for_tokens:
            baseline = (run.get("objective_contract") or {}).get("baseline_model")
            baseline_hint = ("研究契约已预先指定共同基线，但当前可见证据尚无真实基线结果。"
                             "先将这份内置基线冻结为首提案并检验，后续自行提出改进；确定性基线重复执行不能算独立统计复核。"
                             f"基线模型：{json.dumps(baseline, ensure_ascii=False)}。"
                             if baseline and assessment.get("baseline", {}).get("status") != "available" else "")
            prompt = (f"独立提案阶段，轨迹 {tid}。下方已投递协议与消息摘要，按需读取完整 research_protocol，可自主检索/阅读资料，"
                      "在固定目标、输入白名单和评价口径内，自主决定模型与参数，也可经 research_lab_submit 实现模型。"
                      + baseline_hint +
                      "不预设五个候选必须选择不同答案。先 research_idea_create 保存真实来源、理由、预测及反证条件，"
                      "再 research_proposal_commit 冻结一份可执行提案（含 idea_id、model、purpose、idempotency_key）。"
                      "本回合只完成首提案，不调用 research_experiment_run。提交后结束回复，软件等所有首提案齐备后继续。"
                      "没有论文依据可诚实登记 conjecture；检索无结果也需如实记录，不编造阅读。")
        elif ready_to_conclude or closing_for_tokens or not budget_left or remaining_turns <= 1:
            prompt = (f"轨迹 {tid} 进入结题阶段，本轮不新增实验或提案。实验请求额度剩余："
                      f"{max(0, min(run['config']['max_experiments_per_track']-len(jobs), run['config']['max_experiments']-run['experiments_reserved']))}。"
                      + ("冻结研究契约在 validate 上已有满足条件且已审阅的方案。核对来源、约束和负结果后明确研究停止理由；"
                         "无需机械用完实验额度。若最终验收面不是 validate，必须说明外部验收尚未完成。" if ready_to_conclude else "")
                      + (f"累计用量已进入报告预留区，剩余 {state.get('tokens_remaining')} tokens，"
                         "这只是保守估算，不保证足够。立即收尾，不检索新资料、不提交模型或新增方案。" if closing_for_tokens else "") +
                      "优先使用会话内已有证据，按需读取缺失历史；补齐发现并 research_report_submit 保存覆盖全部已结算请求的简洁报告，"
                      f"job_ids 至少包括 {json.dumps([j['id'] for j in jobs])}。"
                      "然后 research_stop，理由说明实际目标判断或资源限制，引用真实证据；没有达到目标也需如实结题。")
        elif jobs and report is None:
            prompt = (f"轨迹 {tid} 进入反馈分析阶段，本轮先保存已有成果，不新增实验。"
                      "下方已投递最新实测反馈。用 research_finding_create 补齐缺少的观察、解释与限制，"
                      "再 research_report_submit 保存覆盖全部已结算实验的简短滚动报告。"
                      "明确支持或否定什么、未验证什么、根据哪个诊断继续，以及下一实验怎样区分解释。"
                      "已有发现可按 ID 引用，无需重写。若证据足够结束，research_stop 给出明确理由；"
                      "否则保存后结束本回合，由软件继续。")
        else:
            if state["sharing_ready"] and run["config"]["strategy"] != "independent" and jobs:
                settled_ids = {j["id"] for j in jobs if j["status"] in {"completed", "failed", "cancelled", "interrupted"}}
                pending_review = settled_ids - set(run.get("coordinated_job_ids") or [])
                if run.get("coordination_in_progress") or (pending_review and engine._main_can_review(rid)):
                    store.update_track(rid, tid, {"status": "waiting", "phase": "awaiting_coordination"})
                    await asyncio.sleep(0.2)
                    continue
            prompt = (f"继续自主研究轨迹 {tid}，当前阶段 {state['stage']}；已申请 {len(jobs)} 次实验，"
                      f"轨迹上限 {run['config']['max_experiments_per_track']}。下方已投递新消息和反馈，按需查缺失的历史。"
                      "有尚未执行的冻结提案时，research_experiment_run 必须传 proposal_id 与完全一致的 model。"
                      "先执行并分析反馈。每次实验后 research_finding_create 保存观察、解释、失败/负结果及限制，"
                      "指出哪个真实诊断支持下一方向、预期改善在哪里、怎样反证；未输出的诊断须如实说缺失。"
                      "后续先创建新想法，再冻结新提案；必须引用自己最近实验的 parent_job_ids，并说明为何根据反馈修订。"
                      "可以继续探索、细化或有理由地复现；精确相同方案只允许显式 replicate，不要求人为制造差异。"
                      "每回合至多进行一次正式实验，随后保存发现及覆盖全部实验的简短滚动报告，再结束回复。"
                      "不提前冻结下一轮提案，软件负责下一回合。有报告不等于研究结束；根据研究契约和新增信息价值决定是否继续，"
                      "出现停滞时分析无改善依据或转换假设，确实没有合理后续时先提交覆盖全部实验的报告，"
                      "再调用 research_stop 引用证据说明具体停止理由。只查看 validate，不寻找最终留出。")
        try:
            await engine._turn(rid, tid, prompt)
        except Exception as exc:
            latest = store.get_track(rid, tid)
            failures = latest.get("failures", 0) + 1
            store.update_track(rid, tid, {"failures": failures, "error": str(exc)[:4000]})
            if failures >= run["config"]["max_failures"]:
                store.update_track(rid, tid, {"status": "failed", "phase": "stopped"})
                store.autonomy_state(rid)
                return
            if getattr(engine.sessions.get((rid, tid)), "state", None) in {"closed", "failed"}:
                try:
                    await engine.sessions[rid, tid].close()
                    await engine._make_session(store.get_run(rid), store.get_track(rid, tid))
                    store.event(rid, "instance.recovered", {"reason": str(exc)[:2000]}, tid)
                except Exception as restore_error:
                    store.update_track(rid, tid, {"status": "failed", "error": str(restore_error)[:4000]})
                    store.autonomy_state(rid)
                    return
            await asyncio.sleep(min(2 ** failures, 8))
    state = store.get_run(rid)["status"]
    store.update_track(rid, tid, {"status": "paused" if state in {"pausing", "paused"} else
        "cancelled" if state in {"cancelling", "cancelled"} else "budget_exhausted", "phase": "stopped"})
    store.autonomy_state(rid)
