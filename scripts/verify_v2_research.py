"""V2 六 Codex 真实研究闭环验收：使用仓库 tests/ 中的合成冷机 fixture。

会启动隔离后台服务并消耗真实 Codex 用量；不使用用户数据、不查外网。
五个候选各做 linear、ridge 两次内置模型实验，验证来源/反馈/报告全链路。
这是一项程序验收，不是多智能体研究质量或论文结论的基准。
完整验收示例（真实消耗）：.venv/Scripts/python scripts/verify_v2_research.py --token-budget 2000000
恢复同一隔离研究：... --resume-root research/v2_validation/e2e-... --resume-token-budget 1500000
恢复预算必须由执行者明确提供；原始失败证据不被覆盖。
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from thermoforge_v2.client import V2Client


ROOT = Path(__file__).resolve().parents[1]
GUIDANCE = """本次为有界程序验收，数据完全合成，五候选执行相同两阶段任务。不要查文献/外网，
不要提交lab源码，不新增第三次实验。主智能体只协调和比较，不占用实验预算。
每个候选第一轮：读取协议和消息；用research_idea_create登记origin=conjecture的线性基线想法，
说明理由/预测/反证条件；调用research_experiment_run，model={"category":"data","estimator":"linear","hyperparameters":{}}，
idempotency_key="baseline-linear"；根据工具返回的真实validate指标调用research_finding_create，引用本次job/idea，
明确结果与局限；然后结束这一轮回复，软件会自动推进下一轮，不向用户提问，也不要第一轮就做第二实验。
候选第二轮：回顾自己第一轮实际结果；登记origin=history的新想法，parent_idea_ids和parent_job_ids引用自己的第一轮记录，
理由必须说明上次验证结果如何启发检验正则化，预测允许无改善。
第二实验model={"category":"data","estimator":"ridge","hyperparameters":{"alpha":1.0}}，idempotency_key="feedback-ridge"。
读取第二次真实validate反馈后保存发现，比较两次结果，不能虚构改善；用research_report_submit保存独立报告，
引用两个idea/job及发现，说明想法来源、失败/负结果、仅合成数据验证及无最终留出信息。报告正文100-300字。
主智能体先通过research_send_message给all分发共同任务；中途只做必要简短协调，候选阶段结束后调用research_team，
用research_team_report保存跨五候选综合报告，引用真实证据且解释方向来源；不额外训练，不宣称多智能体优于单智能体。
所有轮次最终文本不超过60字，主要事实通过工具保存。"""


def write(path: Path, value) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def client_for(config, *, autostart=True):
    return V2Client(**config["roots"], autostart=autostart)


def submit_child(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    client = client_for(config)
    prepared = client.prepare(config["run_config"])
    if not prepared.get("ready"):
        raise RuntimeError(str(prepared.get("errors")))
    run = client.start(config["run_config"], config["idempotency_key"])
    descriptor = client._alive()
    # 控制token仅留在服务描述文件，绝不传到 stdout 或验收报告。
    print(json.dumps({"run_id": run["run_id"], "service_pid": descriptor["pid"],
                      "submitting_client_pid": os.getpid()}), flush=True)


def stop_isolated_service(client, expected_pid):
    descriptor = client._alive()
    if descriptor is None:
        return {"pid": expected_pid, "stopped": True, "already_exited": True}
    if descriptor["pid"] != expected_pid or expected_pid <= 0:
        raise RuntimeError("隔离服务 PID 不匹配，拒绝停止其他进程")
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x0001 | 0x00100000, False, expected_pid)
        if not handle:
            raise RuntimeError("无法取得隔离服务句柄")
        try:
            if not kernel.TerminateProcess(handle, 0):
                raise RuntimeError("停止隔离服务失败")
            if kernel.WaitForSingleObject(handle, 10000) != 0:
                raise RuntimeError("隔离服务未在期限内退出")
        finally:
            kernel.CloseHandle(handle)
    else:
        os.kill(expected_pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if client._alive() is None:
            return {"pid": expected_pid, "stopped": True, "verified_roots": True}
        time.sleep(0.2)
    raise RuntimeError("隔离服务仍有响应")


def validate(snapshot, report, instances):
    errors = []
    tracks = snapshot["tracks"]
    candidates = [t for t in tracks if t["role"] == "candidate"]
    if snapshot["run"]["status"] != "completed":
        errors.append("run 未进入 completed")
    if len(candidates) != 5 or len(instances) != 6:
        errors.append("不是完整的 1+5 实例")
    if len({t.get("pid") for t in instances.values()}) != 6 or len({t.get("thread_id") for t in instances.values()}) != 6:
        errors.append("六进程/会话身份不独立")
    for track in candidates:
        tid = track["track_id"]
        jobs = [j for j in snapshot["jobs"] if j["track_id"] == tid]
        ideas = [i for i in snapshot["ideas"] if i["track_id"] == tid]
        findings = [f for f in snapshot["findings"] if f["track_id"] == tid]
        reports = [r for r in snapshot["reports"] if r["track_id"] == tid]
        if len(jobs) != 2 or any(j["status"] != "completed" for j in jobs):
            errors.append(f"{tid} 未完成恰好两次实验")
        if track["turns"] < 2:
            errors.append(f"{tid} 未完成两轮独立反馈")
        estimators = [j.get("request", {}).get("model", {}).get("estimator") for j in jobs]
        if estimators != ["linear", "ridge"]:
            errors.append(f"{tid} 模型顺序不是 linear → ridge")
        if len(ideas) < 2 or ideas[0].get("origin") != "conjecture" or ideas[-1].get("origin") != "history":
            errors.append(f"{tid} 缺少猜想到历史证据的来源变化")
        if len(ideas) >= 2 and jobs:
            if ideas[0]["id"] not in ideas[-1].get("parent_idea_ids", []) or jobs[0]["id"] not in ideas[-1].get("parent_job_ids", []):
                errors.append(f"{tid} 缺少父想法/实验谱系")
        if len(findings) < 2 or not reports:
            errors.append(f"{tid} 缺少发现或独立报告")
        if jobs and not track_has_final_report(snapshot, track):
            errors.append(f"{tid} 缺少覆盖本轨迹全部实验的最终报告")
        for job in jobs:
            result = job.get("result") or {}
            if result.get("feedback_surface") != "validate" or any(k in result for k in ("surfaces", "holdout", "final_evaluation")):
                errors.append(f"{tid} 研究反馈边界错误")
    main_track = next((t for t in tracks if t["role"] == "main"), None)
    if not main_track or not track_has_final_report(snapshot, main_track):
        errors.append("缺少覆盖全部候选证据的主智能体综合报告")
    # 外部报告触发最终留出评价，但不会回流研究工具。
    final = report.get("final_evaluation") or {}
    if final.get("status") != "evaluated" or final.get("feedback_to_agents") is not False:
        errors.append("外部报告未提供隔离的最终留出评价")
    return errors


def run(args):
    fixture_dir = ROOT / "tests"
    if not (fixture_dir / "phase34_helpers.py").is_file():
        raise RuntimeError("本验收依赖仓库 tests/phase34_helpers.py 合成 fixture")
    sys.path.insert(0, str(fixture_dir))
    from phase2_helpers import FEATURES, TARGET
    from phase34_helpers import make_ctx
    from thermoforge_research.tools import tf_goal_create
    folder = ROOT / "research" / "v2_validation" / ("e2e-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    ctx, ref = make_ctx(folder, n_steps=600, actor="v2-validation")
    goal = tf_goal_create(ctx, {
        "name": "V2 合成冷机端到端验收", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES), "acceptance": {"cvrmse_max": 0.5},
    })
    if not goal["ok"]:
        raise RuntimeError(str(goal["summary"]))
    configuration = {
        "roots": {"research_root": str(ctx.research_root), "vault_root": str(ctx.vault_root), "models_root": str(ctx.models_root)},
        "run_config": {"goal_id": goal["id"], "dataset_ref": ref, "candidates": 5,
                       "max_experiments": 10, "max_experiments_per_track": 2, "max_turns": 5,
                       "token_budget": args.token_budget, "turn_timeout_seconds": 180,
                       "experiment_timeout_seconds": 180, "max_failures": 1,
                       "purge_seconds": 0, "embargo_seconds": 2700, "seed": 79,
                       "strategy": "independent", "guidance": GUIDANCE},
        "idempotency_key": "isolated-real-v2-validation",
    }
    configuration_path = folder / "configuration.json"
    write(configuration_path, configuration)
    proof = {"kind": "real_codex_research_e2e", "scientific_quality_benchmark": False,
             "synthetic_data": True, "fixture_steps": 600, "started_at": datetime.now(timezone.utc).isoformat(),
             "passed": False, "errors": [], "instances": {}, "peak_concurrent_turns": 0}
    proof_path = folder / "research-e2e.json"
    client = client_for(configuration, autostart=False)
    service_pid = None
    rid = None
    try:
        submitted = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--submit-child", str(configuration_path)],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if submitted.returncode:
            raise RuntimeError("提交客户端失败：" + submitted.stderr[-4000:])
        receipt = json.loads(submitted.stdout.strip().splitlines()[-1])
        proof["submission"] = receipt | {"client_exited": True, "client_exit_code": submitted.returncode}
        rid, service_pid = receipt["run_id"], receipt["service_pid"]
        proof["run_id"] = rid
        del submitted
        # 原提交客户端已经退出；此处是另一个客户端对象，经HTTP读取同一后台 run。
        snapshot = client.get_run(rid)
        proof["reconnected_same_run"] = snapshot["run"]["run_id"] == rid
        print(json.dumps({"event": "research_started", "run_id": rid, "service_pid": service_pid,
                          "submitting_client_exited": True, "evidence": str(proof_path)}), flush=True)
        deadline, cursor, prior = time.monotonic() + args.timeout, 0, None
        active = set()
        while True:
            page = client.events(rid, after=cursor, limit=1000)
            cursor = page["cursor"]
            for event in page["events"]:
                tid, kind, payload = event["track_id"], event["kind"], event["payload"]
                if kind == "instance.started":
                    proof["instances"][tid] = {k: payload.get(k) for k in ("pid", "thread_id", "model", "reasoning_effort", "server")}
                if kind == "turn/started":
                    active.add(tid)
                    proof["peak_concurrent_turns"] = max(proof["peak_concurrent_turns"], len(active))
                elif kind == "turn/completed":
                    active.discard(tid)
            snapshot = client.get_run(rid)
            state = snapshot["run"]["status"]
            progress = (state, len(snapshot["jobs"]), len(snapshot["reports"]),
                        tuple(t["status"] for t in snapshot["tracks"]))
            if progress != prior:
                print(json.dumps({"event": "progress", "state": state, "jobs": dict(Counter(j["status"] for j in snapshot["jobs"])),
                                  "reports": len(snapshot["reports"]), "tokens_used": snapshot["run"]["tokens_used"],
                                  "tracks": {t["track_id"]: t["status"] for t in snapshot["tracks"]}}), flush=True)
                prior = progress
            write(folder / "snapshot.json", snapshot)
            write(proof_path, proof)
            if state not in {"running", "queued", "pausing", "cancelling"}:
                break
            if time.monotonic() >= deadline:
                client.control(rid, "cancel")
                raise TimeoutError("端到端验收达到时间上限，已请求取消隔离研究")
            time.sleep(2)
        report = client.get_report(rid)
        write(folder / "external-report.json", report)
        proof["errors"] = validate(snapshot, report, proof["instances"])
        proof["status"] = state
        proof["tokens_used"] = snapshot["run"]["tokens_used"]
        proof["counts"] = {k: len(snapshot[k]) for k in ("ideas", "jobs", "findings", "reports", "decisions")}
        proof["report_artifacts"] = report.get("artifacts", [])
        proof["passed"] = not proof["errors"] and proof["reconnected_same_run"]
    except Exception as exc:
        proof["errors"].append(str(exc)[:4000])
    finally:
        if service_pid is None:
            descriptor = client._alive()
            service_pid = descriptor["pid"] if descriptor else None
        if service_pid:
            try:
                if rid:
                    state = client.get_run(rid)["run"]["status"]
                    if state in {"queued", "running", "pausing", "cancelling"}:
                        client.control(rid, "cancel")
                        end = time.monotonic() + 240
                        while time.monotonic() < end and client.get_run(rid)["run"]["status"] in {"running", "pausing", "cancelling"}:
                            time.sleep(1)
                proof["service_cleanup"] = stop_isolated_service(client, service_pid)
            except Exception as exc:
                proof["errors"].append("隔离服务清理失败：" + str(exc)[:1000])
                proof["passed"] = False
        proof["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(proof_path, proof)
    print(json.dumps({"passed": proof["passed"], "evidence": str(proof_path), "errors": proof["errors"],
                      "tokens_used": proof.get("tokens_used"), "counts": proof.get("counts"),
                      "service_cleanup": proof.get("service_cleanup")}, ensure_ascii=False), flush=True)
    return 0 if proof["passed"] else 1


def track_has_final_report(snapshot, track):
    """以证据覆盖判定是否已结题，旧 completed 标签不能替代最终报告。"""
    tid = track["track_id"]
    reports = [r for r in snapshot["reports"] if r["track_id"] == tid]
    if track["role"] == "main":
        jobs = {j["id"] for j in snapshot["jobs"]}
        findings = {f["id"] for f in snapshot["findings"]}
        ideas = {i["id"] for i in snapshot["ideas"]}
        # 主工具采用统一 evidence_ids；候选工具采用分开的 job/idea/finding_ids。
        return any(r.get("kind") == "team" and (jobs | findings | ideas).issubset(
            set(r.get("evidence_ids", []))) for r in reports)
    jobs = {j["id"] for j in snapshot["jobs"] if j["track_id"] == tid}
    findings = {f["id"] for f in snapshot["findings"] if f["track_id"] == tid}
    ideas = {i["id"] for i in snapshot["ideas"] if i["track_id"] == tid}
    return bool(jobs) and any(jobs.issubset(set(r.get("job_ids", [])))
                              and findings.issubset(set(r.get("finding_ids", [])))
                              and ideas.issubset(set(r.get("idea_ids", []))) for r in reports)


def reconcile_usage(folder):
    """仅修复已知隔离验收的两段用量；持有服务锁并以单事务保证幂等。"""
    from thermoforge_v2.service import ServiceLock
    from thermoforge_v2.store import RunStore, encode, now
    folder = folder.resolve()
    if folder != (ROOT / "research" / "v2_validation" / "e2e-c2c2f720").resolve():
        raise ValueError("这次一次性账本修复仅授权 e2e-c2c2f720")
    evidence_path = folder / "usage-segments.json"
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    digest = hashlib.sha256(evidence_bytes).hexdigest()
    rid = evidence["run_id"]
    if rid != "RUN-5dacd6239bed44ec" or len(evidence["segments"]) != 2:
        raise ValueError("修复来源不是已核验的两段隔离运行")
    segments = [part["raw_usage_by_track"] for part in evidence["segments"]]
    combined = {tid: {key: sum(part[tid]["total"].get(key, 0) for part in segments)
                     for key in set().union(*(part[tid]["total"] for part in segments))}
                for tid in segments[0]}
    if combined != evidence["combined_total_by_track"] or sum(v["totalTokens"] for v in combined.values()) != 1587314:
        raise ValueError("用量来源内部不一致，拒绝写入")
    configuration = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    client = client_for(configuration, autostart=False)
    if client.research_root != (folder / "research").resolve():
        raise ValueError("隔离研究根目录不匹配")
    with ServiceLock(client.root / "service.lock"):
        if client._alive() is not None:
            raise RuntimeError("后台服务仍运行，拒绝修改用量账本")
        store = RunStore(client.root)
        before = store.snapshot(rid)
        if before["run"]["status"] in {"running", "queued", "pausing", "cancelling"} or any(t.get("pid") for t in before["tracks"]):
            raise RuntimeError("隔离运行或进程未停止")
        with store._db() as db:
            existing = db.execute("SELECT payload FROM events WHERE run_id=? AND kind='usage.reconciled'", (rid,)).fetchall()
            if existing:
                payloads = [json.loads(row[0]) for row in existing]
                if len(payloads) != 1 or payloads[0].get("evidence_sha256") != digest:
                    raise RuntimeError("账本已有其他修复，拒绝重复累加")
                receipt = payloads[0] | {"already_reconciled": True}
            else:
                run_doc = store._get(db, "runs", "id", rid)
                tracks = [json.loads(row[0]) for row in db.execute("SELECT data FROM tracks WHERE run_id=?", (rid,))]
                if set(t["track_id"] for t in tracks) != set(combined):
                    raise RuntimeError("轨迹集合与用量来源不一致")
                if run_doc["tokens_used"] != 775642 or any(t["usage"]["total"] != segments[-1][t["track_id"]]["total"] for t in tracks):
                    raise RuntimeError("当前账本不等于已核验的第二段原始值，拒绝覆盖未知变更")
                before_path = folder / "usage-reconcile-before.json"
                if not before_path.exists():
                    write(before_path, before)
                receipt = {"reason": "修正 Codex 0.153.4 恢复后进程用量重置造成的历史漏计",
                           "evidence": evidence_path.name, "evidence_sha256": digest,
                           "run_id": rid, "old_tokens_used": run_doc["tokens_used"], "new_tokens_used": 1587314,
                           "old_usage_total_by_track": {t["track_id"]: t["usage"]["total"] for t in tracks},
                           "new_usage_total_by_track": combined, "version_before": run_doc["version"]}
                for track in tracks:
                    track["usage"]["total"] = combined[track["track_id"]]
                    track["usage_reconciled_from_sha256"] = digest
                    track["updated_at"] = now()
                    db.execute("UPDATE tracks SET data=? WHERE run_id=? AND id=?", (encode(track), rid, track["track_id"]))
                run_doc["tokens_used"] = 1587314
                run_doc["usage_reconciled_from_sha256"] = digest
                store._write_run(db, run_doc)
                receipt["version_after"] = run_doc["version"]
                store._event(db, rid, "usage.reconciled", receipt)
        if not (folder / "usage-reconciled.json").exists():
            write(folder / "usage-reconciled.json", receipt)
        if not (folder / "usage-reconcile-after.json").exists():
            write(folder / "usage-reconcile-after.json", store.snapshot(rid))
    print(json.dumps({"event": "usage_reconciled", "run_id": rid, "tokens_used": receipt["new_tokens_used"],
                      "evidence_sha256": digest, "already_reconciled": receipt.get("already_reconciled", False)}), flush=True)
    return 0


def resume(args):
    """重新启动同一个隔离服务，更新预算，再恢复原 run/thread/实验。"""
    folder = args.resume_root.resolve()
    if not folder.is_relative_to((ROOT / "research" / "v2_validation").resolve()):
        raise ValueError("恢复入口仅接受本仓库 v2_validation 隔离验收目录")
    configuration = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    original = json.loads((folder / "research-e2e.json").read_text(encoding="utf-8"))
    rid = original["run_id"]
    prefix = "resume-" + uuid.uuid4().hex[:8]
    proof_path = folder / (prefix + ".json")
    proof = {"kind": "real_codex_research_resume", "scientific_quality_benchmark": False,
             "original_evidence": "research-e2e.json", "run_id": rid,
             "started_at": datetime.now(timezone.utc).isoformat(), "passed": False,
             "errors": [], "instances": {}, "peak_concurrent_turns": 0}
    client = client_for(configuration)
    service_pid = None
    try:
        before = client.get_run(rid)
        service_pid = client._alive()["pid"]
        write(folder / (prefix + "-before.json"), before)
        total = sum((t.get("usage") or {}).get("total", {}).get("totalTokens", 0) for t in before["tracks"])
        if total != before["run"]["tokens_used"]:
            raise RuntimeError("轨迹累计 tokens 与运行总量不一致，拒绝继续消耗")
        if not before["run"]["tokens_used"] < args.resume_token_budget <= 2000000:
            raise ValueError("恢复预算必须高于已使用 tokens 且不超过本次明确授权的 2000000 上限")
        old_jobs = {j["id"]: j for j in before["jobs"] if j["status"] == "completed"}
        proof["usage_before"] = {t["track_id"]: t.get("usage") for t in before["tracks"]}
        proof["tokens_before"] = total
        previous_instances = {t["track_id"]: {
            k: (t.get("backend") or {}).get(k) for k in ("pid", "thread_id", "model", "reasoning_effort", "server")
        } for t in before["tracks"]}
        expected_resumed = {t["track_id"] for t in before["tracks"] if not track_has_final_report(before, t)}
        proof["instances_before"] = previous_instances
        proof["already_completed_tracks"] = sorted(set(previous_instances) - expected_resumed)
        changes = {"token_budget": args.resume_token_budget}
        if args.resume_max_turns is not None:
            changes["max_turns"] = args.resume_max_turns
        if args.reports_only:
            pending_candidates = sorted(expected_resumed - {"main"})
            candidate_guidance = (("、".join(pending_candidates) + "各自使用已有实验、想法和发现，"
                "通过 research_report_submit 保存覆盖本轨迹全部job/idea/finding的最终报告，正文100–200字。")
                if pending_candidates else "候选最终报告已经齐全，本阶段只需主智能体综合。")
            changes["guidance"] = ("仅补齐现有研究报告。已经完成的十次实验足够，不得创建任何新实验、想法、发现或来源，"
                "不得检索外网，不重复训练。" + candidate_guidance +
                "如果上下文保留各记录ID与真实指标，直接复用；缺失时用research_history一次取必要记录。"
                "其他已完成候选保持原报告。主智能体只在未完成候选补齐报告后读取research_team并用research_team_report"
                "保存覆盖五候选十实验的综合报告，正文200–350字，陈述想法来源、真实负结果或改善及合成验收限制，"
                "不声称多智能体优于单智能体。不要额外协调或保存中间决策。最终回复不超过40字。")
        updated = client.control(rid, "update", expected_version=before["run"]["version"], changes=changes)
        resumed = client.control(rid, "resume", expected_version=updated["version"])
        proof["controls"] = [
            {"action": "update", "expected_version": before["run"]["version"], "version": updated["version"],
             "old_token_budget": before["run"]["config"]["token_budget"], "new_token_budget": args.resume_token_budget,
             "changes": changes},
            {"action": "resume", "expected_version": updated["version"], "version": resumed["version"]}]
        print(json.dumps({"event": "research_resumed", "run_id": rid, "service_pid": service_pid,
                          "completed_jobs_reused": list(old_jobs), "evidence": str(proof_path)}), flush=True)
        cursor = max((e["seq"] for e in before["events"]), default=0)
        deadline, prior, active = time.monotonic() + args.timeout, None, set()
        previous_tokens = total
        while True:
            page = client.events(rid, after=cursor, limit=1000)
            cursor = page["cursor"]
            for event in page["events"]:
                tid, kind, payload = event["track_id"], event["kind"], event["payload"]
                if kind == "instance.started":
                    proof["instances"][tid] = {k: payload.get(k) for k in ("pid", "thread_id", "model", "reasoning_effort", "server")}
                if kind == "turn/started":
                    active.add(tid)
                    proof["peak_concurrent_turns"] = max(proof["peak_concurrent_turns"], len(active))
                elif kind == "turn/completed":
                    active.discard(tid)
            snapshot = client.get_run(rid)
            state = snapshot["run"]["status"]
            current_tokens = snapshot["run"]["tokens_used"]
            if current_tokens < previous_tokens:
                write(folder / (prefix + "-snapshot.json"), snapshot)
                client.control(rid, "cancel")
                raise RuntimeError("恢复后累计 tokens 减少，原生 usage 可能重置；已取消以避免突破总预算")
            previous_tokens = current_tokens
            progress = (state, len(snapshot["jobs"]), len(snapshot["reports"]), tuple(t["status"] for t in snapshot["tracks"]))
            if progress != prior:
                print(json.dumps({"event": "resume_progress", "state": state,
                                  "jobs": dict(Counter(j["status"] for j in snapshot["jobs"])),
                                  "reports": len(snapshot["reports"]), "tokens_used": snapshot["run"]["tokens_used"]}), flush=True)
                prior = progress
            write(folder / (prefix + "-snapshot.json"), snapshot)
            write(proof_path, proof)
            if state not in {"running", "queued", "pausing", "cancelling"}:
                break
            if time.monotonic() >= deadline:
                client.control(rid, "cancel")
                raise TimeoutError("恢复验收达到时间上限，已取消隔离研究")
            time.sleep(2)
        report = client.get_report(rid)
        write(folder / (prefix + "-external-report.json"), report)
        # 已完成候选无需重启；初次的 1+5 身份证明仍保留，当前仅恢复未结题轨迹。
        proof["errors"] = validate(snapshot, report, previous_instances | proof["instances"])
        after_jobs = {j["id"]: j for j in snapshot["jobs"]}
        if any(after_jobs.get(key) != value for key, value in old_jobs.items()):
            proof["errors"].append("恢复后已完成实验记录发生变化")
        proof["completed_jobs_reused_unchanged"] = list(old_jobs)
        proof["same_threads_new_processes"] = set(proof["instances"]) == expected_resumed and all(
            previous_instances[tid]["thread_id"] == current["thread_id"]
            and previous_instances[tid]["pid"] != current["pid"] for tid, current in proof["instances"].items())
        if not proof["same_threads_new_processes"]:
            proof["errors"].append("恢复未保持六原会话或未重新启动独立进程")
        proof["status"] = state
        proof["tokens_used"] = snapshot["run"]["tokens_used"]
        proof["monotonic_usage_guard_passed"] = True
        proof["usage_guard_interval_seconds"] = 2
        proof["counts"] = {k: len(snapshot[k]) for k in ("ideas", "jobs", "findings", "reports", "decisions")}
        proof["report_artifacts"] = report.get("artifacts", [])
        proof["passed"] = not proof["errors"]
    except Exception as exc:
        proof["errors"].append(str(exc)[:4000])
    finally:
        if service_pid:
            try:
                state = client.get_run(rid)["run"]["status"]
                if state in {"queued", "running", "pausing", "cancelling"}:
                    client.control(rid, "cancel")
                # 给引擎关闭独占 Codex 进程和清理 PID 的机会，再停止本隔离服务。
                end = time.monotonic() + 240
                while time.monotonic() < end:
                    cleanup = client.get_run(rid)
                    if not any(t.get("pid") for t in cleanup["tracks"]):
                        break
                    time.sleep(1)
                proof["service_cleanup"] = stop_isolated_service(client, service_pid)
            except Exception as exc:
                proof["errors"].append("隔离服务清理失败：" + str(exc)[:1000])
                proof["passed"] = False
        proof["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(proof_path, proof)
    print(json.dumps({"passed": proof["passed"], "evidence": str(proof_path), "errors": proof["errors"],
                      "tokens_used": proof.get("tokens_used"), "counts": proof.get("counts"),
                      "service_cleanup": proof.get("service_cleanup")}, ensure_ascii=False), flush=True)
    return 0 if proof["passed"] else 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    from thermoforge_v2.contracts import RunConfig
    parser.add_argument("--token-budget", type=int, default=RunConfig.model_fields["token_budget"].default,
                        help="新运行的明确累计预算，默认使用产品配置；完整 ultra 验收示例为 2000000")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--submit-child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-root", type=Path, help="恢复指定隔离验收目录中的原 run，不重新训练已完成实验")
    parser.add_argument("--resume-token-budget", type=int, help="明确授权的恢复累计预算，最高 2000000；不修改产品默认")
    parser.add_argument("--resume-max-turns", type=int, help="明确指定本次恢复的 max_turns，不静默修改")
    parser.add_argument("--reports-only", action="store_true", help="仅补未完成候选及主最终报告，不增加实验")
    parser.add_argument("--reconcile-usage-root", type=Path, help="仅为已知 e2e-c2c2f720 离线修复累计账本，不启动服务或模型")
    args = parser.parse_args()
    if args.submit_child:
        submit_child(args.submit_child)
        return 0
    if not 1 <= args.timeout <= 1800:
        parser.error("--timeout 必须为 1–1800 秒")
    if not 1 <= args.token_budget <= 2000000:
        parser.error("--token-budget 必须为 1–2000000")
    if args.reconcile_usage_root:
        if args.resume_root or args.resume_token_budget or args.resume_max_turns or args.reports_only:
            parser.error("离线用量修复不能与模型恢复同时执行")
        return reconcile_usage(args.reconcile_usage_root)
    if bool(args.resume_root) != bool(args.resume_token_budget):
        parser.error("--resume-root 与 --resume-token-budget 必须同时提供")
    if (args.resume_max_turns is not None or args.reports_only) and not args.resume_root:
        parser.error("--resume-max-turns/--reports-only 必须与 --resume-root 一起使用")
    if args.resume_root:
        return resume(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
