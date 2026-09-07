"""Bounded real Codex acceptance of autonomous research on synthetic chillers.

Prepare without starting a service or model:
  .venv/Scripts/python scripts/verify_v2_autonomous.py --prepare-only --token-budget 3000000
Execute the saved configuration (consumes real Codex usage):
  .venv/Scripts/python scripts/verify_v2_autonomous.py --config <configuration.json>

Research agents default to GPT-6 Astra / ultra at standard speed (Fast disabled).
No experimental model family or parameter values are prescribed. Repeated independently
chosen experiments are recorded as observations, not failed diversity tests.
This validates autonomous workflow behavior, not superiority over one agent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from thermoforge_v2.contracts import RunConfig
from thermoforge_v2.profile import CODEX_EFFORT, CODEX_MODEL
from thermoforge_v2.strategy import compare_jobs

from verify_v2_research import (
    ROOT, client_for, stop_isolated_service, submit_child, track_has_final_report, write,
)


TOKEN_CEILING = 3_000_000
DEFAULT_DATA_SEED = 2_026_090_701
ACTIVE = {"running", "queued", "pausing", "cancelling"}
METRICS = {"CVRMSE", "RMSE", "MAE", "MAPE", "NMBE", "R2"}
GUIDANCE = """本次研究使用完全合成的冷机数据。共同目标：在冻结的数据、特征、切分与随机种子下，
提高功率预测的验证精度，解释方法选择及其局限。五位候选各自独立研究，方法、结构和参数由你决定，
可查文献或资料，可使用内置工具或提交自己的实验室模型；没有预定解法，也不要求五个答案必须不同。
主智能体只给共同目标、协调流程和最终比较，不训练、不分配模型路线、不发送其他候选的解法与反馈。
每个候选先研究问题，登记自己的想法及来源、选择理由、预测和反证条件；通过 research_proposal_commit
冻结首份具体可运行模型提案。首轮提案提交后结束当前回复，等待软件在全部候选提案齐备后启动实验。
按冻结提案调用 research_experiment_run，保存真实验证反馈与发现。然后根据自己上一轮真实结果，
引用 parent_job_ids 和相关发现，提出修订或有理由的复现实验，再冻结和执行新提案。
预算允许且没有真实阻碍时，每个候选至少完成两次实验，以验证反馈确实改变或支持了下一步研究；
每个候选最多三次，总计最多十五次。失败也是证据，应先分析，不能伪装成模型收益。
来源必须诚实：论文和文章需登记实际稳定链接、阅读范围及内容；历史想法需引用实际实验；自主猜想可没有文献。
不要编造论文、指标、改善或研究方向差异。遇到数据或工具阻碍，保存实际错误和报告，再明确停止，不能假称完成。
最终候选报告覆盖自己的全部想法、实验和发现，说明为何开始、如何修订、结果、负结果与局限，
提交后用 research_stop 记录有依据的停止理由。主智能体最后用 research_team_report 综合全部候选证据。
研究中只使用工具给出的训练/验证信息，不访问最终留出；最终留出由外部报告在研究结束后单独评价。
这是合成数据的自主研究流程验收，不能据此宣称真实设备收益或多智能体优于单智能体。
主要证据保存在研究工具，正文保持简洁；遇到阶段等待就结束本轮回复，由软件继续，不向用户提问。"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def make_autonomous_fixture(folder, *, data_seed, n_steps=600):
    """Generate fresh seeded observations without changing the legacy fixture.

    The data seed controls independent operating phases and measurement noise;
    it is separate from RunConfig.seed, which controls model training. Generator
    version and hashes stay outside candidate workspaces as external provenance.
    """
    import numpy as np
    from phase2_helpers import OBJECTS, TARGET, _series
    from thermoforge_core.contracts.tfdc import ObjectRecord, TfdcDataset, TfdcManifest, VariableRecord
    from thermoforge_data.importer import import_parsed
    from thermoforge_data.vault import DataVault
    from thermoforge_research.tools import ToolContext

    rng = np.random.Generator(np.random.PCG64(data_seed))
    units = {"evap_chw_flow": "m3/h", "evap_chw_supply_temp": "Cel",
             "evap_chw_return_temp": "Cel", "cw_supply_temp": "Cel", "input_power": "kW"}
    dataset = TfdcDataset(
        manifest=TfdcManifest(contract="TFDC", contract_version="1.0",
                              dataset_id=f"V2_AUTONOMOUS_{data_seed}", dataset_version=1,
                              site_id="V2A", timezone="UTC", time_resolution="15min",
                              source_system="synthetic-autonomous-v1",
                              description="独立 seed 的合成冷机数据；非真实测量，非旧验收数据的复制。"),
        objects=[ObjectRecord(object_id=obj, object_model_id="chiller.v1") for obj in OBJECTS],
        variables=[VariableRecord(variable_id=f"{obj}.{prop}", object_id=obj, property_code=prop,
                                  unit=unit, dtype="float", role="target" if prop == TARGET else "state",
                                  source_kind="estimated", description="由有版本的合成生成器产生，非真实测量")
                   for obj in OBJECTS for prop, unit in units.items()],
    )
    columns, legacy = {}, {}
    index = np.arange(n_steps, dtype=float)
    for equipment, obj in enumerate(OBJECTS):
        phase = rng.uniform(0, 2 * np.pi, size=5)
        flow = (355 + 20 * equipment + 80 * np.sin(index / 17 + phase[0])
                + 28 * np.sin(index / 51 + phase[1]) + rng.normal(0, 2, n_steps))
        supply = 6.4 + .65 * np.sin(index / 31 + phase[2]) + rng.normal(0, .03, n_steps)
        delta = 4.1 + .45 * np.sin(index / 13 + phase[3]) + rng.normal(0, .025, n_steps)
        condenser = 25.3 + 3.7 * np.sin(index / 47 + phase[4]) + rng.normal(0, .05, n_steps)
        cooling = flow * 998 / 3600 * 4.186 * delta
        cop = 4.25 + .065 * supply - .052 * condenser + (.82 + .06 * equipment) * cooling / 6000
        power = cooling / cop * (1 + rng.normal(0, .008, n_steps))
        generated = {"evap_chw_flow": flow, "evap_chw_supply_temp": supply,
                     "evap_chw_return_temp": supply + delta, "cw_supply_temp": condenser,
                     "input_power": power}
        for prop, values in generated.items():
            columns[f"{obj}.{prop}"] = values.tolist()
        for prop, values in _series(n_steps, offset=20.0 * equipment).items():
            legacy[f"{obj}.{prop}"] = values
    # Compare observations without timestamps: changing only dates is not a new holdout.
    def observation_hash(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()

    values_hash, legacy_hash = observation_hash(columns), observation_hash(legacy)
    if values_hash == legacy_hash:
        raise RuntimeError("新生成数据与已经查看留出的旧 fixture 数值相同，拒绝准备验收")
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=15 * step) for step in range(n_steps)]
    result = import_parsed(dataset, timestamps, columns)
    if not result.ok:
        raise RuntimeError("自主合成数据导入失败：" + str(result.diagnostics))
    provenance = {"generator": "synthetic-autonomous-v1", "data_seed": data_seed,
                  "rng": "numpy.random.Generator(PCG64)", "numpy_version": np.__version__,
                  "steps_per_object": n_steps, "objects": list(OBJECTS), "time_start": timestamps[0].isoformat(),
                  "time_resolution": "15min", "observations_sha256": values_hash,
                  "legacy_observations_sha256": legacy_hash, "different_from_legacy_fixture": True,
                  "legacy_fixture_modified": False,
                  "legacy_fixture_source_sha256": hashlib.sha256((ROOT / "tests" / "phase2_helpers.py").read_bytes()).hexdigest(),
                  "generator_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "holdout_handling": "新 seed 产生全部观测；准备阶段只记录哈希，不训练、评估或展示留出值。"}
    vault = DataVault(folder / "vault")
    ref = vault.store(result, lineage={"synthetic_generator": provenance["generator"], "data_seed": data_seed})
    provenance.update(dataset_ref=ref, content_sha256=vault.resolve(ref).content_sha256)
    write(folder / "data-provenance.json", provenance)
    ctx = ToolContext(vault_root=folder / "vault", research_root=folder / "research",
                      models_root=folder / "models", actor="v2-autonomous-validation")
    return ctx, ref, provenance


def prepare(args):
    """Create local fixture/configuration only; no service or Codex is started."""
    sys.path.insert(0, str(ROOT / "tests"))
    from phase2_helpers import FEATURES, TARGET
    from thermoforge_research.tools import tf_goal_create

    folder = ROOT / "research" / "v2_validation" / ("autonomous-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    ctx, ref, data_provenance = make_autonomous_fixture(
        folder, n_steps=600, data_seed=DEFAULT_DATA_SEED if args.data_seed is None else args.data_seed)
    goal = tf_goal_create(ctx, {
        "name": "V2 自主研究：合成冷机功率预测", "object_model": "chiller.v1",
        "purpose": "optimization", "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"evaluated_on": "validate", "cvrmse_max": .05},
        "description": "自主选择预测方法，以真实反馈修订；只验证合成数据流程，不作为生产或多智能体效果基准。",
    })
    if not goal["ok"]:
        raise RuntimeError("合成研究目标创建失败：" + str(goal["summary"]))
    config = {
        "roots": {"research_root": str(ctx.research_root), "vault_root": str(ctx.vault_root),
                  "models_root": str(ctx.models_root)},
        "run_config": RunConfig.model_validate({
            "goal_id": goal["id"], "dataset_ref": ref, "research_mode": "autonomous", "candidates": 5,
            "model": args.model or CODEX_MODEL, "reasoning_effort": args.reasoning_effort or CODEX_EFFORT,
            "max_experiments": 15, "max_experiments_per_track": 3, "max_turns": 8,
            "token_budget": args.token_budget or TOKEN_CEILING, "turn_timeout_seconds": 300,
            "experiment_timeout_seconds": 180, "max_failures": 2,
            "purge_seconds": 0, "embargo_seconds": 2700, "seed": 79,
            "strategy": "independent", "reuse_experiments": False, "guidance": GUIDANCE,
        }).model_dump(mode="json"),
        "idempotency_key": "isolated-autonomous-" + uuid.uuid4().hex,
        "verification": {"kind": "real_codex_autonomous_research", "synthetic_data": True,
                         "fixture_steps_per_object": 600, "fixture_objects": 2,
                         "scientific_quality_benchmark": False, "prepared_at": utc_now(),
                         "minimum_successful_experiments_per_candidate": 2,
                         "data_provenance": data_provenance,
                         "evidence_root": str(folder)},
    }
    path = folder / "configuration.json"
    write(path, config)
    return path


def load_configuration(path):
    path = path.resolve()
    folder = path.parent
    allowed = (ROOT / "research" / "v2_validation").resolve()
    if not folder.is_relative_to(allowed) or not folder.name.startswith("autonomous-"):
        raise ValueError("只接受本仓库 research/v2_validation/autonomous-* 隔离目录的配置")
    config = json.loads(path.read_text(encoding="utf-8"))
    for name, suffix in (("research_root", "research"), ("vault_root", "vault"), ("models_root", "models")):
        if Path(config["roots"][name]).resolve() != folder / suffix:
            raise ValueError("配置根目录与隔离验收目录不符：" + name)
    run = RunConfig.model_validate(config["run_config"])
    if (run.research_mode != "autonomous" or run.candidates != 5 or run.strategy != "independent"
            or run.reuse_experiments or run.token_budget > TOKEN_CEILING
            or run.max_experiments > 15 or run.max_experiments_per_track > 3 or run.max_turns > 8
            or run.turn_timeout_seconds > 300 or run.experiment_timeout_seconds > 180):
        raise ValueError("配置必须保持 1+5 独立自主研究和已声明的有界预算，不复用实验")
    if config.get("verification", {}).get("synthetic_data") is not True:
        raise ValueError("配置必须明确来自此入口的合成验收数据")
    provenance = config["verification"].get("data_provenance") or {}
    if (provenance.get("generator") != "synthetic-autonomous-v1"
            or provenance.get("different_from_legacy_fixture") is not True
            or provenance.get("dataset_ref") != run.dataset_ref):
        raise ValueError("配置必须包含新 seed 生成的数据来源，不能使用旧 acceptance 留出数据")
    dataset, revision = run.dataset_ref.split("@", 1)
    fingerprint = json.loads((folder / "vault" / "datasets" / dataset / revision / "fingerprint.json").read_text(encoding="utf-8"))
    if fingerprint["content_sha256"] != provenance.get("content_sha256"):
        raise ValueError("实际数据指纹与生成来源不一致")
    return config


def validate(snapshot, report, instances):
    """Validate actual evidence; blocked/incomplete research stays a failed run."""
    errors = []
    run = snapshot["run"]
    protocol = run["protocol"]["fingerprint"]
    tracks = snapshot["tracks"]
    candidates = [t for t in tracks if t["role"] == "candidate"]
    expected = {"main", *(f"candidate-{index}" for index in range(1, 6))}
    if run["status"] != "completed":
        errors.append("研究未进入 completed，缺少结果不能视为通过")
    if len(candidates) != 5 or set(instances) != expected:
        errors.append("缺少完整 1+5 Codex 实例证据")
    pids = [i.get("pid") for i in instances.values()]
    threads = [i.get("thread_id") for i in instances.values()]
    if len(pids) != 6 or len(set(pids)) != 6 or any(not isinstance(p, int) or p <= 0 for p in pids):
        errors.append("六个独立 Codex 进程 PID 未验证")
    if len(threads) != 6 or len(set(threads)) != 6 or any(not t for t in threads):
        errors.append("六个独立 Codex 会话未验证")
    if len({i.get("model") for i in instances.values()}) != 1 or any(not i.get("model") for i in instances.values()):
        errors.append("六实例未使用可核实的同一模型")
    if any(i.get("native_shell") is not False or i.get("native_subagents") is not False
           for i in instances.values()):
        errors.append("研究实例的原生 shell/子智能体隔离标志不完整")

    jobs = snapshot.get("jobs", [])
    proposals = snapshot.get("proposals", [])
    ideas = {i["id"]: i for i in snapshot.get("ideas", [])}
    sources = {s["id"]: s for s in snapshot.get("sources", [])}
    if any(j["track_id"] == "main" for j in jobs):
        errors.append("主智能体占用了训练预算")
    first_proposals = {t["track_id"]: min((p["created_at"] for p in proposals
                       if p["track_id"] == t["track_id"]), default=None) for t in candidates}
    if len(first_proposals) != 5 or any(value is None for value in first_proposals.values()):
        errors.append("至少一个候选未保存自己的初始冻结提案")
    elif jobs and max(first_proposals.values()) > min(j["created_at"] for j in jobs):
        errors.append("首轮全部提案齐备前已经预留实验，独立提案屏障失效")
    by_job = {j["id"]: j for j in jobs}
    by_proposal = {p["id"]: p for p in proposals}
    by_finding = {f["id"]: f for f in snapshot.get("findings", [])}
    for track in candidates:
        tid = track["track_id"]
        own_jobs = sorted((j for j in jobs if j["track_id"] == tid), key=lambda j: j["created_at"])
        successful = [j for j in own_jobs if j["status"] == "completed" and j.get("executed") is True]
        if len(successful) < 2:
            errors.append(f"{tid} 未完成至少两次真实成功实验；阻碍保留为失败证据")
        if not track_has_final_report(snapshot, track):
            errors.append(f"{tid} 最终报告未覆盖自己的全部研究证据")
        if not any(s["track_id"] == tid for s in snapshot.get("stops", [])):
            errors.append(f"{tid} 缺少有依据的明确停止决定")
        for index, job in enumerate(own_jobs):
            p = by_proposal.get(job.get("proposal_id"))
            hypothesis = ideas.get(job["idea_id"], {})
            if not p or p["track_id"] != tid or p["created_at"] > job["created_at"]:
                errors.append(f"{tid}/{job['id']} 缺少事前冻结的本轨迹提案")
                continue
            if (not job.get("experiment_fingerprint") or p["experiment_fingerprint"] != job["experiment_fingerprint"]
                    or p["model"] != (job.get("request") or {}).get("model")):
                errors.append(f"{tid}/{job['id']} 提案与实际模型身份不一致")
            if job.get("reused_from_job_id"):
                errors.append(f"{tid}/{job['id']} 独立实验错误复用了其他结果")
            if index:
                previous = own_jobs[index - 1]
                if (previous["id"] not in p.get("parent_job_ids", [])
                        or previous["id"] not in hypothesis.get("parent_job_ids", [])
                        or not any(by_finding.get(f, {}).get("track_id") == tid
                                   and previous["id"] in by_finding[f].get("job_ids", [])
                                   and by_finding[f]["created_at"] <= p["created_at"]
                                   for f in p.get("finding_ids", []))
                        or previous.get("finished_at", float("inf")) > p["created_at"]):
                    errors.append(f"{tid}/{job['id']} 修订没有引用自己上一轮已结算实验和分析")
            result = job.get("result") or {}
            if any(key in result for key in ("surfaces", "holdout", "final_evaluation")):
                errors.append(f"{tid}/{job['id']} 内部结果包含最终留出字段")
            if job["status"] == "completed":
                value = (result.get("metrics") or {}).get("CVRMSE")
                if (result.get("feedback_surface") != "validate" or result.get("protocol_fingerprint") != protocol
                        or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                    errors.append(f"{tid}/{job['id']} 缺少有效的同协议验证指标")
    for hypothesis in ideas.values():
        origin = hypothesis.get("origin")
        refs = hypothesis.get("source_ids") or []
        if origin not in {"literature", "history", "conjecture", "mixed"}:
            errors.append(hypothesis["id"] + " 未明确想法来源")
        if any(not hypothesis.get(field) for field in ("reason", "prediction", "falsification")):
            errors.append(hypothesis["id"] + " 缺少选择理由、预测或反证条件")
        if any(s not in sources for s in refs):
            errors.append(hypothesis["id"] + " 引用了不存在的来源")
        if origin == "literature" and not any(
                sources.get(s, {}).get("kind") in {"paper", "article"}
                and (sources[s].get("url") or sources[s].get("doi")) for s in refs):
            errors.append(hypothesis["id"] + " 文献启发缺少稳定文献来源")
        if origin == "mixed" and not refs:
            errors.append(hypothesis["id"] + " 混合来源没有登记可追溯资料")
        if origin == "history" and not (hypothesis.get("parent_job_ids") or any(
                sources.get(s, {}).get("kind") == "history" and sources[s].get("history_ids") for s in refs)):
            errors.append(hypothesis["id"] + " 历史来源没有实验引用")
        if hypothesis["track_id"] != "main" and any(
                j not in by_job or by_job[j]["track_id"] != hypothesis["track_id"]
                for j in hypothesis.get("parent_job_ids", [])):
            errors.append(hypothesis["id"] + " 独立候选引用了不存在或其他轨迹的实验")
    for source in sources.values():
        if source.get("read_scope") not in {"metadata", "abstract", "excerpt", "history"}:
            errors.append(source["id"] + " 未如实标注可核实的阅读范围")
        if source.get("kind") in {"paper", "article"} and not (source.get("url") or source.get("doi")):
            errors.append(source["id"] + " 文献资料缺少稳定 URL 或 DOI")
        if source.get("kind") == "history" and (not source.get("history_ids") or any(
                j not in by_job or by_job[j]["track_id"] != source["track_id"] for j in source.get("history_ids", []))):
            errors.append(source["id"] + " 历史来源缺少本轨迹实际实验")
    main = next((t for t in tracks if t["role"] == "main"), None)
    if not main or not track_has_final_report(snapshot, main):
        errors.append("缺少覆盖全部候选证据的主智能体综合报告")
    final = report.get("final_evaluation") or {}
    if final.get("status") != "evaluated" or final.get("feedback_to_agents") is not False:
        errors.append("缺少研究结束后才产生、且不回流智能体的最终留出评价")
    return errors


def factual_results(snapshot, report):
    """Only stored measurements, without claims about significance or diversity."""
    jobs = snapshot.get("jobs", [])
    identities = Counter(j["experiment_fingerprint"] for j in jobs
                         if j.get("experiment_fingerprint") and j.get("executed") is True)
    ranked = compare_jobs(jobs, snapshot["run"]["protocol"]["fingerprint"], 1)
    best = ranked["top_k"][0] if ranked["top_k"] else None
    selected = next((j for j in jobs if best and j["id"] == best["job_id"]), None)
    return {
        "candidate_success_counts": dict(Counter(j["track_id"] for j in jobs if j["status"] == "completed")),
        "executed_requests": sum(identities.values()), "unique_exact_experiments": len(identities),
        "duplicate_executions": sum(count - 1 for count in identities.values()),
        "duplicate_groups": [{"experiment_fingerprint": identity,
                              "job_ids": [j["id"] for j in jobs if j.get("experiment_fingerprint") == identity]}
                             for identity, count in identities.items() if count > 1],
        "duplicate_interpretation": "允许独立收敛；不强迫差异，也不把不同身份自动解释为科学创新。",
        "origin_counts": dict(Counter(i.get("origin", "missing") for i in snapshot.get("ideas", []))),
        "selected_validation": ({"job_id": selected["id"], "track_id": selected["track_id"],
                                 "model": selected["request"]["model"],
                                 "metrics": {k: v for k, v in selected["result"]["metrics"].items() if k in METRICS}}
                                if selected else None),
        "external_final_evaluation": report.get("final_evaluation"),
        "feedback_boundary_check_scope": "核对持久化内部实验结果、Codex 权限标志与外部留出隔离标志；不是完整会话内容审计。",
        "source_check_scope": "核对来源类型、稳定引用和实际实验 ID；未独立核实每篇论文内容，保留来源原有 verification 标记。",
    }


def execute(configuration_path, args):
    configuration_path = configuration_path.resolve()
    config = load_configuration(configuration_path)
    folder = configuration_path.parent
    attempt = folder / ("attempt-" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    proof_path = attempt / "autonomous-evidence.json"
    event_path = attempt / "events.jsonl"
    proof = {"kind": "real_codex_autonomous_research", "synthetic_data": True,
             "scientific_quality_benchmark": False, "passed": False, "errors": [],
             "started_at": utc_now(), "configuration": str(configuration_path),
             "configuration_sha256": hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
             "run_config": config["run_config"], "instances": {}, "instance_history": [],
             "peak_concurrent_turns": 0, "original_events": str(event_path),
             "evidence_directory": str(attempt), "events_count": 0}
    write(proof_path, proof)
    client = client_for(config, autostart=False)
    service_pid = rid = None
    cursor, active = 0, set()

    def collect_events():
        nonlocal cursor
        page = client.events(rid, after=cursor, limit=1000)
        cursor = page["cursor"]
        with event_path.open("a", encoding="utf-8", newline="\n") as output:
            for event in page["events"]:
                output.write(json.dumps(event, ensure_ascii=False) + "\n")
                proof["events_count"] += 1
                tid, kind, payload = event["track_id"], event["kind"], event["payload"]
                if kind == "instance.started":
                    instance = {k: payload.get(k) for k in (
                        "pid", "thread_id", "model", "reasoning_effort", "service_tier", "fast_mode",
                        "requested_backend", "server", "permission_profile",
                        "native_shell", "native_subagents", "dynamic_tools", "isolation")}
                    instance["session_id"] = payload.get("session_id") or payload.get("thread_id")
                    proof["instances"].setdefault(tid, instance)
                    proof["instance_history"].append({"track_id": tid, "at": event["at"], **instance})
                if kind == "turn/started":
                    active.add(tid)
                    proof["peak_concurrent_turns"] = max(proof["peak_concurrent_turns"], len(active))
                elif kind == "turn/completed":
                    active.discard(tid)
        return bool(page["events"])

    try:
        # The submission client exits; all six Codex sessions remain owned by the isolated service.
        submitted = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--submit-child", str(configuration_path)],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if submitted.returncode:
            # Preserve diagnostics locally; never send raw subprocess/configuration output to stdout.
            (attempt / "submission-error.txt").write_text(submitted.stderr, encoding="utf-8")
            raise RuntimeError("提交客户端失败，详见隔离目录 submission-error.txt")
        receipt = json.loads(submitted.stdout.strip().splitlines()[-1])
        rid, service_pid = receipt["run_id"], receipt["service_pid"]
        proof.update(run_id=rid, submission=receipt | {"client_exited": True})
        write(proof_path, proof)
        emit(event="autonomous_started", run_id=rid, service_pid=service_pid,
             token_budget=config["run_config"]["token_budget"], evidence=str(proof_path))
        deadline, prior, printed, previous_tokens = time.monotonic() + args.timeout, None, 0, 0
        while True:
            collect_events()
            snapshot = client.get_run(rid)
            run = snapshot["run"]
            if run["tokens_used"] < previous_tokens:
                raise RuntimeError("累计用量倒退，取消验收以防预算失真")
            previous_tokens = run["tokens_used"]
            counts = {key: len(snapshot.get(key, [])) for key in ("proposals", "jobs", "findings", "reports", "stops")}
            main = next((t for t in snapshot["tracks"] if t["role"] == "main"), {})
            progress = (run["status"], run.get("research_stage"), tuple(counts.values()),
                        tuple((t["track_id"], t["status"], t.get("phase"), t["turns"]) for t in snapshot["tracks"]))
            if progress != prior or time.monotonic() - printed >= 30:
                emit(event="autonomous_progress", state=run["status"], stage=run.get("research_stage"),
                     counts=counts, jobs=dict(Counter(j["status"] for j in snapshot["jobs"])),
                     tokens_used=run["tokens_used"], main={k: main.get(k) for k in ("status", "phase", "turns")})
                prior, printed = progress, time.monotonic()
            proof.update(status=run["status"], research_stage=run.get("research_stage"),
                         tokens_used=run["tokens_used"], counts=counts)
            write(attempt / "snapshot.json", snapshot)
            write(proof_path, proof)
            if run["status"] not in ACTIVE:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("自主研究达到有界时间上限，将取消隔离运行并保留未完成证据")
            time.sleep(2)
        while collect_events():
            pass
        report = client.get_report(rid)
        write(attempt / "external-report.json", report)
        proof["errors"] = validate(snapshot, report, proof["instances"])
        proof["results"] = factual_results(snapshot, report)
        proof["report_artifacts"] = report.get("artifacts", [])
        proof["passed"] = not proof["errors"]
    except Exception as exc:
        proof["errors"].append(type(exc).__name__ + ": " + str(exc)[:2000])
    finally:
        try:
            descriptor = client._alive()
            if service_pid is None and descriptor:
                service_pid = descriptor["pid"]
            if rid and descriptor:
                if client.get_run(rid)["run"]["status"] in ACTIVE:
                    client.control(rid, "cancel")
                cleanup_deadline = time.monotonic() + 180
                while True:
                    snapshot = client.get_run(rid)
                    collect_events()
                    remaining = [t.get("pid") for t in snapshot["tracks"] if t.get("pid")]
                    if not remaining or time.monotonic() >= cleanup_deadline:
                        break
                    time.sleep(2)
                write(attempt / "final-snapshot.json", snapshot)
                proof.update(status=snapshot["run"]["status"], tokens_used=snapshot["run"]["tokens_used"])
                if remaining:
                    raise RuntimeError("候选进程尚未关闭；保留隔离服务以继续处理取消，需检查 PID " + str(remaining))
            if service_pid:
                proof["service_cleanup"] = stop_isolated_service(client, service_pid)
        except Exception as exc:
            proof["errors"].append("隔离服务清理失败：" + str(exc)[:1000])
            proof["passed"] = False
        if event_path.exists():
            proof["events_sha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
        proof["finished_at"] = utc_now()
        write(proof_path, proof)
    emit(passed=proof["passed"], evidence=str(proof_path), errors=proof["errors"],
         tokens_used=proof.get("tokens_used"), counts=proof.get("counts"))
    return 0 if proof["passed"] else 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="只生成合成数据和配置，不启动服务或模型")
    parser.add_argument("--config", type=Path, help="执行已保存的隔离配置；不覆盖配置或此前验收证据")
    parser.add_argument("--model", help=f"新配置使用的 Codex 模型；默认 {CODEX_MODEL}")
    parser.add_argument("--reasoning-effort", help=f"新配置使用的思考强度；默认最高档 {CODEX_EFFORT}")
    parser.add_argument("--data-seed", type=int, help="独立合成数据的 seed，默认 2026090701；与模型训练 seed 分开")
    parser.add_argument("--token-budget", type=int, help="累计 token 上限，默认 3000000，最高 3000000")
    parser.add_argument("--timeout", type=float, default=1800, help="有界研究监控秒数，默认 1800，最高 3600")
    parser.add_argument("--submit-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.submit_child:
        load_configuration(args.submit_child)
        submit_child(args.submit_child)
        return 0
    if not 1 <= args.timeout <= 3600:
        parser.error("--timeout 必须为 1–3600 秒")
    if args.token_budget is not None and not 1000 <= args.token_budget <= TOKEN_CEILING:
        parser.error("--token-budget 必须为 1000–3000000")
    if args.data_seed is not None and not 0 <= args.data_seed <= 2**32 - 1:
        parser.error("--data-seed 必须为 0–4294967295")
    if args.config and any(value is not None for value in (args.model, args.reasoning_effort, args.token_budget, args.data_seed)):
        parser.error("--config 使用保存的准确配置，不能同时覆盖模型或 token 预算")
    configuration_path = args.config.resolve() if args.config else prepare(args)
    configuration = load_configuration(configuration_path)
    if args.prepare_only:
        emit(event="prepared_only", configuration=str(configuration_path), model_calls=0,
             token_budget=configuration["run_config"]["token_budget"],
             data_seed=configuration["verification"]["data_provenance"]["data_seed"],
             content_sha256=configuration["verification"]["data_provenance"]["content_sha256"],
             evidence_root=str(configuration_path.parent))
        return 0
    return execute(configuration_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
