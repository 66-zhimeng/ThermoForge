"""有实测依据的候选保留与停滞策略建议，评分器始终由领域服务固定。"""

from __future__ import annotations

import math


def compare_jobs(jobs: list[dict], fingerprint: str, top_k: int = 2) -> dict:
    ranked, excluded = [], []
    for job in jobs:
        result = job.get("result") or {}
        value = (result.get("metrics") or {}).get("CVRMSE")
        if (job.get("status") != "completed"
                or result.get("protocol_fingerprint") != fingerprint
                or result.get("feedback_surface") != "validate"
                or not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            excluded.append(job["id"])
            continue
        row = {"job_id": job["id"], "track_id": job["track_id"],
               "idea_id": job["idea_id"], "CVRMSE": value}
        identity = job.get("experiment_fingerprint")
        if isinstance(identity, str) and identity:
            row["experiment_fingerprint"] = identity
        ranked.append(row)
    ranked.sort(key=lambda row: (row["CVRMSE"], row["job_id"]))
    return {"metric": "CVRMSE", "surface": "validate", "lower_is_better": True,
            "ranked": ranked, "top_k": ranked[:top_k], "excluded": excluded,
            "reason": "仅比较相同冻结协议的有效验证指标；未据此宣称最终留出达标。"}


def strategy_advice(jobs: list[dict], config: dict, fingerprint: str) -> dict:
    comparison = compare_jobs(jobs, fingerprint, config.get("top_k", 2))
    # 多实验 worker 的完成顺序不等于预约顺序。优先按真实归账时间判断
    # 最近反馈；没有结束时间的旧记录使用创建时间，再保留原有顺序。
    def completion_order(indexed):
        index, record = indexed
        for field in ("finished_at", "created_at"):
            value = record.get(field)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value)):
                return value, index
        return float("-inf"), index

    completed = [record for _, record in sorted(
        ((index, j) for index, j in enumerate(jobs)
         if j.get("status") in {"completed", "failed"}), key=completion_order)]
    window = config.get("stagnation_rounds", 2)
    previous = compare_jobs(completed[:-window], fingerprint, 1)["top_k"]
    recent = compare_jobs(completed[-window:], fingerprint, 1)["top_k"]
    stagnant = bool(previous) and (not recent or recent[0]["CVRMSE"] >= previous[0]["CVRMSE"])
    policy = config.get("strategy", "independent")
    selected = "independent" if policy == "independent" else "refine"
    if policy == "adaptive" and stagnant:
        selected = "explore" if len(completed) // window % 2 else "recombine"
    # 实验排行榜保留所有实测结果。新记录按完整实验身份识别重复路线，
    # 避免不同想法 ID 包装的同一实验占满父路线；旧记录保留想法去重。
    # 只识别精确身份，不能把同类算法、不同参数或源码当成同一实验。
    seen_routes, selected_routes, identity_groups = set(), [], {}
    for row in comparison["ranked"]:
        identity = row.get("experiment_fingerprint")
        if identity:
            identity_groups.setdefault(identity, []).append(row)
        route = ("experiment", identity) if identity else ("idea", row["idea_id"])
        if route in seen_routes:
            continue
        seen_routes.add(route)
        if len(selected_routes) < config.get("top_k", 2):
            selected_routes.append(row)
    duplicates = [{"experiment_fingerprint": identity,
                   "representative_job_id": rows[0]["job_id"],
                   "job_ids": [r["job_id"] for r in rows],
                   "idea_ids": list(dict.fromkeys(r["idea_id"] for r in rows)),
                   "track_ids": list(dict.fromkeys(r["track_id"] for r in rows)),
                   "duplicate_count": len(rows) - 1}
                  for identity, rows in identity_groups.items() if len(rows) > 1]
    independent = policy == "independent"
    reason = ("独立模式仅回顾验证结果和精确重复，不向候选分发其他路线反馈。" if independent
              else "验证进展停滞，建议扩展结构或组合已有思路。" if stagnant
              else "依据有效验证结果保留路线；缺少证据时继续独立探索。")
    return {"configured_strategy": policy, "suggested_action": selected,
            "stagnant": stagnant, "comparison": comparison,
            "parents": [] if independent else list(dict.fromkeys(r["idea_id"] for r in selected_routes)),
            "selected_routes": selected_routes,
            "duplicate_experiments": duplicates,
            "duplicate_experiment_count": sum(g["duplicate_count"] for g in duplicates),
            "advice_scope": "retrospective" if independent else "coordination",
            "recent_feedback_job_ids": [j["id"] for j in completed[-window:]],
            "reason": reason,
            "requires_scientific_decision": True}
