"""`tf status` 汇总面板：纯读 research/ 与 models/，不写任何文件。

内容：goals 进展（goal_progress）、最近 N 次实验（状态 + 主测试面
关键指标）、生产模型清单（registry 当前 production）、未审批预处理
规则数、Ledger 停止原因统计。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thermoforge_research.tools import ToolContext


def _primary_metrics(report: dict[str, Any]) -> dict[str, Any]:
    surfaces = (report.get("metrics") or {}).get("surfaces") or {}
    for name in ("C", "A", "validate"):
        surf = surfaces.get(name) or {}
        if surf.get("n_samples"):
            return dict(surf.get("metrics") or {})
    return {}


def build_status(ctx: ToolContext, *, recent: int = 5) -> dict[str, Any]:
    # 纯读面板：research_root 不存在时返回空面板，不创建任何目录
    empty = {"goals": [], "recent_experiments": [], "production_models": [],
             "preprocess": {"rulesets": 0, "pending_rules": 0},
             "stop_reasons": {}}
    if not ctx.research_root.exists():
        return empty

    ledger = ctx.ledger

    goals = []
    stop_reasons: dict[str, int] = {}
    for goal in ledger.list_goals():
        progress = ledger.goal_progress(goal["id"])
        goals.append({
            "goal_id": goal["id"],
            "name": goal.get("name"),
            "status": goal["status"],
            "experiments": progress["experiments"],
            "hypotheses": progress["hypotheses"],
            "models": progress["models"],
            "last_activity": progress["last_activity"],
        })
        if goal["status"] in ("STOPPED", "PUBLISH"):
            last = goal["transitions"][-1]
            reason = str(last.get("reason") or "").split(":", 1)[0]
            stop_reasons[f"{goal['status']}:{reason}"] = (
                stop_reasons.get(f"{goal['status']}:{reason}", 0) + 1
            )

    exp_dir = ctx.research_root / "experiments"
    reports = []
    if exp_dir.is_dir():
        for path in exp_dir.glob("EXP-*/report.json"):
            try:
                with open(path, encoding="utf-8") as fp:
                    report = json.load(fp)
            except (ValueError, OSError):
                continue
            metrics = _primary_metrics(report)
            reports.append({
                "experiment_id": report.get("experiment_id", path.parent.name),
                "status": report.get("status"),
                "started_at": report.get("started_at"),
                "duration_seconds": report.get("duration_seconds"),
                "cvrmse": metrics.get("CVRMSE"),
                "nmbe": metrics.get("NMBE"),
                "mape": metrics.get("MAPE"),
                "physics_overall_rate": (report.get("physics") or {})
                .get("overall_rate"),
            })
    reports.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)

    models = []
    if ctx.models_root.is_dir():
        for index_path in sorted(ctx.models_root.glob("*/registry.json")):
            with open(index_path, encoding="utf-8") as fp:
                index = json.load(fp)
            models.append({
                "model_id": index.get("model_id", index_path.parent.name),
                "production": index.get("production"),
                "versions": len(index.get("versions") or {}),
            })

    rulesets = (
        ctx.preprocess_store.list_rulesets()
        if (ctx.research_root / "preprocess").exists() else []
    )
    pending_rules = sum(
        1 for rs in rulesets for r in rs["rules"] if r["status"] == "proposed"
    )

    return {
        "goals": goals,
        "recent_experiments": reports[:recent],
        "production_models": models,
        "preprocess": {"rulesets": len(rulesets),
                       "pending_rules": pending_rules},
        "stop_reasons": stop_reasons,
    }


def render_json(status: dict[str, Any]) -> str:
    return json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True)


def render_text(status: dict[str, Any]) -> str:
    lines = ["ThermoForge 状态面板", "=" * 60, "", "◆ Research Goals"]
    if not status["goals"]:
        lines.append("  （无）")
    for g in status["goals"]:
        exp = g["experiments"]
        lines.append(
            f"  {g['goal_id']}  [{g['status']}] {g['name']}\n"
            f"    实验 {exp['total']}（{exp['by_status']}）· "
            f"假设 {g['hypotheses']['total']} · 模型 {g['models']['total']} · "
            f"最近活动 {g['last_activity']}"
        )
    lines += ["", "◆ 最近实验"]
    if not status["recent_experiments"]:
        lines.append("  （无）")
    for r in status["recent_experiments"]:
        cv = r["cvrmse"]
        nmbe = r["nmbe"]
        lines.append(
            f"  {r['experiment_id']}  [{r['status']}] "
            f"CVRMSE={f'{cv:.4f}' if isinstance(cv, float) else '-'} "
            f"NMBE={f'{nmbe:+.4f}' if isinstance(nmbe, float) else '-'} "
            f"（{r['duration_seconds']:.1f}s）"
            if isinstance(r.get("duration_seconds"), float)
            else f"  {r['experiment_id']}  [{r['status']}]"
        )
    lines += ["", "◆ 生产模型"]
    if not status["production_models"]:
        lines.append("  （无）")
    for m in status["production_models"]:
        lines.append(f"  {m['model_id']}  production={m['production']} "
                     f"（共 {m['versions']} 个版本）")
    pp = status["preprocess"]
    lines += [
        "",
        "◆ 预处理规则",
        f"  规则集 {pp['rulesets']} 个，待审批规则 {pp['pending_rules']} 条",
        "",
        "◆ 目标停止原因统计",
    ]
    if not status["stop_reasons"]:
        lines.append("  （无停止/发布记录）")
    for reason, count in sorted(status["stop_reasons"].items()):
        lines.append(f"  {reason}: {count}")
    return "\n".join(lines)
