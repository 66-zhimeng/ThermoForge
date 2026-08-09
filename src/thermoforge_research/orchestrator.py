"""Research Orchestrator（research-loop.md §2/§9、architecture.md §3/§5）。

编排循环：Research Goal → 假设 → 实验计划 → 执行 → 结果分析 → 下一轮。
按 §2 状态机在 goal 实体上留痕（HYPOTHESIS_GENERATION /
EXPERIMENT_RUNNING / RESULT_ANALYSIS / MODEL_REVIEW → PUBLISH|STOPPED）。

铁律：

- **编排层只拿工具信封与摘要**（architecture §3：Agent 不直接读取
  海量原始数据）——本模块只调用 `tools.py` 的信封函数与 Ledger 的
  结构化查询，不触碰 parquet 原始行。
- 每轮实验必须引用已有证据：非首个假设的 `basis` 由 planner 提供，
  Ledger 强制非空（research-loop §3）。
- **停止条件**（§9）全部实现并输出结构化原因，见 `STOP_REASONS`；
  预算耗尽伴随 TFX-906 WARN 诊断（conventions §7.9）。

planner 协议（Agent 决策点注入）::

    planner(round_index, evidence) -> plan | None
    plan = {
        "statement": str,            # 假设陈述
        "basis": [finding/exp IDs],  # 证据引用（非首个假设必填）
        "view_id": "VIEW-0001",      # 已登记的 Dataset View
        "model": {...Experiment ModelSpec...},
        "validation": {...},          # 可选，默认 0.70/0.15/0.15
        "y_floor": float,             # 可选
    }
    # planner 返回 None 表示无可行假设 → no_information_gain 停止
    # plan["needs_human"] = "原因"   → human_confirmation_required 停止

`evidence` 只含信封摘要：各轮指标、最优指标、失败原因、预检画像。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .envelope import finalize_envelope, make_envelope
from .runner import current_environment_lock
from .tools import (
    ToolContext,
    tf_dataset_modelability,
    tf_dataset_profile,
    tf_dataset_schema,
    tf_experiment_plan,
    tf_experiment_run,
    tf_hypothesis_create,
    tf_research_status,
)

# research-loop §9 停止条件
STOP_ACCEPTANCE_MET = "acceptance_met"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_NO_GAIN = "no_information_gain"
STOP_INSUFFICIENT_COVERAGE = "insufficient_data_coverage"
STOP_MISSING_VARIABLES = "missing_required_variables"
STOP_HUMAN_CONFIRMATION = "human_confirmation_required"
# gap-analysis G3：可建模性门禁未过（存在 blocker），不得进入建模
STOP_MODELABILITY_FAILED = "modelability_failed"

STOP_REASONS = (
    STOP_ACCEPTANCE_MET,
    STOP_BUDGET_EXHAUSTED,
    STOP_NO_GAIN,
    STOP_INSUFFICIENT_COVERAGE,
    STOP_MISSING_VARIABLES,
    STOP_HUMAN_CONFIRMATION,
    STOP_MODELABILITY_FAILED,
)

_DEFAULT_SPLIT = {"train": 0.70, "validate": 0.15, "test": 0.15}
_ALL_METRICS = ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"]

Planner = Callable[[int, Mapping[str, Any]], Mapping[str, Any] | None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResearchOrchestrator:
    """单 Agent 顺序编排器（architecture §5：一期单 Agent 多角色）。

    ::

        orch = ResearchOrchestrator(ctx, goal_id, planner,
                                    dataset_ref="DC01_2026_CHILLER@rev_0001")
        result = orch.run()
        result["stop"]["reason"]   # STOP_REASONS 之一
    """

    def __init__(
        self,
        ctx: ToolContext,
        goal_id: str,
        planner: Planner,
        *,
        dataset_ref: str,
        no_gain_rounds: int = 3,
        min_gain_rel: float = 0.01,
        min_rows: int = 100,
        min_coverage_days: float = 1.0,
        max_rounds: int = 50,
        random_seed: int = 20260808,
    ):
        self.ctx = ctx
        self.goal_id = goal_id
        self.planner = planner
        self.dataset_ref = dataset_ref
        self.no_gain_rounds = int(no_gain_rounds)
        self.min_gain_rel = float(min_gain_rel)
        self.min_rows = int(min_rows)
        self.min_coverage_days = float(min_coverage_days)
        self.max_rounds = int(max_rounds)
        self.random_seed = int(random_seed)

        self.rounds: list[dict[str, Any]] = []
        self.best_cvrmse: float | None = None
        self._no_gain_streak = 0

    # ---------------------------------------------------------------- 主循环

    def run(self) -> dict[str, Any]:
        goal = self.ctx.ledger.get(self.goal_id)
        definition = goal.get("definition") or {}

        stop = self._precheck_variables(definition)
        stop = stop or self._precheck_coverage()
        stop = stop or self._precheck_modelability(definition)
        if stop:
            return self._finish(stop)

        for round_index in range(1, self.max_rounds + 1):
            stop = self._check_budget(definition)
            if stop:
                return self._finish(stop)
            stop = self._run_round(round_index, definition)
            if stop:
                return self._finish(stop)
        return self._finish(self._stop(
            STOP_BUDGET_EXHAUSTED,
            f"达到编排器硬上限 max_rounds={self.max_rounds}",
        ))

    # ---------------------------------------------------------------- 预检

    def _precheck_variables(
        self, definition: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """必需变量缺失（§9）：candidate_inputs 封闭白名单 + target 必须在
        数据版本中存在（DD-16）。"""
        env = tf_dataset_schema(self.ctx, self.dataset_ref)
        if not env["ok"]:
            return self._stop(
                STOP_MISSING_VARIABLES,
                f"无法读取数据版本 Schema: {env['summary'].get('error')}",
            )
        summary = env["summary"]
        # 紧凑清单优先（宽表 variables 明细可能被 32KB 截断）
        available_ids = set(summary.get("variable_ids") or [])
        available_props = set(summary.get("property_codes") or [])
        if not available_ids:
            variables = [v for v in summary.get("variables", [])
                         if isinstance(v, dict)]
            available_ids = {v["variable_id"] for v in variables}
            available_props = {v["property_code"] for v in variables}
        missing: list[str] = []
        for item in [definition.get("target"),
                     *(definition.get("candidate_inputs") or [])]:
            if item is None:
                continue
            item = str(item)
            ok = (item in available_ids) if "." in item \
                else (item in available_props)
            if not ok:
                missing.append(item)
        if missing:
            return self._stop(
                STOP_MISSING_VARIABLES,
                f"数据版本 {self.dataset_ref} 缺少必需变量: {missing}（§9）",
                evidence={"missing": missing},
            )
        return None

    def _precheck_coverage(self) -> dict[str, Any] | None:
        """数据覆盖不足（§9）：行数或时间跨度低于下限则无法验证关键工况。"""
        env = tf_dataset_profile(self.ctx, self.dataset_ref)
        if not env["ok"]:
            return self._stop(
                STOP_INSUFFICIENT_COVERAGE,
                f"无法读取数据画像: {env['summary'].get('error')}",
            )
        summary = env["summary"]
        rows = int(summary.get("record_count") or 0)
        span_days = 0.0
        t0, t1 = (summary.get("time_range") or [None, None])
        if t0 and t1:
            from thermoforge_core.timeutil import parse_timestamp

            span_days = (parse_timestamp(t1)
                         - parse_timestamp(t0)).total_seconds() / 86400.0
        if rows < self.min_rows or span_days < self.min_coverage_days:
            return self._stop(
                STOP_INSUFFICIENT_COVERAGE,
                f"数据覆盖不足: {rows} 行 / {span_days:.2f} 天 "
                f"（下限 {self.min_rows} 行 / {self.min_coverage_days} 天，§9）",
                evidence={"rows": rows, "span_days": span_days},
            )
        return None

    # ---------------------------------------------------------------- G3 门禁

    def _precheck_modelability(
        self, definition: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """可建模性门禁（G3）：DATA_PROFILING 之后、首个实验之前执行。

        报告 verdict=FAIL（存在 blocker）时不允许登记实验，停止原因
        `modelability_failed`（结构化，含 blocker 清单与报告 artifact）。
        """
        ledger = self.ctx.ledger
        ledger.transition(
            self.goal_id, "MODELABILITY_ASSESSMENT",
            reason="G3 可建模性门禁：进入建模前的语义层判定",
            actor=self.ctx.actor,
        )
        env = tf_dataset_modelability(self.ctx, self.dataset_ref,
                                      goal_id=self.goal_id)
        artifact = next((a for a in env.get("artifacts", [])
                         if a.get("kind") == "modelability_report"), None)
        if env["status"] == "FAIL":
            blockers = env["summary"]["blockers"]
            return self._stop(
                STOP_MODELABILITY_FAILED,
                f"可建模性报告存在阻断级检查未过: {blockers}（G3 门禁）",
                evidence={"blockers": blockers,
                          "warnings": env["summary"]["warnings"],
                          "checks": env["summary"]["checks"],
                          "report_artifact": (artifact or {}).get("path")},
            )
        ledger.transition(
            self.goal_id, "BASELINE_MODELING",
            reason=f"可建模性门禁通过（warnings={env['summary']['warnings']}）",
            actor=self.ctx.actor,
        )
        return None

    # ---------------------------------------------------------------- 预算

    def _check_budget(
        self, definition: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """预算耗尽（§9）：实验数 / 研究时长 / 计算量，TFX-906 WARN 触发停止。"""
        status = tf_research_status(self.ctx, self.goal_id)
        budget = status["summary"]["budget"]
        used = int(budget["experiments_used"])
        max_exp = budget.get("experiments_max")
        reasons: list[str] = []
        if max_exp is not None and used >= int(max_exp):
            reasons.append(f"实验数 {used} >= max_experiments={max_exp}")
        days_max = budget.get("days_max")
        if days_max is not None and budget.get("days_elapsed", 0.0) >= float(days_max):
            reasons.append(
                f"研究时长 {budget['days_elapsed']:.2f} 天 >= "
                f"max_duration_days={days_max}"
            )
        compute_max = definition.get("compute_budget_hours")
        if compute_max is not None:
            spent_hours = sum(
                float(r.get("duration_seconds") or 0.0) for r in self.rounds
            ) / 3600.0
            if spent_hours >= float(compute_max):
                reasons.append(
                    f"计算量 {spent_hours:.3f} h >= compute_budget_hours={compute_max}"
                )
        if reasons:
            return self._stop(
                STOP_BUDGET_EXHAUSTED, "; ".join(reasons) + "（§9）",
                diagnostic={"code": "TFX-906", "level": "WARN", "count": 1,
                            "message": "; ".join(reasons)},
            )
        return None

    # ---------------------------------------------------------------- 单轮

    def _run_round(
        self, round_index: int, definition: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        ledger = self.ctx.ledger
        evidence = self._evidence()
        plan = self.planner(round_index, evidence)
        if plan is None:
            return self._stop(
                STOP_NO_GAIN, "planner 无可行假设（§2：STOPPED: no useful hypothesis）",
            )
        if plan.get("needs_human"):
            return self._stop(
                STOP_HUMAN_CONFIRMATION, str(plan["needs_human"]),
            )

        ledger.transition(
            self.goal_id, "HYPOTHESIS_GENERATION",
            reason=f"第 {round_index} 轮假设生成", actor=self.ctx.actor,
        )
        hyp_env = tf_hypothesis_create(
            self.ctx, self.goal_id, str(plan["statement"]),
            basis=list(plan.get("basis") or ()),
        )
        if not hyp_env["ok"]:
            return self._record_failure(
                round_index, None, None,
                f"假设创建失败: {hyp_env['summary'].get('error')}",
            )

        ledger.transition(
            self.goal_id, "EXPERIMENT_DESIGN",
            reason=f"第 {round_index} 轮实验设计", actor=self.ctx.actor,
        )
        exp_doc = {
            "goal_id": self.goal_id,
            "hypothesis_id": hyp_env["id"],
            "dataset_view": str(plan["view_id"]),
            "model": dict(plan["model"]),
            "target": str(definition.get("target")),
            "validation": dict(plan.get("validation") or {
                "temporal_split": dict(_DEFAULT_SPLIT),
            }),
            "metrics": list(_ALL_METRICS),
            "physics_tests": {"enabled": True},
            "runtime": {
                "environment_lock": current_environment_lock()[0],
                "random_seed": self.random_seed,
            },
        }
        plan_env = tf_experiment_plan(self.ctx, exp_doc)
        if not plan_env["ok"]:
            return self._record_failure(
                round_index, hyp_env["id"], None,
                f"实验计划登记失败: {plan_env['summary'].get('error')}",
            )

        exp_id = plan_env["id"]
        ledger.transition(
            self.goal_id, "EXPERIMENT_RUNNING",
            reason=f"第 {round_index} 轮实验执行: {exp_id}",
            actor=self.ctx.actor, outputs=[exp_id],
        )
        run_env = tf_experiment_run(
            self.ctx, exp_id,
            y_floor=(float(plan["y_floor"]) if plan.get("y_floor") else None),
        )
        ledger.transition(
            self.goal_id, "RESULT_ANALYSIS",
            reason=f"第 {round_index} 轮结果分析: {exp_id}",
            actor=self.ctx.actor, inputs=[exp_id],
        )
        if not run_env["ok"]:
            return self._record_failure(
                round_index, hyp_env["id"], exp_id,
                f"实验失败: {run_env['summary'].get('error_code')}",
                duration=run_env["summary"].get("duration_seconds"),
            )

        summary = run_env["summary"]
        primary = self._primary_metrics(summary)
        finding = ledger.create_finding(
            f"第 {round_index} 轮 {exp_id}: "
            f"主测试面指标 {primary or '无可用指标'}",
            actor=self.ctx.actor, supported_by=[exp_id],
            hypothesis_id=hyp_env["id"],
            reason="编排器结构化发现（research-loop §3）",
        )
        record = {
            "round": round_index,
            "hypothesis_id": hyp_env["id"],
            "experiment_id": exp_id,
            "finding_id": finding["id"],
            "status": "completed",
            "primary_metrics": primary,
            "physics_overall_rate": summary.get("physics_overall_rate"),
            "duration_seconds": summary.get("duration_seconds"),
        }
        self.rounds.append(record)

        ledger.transition(
            self.goal_id, "MODEL_REVIEW",
            reason=f"第 {round_index} 轮模型评审: {exp_id}",
            actor=self.ctx.actor, inputs=[exp_id, finding["id"]],
        )
        acceptance = definition.get("acceptance") or {}
        unmet = self._acceptance_unmet(acceptance, summary)
        if not unmet:
            if (definition.get("approval_required") or []):
                return self._stop(
                    STOP_HUMAN_CONFIRMATION,
                    f"验收条件已满足，但目标要求人工审批: "
                    f"{definition['approval_required']}（§9）",
                    evidence={"candidate_experiment": exp_id,
                              "primary_metrics": primary},
                )
            model = ledger.register_model(
                f"{self.goal_id}-candidate", exp_id, actor=self.ctx.actor,
                metrics=primary or {},
                reason="验收条件满足，登记发布候选",
            )
            ledger.create_decision(
                f"goal {self.goal_id}", "accept",
                actor=self.ctx.actor,
                rationale="硬性验收条件全部满足（§9），登记发布候选",
                references=[exp_id, model["id"]],
            )
            return self._stop(
                STOP_ACCEPTANCE_MET,
                f"硬性验收条件全部满足，发布候选 {model['id']}（实验 {exp_id}）",
                evidence={"candidate_experiment": exp_id,
                          "candidate_model": model["id"],
                          "primary_metrics": primary},
            )

        # 信息增益跟踪：主测试面 CVRMSE 相对改善 < min_gain_rel 计一轮无增益
        cvrmse = (primary or {}).get("CVRMSE")
        if cvrmse is not None and (
            self.best_cvrmse is None
            or float(cvrmse) < self.best_cvrmse * (1.0 - self.min_gain_rel)
        ):
            self.best_cvrmse = float(cvrmse)
            self._no_gain_streak = 0
        else:
            self._no_gain_streak += 1
        if self._no_gain_streak >= self.no_gain_rounds:
            return self._stop(
                STOP_NO_GAIN,
                f"连续 {self._no_gain_streak} 轮无显著信息增益"
                f"（CVRMSE 相对改善 < {self.min_gain_rel}，§9）",
                evidence={"best_cvrmse": self.best_cvrmse},
            )
        return None

    def _record_failure(
        self, round_index: int, hypothesis_id: str | None,
        experiment_id: str | None, detail: str,
        duration: float | None = None,
    ) -> dict[str, Any] | None:
        self.rounds.append({
            "round": round_index,
            "hypothesis_id": hypothesis_id,
            "experiment_id": experiment_id,
            "status": "failed",
            "detail": detail,
            "duration_seconds": duration,
        })
        self._no_gain_streak += 1
        if self._no_gain_streak >= self.no_gain_rounds:
            return self._stop(
                STOP_NO_GAIN,
                f"连续 {self._no_gain_streak} 轮失败/无增益（§9）",
            )
        return None

    # ---------------------------------------------------------------- 判定

    @staticmethod
    def _primary_metrics(summary: Mapping[str, Any]) -> dict[str, Any] | None:
        surfaces = summary.get("surfaces") or {}
        for name in ("C", "A", "validate"):
            surf = surfaces.get(name) or {}
            if surf.get("n_samples"):
                return dict(surf.get("metrics") or {})
        return None

    @staticmethod
    def _acceptance_unmet(
        acceptance: Mapping[str, Any], summary: Mapping[str, Any]
    ) -> list[str]:
        """硬性验收条件（与发布门禁 TFM-1003 同口径：主测试面 C→A→validate）。"""
        metrics = ResearchOrchestrator._primary_metrics(summary) or {}
        values: dict[str, Any] = dict(metrics)
        if summary.get("physics_overall_rate") is not None:
            values["physics_violation_rate"] = summary["physics_overall_rate"]
        unmet: list[str] = []
        checks = (
            ("cvrmse_max", "CVRMSE", lambda v, lim: v <= lim),
            ("mape_max", "MAPE", lambda v, lim: v <= lim),
            ("nmbe_abs_max", "NMBE", lambda v, lim: abs(v) <= lim),
            ("physics_violation_rate_max", "physics_violation_rate",
             lambda v, lim: v <= lim),
        )
        for key, metric, ok_fn in checks:
            limit = acceptance.get(key)
            if limit is None:
                continue
            value = values.get(metric)
            if value is None or not ok_fn(value, float(limit)):
                unmet.append(f"{metric}={value!r} 未满足 {key}={limit}")
        if acceptance.get("extrapolation_required"):
            surfaces = summary.get("surfaces") or {}
            if not (surfaces.get("C") or {}).get("n_samples"):
                unmet.append("extrapolation_required 但面 C（未见×未来）无样本")
        return unmet

    # ---------------------------------------------------------------- 收尾

    def _evidence(self) -> dict[str, Any]:
        """供 planner 的证据摘要（仅信封级信息，无原始数据）。"""
        return {
            "goal_id": self.goal_id,
            "rounds": [dict(r) for r in self.rounds],
            "best_cvrmse": self.best_cvrmse,
            "no_gain_streak": self._no_gain_streak,
        }

    def _stop(
        self,
        reason: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if reason not in STOP_REASONS:
            raise ValueError(f"未登记的停止原因: {reason!r}（允许 {STOP_REASONS}）")
        return {
            "reason": reason,
            "detail": detail,
            "evidence": dict(evidence or {}),
            "diagnostic": dict(diagnostic) if diagnostic else None,
            "at": _utcnow().isoformat(),
        }

    def _finish(self, stop: Mapping[str, Any]) -> dict[str, Any]:
        """停止收尾：goal 状态机落 STOPPED/PUBLISH 并留痕（§2/§9）。"""
        ledger = self.ctx.ledger
        terminal = ("PUBLISH" if stop["reason"] == STOP_ACCEPTANCE_MET
                    else "STOPPED")
        ledger.transition(
            self.goal_id, terminal,
            reason=f"{stop['reason']}: {stop['detail']}",
            actor=self.ctx.actor,
            error=stop["detail"] if terminal == "STOPPED" else None,
        )
        envelope = make_envelope(
            "tf_orchestration_run",
            ok=stop["reason"] == STOP_ACCEPTANCE_MET,
            id=self.goal_id, status=terminal,
            inputs={"goal_id": self.goal_id, "dataset_ref": self.dataset_ref},
            summary={
                "stop": dict(stop),
                "rounds": self.rounds,
                "best_cvrmse": self.best_cvrmse,
            },
            diagnostics=[stop["diagnostic"]] if stop.get("diagnostic") else [],
        )
        return finalize_envelope(envelope, self.ctx.artifacts_root)
