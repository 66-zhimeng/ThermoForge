"""实验工件读取：报告、指标、切分、物理检查、预测点。

只读，不写任何东西——已完成的实验是不可变工件（TFX-904）。

关于 R²：I-54 之前跑的实验，`metrics.json` 里没有 R²。这里**不改写历史
产物**，而是在读的时候用 `predictions.parquet` 调 `metrics.r2()`（与实验
内部同一个实现）现算补齐，并标记 `r2_backfilled=True`，界面上如实标注
「由预测点现算」，不让人误以为是当时记录的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from thermoforge_research.metrics import r2

from ..context import experiments_root

# 主测试面优先级：C（最终测试面）> A > validate。与 cli/status.py 同调。
SURFACE_PRIORITY = ("C", "A", "validate")
# 建模路线类别（契约取值 → 中文）
CATEGORY_LABELS = {"physics": "物理", "data": "数据", "hybrid": "混合"}

# 实验状态：产物里存英文，界面与报告一律显示中文
STATUS_LABELS = {
    "completed": "已完成",
    "failed": "失败",
    "running": "运行中",
    "planned": "已计划",
}

SURFACE_LABELS = {
    "train": "训练面",
    "validate": "验证面",
    "A": "测试面 A（时间外推）",
    "B": "测试面 B（设备留一）",
    "C": "测试面 C（最终）",
}


@dataclass(frozen=True)
class ExperimentSummary:
    """实验清单里的一行。指标取主测试面。"""

    experiment_id: str
    status: str
    goal_id: str | None
    hypothesis_id: str | None
    dataset_view: str | None
    model_label: str
    started_at: str | None
    duration_seconds: float | None
    surface: str | None
    metrics: dict[str, float | None]
    physics_rate: float | None
    error_code: str | None
    failure_reason: str | None

    def metric(self, name: str) -> float | None:
        return self.metrics.get(name)


@dataclass(frozen=True)
class ExperimentDetail:
    """单个实验的全部工件（预测点按需加载，不在这里）。"""

    experiment_id: str
    report: dict[str, Any]
    spec: dict[str, Any]
    split: dict[str, Any]
    split_profile: dict[str, Any]
    physics: dict[str, Any]
    environment: dict[str, Any]
    directory: Path
    r2_backfilled: bool = False
    surfaces: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.report.get("status") or "unknown")

    @property
    def model_label(self) -> str:
        return model_label(self.spec)

    @property
    def target(self) -> str:
        return str((self.spec.get("experiment") or {}).get("target") or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fp:
            doc = json.load(fp)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def model_label(spec: dict[str, Any]) -> str:
    """把 ModelSpec 压成一行人能读的标签，如「混合 · cooling_balance_v2 + xgboost 残差」。

    类别翻成中文，但方程版本、estimator 名保留原样——那些是契约取值和
    算法专名，翻译反而对不上文档和代码。
    """
    model = ((spec.get("experiment") or {}).get("model") or {})
    category = str(model.get("category") or "?")
    parts = [CATEGORY_LABELS.get(category, category)]
    if model.get("physics"):
        parts.append(str(model["physics"]))
    if model.get("estimator"):
        parts.append(str(model["estimator"]))
    if model.get("residual"):
        parts.append(f"+ {model['residual']} 残差")
    return " · ".join(parts)


def primary_surface(report: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """主测试面：C > A > validate，取第一个有样本的。"""
    surfaces = (report.get("metrics") or {}).get("surfaces") or {}
    for name in SURFACE_PRIORITY:
        surface = surfaces.get(name) or {}
        if surface.get("n_samples"):
            return name, surface
    return None, {}


def list_experiments(limit: int | None = None) -> list[ExperimentSummary]:
    """所有实验，按开始时间倒序。目录里没有 report.json 的（跑挂了）跳过。"""
    root = experiments_root()
    if not root.is_dir():
        return []
    rows: list[ExperimentSummary] = []
    for report_path in root.glob("EXP-*/report.json"):
        report = _read_json(report_path)
        if not report:
            continue
        spec = _read_json(report_path.parent / "spec.json")
        name, surface = primary_surface(report)
        rows.append(ExperimentSummary(
            experiment_id=str(report.get("experiment_id")
                              or report_path.parent.name),
            status=str(report.get("status") or "unknown"),
            goal_id=report.get("goal_id"),
            hypothesis_id=report.get("hypothesis_id"),
            dataset_view=report.get("dataset_view"),
            model_label=model_label(spec),
            started_at=report.get("started_at"),
            duration_seconds=report.get("duration_seconds"),
            surface=name,
            metrics=dict(surface.get("metrics") or {}),
            physics_rate=(report.get("physics") or {}).get("overall_rate"),
            error_code=report.get("error_code"),
            failure_reason=report.get("failure_reason"),
        ))
    rows.sort(key=lambda r: str(r.started_at or ""), reverse=True)
    return rows[:limit] if limit else rows


def load_detail(experiment_id: str) -> ExperimentDetail | None:
    """单个实验的全部工件；顺带把缺失的 R² 从预测点补齐。"""
    directory = experiments_root() / experiment_id
    report = _read_json(directory / "report.json")
    if not report:
        return None
    surfaces = dict((report.get("metrics") or {}).get("surfaces") or {})
    backfilled = _backfill_r2(directory, surfaces)
    return ExperimentDetail(
        experiment_id=experiment_id,
        report=report,
        spec=_read_json(directory / "spec.json"),
        split=_read_json(directory / "split.json"),
        split_profile=_read_json(directory / "split_profile.json"),
        physics=_read_json(directory / "physics_report.json"),
        environment=_read_json(directory / "environment.json"),
        directory=directory,
        r2_backfilled=backfilled,
        surfaces=surfaces,
    )


def _backfill_r2(directory: Path, surfaces: dict[str, Any]) -> bool:
    """给缺 R² 的面用预测点现算（I-54）。原地改的是内存副本，不写盘。"""
    missing = [name for name, surface in surfaces.items()
               if isinstance(surface, dict)
               and "R2" not in (surface.get("metrics") or {})]
    if not missing:
        return False
    frame = load_predictions(directory.name)
    if frame is None or frame.empty:
        return False
    filled = False
    for name in missing:
        subset = frame[frame["surface"] == name]
        if subset.empty:
            continue
        value = r2(subset["y_true"].to_numpy(dtype=float),
                   subset["y_pred"].to_numpy(dtype=float))
        surfaces[name] = {**surfaces[name],
                          "metrics": {**(surfaces[name].get("metrics") or {}),
                                      "R2": value}}
        filled = True
    return filled


def load_predictions(experiment_id: str) -> pd.DataFrame | None:
    """预测点（surface / object_id / timestamp / y_true / y_pred）。"""
    path = experiments_root() / experiment_id / "predictions.parquet"
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def surface_options(detail: ExperimentDetail) -> list[str]:
    """该实验有哪些面可看，按 train → validate → A/B/C 排。"""
    order = ["train", "validate", "A", "B", "C"]
    present = [name for name in order if name in detail.surfaces]
    extra = sorted(set(detail.surfaces) - set(present))
    return present + extra


def comparison_frame(summaries: list[ExperimentSummary]) -> pd.DataFrame:
    """多实验对比表：一行一个实验，列是指标。"""
    rows = []
    for item in summaries:
        rows.append({
            "实验": item.experiment_id,
            "模型": item.model_label,
            "面": SURFACE_LABELS.get(item.surface or "", item.surface or "—"),
            "CVRMSE": item.metric("CVRMSE"),
            "R2": item.metric("R2"),
            "NMBE": item.metric("NMBE"),
            "MAPE": item.metric("MAPE"),
            "RMSE": item.metric("RMSE"),
            "MAE": item.metric("MAE"),
            "物理违规率": item.physics_rate,
            "状态": STATUS_LABELS.get(item.status, item.status),
            "假设": item.hypothesis_id,
        })
    return pd.DataFrame(rows)


def enrich_with_r2(summaries: list[ExperimentSummary]) -> list[ExperimentSummary]:
    """给清单补 R²（逐个读预测点，代价随实验数线性增长，只在对比页用）。"""
    out: list[ExperimentSummary] = []
    for item in summaries:
        if "R2" in item.metrics or item.surface is None:
            out.append(item)
            continue
        detail = load_detail(item.experiment_id)
        metrics = dict(item.metrics)
        if detail is not None:
            surface = detail.surfaces.get(item.surface) or {}
            value = (surface.get("metrics") or {}).get("R2")
            if value is not None:
                metrics["R2"] = value
        out.append(replace(item, metrics=metrics))
    return out
