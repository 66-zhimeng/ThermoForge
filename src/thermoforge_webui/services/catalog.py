"""数据目录：数据集、修订版、变量、画像、时序预览。

界面直接读 Vault 是允许的——**「Agent 不直接读原始数据」约束的是 Agent，
不是人**。使用者要看的就是原始点，经信封反而看不到（信封有 32KB 上限、
采样上限 200 行）。所以这里走 `ctx.vault`，而画像/schema 仍走工具信封，
因为那本来就是为「压缩成能读的摘要」设计的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from thermoforge_research.tools import (
    ToolContext,
    tf_dataset_list,
    tf_dataset_profile,
    tf_dataset_schema,
)

# 时序预览的点数上限：35040 行 × 多条曲线在浏览器里会明显发卡，
# 而看趋势并不需要每个点。超过就等间隔抽稀，并在图上注明。
PREVIEW_MAX_POINTS = 4000


@dataclass(frozen=True)
class RevisionRow:
    dataset_id: str
    revision: str
    ref: str
    rows: int | None
    time_range: tuple[str | None, str | None]
    content_sha256: str
    created_at: str | None


def list_revisions(ctx: ToolContext) -> list[RevisionRow]:
    """所有数据集的所有修订版，新的在前。"""
    envelope = tf_dataset_list(ctx)
    if not envelope.get("ok"):
        return []
    rows: list[RevisionRow] = []
    for dataset in (envelope["summary"].get("datasets") or []):
        for revision in dataset.get("revisions") or []:
            time_range = revision.get("time_range") or [None, None]
            rows.append(RevisionRow(
                dataset_id=str(dataset.get("dataset_id")),
                revision=str(revision.get("revision")),
                ref=str(revision.get("ref")),
                rows=revision.get("rows"),
                time_range=(time_range[0], time_range[1]),
                content_sha256=str(revision.get("content_sha256") or ""),
                created_at=revision.get("created_at"),
            ))
    rows.sort(key=lambda r: (r.dataset_id, r.revision), reverse=True)
    return rows


def dataset_refs(ctx: ToolContext) -> list[str]:
    return [row.ref for row in list_revisions(ctx)]


def schema_of(ctx: ToolContext, ref: str) -> dict[str, Any]:
    envelope = tf_dataset_schema(ctx, ref)
    return envelope.get("summary") or {} if envelope.get("ok") else {}


def profile_of(ctx: ToolContext, ref: str) -> dict[str, Any]:
    envelope = tf_dataset_profile(ctx, ref)
    return envelope.get("summary") or {} if envelope.get("ok") else {}


def variables_frame(schema: dict[str, Any]) -> pd.DataFrame:
    """变量表。`source_kind` 单独一列且排在前面——measured 还是 derived
    直接决定这条变量能不能进候选输入白名单（DD-16）。"""
    rows = []
    for variable in schema.get("variables") or []:
        rows.append({
            "变量": variable.get("variable_id"),
            "来源": ("实测" if variable.get("source_kind") == "measured"
                     else "派生"),
            "对象": variable.get("object_id"),
            "属性": variable.get("property_code"),
            "单位": variable.get("unit"),
            "类型": variable.get("dtype"),
            "角色": variable.get("role"),
            "下限": variable.get("min_value"),
            "上限": variable.get("max_value"),
            "采样周期": variable.get("sample_period"),
        })
    return pd.DataFrame(rows)


def profile_frame(profile: dict[str, Any]) -> pd.DataFrame:
    """画像表：缺失率、分位、离群数。分位点是固定的 7 个（信封约定）。"""
    rows = []
    for variable in profile.get("variables") or []:
        distribution = variable.get("distribution") or {}
        rows.append({
            "变量": variable.get("variable_id"),
            "单位": variable.get("unit"),
            "有效样本": variable.get("count"),
            "缺失率": variable.get("missing_rate"),
            "离群数(IQR)": variable.get("outlier_count_iqr"),
            "恒定": variable.get("constant"),
            **{key: distribution.get(key) for key in
               ("min", "p01", "p25", "p50", "p75", "p99", "max")},
        })
    return pd.DataFrame(rows)


def load_wide(ctx: ToolContext, ref: str) -> pd.DataFrame:
    """整个修订版的宽表（timestamp + 每个 variable_id 一列）。"""
    return ctx.vault.load_data(ref).to_pandas()


@dataclass(frozen=True)
class SeriesPreview:
    frame: pd.DataFrame
    total_rows: int
    stride: int

    @property
    def downsampled(self) -> bool:
        return self.stride > 1


def series_preview(ctx: ToolContext, ref: str, variable_ids: list[str],
                   *, start: str | None = None, end: str | None = None,
                   max_points: int = PREVIEW_MAX_POINTS) -> SeriesPreview:
    """时序预览：选定变量 + 时间窗，超过点数上限就等间隔抽稀。

    抽稀用等间隔而不是聚合平均：这里的用途是「看数据长什么样、有没有
    断档和跳变」，平均会把毛刺抹掉，恰恰藏起了要看的东西。
    """
    frame = load_wide(ctx, ref)
    columns = [col for col in variable_ids if col in frame.columns]
    frame = frame[["timestamp", *columns]] if columns else frame[["timestamp"]]
    if start:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
    total = len(frame)
    stride = max(1, -(-total // max_points)) if total else 1
    if stride > 1:
        frame = frame.iloc[::stride]
    return SeriesPreview(frame=frame.reset_index(drop=True),
                         total_rows=total, stride=stride)


def measured_variable_ids(schema: dict[str, Any]) -> list[str]:
    """只有实测变量能进候选输入白名单（DD-16 的目标级校验）。"""
    return [str(v.get("variable_id")) for v in schema.get("variables") or []
            if v.get("source_kind") == "measured"]


def property_codes(schema: dict[str, Any]) -> list[str]:
    return [str(code) for code in schema.get("property_codes") or []]
