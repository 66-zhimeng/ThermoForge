"""数据画像与质量报告（data-contract.md §8）。

输出：时间范围/记录数/采样间隔及漂移、缺失率、重复/乱序/空洞、
每变量 min/max/分位数/异常值、范围违规、单位与派生一致性问题、
设备/变量/工况覆盖、指纹与版本。指纹字段由 vault 落盘时补全。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pyarrow as pa

from thermoforge_core.contracts.tfdc import TfdcDataset, VariableRecord
from thermoforge_core.errors import Diagnostic
from thermoforge_core.timeutil import parse_time_resolution

# 分位数点位固定（implementation-notes §11）
QUANTILE_POINTS = (0.01, 0.25, 0.5, 0.75, 0.99)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _variable_profile(var: VariableRecord, values: Sequence[Any]) -> dict[str, Any]:
    n = len(values)
    non_null = [v for v in values if v is not None]
    entry: dict[str, Any] = {
        "variable_id": var.variable_id,
        "unit": var.unit,
        "dtype": var.dtype,
        "role": var.role,
        "source_kind": var.source_kind,
        "count": len(non_null),
        "missing": n - len(non_null),
        "missing_rate": (n - len(non_null)) / n if n else 0.0,
    }
    if var.dtype in ("float", "integer") and non_null:
        nums = sorted(float(v) for v in non_null)
        q1, q3 = _quantile(nums, 0.25), _quantile(nums, 0.75)
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = sum(1 for v in nums if v < lo_fence or v > hi_fence)
        entry.update({
            "min": nums[0],
            "max": nums[-1],
            "quantiles": {
                f"p{int(q * 100):02d}": _quantile(nums, q) for q in QUANTILE_POINTS
            },
            "outlier_count_iqr": outliers,
            "constant": nums[0] == nums[-1],
        })
    elif var.dtype == "boolean" and non_null:
        true_ratio = sum(1 for v in non_null if v) / len(non_null)
        entry["true_ratio"] = true_ratio
        entry["constant"] = all(v == non_null[0] for v in non_null)
    return entry


def build_profile(
    dataset: TfdcDataset,
    table: pa.Table,
    diagnostics: Sequence[Diagnostic],
    *,
    fingerprints: Mapping[str, str] | None = None,
    importer_version: str = "",
    degradations: Sequence[str] = (),
) -> dict[str, Any]:
    """按 data-contract.md §8 生成质量报告（JSON 可序列化 dict）。"""
    ts = table.column("timestamp").to_pylist()
    manifest = dataset.manifest
    resolution_s = parse_time_resolution(manifest.time_resolution)
    diffs = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]

    diag_counts: dict[str, int] = {}
    for d in diagnostics:
        diag_counts[d.code] = diag_counts.get(d.code, 0) + d.count

    variables = {v.variable_id: v for v in dataset.variables}
    var_profiles = [
        _variable_profile(variables[name], table.column(name).to_pylist())
        for name in table.column_names
        if name != "timestamp" and name in variables
    ]

    # 工况覆盖：布尔运行状态变量的 true 比例
    operating: dict[str, float] = {}
    for v in dataset.variables:
        if v.dtype == "boolean" and v.property_code.startswith("status"):
            if v.variable_id in table.column_names:
                vals = [x for x in table.column(v.variable_id).to_pylist()
                        if x is not None]
                if vals:
                    operating[v.variable_id] = sum(vals) / len(vals)

    report: dict[str, Any] = {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "contract_version": manifest.contract_version,
        "importer_version": importer_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "time_range": [
            ts[0].isoformat() if ts else None,
            ts[-1].isoformat() if ts else None,
        ],
        "record_count": len(ts),
        "declared_resolution": manifest.time_resolution,
        "resolution_seconds": resolution_s,
        "interval_stats": {
            "min_s": min(diffs) if diffs else None,
            "median_s": sorted(diffs)[len(diffs) // 2] if diffs else None,
            "max_s": max(diffs) if diffs else None,
            "off_resolution_fraction": (
                sum(1 for d in diffs if d != resolution_s) / len(diffs)
                if diffs else 0.0
            ),
        },
        "duplicated_timestamps": diag_counts.get("TFDC-503", 0),
        "out_of_order_timestamps": diag_counts.get("TFDC-504", 0),
        "gap_count": diag_counts.get("TFDC-505", 0),
        "range_violations": diag_counts.get("TFDC-601", 0),
        "dtype_rejections": diag_counts.get("TFDC-404", 0),
        "derived_inconsistencies": diag_counts.get("TFDC-604", 0),
        "variables": var_profiles,
        "coverage": {
            "object_count": len(dataset.objects),
            "variable_count": len(dataset.variables),
            "objects_by_model": _count_by_model(dataset),
            "operating_true_ratio": operating,
        },
        "degradations": list(degradations),
        "diagnostics_summary": [
            {
                "code": d.code,
                "level": d.level.value,
                "location": d.location,
                "count": d.count,
                "message": d.message,
            }
            for d in diagnostics
        ],
        "fingerprints": dict(fingerprints or {}),
    }
    return report


def _count_by_model(dataset: TfdcDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in dataset.objects:
        counts[o.object_model_id] = counts.get(o.object_model_id, 0) + 1
    return counts
