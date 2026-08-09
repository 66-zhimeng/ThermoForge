"""Dataset View（data-contract.md §7、conventions.md §3.4/§5.1）。

- View 定义以 YAML 表达；`view_hash = sha256(dataset_revision_id + "\\n"
  + canonical_json(view))`，集合语义字段 `objects`/`features` 哈希前排序。
- 物化缓存目录名为 view_hash 前 16 位，目录内存放完整定义与完整哈希，
  加载时重新校验（TFV-801）；相同 revision + 定义复用同一份 Parquet。
- 多实例转长表：`object_id` 为分组键，列名为 `property_code`。
- 重采样：左闭右开、标签取左边界、min_count 规则、聚合白名单
  （mean/sum/min/max/first/last/median）。窗口聚合是纯函数
  `resample_window`，离线/在线共用（implementation-notes §3.2）。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from thermoforge_core.canonical import canonical_json, sha256_hex
from thermoforge_core.errors import Level
from thermoforge_core.naming import is_property_code
from thermoforge_core.timeutil import parse_time_resolution

from .vault import DataVault, VaultError

AGGREGATIONS = ("mean", "sum", "min", "max", "first", "last", "median")

# conventions.md §5.1 规则 6：View 定义中声明为集合语义的字段
VIEW_SET_FIELDS = frozenset({"objects", "features"})

_US = 1_000_000


# ---------------------------------------------------------------- 纯函数：窗口聚合


def resample_window(
    timestamps_us: Sequence[int],
    values: Sequence[Any],
    window_us: int,
    min_count: int,
    aggregation: str,
) -> tuple[list[int], list[Any]]:
    """窗口聚合纯函数（离线/在线共用，implementation-notes §3.2）。

    - 左闭右开，标签取左边界：桶 `[k·window, (k+1)·window)` 的标签为 `k·window`，
      桶对齐 Unix  epoch 的整数倍。
    - 聚合只统计非 null 样本；非 null 样本数 < `min_count` 时输出 null。
    - `aggregation` 必须在白名单内（conventions §3.4）。
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"聚合方式不在白名单 {AGGREGATIONS}: {aggregation!r}"
        )
    if len(timestamps_us) != len(values):
        raise ValueError("timestamps 与 values 长度不一致")
    if window_us <= 0:
        raise ValueError("window_us 必须为正")
    buckets: dict[int, list[float]] = {}
    first_seen: dict[int, tuple[int, float]] = {}
    last_seen: dict[int, tuple[int, float]] = {}
    for i, (t, v) in enumerate(zip(timestamps_us, values)):
        if v is None:
            continue
        key = (t // window_us) * window_us
        fv = float(v)
        buckets.setdefault(key, []).append(fv)
        if key not in first_seen:
            first_seen[key] = (i, fv)
        last_seen[key] = (i, fv)
    out_ts: list[int] = []
    out_vals: list[Any] = []
    for key in sorted(buckets):
        out_ts.append(key)
        vals = buckets[key]
        if len(vals) < min_count:
            out_vals.append(None)
            continue
        if aggregation == "mean":
            out_vals.append(sum(vals) / len(vals))
        elif aggregation == "sum":
            out_vals.append(sum(vals))
        elif aggregation == "min":
            out_vals.append(min(vals))
        elif aggregation == "max":
            out_vals.append(max(vals))
        elif aggregation == "first":
            out_vals.append(first_seen[key][1])
        elif aggregation == "last":
            out_vals.append(last_seen[key][1])
        else:  # median
            s = sorted(vals)
            mid = len(s) // 2
            out_vals.append(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2)
    return out_ts, out_vals


# ---------------------------------------------------------------- View 定义与哈希


def load_view_definition(path: str | Path) -> dict[str, Any]:
    """读取 YAML View 定义。"""
    with open(path, encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    if not isinstance(doc, dict):
        raise ValueError("View 定义必须是 YAML mapping")
    validate_view_definition(doc)
    return doc


def validate_view_definition(doc: Mapping[str, Any]) -> None:
    if "dataset" not in doc or "@" not in str(doc["dataset"]):
        raise ValueError("View 必须引用完整版本 dataset@rev_NNNN（data-contract §6）")
    for key in ("features",):
        if key not in doc or not doc[key]:
            raise ValueError(f"View 缺少必需字段: {key}")
        for feat in doc[key]:
            if not is_property_code(str(feat)):
                raise ValueError(f"非法 feature property_code: {feat!r}")
    if "resolution" in doc and doc["resolution"] is not None:
        parse_time_resolution(str(doc["resolution"]))  # 非法即抛
    if "aggregation" in doc and doc["aggregation"] is not None:
        if doc["aggregation"] not in AGGREGATIONS:
            raise ValueError(f"聚合方式不在白名单: {doc['aggregation']!r}")


def view_hash(dataset_revision: str, view_definition: Mapping[str, Any]) -> str:
    """`sha256(dataset_revision_id + "\\n" + canonical_json(view))`（§5.1）。"""
    return sha256_hex(
        dataset_revision + "\n" + canonical_json(dict(view_definition),
                                                 VIEW_SET_FIELDS)
    )


# ---------------------------------------------------------------- 物化


@dataclass(frozen=True)
class MaterializedView:
    view_hash: str
    path: Path
    table: pa.Table
    reused: bool  # True = 命中缓存复用


def materialize_view(
    vault: DataVault,
    view_definition: Mapping[str, Any],
    cache_root: str | Path,
) -> MaterializedView:
    """物化 Dataset View 为长表 Parquet 缓存。

    缓存目录 `<cache_root>/<view_hash 前 16 位>/`，内含完整定义
    （`view.json`，含完整哈希）与 `data.parquet`。
    """
    doc = dict(view_definition)
    validate_view_definition(doc)
    ref = str(doc["dataset"])
    digest = view_hash(ref, doc)
    cache_dir = Path(cache_root) / digest[:16]
    meta_path = cache_dir / "view.json"
    data_path = cache_dir / "data.parquet"

    if cache_dir.exists():
        return _load_cached(cache_dir, meta_path, data_path, digest, ref, doc)

    table = _build_long_table(vault, ref, doc)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, data_path)
    meta = {
        "view_hash": digest,
        "dataset": ref,
        "definition": json.loads(canonical_json(doc, VIEW_SET_FIELDS)),
    }
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(canonical_json(meta))
        fp.write("\n")
    os.replace(tmp, meta_path)
    return MaterializedView(digest, cache_dir, table, reused=False)


def _load_cached(
    cache_dir: Path,
    meta_path: Path,
    data_path: Path,
    digest: str,
    ref: str,
    doc: Mapping[str, Any],
) -> MaterializedView:
    """加载缓存并重校验完整哈希（TFV-801，implementation-notes §3.1）。"""
    if not meta_path.exists() or not data_path.exists():
        raise VaultError("TFV-801", f"View 缓存不完整: {cache_dir}")
    with open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)
    stored_hash = meta.get("view_hash")
    recomputed = view_hash(str(meta.get("dataset")), meta.get("definition", {}))
    if stored_hash != digest or recomputed != digest or str(meta.get("dataset")) != ref:
        raise VaultError(
            "TFV-801",
            f"缓存物化结果与 View 定义哈希不符: {cache_dir}",
        )
    return MaterializedView(digest, cache_dir, pq.read_table(data_path), reused=True)


def _build_long_table(
    vault: DataVault, ref: str, doc: Mapping[str, Any]
) -> pa.Table:
    data = vault.load_data(ref)
    variables = {r["variable_id"]: r for r in vault.load_variables(ref)}
    objects = vault.load_objects(ref)

    scope_model = (doc.get("scope") or {}).get("object_model")
    if scope_model:
        candidates = [o["object_id"] for o in objects
                      if o["object_model_id"] == scope_model]
    else:
        candidates = [o["object_id"] for o in objects]
    requested = [str(o) for o in doc.get("objects") or candidates]
    object_ids = [o for o in requested if o in candidates]
    if not object_ids:
        raise VaultError("TFV-803", f"View 范围内无对象: {doc.get('objects')!r}")

    features = [str(f) for f in doc["features"]]
    target = str(doc["target"]) if doc.get("target") else None
    needed = features + ([target] if target else [])
    filters = {str(k): v for k, v in (doc.get("filter") or {}).items()}
    needed_with_filter = sorted(set(needed) | set(filters))

    # TFV-802：特征/过滤字段在该数据版本中不存在
    for obj in object_ids:
        for prop in needed_with_filter:
            if f"{obj}.{prop}" not in variables:
                raise VaultError(
                    "TFV-802",
                    f"View 请求的变量在该数据版本中不存在: {obj}.{prop}",
                )

    resolution = str(doc["resolution"]) if doc.get("resolution") else None
    default_agg = str(doc.get("aggregation") or "mean")
    base_ts = data.column("timestamp").to_pylist()
    base_ts_us = [int(t.timestamp() * _US) for t in base_ts]

    out_object: list[str] = []
    out_ts_us: list[int] = []
    out_cols: dict[str, list[Any]] = {p: [] for p in needed}

    for obj in object_ids:
        # 过滤（如 status_run == true）
        keep_mask = [True] * len(base_ts)
        for prop, expected in filters.items():
            col = data.column(f"{obj}.{prop}").to_pylist()
            for i, v in enumerate(col):
                if v is None or v != expected:
                    keep_mask[i] = False
        obj_ts = [t for t, k in zip(base_ts_us, keep_mask) if k]

        series: dict[str, dict[int, Any]] = {}
        axis: set[int] = set()
        for prop in needed:
            col = data.column(f"{obj}.{prop}").to_pylist()
            vals = [v for v, k in zip(col, keep_mask) if k]
            if resolution:
                window_us = parse_time_resolution(resolution) * _US
                agg = variables[f"{obj}.{prop}"].get("aggregation") or default_agg
                min_count = _min_count(doc, base_ts_us, window_us)
                ts_out, vals_out = resample_window(
                    obj_ts, vals, window_us, min_count, agg
                )
                series[prop] = dict(zip(ts_out, vals_out))
                axis.update(ts_out)
            else:
                series[prop] = dict(zip(obj_ts, vals))
                axis.update(obj_ts)
        obj_axis = sorted(axis)
        out_object.extend([obj] * len(obj_axis))
        out_ts_us.extend(obj_axis)
        for prop in needed:
            out_cols[prop].extend(series[prop].get(t) for t in obj_axis)

    if not out_object:
        raise VaultError("TFV-803", "View 过滤后无样本")

    from datetime import datetime, timezone

    arrays = {
        "object_id": pa.array(out_object, type=pa.string()),
        "timestamp": pa.array(
            [datetime.fromtimestamp(t / _US, tz=timezone.utc) for t in out_ts_us],
            type=pa.timestamp("us", tz="UTC"),
        ),
    }
    for prop in needed:
        dtype = variables[f"{object_ids[0]}.{prop}"]["dtype"]
        arrays[prop] = pa.array(out_cols[prop], type=_arrow_type(dtype))
    return pa.table(arrays)


def _min_count(doc: Mapping[str, Any], base_ts_us: Sequence[int],
               window_us: int) -> int:
    """桶内最小有效样本数（§3.4 [草案：期望样本数的 50%]）。"""
    if doc.get("min_count") is not None:
        return int(doc["min_count"])
    if len(base_ts_us) > 1:
        base = min(b - a for a, b in zip(base_ts_us, base_ts_us[1:]) if b > a)
        expected = max(1, window_us // base)
    else:
        expected = 1
    return max(1, math.ceil(expected * 0.5))


def _arrow_type(dtype: str) -> pa.DataType:
    return {
        "float": pa.float64(),
        "integer": pa.int64(),
        "boolean": pa.bool_(),
        "string": pa.string(),
    }[dtype]
