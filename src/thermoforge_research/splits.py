"""数据切分与泄漏检测（implementation-notes.md §4、research-loop.md §6）。

时间切分按**时间边界**而不是按行数（§4.1）：

1. 按 timestamp 排序；
2. 取累积样本数分位（train / train+validate）对应的时间点；
3. 边界**向下对齐**到 resolution 整数倍（Unix epoch 起算）；
4. train = [t0, b1)，validate = [b1, b2)，test = [b2, t_end]。

边界时间点必须记录进实验制品，不能只记录比例（§4.1）。

Purge / Embargo（§4.2）：

- **Purge**：训练集末尾剔除 `max(lag, window)` 时长 → train = [t0, b1 − P)。
- **Embargo**：验证集起始跳过 E（默认 45 min）→ validate = [b1 + E, b2)。

泄漏检测（TFX-903）：训练与评估在时间或设备维度重叠，或
purge/embargo 后的间隙小于特征回看跨度（未 purge 的滚动特征），即报。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .errors import ResearchError

DEFAULT_EMBARGO_SECONDS = 45 * 60  # §4.2 [草案：30–60 min]，默认 45 min


def _to_epoch_seconds(ts: datetime) -> float:
    if ts.tzinfo is None:
        raise ValueError("timestamp 必须带时区（内部表示为 UTC，conventions §3.2）")
    return ts.timestamp()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class TemporalSplitResult:
    """时间切分结果。索引为原始序列中的位置（不是排序后位置）。"""

    train_idx: list[int]
    validate_idx: list[int]
    test_idx: list[int]
    boundaries: dict[str, Any] = field(default_factory=dict)


def temporal_split(
    timestamps: Sequence[datetime],
    resolution_seconds: int,
    train: float = 0.70,
    validate: float = 0.15,
    test: float = 0.15,
    *,
    purge_seconds: float = 0.0,
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
) -> TemporalSplitResult:
    """时间边界切分（§4.1）+ purge/embargo（§4.2）。

    - `resolution_seconds`：边界向下对齐的粒度（数据采样周期）。
    - `purge_seconds`：特征回看跨度 `max(lag, window)`，无滞后特征时传 0。
    - `embargo_seconds`：验证集起始额外跳过，默认 45 min；传 0 关闭。
    """
    if not timestamps:
        raise ValueError("timestamps 不能为空")
    if resolution_seconds <= 0:
        raise ValueError("resolution_seconds 必须为正")
    if abs(train + validate + test - 1.0) > 1e-9:
        raise ValueError("train + validate + test 必须为 1")
    if purge_seconds < 0 or embargo_seconds < 0:
        raise ValueError("purge/embargo 不能为负")

    n = len(timestamps)
    epochs = [_to_epoch_seconds(t) for t in timestamps]
    order = sorted(range(n), key=lambda i: (epochs[i], i))
    sorted_epochs = [epochs[i] for i in order]

    def _floor_boundary(frac: float) -> float:
        k = min(n - 1, int(frac * n))
        raw = sorted_epochs[k]
        # 向下对齐到 resolution 整数倍（§4.1 步骤 3）
        return raw - (raw % resolution_seconds)

    t0 = sorted_epochs[0]
    t_end = sorted_epochs[-1]
    b1 = _floor_boundary(train)
    b2 = _floor_boundary(train + validate)
    if not (t0 < b1 <= b2 <= t_end + resolution_seconds):
        raise ValueError(
            f"切分比例在 resolution={resolution_seconds}s 下无法形成非空区间: "
            f"t0={_iso(t0)} b1={_iso(b1)} b2={_iso(b2)}"
        )

    train_end = b1 - purge_seconds  # purge：训练集末尾剔除回看跨度
    validate_start = b1 + embargo_seconds  # embargo：验证集起始跳过 E

    train_idx: list[int] = []
    validate_idx: list[int] = []
    test_idx: list[int] = []
    for pos, i in enumerate(order):
        t = sorted_epochs[pos]
        if t < train_end:
            train_idx.append(i)
        elif validate_start <= t < b2:
            validate_idx.append(i)
        elif t >= b2:
            test_idx.append(i)
        # [train_end, validate_start) 区间被 purge/embargo 剔除

    boundaries = {
        "t0": _iso(t0),
        "b1": _iso(b1),
        "b2": _iso(b2),
        "t_end": _iso(t_end),
        "train_range": [_iso(t0), _iso(train_end)],
        "validate_range": [_iso(validate_start), _iso(b2)],
        "test_range": [_iso(b2), _iso(t_end)],
        "fractions": {"train": train, "validate": validate, "test": test},
        "resolution_seconds": resolution_seconds,
        "purge_seconds": purge_seconds,
        "embargo_seconds": embargo_seconds,
        "counts": {
            "train": len(train_idx),
            "validate": len(validate_idx),
            "test": len(test_idx),
            "purged_or_embargoed": n - len(train_idx) - len(validate_idx) - len(test_idx),
        },
    }
    return TemporalSplitResult(train_idx, validate_idx, test_idx, boundaries)


def check_no_leakage(
    timestamps: Sequence[datetime],
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    *,
    min_gap_seconds: float = 0.0,
    object_ids: Sequence[str] | None = None,
    eval_label: str = "评估集",
) -> None:
    """泄漏检测：任一违规报 TFX-903（§4.2、conventions §7.9）。

    - 时间重叠：max(train) >= min(eval)。
    - 间隙不足：min(eval) − max(train) < `min_gap_seconds`
      （特征回看跨度；含 lag/滚动特征而未 purge 时在此被捕获）。
    - 设备重叠（`object_ids` 提供时）：留一设备验证下两侧设备集合相交。
    """
    if not train_idx or not eval_idx:
        return
    epochs = [_to_epoch_seconds(t) for t in timestamps]
    train_max = max(epochs[i] for i in train_idx)
    eval_min = min(epochs[i] for i in eval_idx)
    if train_max >= eval_min:
        raise ResearchError(
            "TFX-903",
            f"训练集与{eval_label}时间重叠: train_max={_iso(train_max)} "
            f">= eval_min={_iso(eval_min)}",
        )
    gap = eval_min - train_max
    if gap < min_gap_seconds:
        raise ResearchError(
            "TFX-903",
            f"训练/{eval_label}间隙 {gap:.0f}s 小于特征回看跨度 "
            f"{min_gap_seconds:.0f}s：含滞后/滚动特征时必须 purge（§4.2）",
        )
    if object_ids is not None:
        shared = {object_ids[i] for i in train_idx} & {object_ids[i] for i in eval_idx}
        if shared:
            raise ResearchError(
                "TFX-903",
                f"留一设备验证下训练集与{eval_label}设备重叠: {sorted(shared)}",
            )


def equipment_holdout_split(
    object_ids: Sequence[str], holdout_objects: Sequence[str]
) -> tuple[list[int], list[int]]:
    """留一设备切分（research-loop §6）：返回 (训练索引, 留出索引)。"""
    holdout = set(holdout_objects)
    unknown = holdout - set(object_ids)
    if unknown:
        raise ValueError(f"holdout 对象不在数据中: {sorted(unknown)}")
    train_idx = [i for i, o in enumerate(object_ids) if o not in holdout]
    test_idx = [i for i, o in enumerate(object_ids) if o in holdout]
    return train_idx, test_idx


def build_test_surfaces(
    timestamps: Sequence[datetime],
    object_ids: Sequence[str],
    split: TemporalSplitResult,
    holdout_objects: Sequence[str] = (),
) -> dict[str, list[int]]:
    """三维测试面（§4.3）：A 已见设备×未来时间 / B 未见设备×已见时间 / C 未见×未来。

    未启用设备留出时只有面 A（= test 集）。面 B 的「已见时间」取
    train+validate 的时间范围（留出设备在该时段的数据）。
    """
    holdout = set(holdout_objects)
    surfaces: dict[str, list[int]] = {"A": list(split.test_idx)}
    if holdout:
        seen_time_max = max(
            (_to_epoch_seconds(timestamps[i])
             for i in split.train_idx + split.validate_idx),
            default=None,
        )
        test_min = min(
            (_to_epoch_seconds(timestamps[i]) for i in split.test_idx),
            default=None,
        )
        epochs = [_to_epoch_seconds(t) for t in timestamps]
        surfaces["B"] = [
            i for i, o in enumerate(object_ids)
            if o in holdout and seen_time_max is not None and epochs[i] <= seen_time_max
        ]
        surfaces["C"] = [
            i for i, o in enumerate(object_ids)
            if o in holdout and test_min is not None and epochs[i] >= test_min
        ]
    return surfaces
