"""在线推理运行时（model-package.md §6、implementation-notes.md §9）。

- 线上数据契约延续 TFDC `variable_id`（§6）：输入消息形如
  ``{"timestamp": ..., "values": {"CH-01.evap_chw_flow": 521.4, ...}}``，
  由 `DeploymentBinding` 解析为签名 `property_code`；模型内部不依赖现场地址。
- 超范围策略按**每个输入**声明（§9.2）：reject / clamp /
  passthrough_with_flag；越界事件计数并可通过 `on_range_event` 回流
  Research Ledger。
- 历史依赖（§9.1）：签名可声明 `history_required` 与 `cold_start`；
  历史缓冲不足时返回明确的 ``not_ready``，绝不用部分数据凑预测。
- 时间与幂等（§9.4）：按 timestamp 组织输入；重复时间戳幂等（相同值
  直接重放首次结果，冲突值丢弃并计数）；超出迟到窗口的消息丢弃并计数。
- 延迟口径（§9.3）：`measure_latency` 固定 batch=1，默认预热 100 次、
  测 1000 次，报告 p50/p95/p99。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from thermoforge_core.timeutil import parse_time_resolution, parse_timestamp

from .artifact import load_model_artifact
from .binding import DeploymentBinding
from .package import load_constraints, load_signature, verify_package

LATENCY_WARMUP = 100  # §9.3 [草案]
LATENCY_SAMPLES = 1000


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def measure_latency(
    package_dir: str | Path,
    *,
    warmup: int = LATENCY_WARMUP,
    samples: int = LATENCY_SAMPLES,
) -> dict[str, Any]:
    """batch=1 延迟测量（§9.3）：先预热 `warmup` 次丢弃，再测 `samples` 次。"""
    package_dir = Path(package_dir)
    model = load_model_artifact(package_dir / "artifact")
    golden = pd.read_parquet(package_dir / "golden.parquet")
    input_cols = [c for c in golden.columns if not c.startswith("output__")]
    row = golden[input_cols].iloc[[0]]
    for _ in range(int(warmup)):
        model.predict(row)
    timings: list[float] = []
    for _ in range(int(samples)):
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return {
        "batch": 1,
        "warmup": int(warmup),
        "samples": int(samples),
        "p50_ms": _percentile(timings, 0.50),
        "p95_ms": _percentile(timings, 0.95),
        "p99_ms": _percentile(timings, 0.99),
        "max_ms": timings[-1],
        "min_ms": timings[0],
    }


class InferenceSession:
    """一个已部署模型包的推理会话。

    用法::

        session = InferenceSession(pkg, binding=DeploymentBinding("CH-01"))
        result = session.handle({
            "contract": "TFDC", "version": "1.0",
            "timestamp": "2026-08-08T12:00:00+08:00",
            "values": {"CH-01.evap_chw_flow": 521.4, ...},
        })
    """

    def __init__(
        self,
        package_dir: str | Path,
        *,
        binding: DeploymentBinding | None = None,
        on_range_event: Callable[[dict[str, Any]], None] | None = None,
        late_tolerance_seconds: float | None = None,
        verify: bool = True,
    ):
        self.package_dir = Path(package_dir)
        if verify:
            verify_package(self.package_dir)  # TFM-1002
        self.signature, self.history_required, self.cold_start = load_signature(
            self.package_dir
        )
        if self.history_required and "window" not in self.history_required:
            raise ValueError("history_required 必须声明 window（§9.1）")
        self.constraints = load_constraints(self.package_dir)
        self.model = load_model_artifact(self.package_dir / "artifact")
        self.binding = binding
        self.on_range_event = on_range_event
        self.late_tolerance_seconds = late_tolerance_seconds

        self._constraint_by_prop = {
            c.property_code: c for c in self.constraints.inputs
        }
        self._feature_order = [p.property_code for p in self.signature.inputs]
        self._required = {
            p.property_code for p in self.signature.inputs if p.required
        }
        # 历史缓冲与幂等状态（§9.4）
        self._buffer: dict[float, dict[str, Any]] = {}  # epoch → 原始已解析输入
        self._results: dict[str, dict[str, Any]] = {}  # ts_iso → 首次结果
        self._latest_epoch: float | None = None
        # 计数器（越界/迟到/重复，回流 Ledger 的漂移证据，§9.2）
        self.range_event_counts: dict[str, int] = {}
        self.late_dropped = 0
        self.duplicate_timestamps = 0
        self.duplicate_conflicts = 0

    # ---------------------------------------------------------------- 主入口

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """处理一条线上消息（model-package §6 格式），返回结构化结果。"""
        ts = parse_timestamp(str(message["timestamp"]))
        ts_iso = ts.isoformat()
        epoch = ts.timestamp()
        resolved = self._resolve(dict(message.get("values") or {}))

        # 重复时间戳幂等（§9.4）：相同值重放首次结果，冲突值丢弃并计数
        if ts_iso in self._results:
            self.duplicate_timestamps += 1
            replay = dict(self._results[ts_iso])
            replay["flags"] = dict(replay.get("flags") or {})
            if resolved != self._buffer[epoch]:
                self.duplicate_conflicts += 1
                replay["flags"]["duplicate_conflict"] = True
            else:
                replay["flags"]["idempotent_replay"] = True
            return replay

        # 迟到窗口：超出则丢弃并计数（§9.4）
        if (
            self.late_tolerance_seconds is not None
            and self._latest_epoch is not None
            and epoch < self._latest_epoch - self.late_tolerance_seconds
        ):
            self.late_dropped += 1
            return {"status": "dropped", "reason": "late_arrival",
                    "timestamp": ts_iso}

        missing = [p for p in self._required if resolved.get(p) is None]
        if missing:
            return self._remember(ts_iso, epoch, resolved, {
                "status": "rejected", "reason": "missing_inputs",
                "missing": missing, "timestamp": ts_iso,
            })

        # 超范围策略（§9.2）：reject 直接返回；clamp/passthrough 置 flags
        props = dict(resolved)
        reject, flags = self._apply_range_policies(props)
        if reject is not None:
            return self._remember(ts_iso, epoch, resolved,
                                  {**reject, "timestamp": ts_iso})

        # 历史依赖（§9.1）：缓冲不足 → not_ready，不凑数
        if self.history_required and not self._history_satisfied(epoch):
            if str(self.cold_start or "reject") != "degrade_to_steady_state":
                return self._remember(ts_iso, epoch, resolved, {
                    "status": "not_ready", "reason": "insufficient_history",
                    "have_samples": self._history_count(epoch),
                    "need_samples": int(self.history_required["min_samples"]),
                    "timestamp": ts_iso,
                })
            flags["degraded"] = True  # degrade_to_steady_state：显式标记降级

        row = pd.DataFrame([{f: _as_float(props.get(f))
                             for f in self._feature_order}])
        start = time.perf_counter()
        preds = np.asarray(self.model.predict(row), dtype=np.float64)
        latency_ms = (time.perf_counter() - start) * 1000.0

        outputs = {
            port.property_code: float(preds[i])
            for i, port in enumerate(self.signature.outputs)
        }
        lo = self.constraints.output_min_value
        hi = self.constraints.output_max_value
        for name, value in outputs.items():
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                flags["output_out_of_range"] = True
                self._count_range_event(name, value, "output_range")

        return self._remember(ts_iso, epoch, resolved, {
            "status": "ok",
            "timestamp": ts_iso,
            "predictions": outputs,
            "flags": flags,
            "latency_ms": latency_ms,
        })

    # ---------------------------------------------------------------- 内部

    def _resolve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """variable_id → property_code（binding）；无 binding 时直通。"""
        if self.binding is None:
            return dict(values)
        return self.binding.resolve(values)

    def _remember(
        self,
        ts_iso: str,
        epoch: float,
        resolved: Mapping[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        self._buffer[epoch] = dict(resolved)
        self._results[ts_iso] = result
        if self._latest_epoch is None or epoch > self._latest_epoch:
            self._latest_epoch = epoch
        return result

    def _count_range_event(self, prop: str, value: Any, action: str) -> None:
        self.range_event_counts[prop] = self.range_event_counts.get(prop, 0) + 1
        if self.on_range_event is not None:
            self.on_range_event({
                "property_code": prop, "value": value, "action": action,
            })

    def _apply_range_policies(
        self, props: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """逐输入应用超范围策略（§9.2）。返回 (reject 结果 | None, flags)。"""
        flags: dict[str, Any] = {}
        for prop, constraint in self._constraint_by_prop.items():
            value = props.get(prop)
            if value is None:
                continue
            v = _as_float(value)
            lo, hi = constraint.min_value, constraint.max_value
            if not ((lo is not None and v < lo) or (hi is not None and v > hi)):
                continue
            policy = constraint.out_of_range
            self._count_range_event(prop, value, policy)
            if policy == "reject":
                return {
                    "status": "rejected", "reason": "out_of_range",
                    "input": prop, "value": v, "bounds": [lo, hi],
                }, flags
            if policy == "clamp":
                props[prop] = min(max(v, lo), hi)  # type: ignore[arg-type]
                flags.setdefault("clamped", []).append(prop)
            else:  # passthrough_with_flag
                flags["extrapolated"] = True
        return None, flags

    def _history_count(self, epoch: float) -> int:
        window_s = parse_time_resolution(str(self.history_required["window"]))
        lo = epoch - window_s
        return sum(1 for t in self._buffer if lo < t <= epoch)

    def _history_satisfied(self, epoch: float) -> bool:
        # 当前消息尚未入缓冲，故计数时 +1 计入当前样本
        need = int(self.history_required.get("min_samples", 1))
        return self._history_count(epoch) + 1 >= need

    # ---------------------------------------------------------------- 统计

    def stats(self) -> dict[str, Any]:
        """越界/迟到/重复计数（回流 Research Ledger 的漂移证据，§9.2）。"""
        return {
            "range_event_counts": dict(self.range_event_counts),
            "late_dropped": self.late_dropped,
            "duplicate_timestamps": self.duplicate_timestamps,
            "duplicate_conflicts": self.duplicate_conflicts,
            "buffered_timestamps": len(self._buffer),
        }


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)
