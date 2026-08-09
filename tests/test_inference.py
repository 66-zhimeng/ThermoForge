"""在线推理运行时（model-package §6、implementation-notes §9）。

- 超范围三策略：reject / clamp / passthrough_with_flag，越界计数回流；
- history_required：历史不足返回明确的 not_ready，不凑数预测；
- 幂等：重复时间戳相同值重放、冲突值丢弃并计数；迟到丢弃计数；
- 延迟测量结构：batch=1、p50/p95/p99；
- 线上契约延续 variable_id，经 DeploymentBinding 解析。
"""

from __future__ import annotations

import pytest

from thermoforge_runtime.binding import DeploymentBinding
from thermoforge_runtime.errors import ModelRegistryError
from thermoforge_runtime.inference import InferenceSession, measure_latency

from phase34_helpers import TARGET, build_test_package

BASE_TS = "2026-08-01T00:00:00+00:00"


def _msg(ts, **props):
    values = {f"CH-01.{k}": v for k, v in props.items()}
    return {"contract": "TFDC", "version": "1.0", "timestamp": ts,
            "values": values}


def _in_range(**overrides):
    props = {
        "evap_chw_flow": 400.0,
        "evap_chw_supply_temp": 6.5,
        "evap_chw_return_temp": 10.7,
        "cw_supply_temp": 25.0,
    }
    props.update(overrides)
    return props


@pytest.fixture()
def pkg(tmp_path):
    return build_test_package(tmp_path / "pkg")


@pytest.fixture()
def session(pkg):
    return InferenceSession(pkg, binding=DeploymentBinding("CH-01"))


def test_happy_path_with_binding(session):
    result = session.handle(_msg(BASE_TS, **_in_range()))
    assert result["status"] == "ok"
    assert TARGET in result["predictions"]
    assert result["predictions"][TARGET] > 0
    assert result["latency_ms"] >= 0


def test_reject_policy(session):
    result = session.handle(_msg(BASE_TS, **_in_range(evap_chw_flow=600.0)))
    assert result["status"] == "rejected"
    assert result["reason"] == "out_of_range"
    assert result["input"] == "evap_chw_flow"
    assert session.range_event_counts["evap_chw_flow"] == 1


def test_clamp_policy_flags_and_counts(session):
    result = session.handle(
        _msg(BASE_TS, **_in_range(evap_chw_supply_temp=9.9)))
    assert result["status"] == "ok"
    assert "evap_chw_supply_temp" in result["flags"]["clamped"]
    assert session.range_event_counts["evap_chw_supply_temp"] == 1


def test_passthrough_with_flag_policy(session):
    result = session.handle(
        _msg(BASE_TS, **_in_range(evap_chw_return_temp=15.0)))
    assert result["status"] == "ok"
    assert result["flags"]["extrapolated"] is True
    assert session.range_event_counts["evap_chw_return_temp"] == 1


def test_missing_required_input_rejected(session):
    props = _in_range()
    del props["cw_supply_temp"]
    result = session.handle(_msg(BASE_TS, **props))
    assert result["status"] == "rejected"
    assert result["reason"] == "missing_inputs"
    assert result["missing"] == ["cw_supply_temp"]


def test_not_ready_until_history_satisfied(tmp_path):
    pkg = build_test_package(
        tmp_path / "pkg",
        history_required={"window": "30min", "resolution": "1min",
                          "min_samples": 3},
        cold_start="reject",
    )
    session = InferenceSession(pkg)
    ts = lambda m: f"2026-08-01T00:{m:02d}:00+00:00"  # noqa: E731
    r1 = session.handle({"timestamp": ts(0), "values": _in_range()})
    assert r1["status"] == "not_ready"
    assert r1["reason"] == "insufficient_history"
    assert r1["need_samples"] == 3
    r2 = session.handle({"timestamp": ts(1), "values": _in_range()})
    assert r2["status"] == "not_ready"
    r3 = session.handle({"timestamp": ts(2), "values": _in_range()})
    assert r3["status"] == "ok"  # 缓冲满足后才出预测


def test_duplicate_timestamp_idempotent_replay(session):
    first = session.handle(_msg(BASE_TS, **_in_range()))
    replay = session.handle(_msg(BASE_TS, **_in_range()))
    assert replay["flags"]["idempotent_replay"] is True
    assert replay["predictions"] == first["predictions"]
    assert session.duplicate_timestamps == 1
    assert session.duplicate_conflicts == 0


def test_duplicate_timestamp_conflict_counted(session):
    session.handle(_msg(BASE_TS, **_in_range()))
    conflict = session.handle(_msg(BASE_TS, **_in_range(evap_chw_flow=401.0)))
    assert conflict["flags"]["duplicate_conflict"] is True
    assert session.duplicate_conflicts == 1


def test_late_arrival_dropped(pkg):
    session = InferenceSession(pkg, late_tolerance_seconds=300.0)
    session.handle(_msg("2026-08-01T01:00:00+00:00", **_in_range()))
    late = session.handle(_msg(BASE_TS, **_in_range()))
    assert late["status"] == "dropped"
    assert late["reason"] == "late_arrival"
    assert session.late_dropped == 1


def test_range_events_flow_to_ledger_sink(pkg):
    events = []
    session = InferenceSession(pkg, binding=DeploymentBinding("CH-01"),
                               on_range_event=events.append)
    session.handle(_msg(BASE_TS, **_in_range(evap_chw_flow=600.0)))
    assert events == [{"property_code": "evap_chw_flow", "value": 600.0,
                       "action": "reject"}]
    stats = session.stats()
    assert stats["range_event_counts"]["evap_chw_flow"] == 1


def test_corrupt_package_rejected_tfm1002(pkg):
    (pkg / "metrics.json").write_text("tampered", encoding="utf-8",
                                      newline="\n")
    with pytest.raises(ModelRegistryError) as excinfo:
        InferenceSession(pkg)
    assert excinfo.value.code == "TFM-1002"


def test_measure_latency_structure(pkg):
    result = measure_latency(pkg, warmup=5, samples=30)
    assert result["batch"] == 1
    assert result["warmup"] == 5 and result["samples"] == 30
    assert 0 <= result["min_ms"] <= result["p50_ms"] <= result["p95_ms"]
    assert result["p95_ms"] <= result["p99_ms"] <= result["max_ms"]
