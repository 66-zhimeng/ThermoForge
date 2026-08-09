"""Dataset View 测试：哈希稳定性、物化复用、TFV-801 校验、重采样边界。

重采样约定（conventions §3.4）：左闭右开、标签取左边界、min_count 规则、
聚合白名单。窗口聚合纯函数 `resample_window` 离线/在线共用（§3.2）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_data.importer import import_parsed
from thermoforge_data.vault import DataVault, VaultError
from thermoforge_data.views import (
    materialize_view,
    resample_window,
    view_hash,
)

UTC = timezone.utc


def _dataset() -> TfdcDataset:
    manifest = TfdcManifest(
        contract="TFDC", contract_version="1.0", dataset_id="TEST01_VIEW",
        dataset_version=1, site_id="T01", timezone="Asia/Shanghai",
        time_resolution="60s",
    )
    objects = [
        ObjectRecord(object_id="CH-01", object_model_id="chiller.v1"),
        ObjectRecord(object_id="CH-02", object_model_id="chiller.v1"),
    ]
    variables = []
    for obj in ("CH-01", "CH-02"):
        for prop, unit, role in (
            ("evap_chw_supply_temp", "Cel", "state"),
            ("evap_chw_flow", "m3/h", "state"),
            ("input_power", "kW", "target"),
        ):
            variables.append(VariableRecord(
                variable_id=f"{obj}.{prop}", object_id=obj, property_code=prop,
                unit=unit, dtype="float", role=role, source_kind="measured",
            ))
    return TfdcDataset(manifest=manifest, objects=objects, variables=variables)


def _stored_ref(tmp_path) -> tuple[DataVault, str]:
    base = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(minutes=i) for i in range(6)]
    columns = {}
    for obj in ("CH-01", "CH-02"):
        columns[f"{obj}.evap_chw_supply_temp"] = [6.0 + i * 0.1 for i in range(6)]
        columns[f"{obj}.evap_chw_flow"] = [500.0 + i for i in range(6)]
        columns[f"{obj}.input_power"] = [400.0 + i for i in range(6)]
    result = import_parsed(_dataset(), ts, columns)
    assert result.ok, result.diagnostics
    vault = DataVault(tmp_path / "vault")
    ref = vault.store(result)
    return vault, ref


def _view_def(ref: str, **overrides):
    doc = {
        "view_id": "VIEW-0001",
        "dataset": ref,
        "scope": {"object_model": "chiller.v1"},
        "objects": ["CH-01", "CH-02"],
        "resolution": "2min",
        "features": ["evap_chw_supply_temp", "evap_chw_flow"],
        "target": "input_power",
    }
    doc.update(overrides)
    return doc


# ---- view_hash ----

def test_view_hash_stable_and_set_fields_sorted():
    ref = "TEST01_VIEW@rev_0001"
    h1 = view_hash(ref, _view_def(ref))
    h2 = view_hash(ref, _view_def(ref, features=["evap_chw_flow",
                                                  "evap_chw_supply_temp"],
                                 objects=["CH-02", "CH-01"]))
    assert h1 == h2  # objects/features 为集合语义字段（§5.1 规则 6）
    # 不同 revision → 不同 hash（§5.1：view_hash 必须包含数据集 revision）
    assert view_hash("TEST01_VIEW@rev_0002", _view_def(ref)) != h1


# ---- 物化与复用 ----

def test_materialize_and_reuse(tmp_path):
    vault, ref = _stored_ref(tmp_path)
    cache = tmp_path / "view_cache"
    mv1 = materialize_view(vault, _view_def(ref), cache)
    assert not mv1.reused
    assert mv1.path.name == mv1.view_hash[:16]
    # 长表：object_id 为分组键，列名为 property_code
    assert set(mv1.table.column_names) == {
        "object_id", "timestamp", "evap_chw_supply_temp", "evap_chw_flow",
        "input_power",
    }
    # 6 行 60s → 每对象 3 个 2min 桶 × 2 对象
    assert mv1.table.num_rows == 6
    mv2 = materialize_view(vault, _view_def(ref), cache)
    assert mv2.reused
    assert mv2.view_hash == mv1.view_hash
    assert mv2.table.equals(mv1.table)


def test_materialize_hash_mismatch_tampered(tmp_path):
    vault, ref = _stored_ref(tmp_path)
    cache = tmp_path / "view_cache"
    mv = materialize_view(vault, _view_def(ref), cache)
    meta_path = mv.path / "view.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["definition"]["features"] = ["evap_chw_flow"]  # 篡改缓存定义
    meta_path.write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    with pytest.raises(VaultError) as exc_info:
        materialize_view(vault, _view_def(ref), cache)
    assert exc_info.value.code == "TFV-801"


def test_view_feature_unavailable(tmp_path):
    vault, ref = _stored_ref(tmp_path)
    with pytest.raises(VaultError) as exc_info:
        materialize_view(
            vault, _view_def(ref, features=["no_such_feature"]),
            tmp_path / "cache",
        )
    assert exc_info.value.code == "TFV-802"


def test_view_empty_result(tmp_path):
    vault, ref = _stored_ref(tmp_path)
    with pytest.raises(VaultError) as exc_info:
        materialize_view(
            vault, _view_def(ref, objects=["CH-09"]),  # 不属于 scope 的对象
            tmp_path / "cache",
        )
    assert exc_info.value.code == "TFV-803"


# ---- resample_window 纯函数 ----

def test_resample_left_closed_left_label():
    # 桶 [0, 120) 与 [120, 240)，标签取左边界；t=120 落入第二个桶
    ts = [0, 60, 119, 120, 180]
    vals = [1.0, 2.0, 3.0, 10.0, 20.0]
    out_ts, out_vals = resample_window(ts, vals, 120, 1, "mean")
    assert out_ts == [0, 120]
    assert out_vals == [2.0, 15.0]


def test_resample_min_count():
    ts = [0, 60, 120]
    vals = [1.0, 2.0, 5.0]
    out_ts, out_vals = resample_window(ts, vals, 120, 2, "mean")
    assert out_ts == [0, 120]
    assert out_vals[0] == 1.5  # 2 个有效样本 ≥ min_count
    assert out_vals[1] is None  # 桶内有效样本 1 < min_count 2 → null


def test_resample_aggregation_whitelist():
    ts = [0, 30, 60]
    vals = [3.0, 1.0, 2.0]
    assert resample_window(ts, vals, 120, 1, "sum")[1] == [6.0]
    assert resample_window(ts, vals, 120, 1, "min")[1] == [1.0]
    assert resample_window(ts, vals, 120, 1, "max")[1] == [3.0]
    assert resample_window(ts, vals, 120, 1, "first")[1] == [3.0]
    assert resample_window(ts, vals, 120, 1, "last")[1] == [2.0]
    assert resample_window(ts, vals, 120, 1, "median")[1] == [2.0]
    with pytest.raises(ValueError):
        resample_window(ts, vals, 120, 1, "p95")
