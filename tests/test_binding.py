"""部署绑定（model-package.md §3/§6）：property_code ↔ variable_id。"""

from __future__ import annotations

import pytest

from thermoforge_core.contracts.tfdc import BindingRecord
from thermoforge_runtime.binding import DeploymentBinding


def test_default_convention_mapping():
    b = DeploymentBinding("CH-01")
    assert b.variable_id_for("evap_chw_flow") == "CH-01.evap_chw_flow"
    props = b.resolve({"CH-01.evap_chw_flow": 521.4,
                       "CH-01.cw_supply_temp": 29.7})
    assert props == {"evap_chw_flow": 521.4, "cw_supply_temp": 29.7}


def test_explicit_bindings_override_and_cross_object():
    b = DeploymentBinding("CH-01", {"outdoor_wet_bulb_temp": "MT-01.wet_bulb"})
    assert b.variable_id_for("outdoor_wet_bulb_temp") == "MT-01.wet_bulb"
    props = b.resolve({"MT-01.wet_bulb": 28.0, "CH-01.evap_chw_flow": 1.0})
    assert props == {"outdoor_wet_bulb_temp": 28.0, "evap_chw_flow": 1.0}


def test_unrelated_variable_ids_are_ignored():
    b = DeploymentBinding("CH-01")
    assert b.resolve({"CH-02.evap_chw_flow": 1.0, "garbage": 2}) == {}


def test_model_never_sees_field_addresses():
    """模型内部只见 property_code；bindings 是 Adapter 边界的投影。"""
    b = DeploymentBinding("CH-01", {"evap_chw_flow": "CH-01.evap_chw_flow"})
    assert b.as_dict() == {"object_id": "CH-01",
                           "bindings": {"evap_chw_flow": "CH-01.evap_chw_flow"}}


def test_from_binding_records_picks_non_default_only():
    records = [
        BindingRecord(variable_id="CH-01.evap_chw_flow", adapter="bacnet",
                      source_ref="analog-input,1"),
        BindingRecord(variable_id="MT-01.wet_bulb", adapter="mqtt",
                      source_ref="site/wb"),
    ]
    # 属于 CH-01 且符合默认约定 → 无显式映射
    assert DeploymentBinding.from_binding_records("CH-01", records).as_dict() == {
        "object_id": "CH-01", "bindings": {}}


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        DeploymentBinding("CH 01")  # 非法 object_id
    with pytest.raises(ValueError):
        DeploymentBinding("CH-01", {"Bad Prop": "CH-01.x"})
    with pytest.raises(ValueError):
        DeploymentBinding(
            "CH-01", {"a": "CH-01.dup", "b": "CH-01.dup"})  # 重复 variable_id
