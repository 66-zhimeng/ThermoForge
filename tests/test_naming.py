"""命名规则测试（conventions.md §1.1–1.2）。"""

import pytest

from thermoforge_core.naming import (
    NamingError,
    is_dataset_id,
    is_object_id,
    is_object_model_id,
    is_property_code,
    is_variable_id,
    split_variable_id,
    validate_variable_id,
    variable_id_for,
)


@pytest.mark.parametrize(
    "value",
    ["a", "input_power", "evap_chw_supply_temp", "x1", "a" * 64],
)
def test_property_code_valid(value):
    assert is_property_code(value)


@pytest.mark.parametrize(
    "value",
    [
        "", "1abc", "Abc", "input-power", "input power", "电流百分比",
        "_lead", "a.b", "a" * 65,
    ],
)
def test_property_code_invalid(value):
    assert not is_property_code(value)


@pytest.mark.parametrize("value", ["CH-01", "SYS-CHW", "chiller_01", "A", "x" * 64])
def test_object_id_valid(value):
    assert is_object_id(value)


@pytest.mark.parametrize("value", ["", "CH.01", "CH 01", "冷冻水", "x" * 65])
def test_object_id_invalid(value):
    # object_id 不得包含 '.'，否则 variable_id 无法唯一切分
    assert not is_object_id(value)


@pytest.mark.parametrize(
    "value",
    ["CH-01.input_power", "SYS-CHW.header_supply_temp", "A.b"],
)
def test_variable_id_valid(value):
    assert is_variable_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "CH-01",  # 缺 property
        "CH-01.Input_power",  # property 含大写
        "CH-01.input-power",  # property 含连字符
        ".input_power",
        "CH-01.",
        "CH.01.input_power",  # object_id 含点
    ],
)
def test_variable_id_invalid(value):
    assert not is_variable_id(value)


@pytest.mark.parametrize("value", ["chiller.v1", "chilled_header.v12", "a.v0"])
def test_object_model_id_valid(value):
    assert is_object_model_id(value)


@pytest.mark.parametrize("value", ["chiller", "chiller.v", "Chiller.v1", "chiller.V1", "chiller.1"])
def test_object_model_id_invalid(value):
    assert not is_object_model_id(value)


@pytest.mark.parametrize("value", ["DC01_2026_CHILLER", "A", "X" * 64])
def test_dataset_id_valid(value):
    assert is_dataset_id(value)


@pytest.mark.parametrize("value", ["", "dc01", "1DC", "DC-01", "X" * 65])
def test_dataset_id_invalid(value):
    assert not is_dataset_id(value)


def test_split_variable_id_first_dot():
    # 按第一个 '.' 切分（§1.1）
    assert split_variable_id("CH-01.input_power") == ("CH-01", "input_power")


def test_split_variable_id_rejects_malformed():
    with pytest.raises(NamingError):
        split_variable_id("CH.01.x")
    with pytest.raises(NamingError):
        validate_variable_id("bad")


def test_variable_id_for_roundtrip():
    vid = variable_id_for("CH-01", "input_power")
    assert vid == "CH-01.input_power"
    assert split_variable_id(vid) == ("CH-01", "input_power")
