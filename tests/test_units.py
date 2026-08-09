"""单位规范测试（conventions.md §2）。"""

import pytest

from thermoforge_core.units import (
    IncompatibleUnitError,
    UnknownUnitError,
    conversion_factor,
    convert_value,
    normalize_unit,
    quantity_kind_of,
)


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("Cel", "Cel"), ("℃", "Cel"), ("°C", "Cel"), ("C", "Cel"), ("degC", "Cel"),
        ("kW", "kW"), ("KW", "kW"), ("kw", "kW"),
        ("kW.h", "kW.h"), ("kWh", "kW.h"), ("kwh", "kW.h"),
        ("m3/h", "m3/h"), ("m³/h", "m3/h"), ("CMH", "m3/h"), ("m3/hr", "m3/h"),
        ("kg/s", "kg/s"),
        ("kPa", "kPa"), ("KPA", "kPa"), ("kpa", "kPa"),
        ("%", "%"), ("%RH", "%"), ("RH", "%"),
        ("1", "1"), ("-", "1"), ("ratio", "1"),
        ("Hz", "Hz"), ("s", "s"),
    ],
)
def test_normalize_unit_alias(alias, canonical):
    assert normalize_unit(alias) == canonical


def test_temperature_difference_alias_cel_to_k():
    # §2.1：温差上下文允许 Cel 作为 K 的别名
    assert normalize_unit("Cel", "temperature_difference") == "K"
    assert normalize_unit("K") == "K"


def test_unknown_unit_raises_tfdc_401():
    with pytest.raises(UnknownUnitError) as exc:
        normalize_unit("BTU/h")
    assert exc.value.code == "TFDC-401"


def test_temperature_affine_conversion():
    # §2.2：温度是仿射换算 K = Cel + 273.15
    assert convert_value(20.0, "Cel", "K", "temperature") == pytest.approx(293.15)
    assert convert_value(293.15, "K", "Cel", "temperature") == pytest.approx(20.0)
    scale, offset = conversion_factor("Cel", "K", "temperature")
    assert (scale, offset) == (1.0, 273.15)


def test_temperature_difference_linear_conversion():
    # §2.2：温差是线性换算 ΔK = ΔCel，不得引入 273.15 偏移
    assert convert_value(5.0, "Cel", "K", "temperature_difference") == 5.0
    scale, offset = conversion_factor("K", "Cel", "temperature_difference")
    assert (scale, offset) == (1.0, 0.0)


def test_temperature_and_difference_not_mixable_without_kind():
    # 未声明 quantity_kind 时，Cel（温度）与 K（温差）不互通
    with pytest.raises(IncompatibleUnitError) as exc:
        convert_value(20.0, "Cel", "K")
    assert exc.value.code == "TFDC-402"


def test_percent_dimensionless_factor_100():
    # §2.3：% 与无量纲比率换算系数为 100
    assert convert_value(50.0, "%", "1") == pytest.approx(0.5)
    assert convert_value(0.5, "1", "%") == pytest.approx(50.0)


def test_incompatible_kinds_raise_tfdc_402():
    with pytest.raises(IncompatibleUnitError) as exc:
        convert_value(1.0, "kW", "Cel")
    assert exc.value.code == "TFDC-402"


def test_quantity_kind_of():
    assert quantity_kind_of("kW") == "power"
    assert quantity_kind_of("Cel") == "temperature"
    assert quantity_kind_of("K") == "temperature_difference"


def test_identity_conversion():
    assert conversion_factor("kW", "kW") == (1.0, 0.0)
    assert convert_value(3.14, "kWh", "kW.h") == pytest.approx(3.14)
