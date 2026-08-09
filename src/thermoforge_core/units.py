"""单位规范（conventions.md §2）。

单位字符串采用 UCUM 区分大小写形式。别名映射是登记在册的确定性规则表，
未登记的单位字符串抛出 `UnknownUnitError`（对应 TFDC-401），不做模糊匹配。

温度与温差不能共用转换函数（§2.2）：
- `temperature`：仿射换算，K = Cel + 273.15
- `temperature_difference`：线性换算，ΔK = ΔCel
"""

from __future__ import annotations

from dataclasses import dataclass, field


class UnitError(ValueError):
    """单位相关错误，携带 conventions.md §7 的错误码。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class UnknownUnitError(UnitError):
    """TFDC-401 UNIT_UNKNOWN：单位字符串不在规范表或别名表。"""

    def __init__(self, unit: str):
        super().__init__("TFDC-401", f"未登记的单位字符串: {unit!r}")


class IncompatibleUnitError(UnitError):
    """TFDC-402 UNIT_INCOMPATIBLE：量纲不一致，无法换算。"""


class UnregisteredConversionError(UnitError):
    """TFDC-403 UNIT_CONVERSION_UNREGISTERED：量纲一致但无登记换算规则。"""


@dataclass(frozen=True)
class UnitDef:
    """规范单位表中的一行（conventions.md §2.1）。"""

    canonical: str  # UCUM 规范写法
    quantity_kind: str  # 量纲语义
    aliases: tuple[str, ...] = field(default=())


# conventions.md §2.1 规范单位表
UNIT_TABLE: tuple[UnitDef, ...] = (
    UnitDef("Cel", "temperature", ("℃", "°C", "C", "degC")),
    # 温差输入别名允许 Cel：温差换算是线性的，ΔCel 与 ΔK 数值相等（§2.2）
    UnitDef("K", "temperature_difference", ("Cel",)),
    UnitDef("kW", "power", ("KW", "kw")),
    UnitDef("kW.h", "energy", ("kWh", "kwh")),
    UnitDef("m3/h", "volume_flow", ("m³/h", "CMH", "m3/hr")),
    UnitDef("kg/s", "mass_flow", ()),
    # 压力与压差共用规范单位 kPa（§2.1）
    UnitDef("kPa", "pressure", ("KPA", "kpa")),
    UnitDef("%", "percent", ("%RH", "RH")),
    UnitDef("1", "dimensionless", ("-", "ratio")),
    UnitDef("Hz", "frequency", ()),
    UnitDef("s", "time", ()),
)

# 别名 → 规范单位；注意 "Cel" 本身是规范单位，不作为 K 的别名查表入口
_ALIAS_MAP: dict[str, str] = {}
_CANONICAL_SET: set[str] = set()
for _u in UNIT_TABLE:
    _CANONICAL_SET.add(_u.canonical)
    for _a in _u.aliases:
        if _a in _CANONICAL_SET:
            continue  # 规范单位优先（如 Cel）
        _ALIAS_MAP[_a] = _u.canonical

# 规范单位 → 量纲
_QUANTITY_KIND: dict[str, str] = {_u.canonical: _u.quantity_kind for _u in UNIT_TABLE}

# 温差上下文中的别名：quantity_kind == temperature_difference 时 Cel 视作 K
_TEMPERATURE_DIFFERENCE_ALIASES: dict[str, str] = {"Cel": "K"}


def normalize_unit(unit: str, quantity_kind: str | None = None) -> str:
    """把输入单位字符串规范化为 UCUM 规范写法。

    `quantity_kind == "temperature_difference"` 时，`Cel` 按 §2.1 温差行的
    别名规则归一为 `K`。未登记的单位抛 `UnknownUnitError`（TFDC-401）。
    """
    if not isinstance(unit, str):
        raise UnknownUnitError(str(unit))
    if quantity_kind == "temperature_difference" and unit in _TEMPERATURE_DIFFERENCE_ALIASES:
        return _TEMPERATURE_DIFFERENCE_ALIASES[unit]
    if unit in _CANONICAL_SET:
        return unit
    if unit in _ALIAS_MAP:
        return _ALIAS_MAP[unit]
    raise UnknownUnitError(unit)


def quantity_kind_of(unit: str) -> str:
    """返回规范单位对应的量纲语义。输入必须先规范化或本身即规范单位。"""
    canonical = normalize_unit(unit)
    return _QUANTITY_KIND[canonical]


def conversion_factor(from_unit: str, to_unit: str, quantity_kind: str | None = None) -> tuple[float, float]:
    """返回 `(scale, offset)`，使 `to_value = from_value * scale + offset`。

    - 温度（temperature）：仿射，Cel → K 为 (1.0, 273.15)。
    - 温差（temperature_difference）：线性，Cel → K 为 (1.0, 0.0)。
    - `%` ↔ `1`：系数 100（§2.3）。
    """
    src = normalize_unit(from_unit, quantity_kind)
    dst = normalize_unit(to_unit, quantity_kind)
    if src == dst:
        return (1.0, 0.0)

    src_kind = _QUANTITY_KIND[src]
    dst_kind = _QUANTITY_KIND[dst]
    kind = quantity_kind or src_kind

    # 允许 percent ↔ dimensionless 这一对显式登记的跨量纲换算（§2.3）
    if {src_kind, dst_kind} == {"percent", "dimensionless"}:
        scale = 100.0 if src_kind == "dimensionless" else 0.01
        return (scale, 0.0)

    if kind in ("temperature", "temperature_difference") and {src, dst} <= {"Cel", "K"}:
        if quantity_kind is None and src_kind != dst_kind:
            # 未声明 quantity_kind 时，温度（Cel）与温差（K）不互通，
            # 防止仿射/线性语义被静默混用（§2.2）
            raise IncompatibleUnitError(
                "TFDC-402",
                f"量纲语义不一致: {src_kind}({src}) -> {dst_kind}({dst})，"
                "请显式声明 quantity_kind",
            )
        if kind == "temperature":
            # 仿射换算：K = Cel + 273.15（§2.2）
            to_kelvin = {"Cel": (1.0, 273.15), "K": (1.0, 0.0)}
            s1, o1 = to_kelvin[src]
            s2, o2 = to_kelvin[dst]
            return (s1 / s2, (o1 - o2) / s2)
        # 线性换算：ΔK = ΔCel（§2.2）
        return (1.0, 0.0)

    if src_kind != dst_kind:
        raise IncompatibleUnitError(
            "TFDC-402",
            f"量纲不一致，无法换算: {src_kind}({src}) -> {dst_kind}({dst})",
        )
    raise UnregisteredConversionError("TFDC-403", f"无登记的换算规则: {src} -> {dst}")


def convert_value(value: float, from_unit: str, to_unit: str, quantity_kind: str | None = None) -> float:
    """按登记的确定性规则换算数值，使用 IEEE-754 double，不做舍入（§2.4）。"""
    scale, offset = conversion_factor(from_unit, to_unit, quantity_kind)
    return value * scale + offset
