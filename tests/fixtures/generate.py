"""生成 TFDC-XLSX 固定测试数据（tests/fixtures/）。

- `valid_minimal.xlsx`：合法的最小工作簿（七张表齐全，timestamp 为文本格式）。
- `invalid_*.xlsx`：每个变体精确触发一个错误码，期望码见 `expected_codes.json`。

xlsx 是 ZIP 容器，字节级不稳定（内部时间戳），因此重新生成的文件
字节可能不同，但逻辑内容必须一致；断言只针对内容与错误码。

用法：`.venv/Scripts/python tests/fixtures/generate.py`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openpyxl import Workbook

FIXTURES_DIR = Path(__file__).resolve().parent

TZ = timezone(timedelta(hours=8))
BASE_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=TZ)

MANIFEST_ROWS = [
    ("contract", "TFDC"),
    ("contract_version", "1.0"),
    ("dataset_id", "DC01_2026_CHILLER"),
    ("dataset_version", "1"),
    ("site_id", "DC01"),
    ("timezone", "Asia/Shanghai"),
    ("time_resolution", "60s"),
    ("object_model_version", "DC-HVAC-1.0"),
    ("source_system", "BMS"),
    ("created_at", "2026-08-08T12:00:00+08:00"),
    ("description", "minimal fixture"),
]

OBJECTS_HEADER = ["object_id", "object_model_id", "object_name", "parent_id", "system_id"]
OBJECTS_ROWS = [
    ["CH-01", "chiller.v1", "冷水机1", "SYS-CHILLER", "CHW"],
    ["SYS-CHW", "chilled_header.v1", "冷冻水总管", None, "CHW"],
]

VARIABLES_HEADER = [
    "variable_id", "object_id", "property_code", "unit", "dtype", "role", "source_kind",
]
VARIABLES_ROWS = [
    ["CH-01.evap_chw_supply_temp", "CH-01", "evap_chw_supply_temp", "Cel", "float", "state", "measured"],
    ["CH-01.evap_chw_flow", "CH-01", "evap_chw_flow", "m3/h", "float", "state", "measured"],
    ["CH-01.input_power", "CH-01", "input_power", "kW", "float", "target", "measured"],
]

DATA_HEADER = [
    "timestamp",
    "CH-01.evap_chw_supply_temp",
    "CH-01.evap_chw_flow",
    "CH-01.input_power",
]
DATA_VALUES = [
    (6.8, 521.2, 412.5),
    (6.8, 522.1, 411.7),
    (6.9, 520.8, 413.2),
    (7.0, 519.5, 415.0),
]

PARAMETERS_HEADER = ["object_id", "parameter_code", "value", "unit"]
PARAMETERS_ROWS = [
    ["CH-01", "rated_capacity", 3500, "kW"],
    ["CH-01", "rated_power", 620, "kW"],
]

RELATIONS_HEADER = ["from_object", "relation", "to_object", "port_from", "port_to", "medium", "direction"]
RELATIONS_ROWS = [
    ["CH-01", "chilled_water_to", "SYS-CHW", "evap_out", "supply_in", "water", "forward"],
]

BINDINGS_HEADER = ["variable_id", "adapter", "source_ref"]
BINDINGS_ROWS = [
    ["CH-01.evap_chw_supply_temp", "BACnet", "device=3101,AI=7"],
    ["CH-01.input_power", "OPCUA", "ns=2;s=CH01.Power"],
]


def _write_sheet(wb: Workbook, name: str, header: list[str], rows: list[list]) -> None:
    ws = wb.create_sheet(name)
    ws.append(header)
    for row in rows:
        ws.append(row)


def _timestamp_text(i: int) -> str:
    return (BASE_TS + timedelta(seconds=60 * i)).isoformat()


def _write_data_sheet(wb: Workbook, header: list[str] | None = None, rows: list[list] | None = None) -> None:
    """data 表：timestamp 列必须是**文本格式**单元格（conventions.md §3.1）。"""
    ws = wb.create_sheet("data")
    ws.append(header or DATA_HEADER)
    for row in rows if rows is not None else [
        [_timestamp_text(i), *vals] for i, vals in enumerate(DATA_VALUES)
    ]:
        ws.append(row)
        for cell in ws[ws.max_row]:
            if cell.column == 1 and isinstance(cell.value, str):
                cell.number_format = "@"  # 文本格式


def build_valid() -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    _write_sheet(wb, "manifest", ["key", "value"], MANIFEST_ROWS)
    _write_sheet(wb, "objects", OBJECTS_HEADER, OBJECTS_ROWS)
    _write_sheet(wb, "variables", VARIABLES_HEADER, VARIABLES_ROWS)
    _write_data_sheet(wb)
    _write_sheet(wb, "parameters", PARAMETERS_HEADER, PARAMETERS_ROWS)
    _write_sheet(wb, "relations", RELATIONS_HEADER, RELATIONS_ROWS)
    _write_sheet(wb, "bindings", BINDINGS_HEADER, BINDINGS_ROWS)
    return wb


def build_missing_sheet() -> Workbook:
    """TFDC-103：缺少 variables 表。"""
    wb = build_valid()
    wb.remove(wb["variables"])
    return wb


def build_merged_cell() -> Workbook:
    """TFDC-105：objects 表存在合并单元格。"""
    wb = build_valid()
    wb["objects"].merge_cells("D2:E2")
    return wb


def build_duplicated_header() -> Workbook:
    """TFDC-107：data 表头列名重复。"""
    wb = build_valid()
    ws = wb["data"]
    ws.cell(row=1, column=3, value=DATA_HEADER[2 - 1])  # 第 3 列改为与第 2 列同名
    return wb


def build_formula() -> Workbook:
    """TFDC-108：data 中存在公式。"""
    wb = build_valid()
    ws = wb["data"]
    ws.cell(row=2, column=4, value="=C2*0.79")
    return wb


def build_bad_timezone() -> Workbook:
    """TFDC-204：manifest.timezone 非 IANA 时区名。"""
    wb = build_valid()
    ws = wb["manifest"]
    for row in ws.iter_rows(min_row=2):
        if row[0].value == "timezone":
            row[1].value = "CST"
    return wb


def build_bad_resolution() -> Workbook:
    """TFDC-205：time_resolution 不符合简写格式。"""
    wb = build_valid()
    ws = wb["manifest"]
    for row in ws.iter_rows(min_row=2):
        if row[0].value == "time_resolution":
            row[1].value = "PT60S"
    return wb


def build_malformed_variable_id() -> Workbook:
    """TFDC-304：data 列名不符合 variable_id 正则（含大写 property）。"""
    wb = build_valid()
    ws = wb["data"]
    ws.cell(row=1, column=2, value="CH-01.Evap_Chw_Supply_Temp")
    return wb


def build_undeclared_variable() -> Workbook:
    """TFDC-305：data 列未在 variables 声明。"""
    wb = build_valid()
    ws = wb["data"]
    ws.cell(row=1, column=5, value="CH-01.cw_supply_temp")
    for i in range(2, 6):
        ws.cell(row=i, column=5, value=29.7)
    return wb


def build_duplicated_variable() -> Workbook:
    """TFDC-307：variables 中 variable_id 重复声明。"""
    wb = build_valid()
    wb["variables"].append(VARIABLES_ROWS[0])
    return wb


def build_unknown_unit() -> Workbook:
    """TFDC-401：variables 中单位未登记。"""
    wb = build_valid()
    ws = wb["variables"]
    ws.cell(row=2, column=4, value="BTU/h")
    return wb


def build_naive_timestamp() -> Workbook:
    """TFDC-502：timestamp 为 Excel 日期类型单元格（序列号），非文本。"""
    wb = build_valid()
    ws = wb["data"]
    for i in range(2, 6):
        cell = ws.cell(row=i, column=1)
        cell.value = BASE_TS.replace(tzinfo=None) + timedelta(seconds=60 * (i - 2))
        cell.number_format = "yyyy-mm-dd hh:mm:ss"
    return wb


def build_duplicated_timestamp() -> Workbook:
    """TFDC-503（ERROR 分支）：重复时间戳且值不一致。"""
    wb = build_valid()
    rows = [[_timestamp_text(i), *vals] for i, vals in enumerate(DATA_VALUES)]
    rows[3] = [_timestamp_text(2), 7.1, 518.0, 416.6]  # 与第 3 行同戳不同值
    wb.remove(wb["data"])
    _write_data_sheet(wb, rows=rows)
    return wb


BUILDERS = {
    "valid_minimal.xlsx": (build_valid, []),
    "invalid_TFDC-103_missing_sheet.xlsx": (build_missing_sheet, ["TFDC-103"]),
    "invalid_TFDC-105_merged_cell.xlsx": (build_merged_cell, ["TFDC-105"]),
    "invalid_TFDC-107_duplicated_header.xlsx": (build_duplicated_header, ["TFDC-107"]),
    "invalid_TFDC-108_formula.xlsx": (build_formula, ["TFDC-108"]),
    "invalid_TFDC-204_bad_timezone.xlsx": (build_bad_timezone, ["TFDC-204"]),
    "invalid_TFDC-205_bad_resolution.xlsx": (build_bad_resolution, ["TFDC-205"]),
    "invalid_TFDC-304_malformed_variable_id.xlsx": (build_malformed_variable_id, ["TFDC-304"]),
    "invalid_TFDC-305_undeclared_variable.xlsx": (build_undeclared_variable, ["TFDC-305"]),
    "invalid_TFDC-307_duplicated_variable.xlsx": (build_duplicated_variable, ["TFDC-307"]),
    "invalid_TFDC-401_unknown_unit.xlsx": (build_unknown_unit, ["TFDC-401"]),
    "invalid_TFDC-502_naive_timestamp.xlsx": (build_naive_timestamp, ["TFDC-502"]),
    "invalid_TFDC-503_duplicated_timestamp.xlsx": (build_duplicated_timestamp, ["TFDC-503"]),
}


def main() -> None:
    expected: dict[str, list[str]] = {}
    for filename, (builder, codes) in BUILDERS.items():
        wb = builder()
        wb.save(FIXTURES_DIR / filename)
        expected[filename] = codes
        print(f"written: {filename}")
    meta = FIXTURES_DIR / "expected_codes.json"
    with open(meta, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(expected, fp, ensure_ascii=False, sort_keys=True, indent=2)
        fp.write("\n")
    print(f"written: {meta.name}")


if __name__ == "__main__":
    main()
