"""旧格式工作簿 → Data Vault。

走 `thermoforge_data.legacy` 适配器的标准路径
（load_workbook_model → convert_legacy_tables → import_parsed → vault.store），
另加两处保守处理，理由见 docs/data-processing-handbook.md §1.3：

1. 剔除派生汇总表（默认 `Sheet1`）——全部列是其它实测列的求和，属派生量。
2. 时间戳非单调的表按断点截断——只丢弃，不修复。时间戳订正属数据订正，
   必须走预处理规则库 + 人工审批，不能藏在导入脚本里。

用法::

    python scripts/import_legacy_workbook.py <工作簿.xlsx> [--dataset-id WX_2025_HVAC]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from thermoforge_data.importer import default_registry, import_parsed
from thermoforge_data.legacy import convert_legacy_tables, load_workbook_model
from thermoforge_research.tools import ToolContext

STEP = dt.timedelta(minutes=15)
DEFAULT_DROP_SHEETS = ("Sheet1",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, help="旧格式工作簿路径")
    parser.add_argument("--dataset-id", default="WX_2025_HVAC")
    parser.add_argument("--site-id", default="WX")
    parser.add_argument("--time-resolution", default="900s")
    parser.add_argument("--vault-root", type=Path, default=Path("vault"))
    parser.add_argument("--research-root", type=Path, default=Path("research"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--drop-sheet", action="append", default=None,
        help=f"剔除的表名，可多次给出（默认 {list(DEFAULT_DROP_SHEETS)}）")
    return parser.parse_args(argv)


def truncate_at_break(rows: list[tuple], step: dt.timedelta) -> tuple[list, int]:
    """在首个非等间隔步进处截断，返回（保留的行, 丢弃行数）。"""
    brk = next((i for i in range(1, len(rows))
                if rows[i][0] - rows[i - 1][0] != step), None)
    if brk is None:
        return rows, 0
    return rows[:brk], len(rows) - brk


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    drop = set(args.drop_sheet if args.drop_sheet is not None
               else DEFAULT_DROP_SHEETS)

    registry = default_registry()
    sheets = load_workbook_model(args.path)
    notes: list[str] = []

    for name in sorted(set(sheets) & drop):
        del sheets[name]
        notes.append(f"剔除派生汇总表 {name}")
        print(f"[剔除] {name}")

    for name, sheet in sheets.items():
        rows = [r for r in sheet.data_rows
                if not (r[0] is None and all(v is None for v in r[1:]))]
        kept, cut = truncate_at_break(rows, STEP)
        if not cut:
            continue
        note = (f"{name}: 时间戳自第 {len(kept)} 行起非单调"
                f"（{kept[-1][0]} → {rows[len(kept)][0]}），截断丢弃 {cut} 行；"
                f"保留 {kept[0][0]} → {kept[-1][0]}")
        print(f"[截断] {note}")
        notes.append(note)
        sheet.data_rows = kept

    conv = convert_legacy_tables(
        sheets, registry,
        dataset_id=args.dataset_id, site_id=args.site_id,
        time_resolution=args.time_resolution, source_name=args.path.name,
    )
    print(f"转换: {len(conv.columns)} 列 / {len(conv.timestamps)} 时刻")

    result = import_parsed(
        conv.dataset, conv.timestamps, conv.columns,
        registry=registry,
        pre_diagnostics=conv.pre_diagnostics,
        degradations=[*conv.degradations, *notes],
        source_path=args.path,
    )

    levels: dict[str, int] = {}
    for d in result.diagnostics:
        key = getattr(d.level, "value", str(d.level))
        levels[key] = levels.get(key, 0) + 1
    print(f"ok={result.ok}  诊断={levels}")

    if not result.ok:
        for d in result.diagnostics:
            if getattr(d.level, "value", str(d.level)) == "ERROR":
                print(f"  ERROR {d.code} {d.location}", file=sys.stderr)
        print("存在 ERROR 级诊断，未写入 vault", file=sys.stderr)
        return 1

    lineage = conv.lineage(args.path)
    lineage["import_notes"] = notes
    ctx = ToolContext(
        vault_root=args.vault_root, research_root=args.research_root,
        models_root=args.models_root, actor="human",
    )
    ref = ctx.vault.store(result, source_path=args.path, lineage=lineage)
    stamps = result.table.column("timestamp").to_pylist()
    print(f"\nVAULT REF: {ref}")
    print(f"rows={result.table.num_rows} "
          f"variables={len(result.dataset.variables)} "
          f"objects={len(result.dataset.objects)}")
    print(f"time_range: {stamps[0].isoformat()} → {stamps[-1].isoformat()}")
    print(json.dumps({"ref": ref, "notes": notes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
