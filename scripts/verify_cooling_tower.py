# -*- coding: utf-8 -*-
"""Verify the filled 冷却塔 sheet in the processed workbook."""
import sys
from openpyxl import load_workbook

sys.stdout.reconfigure(encoding="utf-8")

DST = "data/数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx"
N_ROWS = 35040

wb = load_workbook(DST, read_only=True, data_only=True)
ct = wb["冷却塔"]
rows = list(ct.iter_rows(min_row=5, max_row=5 + N_ROWS - 1, min_col=1, max_col=22,
                         values_only=True))
wb.close()
print("rows read:", len(rows))

# stats per tower: supply_t(B,F,K,P), return_t(C,G,L,Q), instant_flow(F? no)
# col layout: A time; tower t at cols 2+5t .. 6+5t: supply_t, return_t, supply_p, return_p, instant_flow
import statistics
for t in range(4):
    base = 1 + 5 * t
    sup = [r[base] for r in rows if r[base] is not None]
    ret = [r[base + 1] for r in rows if r[base + 1] is not None]
    flo = [r[base + 4] for r in rows]
    n_none = sum(1 for r in rows if r[base] is None)
    nz = [v for v in flo if isinstance(v, (int, float)) and v != 0]
    print(f"ct_0{t+1}: nonzero-flow rows={len(nz)}, supply_t None={n_none}, "
          f"flow p50={statistics.median(nz):.1f} min={min(nz):.1f} max={max(nz):.1f}" if nz else f"ct_0{t+1}: no flow",
          f"| supply_t range {min(sup)}..{max(sup)}" if sup else "| no supply_t")

# sample rows
print("\nsample row 0 :", rows[0][:11])
print("sample row 1000:", rows[1000][:11])
# check a running row where n_running known: find row where exactly 3 towers running
for i, r in enumerate(rows):
    running = [t for t in range(4) if r[1 + 5 * t] != 0]
    if len(running) == 3:
        print(f"row {i}: towers running {running}, ct_01 supply_t={r[1]}, return_t={r[2]}, flow={r[5]}")
        break

# pressure cols must be empty
p_empty = all(r[c] is None for r in rows[:5000] for c in (3, 4, 8, 9, 13, 14, 18, 19))
print("pressure cols empty (first 5000 rows):", p_empty)
