# -*- coding: utf-8 -*-
"""Fill the 冷却塔 sheet of the workbook.

Rules (from requirement, 2026-08-08):
- ct.supply_t  = 冷却水总管.t_supply  (tower outlet = cooled water back to chiller)
- ct.return_t  = 冷却水总管.t_return  (tower inlet = hot water from condenser)
- ct.instant_flow = 冷却水总管.f / n_running_towers
- tower N is running iff any of its fans ct_0N_f_01..03 has status_run == 1
- when a tower is not running -> 0 for all three values (workbook idiom IF(status=1, x, 0))
- supply_p / return_p stay empty (no source data / rule given)

Writes a NEW file; the original workbook is untouched. The fill is done at
ZIP/XML level so the ~3M cached formula values in other sheets are preserved.
"""
import re
import shutil
import sys
import zipfile
from openpyxl import load_workbook

sys.stdout.reconfigure(encoding="utf-8")

SRC = "data/数据处理_算法导入训练_0723_WX_已加入表冷器.xlsx"
DST = "data/数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx"
N_ROWS = 35040
FIRST_DATA_ROW = 5

# ---------------------------------------------------------------- read source
wb = load_workbook(SRC, read_only=True, data_only=True)

cw = wb["冷却水总管"]
cw_f, cw_ts, cw_tr, cw_time = [], [], [], []
for row in cw.iter_rows(min_row=FIRST_DATA_ROW, max_row=FIRST_DATA_ROW + N_ROWS - 1,
                        min_col=1, max_col=4, values_only=True):
    cw_time.append(row[0])
    cw_f.append(row[1])
    cw_ts.append(row[2])
    cw_tr.append(row[3])

fan = wb["冷却塔风机"]
fan_status = [[] for _ in range(12)]  # fan_status[i][row]
fan_time = []
for row in fan.iter_rows(min_row=FIRST_DATA_ROW, max_row=FIRST_DATA_ROW + N_ROWS - 1,
                         min_col=1, max_col=73, values_only=True):
    fan_time.append(row[0])
    for i in range(12):
        fan_status[i].append(row[1 + 6 * i])  # status_run is 1st attr of each fan
wb.close()

assert len(cw_f) == N_ROWS, f"cw rows {len(cw_f)} != {N_ROWS}"
assert len(fan_status[0]) == N_ROWS, f"fan rows {len(fan_status[0])} != {N_ROWS}"
mismatch = sum(1 for a, b in zip(cw_time, fan_time) if a != b)
assert mismatch == 0, f"timestamp mismatch on {mismatch} rows"
print(f"source rows ok: {N_ROWS}, timestamps aligned")

# ------------------------------------------------------------------- compute
# tower_running[t][r]: tower t (0..3) running at row r
tower_running = [[any(fan_status[3 * t + k][r] == 1 for k in range(3))
                  for r in range(N_ROWS)] for t in range(4)]

n_run_dist = {}
values = []  # values[r] = list of 20 numbers (None = leave empty)
for r in range(N_ROWS):
    n_run = sum(tower_running[t][r] for t in range(4))
    n_run_dist[n_run] = n_run_dist.get(n_run, 0) + 1
    row_vals = []
    for t in range(4):
        if tower_running[t][r] and n_run > 0:
            f = cw_f[r]
            flow = (f / n_run) if isinstance(f, (int, float)) else None
            row_vals += [cw_ts[r], cw_tr[r], None, None, flow]
        else:
            row_vals += [0, 0, None, None, 0]
    values.append(row_vals)

print("running-tower count distribution:", dict(sorted(n_run_dist.items())))

# ------------------------------------------------- locate sheet xml inside zip
zin = zipfile.ZipFile(SRC)
wbxml = zin.read("xl/workbook.xml").decode("utf-8")
rels = zin.read("xl/_rels/workbook.xml.rels").decode("utf-8")
m = re.search(r'<sheet[^>]*name="冷却塔"[^>]*r:id="(rId\d+)"', wbxml)
rid = m.group(1)
m = re.search(rf'<Relationship[^>]*Id="{rid}"[^>]*Target="([^"]+)"', rels)
sheet_path = "xl/" + m.group(1)
print("冷却塔 sheet xml:", sheet_path)

sheet = zin.read(sheet_path).decode("utf-8")

# ------------------------------------------------------------------ rewrite
COLS = []
for i in range(2, 22):  # column index 2..21 -> B..U (4 towers x 5 attrs)
    n, s = i, ""
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    COLS.append(s)

row_re = re.compile(r'(<row r="(\d+)"[^>]*>)(.*?)(</row>)', re.S)

def fmt(v):
    if isinstance(v, float):
        return repr(round(v, 10))
    return str(v)

def repl(mo):
    rownum = int(mo.group(2))
    if rownum < FIRST_DATA_ROW:
        return mo.group(0)
    idx = rownum - FIRST_DATA_ROW
    if idx >= N_ROWS:
        return mo.group(0)
    # keep only the column-A (timestamp) cell; drop stray pre-existing cells
    # (original sheet has one junk row of zeros at row 13 -> duplicate-cell risk)
    body = mo.group(3)
    cells = re.findall(r'(<c r="A' + str(rownum) + r'"[^>]*>.*?</c>)', body)
    for col, v in zip(COLS, values[idx]):
        if v is None:
            continue
        cells.append(f'<c r="{col}{rownum}"><v>{fmt(v)}</v></c>')
    return mo.group(1) + "".join(cells) + mo.group(4)

new_sheet, n_sub = row_re.subn(repl, sheet)
print(f"rows processed by regex: {n_sub}")

# update <dimension> if present and too small
new_sheet = re.sub(r'<dimension ref="[^"]*"/>',
                   f'<dimension ref="A1:U{FIRST_DATA_ROW + N_ROWS - 1}"/>',
                   new_sheet, count=1)

# ------------------------------------------------------------------- write
shutil.copy(SRC, DST)
# rewrite the single entry: copy all entries except target, then append it
src_z = zipfile.ZipFile(SRC)
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in src_z.infolist():
        if item.filename == sheet_path:
            continue
        zout.writestr(item, src_z.read(item.filename))
    zout.writestr(sheet_path, new_sheet.encode("utf-8"))
src_z.close()
zin.close()
print("written:", DST)
