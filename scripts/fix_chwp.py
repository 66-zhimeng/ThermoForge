# -*- coding: utf-8 -*-
"""Fix the 冷冻水泵 sheet of the processed workbook.

Defects (docs/data-survey.md F7):
- 2,049 duplicated rows: the block 2025-12-26 17:00..23:30 was appended ~76
  extra times (rows 34564..36612). The clean prefix rows 5..34563 is a unique
  monotonic series ending 2025-12-26 23:30.
- Time axis stops at 12-26 instead of 12-31: 481 canonical timestamps missing.
- Header rows 2/3 swapped: the row labelled 物模型 holds instance names.

Fix:
- keep rows 5..34563 verbatim (first occurrence of every timestamp wins);
- drop rows 34564..36612 (the duplicated blocks);
- append rows 34564..35044 with canonical timestamps only (values blank,
  no interpolation per data-contract);
- swap inner content of header rows 2 and 3 back to 设备名称 / 物模型 order.

Operates in place on the processed file; all other sheets untouched.
"""
import os
import re
import sys
import zipfile
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

DST = "data/数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx"
TMP = DST + ".tmp"
LAST_CLEAN_ROW = 34563          # 2025-12-26 23:30
FIRST_NEW_ROW = 34564           # 2025-12-26 23:45
LAST_ROW = 35044                # 2025-12-31 23:45
EPOCH = datetime(1899, 12, 30)  # Excel 1900 date system

zin = zipfile.ZipFile(DST)
wbxml = zin.read("xl/workbook.xml").decode("utf-8")
rels = zin.read("xl/_rels/workbook.xml.rels").decode("utf-8")
rid = re.search(r'<sheet[^>]*name="冷冻水泵"[^>]*r:id="(rId\d+)"', wbxml).group(1)
sheet_path = "xl/" + re.search(rf'<Relationship[^>]*Id="{rid}"[^>]*Target="([^"]+)"', rels).group(1)
print("sheet xml:", sheet_path)
xml = zin.read(sheet_path).decode("utf-8")

# --- fix header labels: rows 2/3 values were already in the right rows
# (r2 = instance names chwp_01.., r3 = model name chilled_water_pump); only the
# column-A labels 设备名称/物模型 were swapped. Exchange just the label cells.
a2 = re.search(r'<c r="A2"([^>]*)><v>(\d+)</v></c>', xml)
a3 = re.search(r'<c r="A3"([^>]*)><v>(\d+)</v></c>', xml)
xml = xml[:a2.start()] + f'<c r="A2"{a2.group(1)}><v>{a3.group(2)}</v></c>' + xml[a2.end():]
a3b = re.search(r'<c r="A3"([^>]*)><v>(\d+)</v></c>', xml)
xml = xml[:a3b.start()] + f'<c r="A3"{a3b.group(1)}><v>{a2.group(2)}</v></c>' + xml[a3b.end():]
print("header labels A2/A3 swapped back to 设备名称/物模型")

# --- split: everything up to end of LAST_CLEAN_ROW stays; old tail dropped
m_end = re.search(rf'<row r="{LAST_CLEAN_ROW}".*?</row>', xml, re.S)
head = xml[:m_end.end()]
# find where sheetData content resumes after old rows (mergeCells etc.)
m_tail = re.search(r'</sheetData>', xml)
tail = xml[m_tail.start():]

# --- build 481 new timestamp-only rows
dt = datetime(2025, 12, 26, 23, 45)
a_style = re.search(r'<c r="A100" s="(\d+)"', xml).group(1)
rows_out = []
for i in range(LAST_ROW - FIRST_NEW_ROW + 1):
    rownum = FIRST_NEW_ROW + i
    serial = (dt - EPOCH).total_seconds() / 86400.0
    rows_out.append(
        f'<row r="{rownum}" spans="1:25">'
        f'<c r="A{rownum}" s="{a_style}"><v>{serial:.10f}</v></c></row>')
    dt = dt + timedelta(minutes=15)

new_xml = head + "".join(rows_out) + tail
new_xml = re.sub(r'<dimension ref="[^"]*"/>', f'<dimension ref="A1:Y{LAST_ROW}"/>',
                 new_xml, count=1)

# --- rewrite zip
with zipfile.ZipFile(TMP, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename == sheet_path:
            continue
        zout.writestr(item, zin.read(item.filename))
    zout.writestr(sheet_path, new_xml.encode("utf-8"))
zin.close()
os.replace(TMP, DST)
print("fixed in place:", DST)
print(f"rows: kept 5..{LAST_CLEAN_ROW}, appended {LAST_ROW - FIRST_NEW_ROW + 1} blank rows "
      f"({FIRST_NEW_ROW}..{LAST_ROW})")
