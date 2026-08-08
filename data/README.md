# Data

This directory contains example and research data used by ThermoForge.

The workbook below is included as a source data artifact for algorithm import and training workflows:

- `数据处理_算法导入训练_0723_WX_已加入表冷器.xlsx`（原始文件，未改动）
- `数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx`（处理副本）

处理副本相对原始文件的改动（处理脚本见 `scripts/fill_cooling_tower.py` 与 `scripts/fix_chwp.py`）：

- **冷却塔表已填充**（原全空）：`supply_t`/`return_t` 取自冷却水总管 `t_supply`/`t_return`；
  `instant_flow` = 冷却水总管 `f` ÷ 运行塔数（塔运行 = 该塔任一台风机 `status_run = 1`）；
  未运行的塔写 0；`supply_p`/`return_p` 无来源保持空。继承了总管流量的负值缺陷。
- **冷冻水泵表时间轴已修复**：删除重复粘贴块（12-26 17:00~23:30 约 76 次重复，2,049 行），
  保留干净前缀（重复时间戳取首次出现），并按其他表的时间轴补齐 12-26 23:45 ~ 12-31 23:45
  共 481 行（数据列留空，未插值）。同时修正了该表「设备名称/物模型」两行标签互换的问题。
- 集总负载表按需求方意见暂不处理、不参与训练。

A structural and numerical survey of this workbook is documented in
[docs/data-survey.md](../docs/data-survey.md). **Read it before modelling.** Key points:

- The workbook is not in TFDC form: 4-row multi-level headers, 111 merged cells,
  2,955,883 formula cells, and timestamps stored as timezone-less Excel serials.
- Most columns are computed rather than measured. In particular
  `冷水主机.load` is exactly `电流百分比 × 96.72`, so using it to predict chiller
  power is circular reasoning.
- Several sheets are incomplete: `冷却塔` is entirely empty, `集总负载` has 6 of 7
  properties empty, and `冷冻水泵` has a corrupted time axis (2,049 duplicate
  timestamps, 76 backward jumps).
- Multiple indicators suggest the data is simulation-generated rather than field-
  measured; this is **unconfirmed** and tracked as
  [open question Q10](../docs/open-questions.md).

Please verify that any redistributed data complies with applicable privacy, confidentiality, and third-party data licenses before using it outside this repository.
