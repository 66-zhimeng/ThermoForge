# Data

This directory contains example and research data used by ThermoForge.

The workbook below is included as a source data artifact for algorithm import and training workflows:

- `数据处理_算法导入训练_0723_WX_已加入表冷器.xlsx`

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
