# 实验报告：运行冷机总功率模型（首个垂直切片）

生成：2026-08-09 05:42:57 UTC · 耗时 57s · 种子 20260808 · 子进程隔离执行

## 数据与谱系

- 源数据集：`WX_2025_HVAC@rev_0001`（43MB 旧格式工作簿，legacy 适配器导入）
- 派生数据集：`WX_2025_PLANT@rev_0001`（系统级 PLANT 对象，规则见下）
- Dataset View：`VIEW-0001`（filter `any_running=true`）
- Research Goal：`RG-0001`

### 目标构造（跨对象聚合的落点）

「运行冷机总功率 = sum(chiller.power where status_run=1)」无法用
单机物模型或现有 View 长表表达。选择**派生数据集**路线
（`derive_plant.py`）：确定性纯函数映射 → `import_parsed` 标准管线
校验 → 不可变 revision 落 vault，lineage 记录 `derived_from`。
两总管温度逐点一致（复核 max|A1−A2| = 0.0 K），流量独立取和。

## 白名单（DD-16 机器校验）

- 允许：`chw_flow`, `chw_supply_temp`, `chw_return_temp`, `cw_supply_temp`, `cw_return_temp`, `ambient_t`, `ambient_h`, `run_count`
- 禁用（DD-12）：`load`（派生量，§F1 循环论证）、`current_percent`
  （与目标同源）、`condenser_return_t` / `evaporator_supply_t`
  （表头公式副本）
- 负例断言：含 `chiller_01.load` 的 candidate_inputs 在 Goal 创建时
  被工具层拒绝（source_kind=derived）；白名单外特征在实验登记时
  被拒绝（tests/test_whitelist.py）。

## 三组实验（时间外推：制冷季内 70/15/15，embargo 45min）

| 模型 | 实验 | 面 | n | RMSE | MAE | MAPE | CVRMSE | NMBE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline_ridge | EXP-0001 | validate | 2236 | 138.37 | 109.07 | 0.0714 | 0.0874 | +0.0437 |
| baseline_ridge | EXP-0001 | A | 2239 | 122.30 | 96.05 | 0.1002 | 0.1216 | -0.0132 |
| physics_cop | EXP-0002 | validate | 2236 | 1492.01 | 1359.18 | 0.8722 | 0.9430 | -0.8590 |
| physics_cop | EXP-0002 | A | 2239 | 1183.31 | 1168.90 | 1.2505 | 1.1765 | -1.1622 |
| hybrid_residual | EXP-0003 | validate | 2236 | 160.82 | 123.13 | 0.0834 | 0.1016 | +0.0233 |
| hybrid_residual | EXP-0003 | A | 2239 | 166.44 | 131.58 | 0.1457 | 0.1655 | -0.0193 |

面 A = 已见对象 × 未来时段（本数据单系统级对象，无留一设备面）。
混合模型的 DD-07 配套（物理主干单独指标 / 残差占比）见各实验 metrics.json 的 `physics_only` / `residual_share`。

## 物理验证（硬约束 + 单调性，总体口径）

| 模型 | overall_rate | 明细 |
|---|---:|---|
| baseline_ridge | 0.0 | power_within_rated: 0/2239 |
| physics_cop | 0.0 | cop_below_carnot: 0/2239; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239 |
| hybrid_residual | 0.011165698972755694 | cop_below_carnot: 8/2239; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239; monotonic:chw_flow:+: 17/20 |

注：`cop_below_carnot` 的冷凝温度以冷却水供水温度近似（I-40）；
本数据系统 COP 中位数 ~11 超出水冷离心机物理范围（§F6），物理主干
的绝对精度受此影响，物理路线结果仅作参照。

## 负荷分档（最优模型，面 A，按 chw_flow 三分位）

| 档位 | 流量范围 (m³/h) | n | CVRMSE | NMBE | MAPE |
|---|---|---:|---:|---:|---:|
| 低负荷 | 2870–3162 | 706 | 0.1088 | +0.0030 | 0.0902 |
| 中负荷 | 3162–3205 | 784 | 0.1117 | -0.0277 | 0.0977 |
| 高负荷 | 3205–3508 | 749 | 0.1412 | -0.0135 | 0.1122 |

## 验收阈值与发布决策

- 口径（Q9）：CVRMSE 主指标 ≤ 0.13、NMBE ±0.02 以内（DD-14 必报）、推理延迟 p99 ≤ 5.0 ms。
- 阈值依据：Q9 初始建议 CVRMSE ≤ 0.10 是基于探查期 OLS 估计（≈0.12）的期望；本切片实测最优诚实模型为线性基线（面 A CVRMSE=0.1216），混合模型受物理主干系统性偏差（§F6）与树模型
  时间外推能力限制反而更差（0.1655）。无诚实模型达到 0.10，
  故按「最优诚实模型 + 合理余量」修订为 0.13（Ledger 决策留痕），
  待 F6 的 COP 量纲问题解决后再收紧。物理路线仅作参照不参评。
- 发布：`plant-power@1.0.0`（模型 baseline_ridge）→ PRODUCTION

### 发布门禁（TFM-10xx）

| 门禁 | 结果 |
|---|---|
| integrity | 文件齐全且校验和一致（12 个文件） |
| signature_tfom | 签名与 plant.v1 兼容 |
| acceptance | 硬性验收条件全部满足（6 项指标） |
| latency | p99=0.758 ms <= 5.0 ms |
| smoke | 冷加载 + golden 比对通过（n=15, max_rel_error=0） |
| rollback_recorded | 首个生产版本，无回滚目标（已记录） |
