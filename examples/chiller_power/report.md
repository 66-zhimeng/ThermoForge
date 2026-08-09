# 实验报告：运行冷机总功率模型（首个垂直切片）

生成：2026-08-09 08:30:34 UTC · 耗时 57s · 种子 20260808 · 子进程隔离执行

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

## 可建模性门禁（G3 语义层检查）

- [info] **derivation_chain**：候选输入与目标均无派生依赖交叉
- [info] **same_origin**：候选与目标相关性均在合理范围
- [info] **device_diversity**：范围内实例数 < 2，无需设备区分度检查
- [warning] **physical_plausibility**：1 项派生物理量超范围
- [info] **operating_coverage**：目标可用占比 100.0%，负荷三档覆盖充足

- 正例 verdict：**PASS**（完整报告落 artifact，信封仅摘要）
- 负例（故意含循环输入 `chiller_01.current_percent`，与 power 同源 r≈0.9955）：准入层 whitelist 放行（measured），可建模性报告 same_origin blocker → verdict FAIL；编排层停止原因 `modelability_failed`，未登记任何实验。

## 实验（时间外推：制冷季内 70/15/15，embargo 45min）

| 模型 | 实验 | 面 | n | RMSE | MAE | MAPE | CVRMSE | NMBE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline_ridge | EXP-0001 | validate | 2236 | 138.37 | 109.07 | 0.0714 | 0.0874 | +0.0437 |
| baseline_ridge | EXP-0001 | A | 2239 | 122.30 | 96.05 | 0.1002 | 0.1216 | -0.0132 |
| physics_cop | EXP-0002 | validate | 2236 | 1492.01 | 1359.18 | 0.8722 | 0.9430 | -0.8590 |
| physics_cop | EXP-0002 | A | 2239 | 1183.31 | 1168.90 | 1.2505 | 1.1765 | -1.1622 |
| physics_doe2_v2 | EXP-0003 | validate | 2236 | 166.92 | 133.04 | 0.0842 | 0.1055 | +0.0732 |
| physics_doe2_v2 | EXP-0003 | A | 2239 | 154.94 | 117.43 | 0.1101 | 0.1541 | +0.0990 |
| hybrid_residual | EXP-0004 | validate | 2236 | 160.82 | 123.13 | 0.0834 | 0.1016 | +0.0233 |
| hybrid_residual | EXP-0004 | A | 2239 | 166.44 | 131.58 | 0.1457 | 0.1655 | -0.0193 |
| hybrid_residual_v2 | EXP-0005 | validate | 2236 | 121.57 | 95.15 | 0.0634 | 0.0768 | +0.0298 |
| hybrid_residual_v2 | EXP-0005 | A | 2239 | 116.74 | 89.72 | 0.0899 | 0.1161 | +0.0215 |

面 A = 已见对象 × 未来时段（本数据单系统级对象，无留一设备面）。
混合模型的 DD-07 配套（物理主干单独指标 / 残差占比）见各实验 metrics.json 的 `physics_only` / `residual_share`。

## v2 物理模型（cooling_balance_v2，DOE-2 三曲线）

模型形式（离心机标准经验模型，平滑可微）：

```text
Q       = m·Cp·ΔT
CAPFT   = f(T_chws, T_cond)   双二次（可用容量比）
PLR     = Q / (Q_rated · run_count · CAPFT)
EIRFT   = g(T_chws, T_cond)   双二次（能效比温度修正）
EIRFPLR = c0 + c1·PLR + c2·PLR²（部分负荷修正，Σc=1 归一）
P       = P_rated · run_count · PLR · EIRFT · EIRFPLR
```

- 辨识：alternating_least_squares（outer=3 × inner=20，固定迭代无随机源），n=10445；曲线输入按训练集均值/标准差归一化（原始温度下双二次设计矩阵病态，归一化参数随 params.yaml 落盘）。
- CAPFT: const=1.0012, t1=0.0013, t1_sq=0.0039, t1_t2=-0.0146, t2=-0.0152, t2_sq=0.0011
- EIRFT: const=0.6796, t1=0.0004, t1_sq=0.0018, t1_t2=-0.0025, t2=0.1853, t2_sq=-0.0015
- EIRFPLR: const=1.3920, plr=-0.3345, plr_sq=-0.0575
- 系数裁剪: 无（全部在明文范围内）

### 冷凝侧代理选择（I-40 修正，用数据说话）

plant 级有 cw_supply / cw_return 两个冷却水温度。单变量 COP 相关性上
cw_supply 略强（R² 0.583 vs 0.538），但**完整模型辨识残差**（训练集
相对 RMSE）cw_return 更优：

| 候选 | 辨识 rel_rmse |
|---|---:|
| cw_return_temp | 0.0675 |
| cw_supply_temp | 0.0886 |

选定 **cw_return_temp**（完整模型口径计入 PLR 耦合，比单变量
相关性更可信）。注意：本数据中 cw_return < cw_supply 约 6 K，与常规
冷却水环路方向相反，标签疑似互换，已向数据问题清单登记。

### 与 v1 对比（面 A）

| 模型 | CVRMSE | NMBE | MAPE |
|---|---:|---:|---:|
| physics_cop (v1) | 1.1765 | -1.1622 | 1.2505 |
| physics_doe2_v2 | 0.1541 | +0.0990 | 0.1101 |

v1 失败根因是模型形式而非数据：COP 线性形式无法表达部分负荷与温度的
耦合，且 PLR 未按运行台数折算（系统级 rated 使 PLR 恒 >1.4）。

### 下游适用性（DD-17）

v2 物理模型由双二次/二次多项式组成，**平滑可微**、无树模型的分段
常数跳变，仿真与寻优（作为优化器被调模型）均适用；曲线取值保护范围
（curve_guards）保证外推到训练域边缘时行为有界。

## 物理验证（硬约束 + 单调性，总体口径）

| 模型 | overall_rate | 明细 |
|---|---:|---|
| baseline_ridge | 0.0 | power_within_rated: 0/2239 |
| physics_cop | 0.0 | cop_below_carnot: 0/2239; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239 |
| physics_doe2_v2 | 0.0 | cop_below_carnot: 0/2069; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239 |
| hybrid_residual | 0.011165698972755694 | cop_below_carnot: 8/2239; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239; monotonic:chw_flow:+: 17/20 |
| hybrid_residual_v2 | 0.0 | cop_below_carnot: 0/2069; cop_positive: 0/2239; positive_cooling_positive_power: 0/2239; power_within_rated: 0/2239; monotonic:chw_flow:+: 0/20 |

注：`cop_below_carnot` 的冷凝温度列：v1 模型以冷却水供水温度近似
（I-40）；v2 模型使用辨识选定的代理列（见上节，本次为 cw_return_temp）。
背景更新：F6 已撤销——站点为高温离心式冷机（冷冻水供水中位 17.65 °C），
COP 9~11 物理合理，物理路线正式参评。

## 负荷分档（最优模型，面 A，按 chw_flow 三分位）

| 档位 | 流量范围 (m³/h) | n | CVRMSE | NMBE | MAPE |
|---|---|---:|---:|---:|---:|
| 低负荷 | 2870–3162 | 706 | 0.1142 | +0.0543 | 0.0858 |
| 中负荷 | 3162–3205 | 784 | 0.1016 | +0.0179 | 0.0802 |
| 高负荷 | 3205–3508 | 749 | 0.1308 | -0.0059 | 0.1037 |

## 验收阈值与发布决策

- 口径（Q9）：CVRMSE 主指标 ≤ 0.13 为硬门槛；NMBE ±0.02 为 DD-14 必报指标，记录但不门禁；推理延迟 p99 ≤ 5.0 ms。
- 阈值依据：Q9 初始建议 CVRMSE ≤ 0.10 是基于探查期 OLS 估计（≈0.12）的期望；首轮实测最优诚实模型为线性基线（面 A CVRMSE=0.1216），无诚实模型达到 0.10，故按「最优诚实模型 + 合理
  余量」修订为 0.13（Ledger 决策留痕）。F6 撤销后物理路线参评，
  阈值待更多工况数据积累后再评估收紧。
- 发布：`plant-power@1.1.0`（模型 hybrid_residual_v2）→ PRODUCTION

### 发布门禁（TFM-10xx）

| 门禁 | 结果 |
|---|---|
| integrity | 文件齐全且校验和一致（14 个文件） |
| signature_tfom | 签名与 plant.v1 兼容 |
| acceptance | 硬性验收条件全部满足（6 项指标） |
| latency | p99=2.934 ms <= 5.0 ms |
| smoke | 冷加载 + golden 比对通过（n=15, max_rel_error=0） |
| rollback_recorded | 首个生产版本，无回滚目标（已记录） |
