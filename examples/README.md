# examples/ — 端到端示例

## chiller_power：运行冷机总功率模型（首个垂直切片）

按 [DD-12](../docs/design-decisions.md)（2026-08-08 修订）与
[data-survey §5](../docs/data-survey.md) 的定义，用真实工作簿端到端跑通
ThermoForge 全链路（Phase 0–4）：

```text
旧格式工作簿 → legacy 适配器 → Vault（不可变 revision）
  → 系统级派生数据集（PLANT 对象）→ Research Goal（DD-16 白名单机器校验）
  → Dataset View → 三组实验（线性基线 / 物理 / 残差混合，子进程隔离、固定种子）
  → 模型比较 + 负荷分档 → 发布门禁 → Model Registry
```

### 运行

```bash
# 仓库根目录；首次运行导入 43MB 工作簿约 30–50s
.venv/Scripts/python examples/chiller_power/run_demo.py
# 重跑幂等：已导入的 revision 按内容指纹复用；--reimport 强制重导
```

### 产物（均 gitignored，报告除外）

- `vault/`：`WX_2025_HVAC@rev_0001`（源）与 `WX_2025_PLANT@rev_0001`（派生）
- `research/`：Ledger（RG/H/EXP/F/D）+ 实验制品（metrics/predictions/物理报告）
- `models/plant-power/1.0.0/`：已发布模型包（checksums + golden）
- `examples/chiller_power/report.md`：实验报告（随脚本重新生成）

### 结果摘要（实测，种子 20260808，时间外推面 A）

| 模型 | CVRMSE | NMBE | MAPE | 结论 |
|---|---:|---:|---:|---|
| baseline_ridge | **0.1216** | −0.0132 | 0.1002 | 达标，发布为 `plant-power@1.0.0` |
| hybrid_residual | 0.1655 | −0.0193 | 0.1457 | 物理主干受 F6 拖累 + 树模型外推弱 |
| physics_cop | 1.1765 | −1.1622 | 1.2505 | 仅作参照（F6 量纲问题未决） |

- 验收：CVRMSE ≤ 0.13、NMBE ±0.02、p99 ≤ 5 ms（Q9 初值 0.10 实测无诚实
  模型可达，修订留痕于 Ledger 决策，依据见报告）。
- 白名单负例：`chiller_01.load`（派生量，F1 循环论证）在 Goal 创建时被
  机器校验拒绝；白名单外特征在实验登记时被拒绝（tests/test_whitelist.py）。
- 发布门禁六项全过：integrity / signature_tfom / acceptance / latency
  （p99=0.411 ms）/ smoke（冷加载 golden 比对 max_rel_error=0）/
  rollback_recorded。

### 文件

- `run_demo.py` — 端到端演示脚本（六个阶段，命令行 `--reimport`）
- `derive_plant.py` — 系统级派生数据集（跨对象聚合目标的契约化落点）
- `tfom/plant.v1.yaml` — 冷站系统级物模型（演示注册表 = 默认 + 此文件）
- `report.md` — 最近一次完整运行的实验报告
