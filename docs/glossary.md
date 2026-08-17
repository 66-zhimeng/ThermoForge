# 术语表

本表汇总散落在各设计文档中的缩写、标识符约定和枚举值，作为统一速查入口。定义以各自主文档为准，本表只做索引和摘要。

## 1. 核心契约

| 术语 | 英文全称 | 说明 | 主文档 |
|---|---|---|---|
| TFOM | ThermoForge Object Model | 物模型。定义对象、属性、单位、类型、角色、合法范围、派生表达式和物理约束。回答「对象和属性是什么意思」 | [data-contract.md §3](./data-contract.md) |
| TFDC | ThermoForge Data Contract | 数据契约。统一 Excel、数据库、API、MQTT、BACnet、OPC UA 等来源的数据语义。回答「数据如何表达与交换」 | [data-contract.md](./data-contract.md) |
| Dataset View | — | 可版本化、可哈希的数据选择/清洗/聚合/特征定义。禁止修改原始数据，一切预处理由 View 表达 | [data-contract.md §7](./data-contract.md) |
| Research Goal | — | 一次持续研究的根对象。自然语言目标必须先规范化为结构化契约才允许执行实验 | [research-loop.md §1](./research-loop.md) |
| Experiment | — | 一次不可变、可复现的模型实验定义 | [research-loop.md §5](./research-loop.md) |
| Model Package | — | 可验证、可部署的模型交付物，含签名、约束、指标、数据谱系和研究谱系 | [model-package.md](./model-package.md) |
| Engineering Rules | — | 精度、物理一致性、时延、可解释性和部署限制等工程约束 | [architecture.md §1](./architecture.md) |

TFDC 的传输载体变体：`TFDC-XLSX`、`TFDC-JSON`、`TFDC-Parquet`、`TFDC-API`、`TFDC-MQTT`。建模逻辑只依赖 TFDC，不依赖数据最初来自哪个系统。

## 2. 变量标识

| 标识 | 形式 | 说明 |
|---|---|---|
| `property_code` | `lower_snake_case` | 跨设备稳定的物模型属性，如 `evap_chw_supply_temp`、`input_power` |
| `object_id` | 站点内唯一 | 对象实例，如 `CH-01`、`SYS-CHW` |
| `object_model_id` | `name.vN` | 对象所属物模型，如 `chiller.v1`、`chilled_header.v1` |
| `variable_id` | `object_id.property_code` | 具体对象实例上的属性，如 `CH-01.input_power`。**大小写敏感** |

`variable_id` 在 Excel、历史库、Dataset View、实验、模型签名、API、SoftPLC 和实时绑定中必须完全一致。通用设备模型的签名使用 `property_code`，部署绑定到具体对象时才解析为 `variable_id`。

## 3. 枚举值

### 变量角色 `role`

| 值 | 含义 |
|---|---|
| `state` | 状态量 |
| `control` | 控制量 |
| `disturbance` | 扰动量 |
| `target` | 建模目标 |
| `context` | 上下文 |
| `derived` | 派生量，由 TFOM `expression` 计算 |

### 数据来源 `source_kind`

`measured` 实测 · `derived` 派生 · `estimated` 估算 · `manual` 人工录入

### 研究状态机

`RESEARCH_CREATED` → `DATA_DISCOVERY` → `DATA_VALIDATION` → `DATA_PROFILING` → `BASELINE_MODELING` → `HYPOTHESIS_GENERATION` → `EXPERIMENT_DESIGN` → `EXPERIMENT_RUNNING` → `RESULT_ANALYSIS` → `MODEL_REVIEW` → `PUBLISH` / `STOPPED`

完整转换规则见 [research-loop.md §2](./research-loop.md)。

### 模型发布状态

`candidate` → `validated` → `approved` → `production` → `deprecated` → `retired`

见 [model-package.md §5](./model-package.md)。

## 4. 标识符前缀

| 前缀 | 对象 | 示例 | 存放位置 |
|---|---|---|---|
| `RG-` | Research Goal | `RG-0001` | `research/goals/` |
| `H-` | Hypothesis 假设 | `H-0011` | `research/hypotheses/` |
| `EXP-` | Experiment 实验 | `EXP-0042` | `research/experiments/` |
| `F-` | Finding 发现 | `F-0017` | `research/findings/` |
| `D-` | Decision 决策 | `D-0009` | `research/decisions/` |
| `M-` | Model 模型条目 | `M-0008` | `research/models/` |
| `VIEW-` | Dataset View | `VIEW-0021` | Dataset Engine |
| `rev_` | 数据集 revision | `DC01_2026_CHILLER@rev_0001` | `vault/datasets/` |

## 5. 系统组件

| 组件 | 职责 |
|---|---|
| TF Harness / ThermoForge Harness | 研究编排、工具调用、会话入口。可替换的控制面，研究事实不存在于其中 |
| TFOM Registry | 管理对象模型与物理约束 |
| TFDC Importer | 导入和校验 Excel |
| Data Vault | 原始数据、Parquet、元数据与谱系的不可变存储 |
| Dataset Engine | 查询、画像、采样、Dataset View 物化 |
| Experiment Runner | 隔离执行建模实验 |
| Validator | 数据、泛化、物理和运行时验证 |
| Model Reviewer / Judge | 独立检查指标、约束、泛化与发布条件 |
| Research Ledger | 记录目标、假设、实验、发现和决策 |
| Model Registry | 模型包版本与发布状态 |
| Adapter | 依据 `bindings` 把现场点位转换为 `variable_id`。绑定信息只存在于 Adapter 边界 |

## 6. Agent 角色

一期可由单个主 Agent 顺序扮演全部角色。

TF Harness · Data Scientist · Physics Scientist · ML Scientist · Hybrid Scientist · Experiment Engineer · Model Reviewer

职责见 [architecture.md §5](./architecture.md)。

## 7. 混合建模方式

| 方式 | 形式 | 说明 |
|---|---|---|
| Residual Hybrid | `Y = Y_physics + ML(X)` | 机器学习拟合物理模型残差 |
| Parameter Hybrid | 物理方程不变 | 模型根据工况预测动态参数 |
| Physics-Constrained | 数据模型直接预测 | 训练或验证中加入守恒、单调性和范围约束 |

## 8. HVAC 与统计缩写

| 缩写 | 含义 |
|---|---|
| CHW | Chilled Water，冷冻水 |
| CW | Condenser Water，冷却水 |
| COP | Coefficient of Performance，能效比 |
| PLR | Part Load Ratio，部分负荷率 |
| BMS | Building Management System，楼宇管理系统 |
| SoftPLC | 软件可编程逻辑控制器 |
| RMSE / MAE / MAPE | 均方根误差 / 平均绝对误差 / 平均绝对百分比误差 |
| CVRMSE | 变异系数均方根误差，建筑能耗校准常用指标 |

## 相关文档

方案层：[范围与前提](./scope.md) · [设计决策记录](./design-decisions.md) · [可行性风险](./risks.md) · [开放议题](./open-questions.md)

设计层：[系统总体设计](./architecture.md) · [TFDC 数据契约](./data-contract.md) · [自主研究闭环](./research-loop.md) · [模型包与部署契约](./model-package.md) · [实施路线图](./roadmap.md)

实现层：[工程约定](./conventions.md) · [实现细则与已知陷阱](./implementation-notes.md) · [文档总览](./README.md)

标识符的完整正则、单位规范表和错误码清单见 [工程约定](./conventions.md)。
