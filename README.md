# ThermoForge

> **Autonomous Physics–Data Hybrid Modeling Research System**
> 面向数据中心 HVAC 的自主物理—数据混合建模研究系统

ThermoForge 接收物模型、历史数据和研究目标，持续提出假设、执行实验、积累证据，并产出可验证、可部署的模型包。

ThermoForge takes an object model, historical data, and a research goal, then continuously forms hypotheses, runs experiments, accumulates evidence, and delivers verifiable, deployable model packages.

---

## 是什么 / 不是什么

| ThermoForge 是 | ThermoForge 不是 |
|---|---|
| 可长期运行的自动建模研究平台 | 一个「让 LLM 写 Python 脚本」的工具 |
| 以契约驱动的可复现实验系统 | 一次性的调参或 AutoML 服务 |
| 物理模型与数据模型的混合建模框架 | 纯黑箱机器学习管线 |
| 交付带谱系、约束和指标的 Model Package | 交付孤立的 `model.pkl` |

## 核心原则

1. **Agent 只做研究推理与编排**，大规模科学计算交给确定性的 Python 计算层。
2. **一套变量语义贯穿全链路**：物模型、Excel、实验、模型签名、API、现场绑定使用同一 `variable_id`。
3. **Excel 是交换入口，不是数据库**，导入后转为可追溯、可复用的数据资产。
4. **研究过程必须持久化**，目标、假设、证据、结论和失败原因不能只存在于 LLM 上下文中。
5. **模型评价是多维的**，精度之外还包括泛化、物理一致性、稳定性、复杂度和运行成本。
6. **原始数据与已完成实验不可原地覆盖**，内容变化一律产生新版本。

完整原则见 [docs/README.md](docs/README.md)。

## 架构速览

```mermaid
flowchart LR
    In["TFOM + TFDC<br/>Research Goal + Engineering Rules"] --> PI["Research Orchestrator"]
    PI --> Exp["Experiment Planner / Runner<br/>物理 · 数据 · 混合"]
    Exp --> Val["Validator + Model Reviewer"]
    Val --> Ledger["Research Ledger"]
    Ledger --> PI
    Val --> Reg["Model Registry"]
    Reg --> Rt["Algorithm Server / SoftPLC"]
    Rt -.漂移与违规证据.-> Ledger
```

研究闭环：`目标 → 假设 → 实验 → 证据 → 结论 → 知识 → 新假设`

详见 [系统总体设计](docs/architecture.md)。

## 关键契约

| 契约 | 职责 | 定义位置 |
|---|---|---|
| **TFOM** | 对象、属性、单位、角色、派生关系和物理约束 | [data-contract.md §3](docs/data-contract.md) |
| **TFDC** | 历史与实时数据如何表达和交换 | [data-contract.md](docs/data-contract.md) |
| **Dataset View** | 可复用、可哈希的数据选择与预处理 | [data-contract.md §7](docs/data-contract.md) |
| **Research Goal** | 研究对象、目标变量、候选输入和验收标准 | [research-loop.md §1](docs/research-loop.md) |
| **Experiment** | 一次可复现的模型实验 | [research-loop.md §5](docs/research-loop.md) |
| **Model Package** | 可验证、可部署的模型交付物 | [model-package.md](docs/model-package.md) |

术语速查见 [术语表](docs/glossary.md)；命名、单位、时间和哈希的规范性细节见 [工程约定](docs/conventions.md)。

## 文档

**新手从 [入门教程](docs/getting-started.md) 开始**：环境准备、五分钟跑通示例、命令行速查、常见任务流程。

设计文档从 [文档总览](docs/README.md) 进入。文档分三层：

**方案层**（当前工作面）

| 文档 | 内容 |
|---|---|
| [范围、非目标与前提](docs/scope.md) | 一期做什么、明确不做什么、方案成立的前提假设、什么算成功 |
| [设计决策记录](docs/design-decisions.md) | 17 条关键决策的备选、理由、代价与复审触发条件 |
| [可行性风险](docs/risks.md) | 12 项风险及其早期验证方式，含 2026-08-08 的核实结果 |
| [数据探查报告](docs/data-survey.md) | 对 `data/` 工作簿的结构与数值探查，及其对方案的影响 |
| [开放议题](docs/open-questions.md) | 尚未**决定**的方案级与契约级问题 |
| [缺口分析](docs/gap-analysis.md) | 尚未被任何文档**覆盖**的空白，按价值链盘点 |
| [待办问题清单](docs/issues.md) | 跨上述文档的统一问题索引，已定级排序，可直接对应 Issue |

**设计层**

| # | 文档 | 内容 |
|---:|---|---|
| 1 | [系统总体设计](docs/architecture.md) | 系统边界、组件、Agent 职责与技术分层 |
| 2 | [TFDC 数据契约](docs/data-contract.md) | 变量体系、Excel 规范、Data Vault、数据版本 |
| 3 | [自主研究闭环](docs/research-loop.md) | 研究状态机、实验协议、验证矩阵与模型评分 |
| 4 | [模型包与部署契约](docs/model-package.md) | 模型签名、制品结构、注册与线上接入 |
| 5 | [实施路线图](docs/roadmap.md) | 阶段划分、验收标准与首个垂直切片 |

**实现层**

| 文档 | 内容 |
|---|---|
| [工程约定](docs/conventions.md) | 命名正则、单位表、时区处理、哈希规范化、完整错误码表 |
| [实现细则与已知陷阱](docs/implementation-notes.md) | 库行为差异、边界情况、泄漏路径与测试基线 |
| [术语表](docs/glossary.md) | 缩写、ID 前缀、枚举值速查 |

## 本地控制台

不想用命令行的话，双击 `启动网页版.bat`（或 `.venv/Scripts/python tools/webui.py`），
浏览器打开 <http://127.0.0.1:8765>：

| 页面 | 做什么 |
|---|---|
| **AI 助手** | **提问就行**：它查数据、给结论，并把你带到该看的页面、选好该看的实验 |
| 数据 | 数据集/修订版浏览、变量表（实测 vs 派生）、画像、原始时序 |
| 数据质量 | 语义门禁体检 → 人话解释 → 处方；能由规则库解决的一键生成提案，由人以 `actor=human` 审批 |
| AI 研究 | 建目标与视图，让 AI 跑 `假设 → 实验 → 证据 → 新假设` 循环；自动或逐轮审批两种模式，实时看进度 |
| 实验结果 | 指标卡、预测/实测时序、散点、残差分布、切分时间轴、物理检查、原始预测点下载、多实验对比 |
| 模型 | 注册表、版本状态机、发布门禁结果 |
| 报告导出 | 自包含 HTML / Markdown+PNG / PDF |

界面**只监听 127.0.0.1**：它能改密钥、跑实验、批准预处理规则，不对同网段开放。

侧栏任何页面都有「问 AI」输入框。副驾有全部工具权限（查数据、体检、建目标、
跑实验、比模型、发布），**唯一必须你亲自点的是预处理规则审批**——那一步要求
`actor=human`，Ledger 里记的必须是人。

## 用外部 Agent 驱动（MCP）

同一套工具也能暴露给 Claude Code、Claude Desktop 或任何 MCP 客户端，
共用同一份工件，不存在第二条数据通路。注册一次：

```bash
claude mcp add thermoforge -- <仓库绝对路径>/.venv/Scripts/python -m thermoforge_mcp
```

之后在 Claude Code 里直接问「ThermoForge 最新实验怎么样」，它会自己调
`tf_status`、`tf_model_compare`、`tf_experiment_report` 等 24 个工具。
审批类工具**不**经 MCP 暴露，理由同上。

## 仓库结构

代码分层见 [architecture.md §7](docs/architecture.md)。

```text
ThermoForge/
├── docs/              # 设计文档与契约说明
├── data/              # 示例与研究数据
├── contracts/         # TFOM / TFDC / Research Goal / Experiment / Model Package Schema
├── src/
│   ├── thermoforge_core/      # 契约 + 规范化 JSON、指纹、ID、单位、时间、错误码
│   ├── thermoforge_data/      # 导入器、Data Vault、Dataset View、预处理规则库
│   ├── thermoforge_models/    # 线性基线、冷机物理模型、残差混合模型
│   ├── thermoforge_research/  # Ledger、实验 Runner、切分、指标、门禁、编排器、工具注册表
│   ├── thermoforge_runtime/   # 模型包构建/校验、注册表、发布门禁、部署绑定、在线推理
│   ├── thermoforge_cli/       # `tf` 命令行
│   ├── thermoforge_agent/     # 内置对话 Agent（OpenAI 兼容 + function calling）
│   ├── thermoforge_webui/     # 本地 Web 控制台（Streamlit）
│   └── thermoforge_mcp/       # MCP 服务端（stdio），把同一套工具给外部 Agent
├── pi/                # Agent extensions / prompts / tools.json
├── tests/             # 约 510 个测试
├── examples/          # 冷水机端到端示例
├── research/          # 运行产物 · Research Ledger（默认不入库）
├── models/            # 运行产物 · 模型注册表（默认不入库）
└── vault/             # 运行产物 · Data Vault（默认不入库）
```

契约、示例、代码和小型固定测试数据应提交到 Git；`vault/` 与运行期 `research/` 制品不提交。

## 项目状态

**Phase 0–4 的 MVP 已实现并通过端到端示例验证**（见 [新手教程](docs/getting-started.md)），方案文档仍在持续演进。关键取舍见 [设计决策记录](docs/design-decisions.md)，未决问题见 [开放议题](docs/open-questions.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| — | 方案论证：范围、决策、风险、开放议题 | 进行中 |
| Phase 0 | 契约定稿：TFOM / TFDC / Research Goal / Experiment / Model Package Schema | ✅ 完成（`contracts/` + `thermoforge_core`） |
| Phase 1 | 数据底座：导入器、指纹、Parquet、Dataset View | ✅ 完成（`thermoforge_data`） |
| Phase 2 | 确定性研究内核：Ledger、Runner、验证套件 | ✅ 完成（`thermoforge_research` / `thermoforge_models`） |
| Phase 3 | Pi/Agent 编排：工具接口、预算与停止条件 | ✅ 完成（23 个工具 + CLI + 编排器 + Web 控制台 + MCP） |
| Phase 4 | 模型注册与部署：Model Package、发布门禁、绑定 | ✅ 完成（`thermoforge_runtime`） |
| Phase 5 | 系统化扩展：容器、队列、多站点、多 Agent | 未开始 |

各阶段验收标准见 [实施路线图](docs/roadmap.md)。

首个目标垂直切片为**冷水机输入功率模型**。数据探查后已修订输入定义与验证方式（见 [DD-12](docs/design-decisions.md)）：原候选输入中的制冷量实为电流百分比的重标定，用它建模构成循环论证。

两个 P0 数据问题已于 2026-08-09 由数据提供方回答（[Q10](docs/open-questions.md)、[I-48](docs/issues.md)），结论直接塑造了建模方式：

- 工作簿的驱动序列是**现场实测**，但其中约 296 万个单元格是 Excel 内的派生计算。因此建模输入必须区分 `source_kind`（measured / derived），派生列不得进入候选输入白名单（[DD-16](docs/design-decisions.md)）。
- COP 9~11 对这个高温离心式站点是**物理有效**的，不是量纲错误——物理建模路线因此解除阻塞。

**下一步**：Phase 5（容器、队列、多站点、多 Agent）；以及示例数据的公开授权与脱敏范围确认。

## 参与贡献

项目处于方案论证阶段，当前最有价值的贡献是**挑战方案本身**，而不是补充实现细节：

- [设计决策记录](docs/design-decisions.md) 中的取舍是否成立，尤其是标注为「建议复审」的 DD-03（自研 Ledger vs 复用 MLflow）和 DD-08（物理模型选型）。
- [可行性风险](docs/risks.md) 是否有遗漏，特别是 R2/R3——若流量不可信或制冷量由功率反算，整个建模目标需要重选。
- [前提假设](docs/scope.md) 在你的场景中是否成立。
- [开放议题](docs/open-questions.md) 的判据是否合理。
- TFOM / TFDC 是否覆盖你所在场景的设备与数据源；Excel 硬性规则在真实 BMS 导出数据上是否可行。
- [工程约定](docs/conventions.md) 与 [实现细则](docs/implementation-notes.md) 中标注 **[草案]** 的取值。

欢迎通过 Issue 提出问题，或在 PR 中直接修订文档。

## 许可证

基于 [MIT License](LICENSE) 发布。
