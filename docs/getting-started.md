# 新手教程：从零使用 ThermoForge

这份教程面向第一次接触本项目的人。读完并照做一遍，你会知道：项目是什么、怎么跑起来、怎么让系统帮你训练一个设备模型、怎么看结果。

---

## 0. 一句话理解这个项目

ThermoForge 是一个**自动建模研究系统**：你给它设备数据（Excel）和一个研究目标（比如「预测冷水机功率」），它自动做实验、验证、评分，最后交付一个可部署的模型包，全过程留痕可复现。

你不需要自己写建模代码。你有两种使用姿势：

- **对 AI 助手说话**（推荐）：在 Kimi CLI 里直接说「帮我训练冷却塔模型」，AI 会调用系统工具完成。本教程的方式三。
- **敲命令行**：用 `tf` 命令直接操作系统。本教程的方式二。

---

## 1. 环境准备（只做一次）

前提：已安装 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
cd ThermoForge
uv sync
```

`uv sync` 会创建 `.venv/` 虚拟环境并装好所有依赖（约 1~2 分钟）。完成后验证：

```bash
.venv/Scripts/tf status
```

看到「ThermoForge 状态面板」就说明环境就绪。

> Windows 注意：命令里的 `.venv/Scripts/tf` 在 PowerShell 中写成 `.venv\Scripts\tf`。

---

## 2. 五分钟跑通现成示例

仓库自带一个完整示例（用真实数据训练「冷站总功率模型」）。直接运行：

```bash
.venv/Scripts/python examples/chiller_power/run_demo.py
```

首次运行约 1~2 分钟（要导入 43MB 的 Excel）。它会自动完成：**导入数据 → 检查数据能不能建模 → 训练三个模型 → 验证评分 → 发布最优模型**。

跑完看结果：

```bash
.venv/Scripts/tf status
```

你会看到研究目标、5 次实验的指标、已发布的生产模型。更详细的实验报告在：

```text
examples/chiller_power/report.md
```

用编辑器打开即可阅读，里面有模型指标对比、物理验证结果和发布决策。

---

## 3. 方式二：命令行速查

所有 `tf` 命令的 stdout 都是 JSON（方便程序处理），加 `--pretty` 变成人类可读格式。

| 你想做什么 | 命令 |
|---|---|
| 看全局状态（目标/实验/生产模型） | `tf status` |
| 看有哪些数据集 | `tf dataset list` |
| 看数据集详情 | `tf dataset get WX_2025_HVAC@rev_0001` |
| 看数据集画像（统计/质量） | `tf dataset profile WX_2025_HVAC@rev_0001` |
| 检查数据能否支撑某建模目标 | `tf dataset modelability ...` |
| 看某次实验详情 | `tf experiment get EXP-0005` |
| 对比几次实验 | `tf model compare EXP-0001,EXP-0005` |
| 创建研究目标 | `tf goal create --params-json '{...}'` |
| 提交预处理规则 | `tf preprocess propose ...` |
| 审批预处理规则（仅人工） | `tf preprocess approve --actor human ...` |

任何命令加 `-h` 看参数说明，例如 `tf goal create -h`。

**退出码约定**（给程序调用用）：命令本身出错 exit 2；工具执行失败 exit 0 但输出 JSON 里 `ok: false`——判断成败看 JSON 里的 `ok` 字段，不看退出码。

---

## 4. 方式三：对 AI 助手说话（推荐）

在 Kimi CLI 中打开本仓库，直接用自然语言下任务，例如：

- 「把 `data/新数据.xlsx` 导入系统」
- 「创建一个研究目标：用冷冻水流量和温度预测冷站功率，验收 CVRMSE ≤ 0.13」
- 「跑一轮实验，对比线性模型和混合模型」
- 「把表现最好的模型发布到生产」
- 「最近五次实验的结果汇总给我」

AI 助手通过 `pi/tools.json` 里登记的 22 个工具完成这些操作，每一步都有结构化记录。要接入你自己的 Agent（Pi Agent），看 `pi/README.md`——CLI 调用和 Python 调用两种方式任选。

### 内置 Agent：`tf agent`（不依赖外部助手）

仓库内置一个可对话的研发 Agent，配好密钥即可使用：

```bash
# 1. 配置密钥（二选一；密钥不会入库）
export TF_AGENT_API_KEY=sk-...        # Moonshot/Kimi、OpenAI、DeepSeek 等均可
# 或：cp pi/agent.example.toml pi/agent.toml，填入 api_key

# 2. 验证连通性
.venv/Scripts/tf agent --check        # 输出 {"ok": true, ...} 即就绪

# 3. 开始对话
.venv/Scripts/tf agent
```

示例对话：

```text
you> 现在有哪些数据集？质量怎么样？
  → 调用 tf_dataset_list → ok
  → 调用 tf_dataset_profile → ok
agent> 当前有 2 个数据集：WX_2025_HVAC（35,040 行）……

you> 把表现最好的实验发布为 plant-power 1.2.0
  → 调用 tf_model_publish → ok
agent> 已发布 plant-power@1.2.0，六项门禁全部通过……
```

环境变量 `TF_AGENT_BASE_URL` / `TF_AGENT_MODEL` 可切换端点与模型（任何
OpenAI 兼容的 chat completions + function calling 端点都能用）。审批类
动作（如预处理规则审批）不会直接执行——Agent 会向你弹确认，输入 `y`
才以 human 身份执行并留痕。会话记录保存在 `research/agent_sessions/`。

---

## 5. 核心概念速查（看懂结果所需的最小词汇表）

| 名词 | 含义 | 存在哪 |
|---|---|---|
| Research Goal（研究目标） | 要研究什么：目标变量、可用输入（白名单）、验收标准 | `research/goals/` |
| Experiment（实验） | 一次可复现的模型训练+验证 | `research/experiments/EXP-XXXX/` |
| Dataset View | 对数据的一份可复用的筛选/预处理定义 | 物化缓存在 `vault/views/` |
| Model Package（模型包） | 可部署的模型交付物：权重+签名+指标+谱系 | `models/<名字>/<版本>/` |
| Ledger（研究账本） | 所有假设、实验、结论、决策的记录 | `research/` |
| CVRMSE / NMBE | 精度指标：相对误差（越小越好）/ 系统偏差（越接近 0 越好） | 实验报告里 |

**白名单规则（重要）**：训练模型只能用 Research Goal 里 `candidate_inputs` 列出的变量，可以跨设备取（如 `cooling_tower.supply_t`）。这是硬约束——防止模型用到「答案泄漏」的变量（比如用制冷量预测功率，而制冷量本身就是功率算出来的）。

---

## 6. 常见任务流程

### 6.1 导入一份新数据

数据需要是 TFDC-XLSX 格式（`manifest`/`objects`/`variables`/`data` 四张表，规范见 `docs/data-contract.md`）。旧格式 Excel 需要先写适配规则（方式三里最省事：直接对 AI 说「导入这份 Excel，字段含义是……」）。

```bash
tf dataset import --path data/你的文件.xlsx
```

导入成功会得到一个数据集 ID（如 `SITE01_2026@rev_0001`）。同一文件重复导入会自动识别、不会产生重复版本。

### 6.2 训练一个设备模型

1. 定义研究目标（哪类设备、预测什么、允许用哪些输入、验收标准）
2. 系统自动做可建模性检查（数据不够、输入泄漏会被拦下）
3. 跑实验（基线/物理/混合模型）
4. 看报告、达标后发布

方式三一句话搞定：「用 XX 数据训练 XX 模型，输入只允许用 A、B、C」。

### 6.3 看结果与迭代

- `tf status`：全局概览
- `research/experiments/EXP-XXXX/report.json`：单次实验完整指标
- `tf model compare`：多次实验对比
- 模型每次发布产生新版本（`models/plant-power/1.1.2` 这样的目录），旧版本可回滚

---

## 7. 遇到问题怎么办

| 症状 | 怎么办 |
|---|---|
| `tf` 命令不存在 | 先 `uv sync`；用 `.venv/Scripts/tf` 全路径 |
| 导入报 `TFDC-xxx` 错误 | 错误码表在 `docs/conventions.md` §7，每个码有明确含义和级别 |
| 实验失败 | `tf experiment get <id>` 看 `diagnostics` 里的结构化错误码 |
| 目标停止了 | `tf research status --goal-id RG-XXXX` 看停止原因（预算耗尽/数据不足/可建模性未过等都有明确说明） |
| 想改契约/规则 | 不要直接改 `docs/` 契约，先提 issue 讨论（见 `docs/issues.md` 的流程） |

进一步阅读：`docs/README.md` 是文档总览（方案层 → 设计层 → 实现层）。
