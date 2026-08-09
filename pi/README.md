# pi/ — ThermoForge 控制面

`pi/` 是 **控制面**（architecture.md §4/§8：Pi Harness 保持轻量、可替换，
研究事实保存在契约与 Research Ledger 中）。本目录不放研究逻辑，只
声明「Agent 如何调用确定性工具」。

## 组成

- `tools.json`：工具名 → Python 入口的声明式清单（与
  `thermoforge_research.tools.TOOL_REGISTRY` 一一对应，测试保证不漂移）。
- `extensions/`：Pi Agent 侧的接入适配示例。

## 接入方式一：内置 Agent（`tf agent`，零宿主）

仓库内置可对话的研发 Agent（`thermoforge_agent` 包），不依赖外部
Agent 宿主：

```bash
export TF_AGENT_API_KEY=sk-...     # 或 cp pi/agent.example.toml pi/agent.toml
.venv/Scripts/tf agent --check     # 验证配置与连通性
.venv/Scripts/tf agent             # 进入对话 REPL（/tools /exit）
```

- 协议：OpenAI 兼容 chat completions + function calling（Moonshot/Kimi、
  OpenAI、DeepSeek 等均可，`TF_AGENT_BASE_URL` / `TF_AGENT_MODEL` 切换）。
- 配置优先级：CLI 参数 > 环境变量 > `pi/agent.toml`（已 gitignore，
  密钥不入库；模板 `pi/agent.example.toml`）。
- 工具清单从 `TOOL_REGISTRY` 自动生成 JSON Schema；
  `tf_preprocess_approve` 等 human-only 工具不直接暴露，模型经
  `tf_human_approval` 发起请求，REPL 弹确认后以 actor=human 执行。
- 系统提示词：`pi/prompts/system.md`；会话日志：
  `research/agent_sessions/*.jsonl`。

## 接入方式二：CLI 进程调用（推荐给外部 Pi Agent）

`tf` 命令（`thermoforge_cli` 包，`pyproject [project.scripts]` 注册）
把每个工具暴露为一个子命令，适合作为 Pi Agent 的 shell 工具：

```bash
tf dataset list                                   # 单行 JSON 信封
tf dataset sample WX_2025_PLANT@rev_0001 --n 50 --variables PLANT.total_power
tf goal create --definition-file goal.yaml --dataset-ref WX_2025_PLANT@rev_0001
tf experiment run EXP-0001
tf model publish EXP-0005 --model-id plant-power --version 1.1.0
tf --actor human preprocess approve WX_FIX        # 审批必须 human actor
tf status                                         # 人类可读面板（--json 机读）
```

机读契约：

- **stdout 只有信封 JSON**（默认单行；`--pretty` 缩进），日志走 stderr，
  可直接 `json.loads`。
- **exit 0** = 命令已执行——包括工具级失败（信封 `ok=false`）。判断成败
  读信封，不靠退出码猜测（architecture §6）。**exit 2** = CLI 自身错误
  （参数解析失败、非法 JSON 参数等）。
- 全局选项 `--vault-root/--research-root/--models-root/--actor` 指向
  工作区（默认仓库根相对路径）。
- 常用参数是显式选项；其余收 `--param k=v`（值按 JSON 解析）或
  `--params-json '<mapping>'`。
- 结构化定义（goal/experiment/view/ruleset）用 `--*-json` 或 `--*-file`
  （YAML 亦可）。

Pi 侧最小适配示例见 `extensions/tf_cli_adapter.py`。

## 接入方式三：Python import（同进程）

```python
from thermoforge_research.tools import ToolContext, TOOL_REGISTRY

ctx = ToolContext(vault_root="vault", research_root="research",
                  models_root="models", actor="agent")
env = TOOL_REGISTRY["tf_dataset_sample"](ctx, "WX_2025_PLANT@rev_0001", n=50)
```

控制面构造一次 `ToolContext`，逐工具调用时把 ctx 作为第一个参数传入，
其余参数来自工具调用的结构化参数。

## 共同约束（两种接入方式一致）

1. 每个工具返回**统一信封**（implementation-notes.md §11）：
   `ok / tool / id / status / inputs / summary / diagnostics / artifacts /
   truncated`。有副作用的工具返回稳定 ID（`dataset@rev_NNNN`、
   `RG-/H-/EXP-/VIEW-`、`model_id@version`），后续调用引用该 ID，
   不得重新描述输入。
2. **Agent 不直接读取海量原始数据**（architecture §3），接口层强制：
   - 响应体 32 KB 上限，超限截断并置 `truncated: true`，完整内容落
     artifact（`research/tool_artifacts/`）供按需读取；
   - `tf_dataset_sample` 硬上限 200 行（等距确定性抽样）；
   - `tf_dataset_profile` 分位数点位固定 min/p1/p25/p50/p75/p99/max；
   - `tf_dataset_query` 只返回聚合统计，不返回原始行；
   - `tf_dataset_schema` 等宽摘要固定携带紧凑清单字段，明细在 artifact。
3. 诊断聚合计数（`count`），不逐行展开；实验失败携带结构化
   `error_code`（TFX-9xx），发布门禁失败携带 TFM-10xx 与门禁明细。
4. 审批类动作（`tf_preprocess_approve`）要求 `--actor human`；
   Agent actor 调用会被 TFPP-006 拒绝。

## 编排

`thermoforge_research.orchestrator.ResearchOrchestrator` 驱动
「目标 → 假设 → 实验 → 证据 → 结论」闭环（research-loop.md §2），
停止条件（§9 + G3 可建模性门禁 `modelability_failed`）全部输出结构化
原因。Agent 的决策点通过 planner 可调用对象注入；编排层自身只消费
工具信封与 Ledger 摘要。
