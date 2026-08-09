# pi/ — ThermoForge 控制面

`pi/` 是 **控制面**（architecture.md §4/§8：Pi Harness 保持轻量、可替换，
研究事实保存在契约与 Research Ledger 中）。本目录不放研究逻辑，只
声明「Agent 如何调用确定性工具」。

## 组成

- `tools.json`：工具名 → Python 入口的声明式清单（与
  `thermoforge_research.tools.TOOL_REGISTRY` 一一对应，测试保证不漂移）。

## 调用模型

```python
from thermoforge_research.tools import ToolContext, TOOL_REGISTRY

ctx = ToolContext(vault_root="vault", research_root="research",
                  models_root="models", actor="agent")
env = TOOL_REGISTRY["tf_dataset_sample"](ctx, "DC01_2026_CHILLER@rev_0001", n=50)
```

1. 控制面（Agent 宿主）构造一次 `ToolContext`，逐工具调用时把 ctx 作为
   第一个参数传入，其余参数来自工具调用的结构化参数。
2. 每个工具返回**统一信封**（implementation-notes.md §11）：
   `ok / tool / id / status / inputs / summary / diagnostics / artifacts /
   truncated`。有副作用的工具返回稳定 ID（`dataset@rev_NNNN`、
   `RG-/H-/EXP-/VIEW-`、`model_id@version`），后续调用引用该 ID，
   不得重新描述输入。
3. **Agent 不直接读取海量原始数据**（architecture §3），接口层强制：
   - 响应体 32 KB 上限，超限截断并置 `truncated: true`，完整内容落
     artifact（`research/tool_artifacts/envelopes/`）供按需读取；
   - `tf_dataset_sample` 硬上限 200 行（等距确定性抽样）；
   - `tf_dataset_profile` 分位数点位固定 min/p1/p25/p50/p75/p99/max；
   - `tf_dataset_query` 只返回聚合统计，不返回原始行。
4. 诊断聚合计数（`count`），不逐行展开；实验失败携带结构化
   `error_code`（TFX-9xx），发布门禁失败携带 TFM-10xx 与门禁明细——
   Agent 不应通过自由文本猜测一次实验是否成功（architecture §6）。

## 编排

`thermoforge_research.orchestrator.ResearchOrchestrator` 驱动
「目标 → 假设 → 实验 → 证据 → 结论」闭环（research-loop.md §2），
停止条件（§9：验收达标 / 预算耗尽 TFX-906 / 连续无信息增益 / 数据
覆盖不足 / 必需变量缺失 / 需人工确认）全部输出结构化原因。
Agent 的决策点通过 planner 可调用对象注入；编排层自身只消费工具
信封与 Ledger 摘要。
