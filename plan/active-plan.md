# Active Plan

## Objective

让模型实验室（`category=lab`）成为**自治**通路：Agent 自己写模型代码、自己在
真实数据上训练与测试、自己读指标改代码，闭环里不含任何人工审批环节。

## Assumptions

- 可复现性由机器判定保证，不由人工审批保证：五连检（接口 / 拟合预测 /
  save 契约 / save-load 往返位级一致 / 同种子重训位级一致）+ 环境锁 +
  子进程隔离 + 源码内容哈希。
- 「这个建模假设好不好」由证据淘汰，不由人排队放行；因此 `validated` 即可
  进正式实验，`deprecate` 是事后否决且不阻塞任何一轮循环。
- **预处理审批的人工门禁保持不变**：它改的是数据（会产生 vault revision），
  与实验室改假设不是一回事（DD-02 修订记录，2026-08-17）。
- 静态扫描是候选代码执行前唯一的自动关卡，是粗筛不是内核沙箱；该风险敞口
  明知并接受（同上）。
- 改模型 = 提交新版本（内容哈希一变即 `v(N+1)`），历史实验因源码快照冻结在
  实验目录而不被改写。

## Tasks

- [x] 门禁语义从 `require_approved` 换成 `require_runnable`（校验通过且未停用）。
- [x] 状态机改为 `validated` / `unvalidated` / `deprecated`，并对取消审批前
  入库的 `proposed`/`approved` 做读时归一。
- [x] `tf_lab_approve`（human-only）替换为 `tf_lab_deprecate`（任意 actor，
  事后否决）；MCP 与 agent 的排除清单只剩 `tf_preprocess_approve`。
- [x] 收紧 AST 扫描：白名单根模块下的禁用子模块 + 远程反序列化 / 动态库加载 /
  数据集下载类调用（扫描现在是执行前唯一自动关卡）。
- [x] 规划器 / 副驾 / WebUI / 技能书 / CLI 提示词全部改口径：过校验即可引用。
- [x] 回归覆盖：自治闭环（v1 跑完 → v2 立刻能跑）、两类门禁拦截、扫描收紧的
  正反例（含 `json.loads` 不误伤）。
- [x] 全量测试 601 passed。

- [x] 修复 `harness/prompts/` 装载在生产路径上断掉的问题（见下）。

## 提示词装载（本次一并修复）

发现四个位点没有一个真正走 `prompts.load()`：`agent.py._default_system_prompt()`
直接读 `system.md` 原文（技能不进提示词），副驾与规划器各用代码内常量
（`copilot.md`/`planner.md` 无人读取），`mcp.md` 根本不存在。`prompts.py`
承诺的「装技能 = 改文件，四处行为一致」一处都没兑现，技能书成了死文档。

修法：四个位点统一走 `prompts.load()`，代码内常量降级为文件缺失时的兜底；
`load()` 增加 `$name` 占位符填充（`safe_substitute`），只用于**代码派生的
事实**（页面清单、界面工具名），决策规则一律写死在文件里；补上 `mcp.md`。
回归测试用哨兵文件替换 `PROMPTS_DIR`，验证四个位点都跟着变。

装载后的提示词规模（含绑定技能）：cli ≈ 7.4k tokens、planner ≈ 6.5k、
mcp ≈ 6.6k、copilot ≈ 0.7k（副驾不绑技能）。mcp 那份进的是 MCP 握手的
`instructions`，对每个外部 agent 会话都生效——嫌重就把 `SKILL_BINDINGS["mcp"]`
改成 `()`，一行的事。

## Change Log

- 2026-08-17: 模型实验室取消人工审批，改为机器判定门禁 + 事后停用；
  同步收紧静态扫描、更新四处提示词口径与 DD-02 修订记录。
- 2026-08-17: 修复提示词装载断链——四个位点统一经 `prompts.load()`，
  技能书（取证 + 系统辨识阶梯）真正进入 CLI agent、规划器与 MCP。
