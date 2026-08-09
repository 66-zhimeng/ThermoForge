"""ThermoForge CLI（`tf`）——Pi Agent 的外部接入面。

每个工具一个子命令，stdout 输出统一信封 JSON（implementation-notes §11）；
`tf status` 为人类可读的汇总面板（`--json` 机读）。

退出码约定（机读契约）：

- 0：命令成功执行——包括工具级失败（信封 ``ok=false``）。Agent 读信封
  判断成败，不靠退出码猜测（architecture §6）。
- 2：CLI 自身错误（参数解析失败、JSON 参数非法、未注册命令等）。

stdout 只输出 JSON（或 status 面板文本），日志与错误走 stderr。
"""
