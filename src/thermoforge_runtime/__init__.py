"""ThermoForge 运行时层（Phase 4：模型注册与部署）。

- `package.py`：Model Package 构建与校验（checksums / golden 预测集）。
- `registry.py`：模型注册、版本状态机、发布门禁、回滚。
- `inference.py`：在线推理运行时（超范围策略 / history_required / 幂等 / 延迟口径）。
- `binding.py`：通用 property_code → 具体 variable_id 的部署绑定。
- `artifact.py`：按格式标识冷加载模型制品（无 pickle）。
- `_smoke.py`：冷加载冒烟子进程入口（TFM-1005）。
"""
