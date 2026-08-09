"""ThermoForge 模型层（Phase 2：三类建模路线模板，research-loop §4）。

- `baseline`：线性基线（系数 JSON 交付，不用 pickle）。
- `physics`：Q = m·Cp·ΔT、P = Q/COP 物理模型，COP 参数辨识（YAML 明文）。
- `hybrid`：残差混合 Y = Y_physics + XGBoost(残差)（XGBoost 原生 .json）。
- `preprocessing`：显式特征顺序 + 训练集 fit 的 scaler（随模型序列化）。

统一接口：fit / predict / save（目录）/ load（目录）。
"""

from . import baseline, hybrid, physics, preprocessing

__all__ = ["baseline", "hybrid", "physics", "preprocessing"]
