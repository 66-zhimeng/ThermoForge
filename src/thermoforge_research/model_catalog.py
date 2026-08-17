"""可执行模型闭集（唯一真源）。

`_child.py` 的分派表、给 LLM 生成的工具 schema、网页规划器的下拉框，
以前各自维护一份名单。名单一分叉，模型看到的 enum 就会漏掉实际可用的
家族 —— 实测 `gordon_ng` / `eps_ntu` 已能执行近一个月，schema 里却只写
`cooling_balance_v1/v2`，Agent 只能靠试错发现它们能用。

这里只放**常量**，不导入 numpy/sklearn/xgboost：schema 生成路径要能在
不装建模依赖的环境里跑通。分派表与本名单的一致性由
`tests/test_model_catalog.py` 机器校验。
"""

from __future__ import annotations

# category=data 的估计器
DATA_ESTIMATORS: tuple[str, ...] = ("ridge", "linear")

# category=physics/hybrid 的物理方程版本：
# 前两个是能量平衡族（需 rated_capacity_kw），后两个是系统辨识族
# （参数少、单调性由方程结构保证，见 models/identification.py）
PHYSICS_BALANCE: tuple[str, ...] = ("cooling_balance_v1", "cooling_balance_v2")
PHYSICS_IDENTIFICATION: tuple[str, ...] = ("gordon_ng", "eps_ntu")
PHYSICS_MODELS: tuple[str, ...] = PHYSICS_BALANCE + PHYSICS_IDENTIFICATION

# category=hybrid 的残差学习器
HYBRID_RESIDUALS: tuple[str, ...] = ("xgboost",)

# 给提示词/schema 用的一句话说明，避免三处各写一遍
MODEL_CATALOG_HINT = (
    "可执行模型闭集：category=data 时 estimator 只能是 "
    + "/".join(DATA_ESTIMATORS)
    + "；category=physics 时 physics 只能是 "
    + "、".join(PHYSICS_MODELS)
    + "（前两个需超参 rated_capacity_kw；后两个是系统辨识族，"
    "用 hyperparameters.inputs 映射列名）"
    + "；category=hybrid 时必须同时给 physics 和 residual="
    + "/".join(HYBRID_RESIDUALS)
    + "。闭集之外的新函数形式走模型实验室：tf_lab_submit 提交代码，"
    "过结构校验即可以 category=lab + hyperparameters.lab 引用（无需审批）。"
    "当前不支持 mlp、neural_network、lightgbm 或纯 data xgboost。"
)
