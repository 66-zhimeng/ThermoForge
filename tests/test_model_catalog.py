"""可执行模型闭集的一致性：真源 ↔ 执行侧分派表 ↔ 模型看到的 schema。

这套断言存在的理由：`gordon_ng` / `eps_ntu` 早已能执行，但 Agent 看到的
schema enum 里只有 `cooling_balance_v1/v2`，于是它只能靠试错才发现这两个
家族可用。名单一分叉，「闭集」就从帮助变成了误导。
"""

from __future__ import annotations

import ast
from pathlib import Path

from thermoforge_agent.schema import build_tool_schemas
from thermoforge_research.model_catalog import (
    DATA_ESTIMATORS,
    HYBRID_RESIDUALS,
    PHYSICS_BALANCE,
    PHYSICS_IDENTIFICATION,
    PHYSICS_MODELS,
)

CHILD = Path(__file__).resolve().parents[1] / "src" / "thermoforge_research" / "_child.py"


def _dict_keys_of_assignment(name: str) -> set[str]:
    """取 `_child.py` 里某个模块级 dict 赋值的字面量键。

    用 AST 而不是 import：`_child` 会拉起 numpy/sklearn/xgboost，
    而这条断言本身与建模依赖无关。
    """
    tree = ast.parse(CHILD.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name in targets and isinstance(node.value, ast.Dict):
            return {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and k.value is not None}
    raise AssertionError(f"未在 _child.py 找到模块级 dict 赋值: {name}")


def test_identification_families_match_catalog():
    assert _dict_keys_of_assignment("_IDENT_FAMILIES") == set(PHYSICS_IDENTIFICATION)


def test_balance_families_match_catalog():
    """`_build_physics` 内的分派表用源码扫描，键含 None（默认版本）。"""
    source = CHILD.read_text(encoding="utf-8")
    for name in PHYSICS_BALANCE:
        assert f'"{name}":' in source, f"执行侧缺少物理方程版本 {name}"


def test_experiment_schema_exposes_every_executable_family():
    """模型看到的 enum 必须覆盖全部可执行家族，一个都不能少。"""
    schemas, _handlers = build_tool_schemas()
    plan = next(s for s in schemas
                if s["function"]["name"] == "tf_experiment_plan")
    model = (plan["function"]["parameters"]["properties"]["definition"]
             ["properties"]["model"])
    assert set(model["properties"]["physics"]["enum"]) == set(PHYSICS_MODELS)
    assert set(model["properties"]["estimator"]["enum"]) == set(DATA_ESTIMATORS)
    assert set(model["properties"]["residual"]["enum"]) == set(HYBRID_RESIDUALS)
    # 说明里也要点名，否则模型只在 enum 里看到、不知道怎么配超参
    for name in PHYSICS_IDENTIFICATION:
        assert name in model["description"]
