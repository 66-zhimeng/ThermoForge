"""JSON Schema 导出：把五份契约的 pydantic 模型写到 contracts/ 目录。

用法：`.venv/Scripts/python scripts/export_schemas.py`

输出文件以 UTF-8、`\\n` 换行写入（implementation-notes §10.3），
键按码点排序，保证导出结果跨平台字节稳定。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment import Experiment
from .model_package import ModelPackage
from .research_goal import ResearchGoal
from .tfdc import TfdcDataset
from .tfom import ObjectModel

# 契约目录名 → pydantic 模型
SCHEMA_TARGETS: dict[str, type] = {
    "tfom": ObjectModel,
    "tfdc": TfdcDataset,
    "research-goal": ResearchGoal,
    "experiment": Experiment,
    "model-package": ModelPackage,
}


def render_schema(model: type) -> str:
    """导出单份 JSON Schema 的规范文本（键排序、UTF-8、无 ASCII 转义）。"""
    schema: dict[str, Any] = model.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def export_schemas(contracts_root: str | Path) -> list[Path]:
    """把全部 Schema 写入 `<contracts_root>/<name>/schema.json`，返回写出的路径。"""
    root = Path(contracts_root)
    written: list[Path] = []
    for name, model in SCHEMA_TARGETS.items():
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "schema.json"
        with open(path, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(render_schema(model))
        written.append(path)
    return written
