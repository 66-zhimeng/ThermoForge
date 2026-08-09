"""模型制品加载（implementation-notes.md §8.1）。

按 `artifact/model.json` 的 `format` 字段分发到 Phase 2 的原生格式加载器：
系数 JSON（线性基线）、YAML 明文参数（物理模型）、XGBoost 原生 .json
（残差混合）。不使用 pickle（跨版本不可加载 + 任意代码执行）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thermoforge_models import baseline, hybrid, physics

_LOADERS = {
    baseline.MODEL_FORMAT: baseline.LinearBaseline.load,
    physics.MODEL_FORMAT: physics.ChillerPhysicsModel.load,
    hybrid.MODEL_FORMAT: hybrid.ResidualHybrid.load,
}


def detect_format(artifact_dir: str | Path) -> str:
    """读取制品目录的模型格式标识。"""
    meta_path = Path(artifact_dir) / "model.json"
    if not meta_path.exists():
        raise ValueError(f"制品目录缺少 model.json: {artifact_dir}")
    with open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)
    fmt = meta.get("format")
    if fmt not in _LOADERS:
        raise ValueError(f"未知模型格式: {fmt!r}（允许 {sorted(_LOADERS)}）")
    return str(fmt)


def load_model_artifact(artifact_dir: str | Path) -> Any:
    """按格式标识冷加载模型（JSON/YAML/xgboost 原生格式，无 pickle）。"""
    fmt = detect_format(artifact_dir)
    return _LOADERS[fmt](Path(artifact_dir))
