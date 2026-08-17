"""模型制品加载（implementation-notes.md §8.1）。

按 `artifact/model.json` 的 `format` 字段分发到 Phase 2 的原生格式加载器：
系数 JSON（线性基线）、YAML 明文参数（物理模型）、XGBoost 原生 .json
（残差混合）。不使用 pickle（跨版本不可加载 + 任意代码执行）。

模型实验室（`thermoforge.lab.` 前缀）的制品自带冻结源码
`artifact/lab_source.py`（实验时由 _child 快照进 model/），加载即导入该
文件并调其 `load_model()`——包自包含，不依赖实验室存储。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from thermoforge_models import baseline, hybrid, physics
from thermoforge_models.lab import MODEL_FORMAT_PREFIX

_LOADERS = {
    baseline.MODEL_FORMAT: baseline.LinearBaseline.load,
    physics.MODEL_FORMAT: physics.ChillerPhysicsModel.load,
    hybrid.MODEL_FORMAT: hybrid.ResidualHybrid.load,
}

LAB_SOURCE_NAME = "lab_source.py"


def _load_lab_artifact(artifact_dir: Path) -> Any:
    """导入打包的实验室源码并调其 load_model()（发布包自包含路径）。"""
    source = artifact_dir / LAB_SOURCE_NAME
    if not source.is_file():
        raise ValueError(
            f"实验室制品缺少 {LAB_SOURCE_NAME}: {artifact_dir}"
            "（lab 模型的方程本体是代码，必须随包发布）"
        )
    spec = importlib.util.spec_from_file_location("tf_lab_packaged", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载实验室源码: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(artifact_dir)


def detect_format(artifact_dir: str | Path) -> str:
    """读取制品目录的模型格式标识。"""
    meta_path = Path(artifact_dir) / "model.json"
    if not meta_path.exists():
        raise ValueError(f"制品目录缺少 model.json: {artifact_dir}")
    with open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)
    fmt = meta.get("format")
    if not (fmt in _LOADERS or str(fmt).startswith(MODEL_FORMAT_PREFIX)):
        raise ValueError(f"未知模型格式: {fmt!r}（允许 {sorted(_LOADERS)} "
                         f"或 {MODEL_FORMAT_PREFIX}* 实验室格式）")
    return str(fmt)


def load_model_artifact(artifact_dir: str | Path) -> Any:
    """按格式标识冷加载模型（JSON/YAML/xgboost 原生格式，无 pickle）。"""
    fmt = detect_format(artifact_dir)
    if fmt.startswith(MODEL_FORMAT_PREFIX):
        return _load_lab_artifact(Path(artifact_dir))
    return _LOADERS[fmt](Path(artifact_dir))
