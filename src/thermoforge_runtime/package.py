"""Model Package 构建与校验（model-package.md §2、implementation-notes.md §8）。

目录结构（§2）::

    <package_dir>/
    ├── model.yaml            # TFMP 契约元数据（含签名与约束）
    ├── signature.yaml        # 模型签名 + history_required/cold_start 扩展（§9.1）
    ├── artifact/             # 模型制品（JSON/YAML/xgboost 原生格式，无 pickle）
    ├── preprocessing.yaml    # 特征顺序 + scaler / 物理输入映射
    ├── constraints.yaml      # 输入范围与超范围策略（§4）
    ├── metrics.json          # 实验指标（分测试面）
    ├── validation.json       # 物理验证与切分记录
    ├── dataset-lineage.json  # 数据谱系（revision + view_hash）
    ├── research-lineage.json # 研究谱系（goal/hypothesis/experiment）
    ├── environment.lock      # 训练环境指纹（JSON）
    ├── checksums.json        # 全文件 sha256
    ├── README.md             # 人类可读摘要
    └── golden.parquet        # golden 预测集（§8.2，覆盖工况边界）

- `checksums.json` 记录包内每个文件（自身除外）的 sha256；
  `verify_package` 重算比对，文件缺失/被篡改/未登记报 **TFM-1002**。
- golden 预测集 = 固定输入样本 + 训练环境输出，输入样本按每个特征的
  min/max 工况边界选取（`boundary_rows`）；冷加载冒烟（TFM-1005）重算比对。
- 所有文本写入 encoding="utf-8"、newline="\\n"（§10.3）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from thermoforge_core.contracts.model_package import (
    Constraints,
    ModelPackage,
    ModelSignature,
)

from .artifact import load_model_artifact
from .errors import ModelRegistryError

PACKAGE_REQUIRED_FILES: tuple[str, ...] = (
    "model.yaml",
    "signature.yaml",
    "preprocessing.yaml",
    "constraints.yaml",
    "metrics.json",
    "validation.json",
    "dataset-lineage.json",
    "research-lineage.json",
    "environment.lock",
    "checksums.json",
    "README.md",
    "golden.parquet",
)

GOLDEN_REL_TOL = 1e-6  # golden 比对容差（§7.3 跨 OS 口径 [草案]）
GOLDEN_ABS_TOL = 1e-9

# signature.yaml 在 ModelSignature 契约之外的扩展键（implementation-notes §9.1）
SIGNATURE_EXTENSION_KEYS = ("history_required", "cold_start")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_",
                               suffix=path.suffix or ".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(text)
    os.replace(tmp, path)


def _write_json(path: Path, doc: Any) -> None:
    _write_text_atomic(
        path, json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False)
        + "\n",
    )


def _write_yaml(path: Path, doc: Any) -> None:
    _write_text_atomic(
        path, yaml.safe_dump(doc, allow_unicode=True, sort_keys=True)
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- golden 样本


def boundary_rows(
    df: pd.DataFrame, features: Sequence[str]
) -> list[dict[str, float]]:
    """golden 输入样本：每个特征的 min/max 所在整行 + 首行（覆盖工况边界）。

    implementation-notes §14 待决策 #6：样本选取覆盖工况边界而非随机抽样。
    """
    rows: list[dict[str, float]] = []
    seen: set[int] = set()
    idxs: list[int] = [0]
    for feat in features:
        col = pd.to_numeric(df[feat], errors="coerce")
        idxs.extend([int(col.idxmin()), int(col.idxmax())])
    for idx in idxs:
        if idx in seen:
            continue
        seen.add(idx)
        row = df.loc[idx]
        rows.append({f: float(row[f]) for f in features})
    return rows


# ---------------------------------------------------------------- 构建


def split_signature_doc(doc: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 signature 文档拆为契约部分与扩展部分（history_required/cold_start）。"""
    base = {k: v for k, v in doc.items() if k not in SIGNATURE_EXTENSION_KEYS}
    ext = {k: doc[k] for k in SIGNATURE_EXTENSION_KEYS if k in doc}
    return base, ext


def build_model_package(
    dest_dir: str | Path,
    *,
    model_id: str,
    version: str,
    signature: Mapping[str, Any],
    artifact_dir: str | Path,
    metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    research_lineage: Mapping[str, Any],
    environment: Mapping[str, Any],
    golden_inputs: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any] | None = None,
    goal_id: str | None = None,
    experiment_id: str | None = None,
    description: str | None = None,
) -> Path:
    """构建模型包目录并返回路径。`dest_dir` 必须不存在或为空。"""
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"模型包目标目录非空: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    sig_base, sig_ext = split_signature_doc(signature)
    sig_model = ModelSignature(**sig_base)
    if sig_model.model_id != model_id or sig_model.version != version:
        raise ValueError("signature 的 model_id/version 与参数不一致")
    constraints_model = Constraints(**dict(constraints or {}))

    package = ModelPackage(
        model_id=model_id,
        version=version,
        status="candidate",
        signature=sig_model,
        constraints=constraints_model,
        goal_id=goal_id,
        experiment_id=experiment_id,
        description=description,
    )

    # artifact/：整体复制实验模型制品
    src_artifact = Path(artifact_dir)
    if not (src_artifact / "model.json").exists():
        raise ValueError(f"制品目录缺少 model.json: {src_artifact}")
    shutil.copytree(src_artifact, dest / "artifact")

    # golden 预测集（§8.2）：固定输入 + 训练环境输出
    if not golden_inputs:
        raise ValueError("golden_inputs 不能为空（golden 预测集是强制制品）")
    feature_order = [p.property_code for p in sig_model.inputs]
    golden_df = pd.DataFrame([dict(r) for r in golden_inputs])
    missing = [f for f in feature_order if f not in golden_df.columns]
    if missing:
        raise ValueError(f"golden 输入缺少签名特征: {missing}")
    model = load_model_artifact(dest / "artifact")
    preds = model.predict(golden_df[feature_order])
    for port in sig_model.outputs:
        golden_df[f"output__{port.property_code}"] = preds
    golden_path = dest / "golden.parquet"
    golden_df.to_parquet(golden_path, compression="zstd", index=False)

    # preprocessing.yaml：特征顺序 + scaler / 物理输入映射（§8.1）
    with open(dest / "artifact" / "model.json", encoding="utf-8") as fp:
        artifact_meta = json.load(fp)
    preprocessing: dict[str, Any] = {"feature_order": feature_order}
    if "scaler" in artifact_meta:
        preprocessing["scaler"] = artifact_meta["scaler"]
    if "inputs" in artifact_meta:
        preprocessing["inputs"] = artifact_meta["inputs"]
    if (src_artifact / "params.yaml").exists():
        with open(src_artifact / "params.yaml", encoding="utf-8") as fp:
            params_doc = yaml.safe_load(fp)
        if isinstance(params_doc, dict) and params_doc.get("inputs"):
            preprocessing["inputs"] = params_doc["inputs"]

    sig_doc = dict(sig_base)
    sig_doc.update(sig_ext)

    _write_yaml(dest / "model.yaml",
                package.model_dump(mode="json", exclude_none=True))
    _write_yaml(dest / "signature.yaml", sig_doc)
    _write_yaml(dest / "preprocessing.yaml", preprocessing)
    _write_yaml(dest / "constraints.yaml",
                constraints_model.model_dump(mode="json", exclude_none=True))
    _write_json(dest / "metrics.json", dict(metrics))
    _write_json(dest / "validation.json", dict(validation))
    _write_json(dest / "dataset-lineage.json", dict(dataset_lineage))
    _write_json(dest / "research-lineage.json", dict(research_lineage))
    _write_json(dest / "environment.lock", dict(environment))
    _write_text_atomic(dest / "README.md", _render_readme(package, sig_ext))
    write_checksums(dest)
    return dest


def _render_readme(package: ModelPackage, sig_ext: Mapping[str, Any]) -> str:
    sig = package.signature
    lines = [
        f"# {package.model_id} {package.version}",
        "",
        f"- object_model: {sig.object_model}",
        f"- inputs: {', '.join(p.property_code for p in sig.inputs)}",
        f"- outputs: {', '.join(p.property_code for p in sig.outputs)}",
        f"- goal_id: {package.goal_id or '-'}",
        f"- experiment_id: {package.experiment_id or '-'}",
    ]
    if sig_ext:
        lines.append(f"- signature extensions: {sorted(sig_ext)}")
    if package.description:
        lines += ["", package.description]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 校验


def compute_checksums(package_dir: str | Path) -> dict[str, str]:
    """包内每个文件（checksums.json 除外）的 sha256，键为相对路径。"""
    root = Path(package_dir)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "checksums.json":
            continue
        out[rel] = _sha256_file(path)
    return out


def write_checksums(package_dir: str | Path) -> None:
    """重算并原子重写 checksums.json（构建/测试修复用）。"""
    _write_json(Path(package_dir) / "checksums.json",
                {"algorithm": "sha256", "files": compute_checksums(package_dir)})


def verify_package(package_dir: str | Path) -> dict[str, Any]:
    """文件齐全 + 校验和一致（model-package §8），失败报 TFM-1002。"""
    root = Path(package_dir)
    if not root.is_dir():
        raise ModelRegistryError("TFM-1002", f"模型包目录不存在: {root}")
    for rel in PACKAGE_REQUIRED_FILES:
        if not (root / rel).is_file():
            raise ModelRegistryError(
                "TFM-1002", f"模型包缺少必需文件: {rel}（§2）"
            )
    if not any((root / "artifact").iterdir()):
        raise ModelRegistryError("TFM-1002", "模型包 artifact/ 为空（§2）")
    with open(root / "checksums.json", encoding="utf-8") as fp:
        recorded = json.load(fp).get("files", {})
    actual = compute_checksums(root)
    missing = sorted(set(recorded) - set(actual))
    extra = sorted(set(actual) - set(recorded))
    mismatch = sorted(
        rel for rel in set(recorded) & set(actual)
        if recorded[rel] != actual[rel]
    )
    if missing or extra or mismatch:
        raise ModelRegistryError(
            "TFM-1002",
            f"模型包校验和不符: missing={missing[:3]} extra={extra[:3]} "
            f"mismatch={mismatch[:3]}",
        )
    return {"ok": True, "files": len(actual)}


# ---------------------------------------------------------------- 读取


def load_signature(
    package_dir: str | Path,
) -> tuple[ModelSignature, dict[str, Any] | None, str | None]:
    """读取 signature.yaml，返回 (契约签名, history_required, cold_start)。"""
    with open(Path(package_dir) / "signature.yaml", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    base, ext = split_signature_doc(doc)
    return (
        ModelSignature(**base),
        ext.get("history_required"),
        ext.get("cold_start"),
    )


def load_constraints(package_dir: str | Path) -> Constraints:
    with open(Path(package_dir) / "constraints.yaml", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp) or {}
    return Constraints(**doc)


def load_model_meta(package_dir: str | Path) -> ModelPackage:
    with open(Path(package_dir) / "model.yaml", encoding="utf-8") as fp:
        return ModelPackage(**yaml.safe_load(fp))
