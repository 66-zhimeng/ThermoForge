"""模型实验室结构校验的子进程入口（model_lab.validate_module 的子进程侧）。

用法：`python -m thermoforge_research._lab_check <candidate.py> <work_dir>`

校验项（全部通过才 ok）：

1. **接口齐全**：模块须定义 `MODEL_FORMAT`（`thermoforge.lab.` 前缀）、
   `build_model(hyperparameters, seed)`、`load_model(directory)`；
   `INPUT_ROLES` 可选（空 = 通用模型，用合成特征列）。
2. **fit/predict 可用**：合成数据（角色名即列名，值为正，兼容 1/x 类
   物理项）上训练并预测，等长且全有限。
3. **save 契约**：save() 必须写 `model.json` 且 `format` 与模块声明一致。
4. **save/load 往返**：load_model 重建后预测与保存前逐位一致。
5. **确定性**：同种子第二次 build+fit 的预测与第一次逐位一致
   （runner §7.3 的机器判定对 lab 模型同样生效）。

报告写 `<work_dir>/lab_check.json`；契约性失败退出码 2，未捕获异常 1。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

MODEL_FORMAT_PREFIX = "thermoforge.lab."

# 与 thermoforge_models.lab 的合成数据约定保持一致（本地重复一份：
# 子进程要能在不 import 建模包重依赖之外的最小面里跑；常量很小）
SYNTHETIC_ROWS = 240
GENERIC_COLUMNS = ("f1", "f2", "f3", "f4")
DISTRACTOR = "zzz_distractor"


def _write_json(path: Path, doc: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(doc, fp, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        fp.write("\n")
    os.replace(tmp, path)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tf_lab_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载候选模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_frame(roles: list[str]):
    """合成确定性数据：角色列（或通用列）+ 干扰列 + object_id/timestamp。"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    columns = list(roles) if roles else list(GENERIC_COLUMNS)
    data: dict[str, Any] = {}
    # 每列不同的正数区间：兼容 1/x、ln 类项；区间固定 → 逐位可复现
    for i, name in enumerate(columns):
        lo = 10.0 + 5.0 * i
        data[name] = rng.uniform(lo, lo + 40.0, SYNTHETIC_ROWS)
    data[DISTRACTOR] = rng.uniform(-1.0, 1.0, SYNTHETIC_ROWS)
    data["object_id"] = "SYN"
    data["timestamp"] = pd.date_range(
        "2025-01-01", periods=SYNTHETIC_ROWS, freq="15min")
    df = pd.DataFrame(data)
    y = np.full(SYNTHETIC_ROWS, 5.0)
    for i, name in enumerate(columns):
        y = y + (i + 1.0) * df[name].to_numpy(np.float64)
    return df, y


def _check_predictions(pred: Any, n: int, label: str) -> tuple[bool, str]:
    import numpy as np

    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape != (n,):
        return False, f"{label}: 预测形状 {arr.shape} 应为 ({n},)"
    if not np.all(np.isfinite(arr)):
        return False, f"{label}: 预测含 NaN/inf"
    return True, ""


def check(source_path: Path, work_dir: Path) -> dict[str, Any]:
    import numpy as np

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    module = _load_module(source_path)

    # 1. 接口
    fmt = getattr(module, "MODEL_FORMAT", None)
    roles = list(getattr(module, "INPUT_ROLES", []) or [])
    interface_ok = (
        isinstance(fmt, str) and fmt.startswith(MODEL_FORMAT_PREFIX)
        and callable(getattr(module, "build_model", None))
        and callable(getattr(module, "load_model", None))
        and all(isinstance(r, str) for r in roles)
    )
    record("interface", interface_ok,
           "" if interface_ok else
           "须定义 MODEL_FORMAT（thermoforge.lab. 前缀）、"
           "build_model(hyperparameters, seed)、load_model(directory)；"
           "INPUT_ROLES 若给必须是字符串列表")
    if not interface_ok:
        return {"ok": False, "checks": checks}

    df, y = _synthetic_frame(roles)
    hp: dict[str, Any] = {}
    if roles:
        hp["inputs"] = ";".join(roles)  # 同名映射约定（parse_inputs）

    # 2. fit/predict
    started = time.monotonic()
    model = module.build_model(hp, seed=42)
    model.fit(df, y)
    fit_seconds = time.monotonic() - started
    pred = model.predict(df)
    ok, detail = _check_predictions(pred, SYNTHETIC_ROWS, "fit/predict")
    record("fit_predict", ok, detail)
    if not ok:
        return {"ok": False, "checks": checks}
    pred = np.asarray(pred, dtype=np.float64)

    # 3. save 契约 + 4. load 往返
    model_dir = work_dir / "check_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    meta_path = model_dir / "model.json"
    save_ok = meta_path.exists()
    if save_ok:
        with open(meta_path, encoding="utf-8") as fp:
            save_ok = json.load(fp).get("format") == fmt
    record("save_contract", save_ok,
           "" if save_ok else "save() 须写 model.json 且 format 与声明一致")
    if not save_ok:
        return {"ok": False, "checks": checks}
    reloaded = module.load_model(model_dir)
    ok, detail = _check_predictions(
        reloaded.predict(df), SYNTHETIC_ROWS, "load_model 往返")
    if ok:
        diff = float(np.max(np.abs(
            np.asarray(reloaded.predict(df), dtype=np.float64) - pred)))
        ok = diff == 0.0
        detail = "" if ok else f"save/load 往返预测不一致（max|Δ|={diff:.3e}）"
    record("roundtrip", ok, detail)

    # 5. 确定性（同种子重训逐位一致）
    model2 = module.build_model(hp, seed=42)
    model2.fit(df, y)
    pred2 = np.asarray(model2.predict(df), dtype=np.float64)
    ok = bool(np.array_equal(pred, pred2))
    record("determinism", ok,
           "" if ok else "同种子重训预测不逐位一致"
           f"（max|Δ|={float(np.max(np.abs(pred - pred2))):.3e}）"
           "——随机源必须用传入的 seed")

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "model_format": fmt,
        "input_roles": roles,
        "n_rows": SYNTHETIC_ROWS,
        "fit_seconds": round(fit_seconds, 6),
    }


def main(argv: list[str]) -> int:
    source_path = Path(argv[1]).resolve()
    work_dir = Path(argv[2]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "lab_check.json"
    try:
        report = check(source_path, work_dir)
    except Exception:
        traceback.print_exc()
        _write_json(report_path, {
            "ok": False, "checks": [],
            "error": traceback.format_exc(limit=3),
        })
        return 1
    _write_json(report_path, report)
    print(f"lab_check ok={report['ok']}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
