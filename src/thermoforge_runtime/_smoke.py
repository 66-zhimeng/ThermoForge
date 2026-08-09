"""冷加载冒烟子进程（TFM-1005，implementation-notes.md §8.2/§8.3）。

由 `registry.ModelRegistry._gate_smoke` 以隔离模式启动::

    python -I -m thermoforge_runtime._smoke <package_dir>

`-I`：忽略 PYTHONPATH 与用户 site，cwd 为空临时目录——仅挂载模型包
目录。在训练进程里跑一遍推理不算冷加载（§8.3）；本子进程模拟干净
环境：重新 import 全部依赖、按 `artifact/model.json` 的格式标识加载、
对 `golden.parquet` 重算预测并与训练环境输出做容差比对
（GOLDEN_REL_TOL / GOLDEN_ABS_TOL，§7.3 跨 OS 口径 [草案]）。

stdout 最后一行为 JSON 结果行，供父进程解析；异常写 stderr。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Sequence

import pandas as pd

from .artifact import load_model_artifact
from .package import GOLDEN_ABS_TOL, GOLDEN_REL_TOL


def run(package_dir: Path) -> dict:
    """冷加载 + golden 容差比对，返回结构化结果。"""
    model = load_model_artifact(package_dir / "artifact")
    golden = pd.read_parquet(package_dir / "golden.parquet")
    input_cols = [c for c in golden.columns if not c.startswith("output__")]
    output_cols = [c for c in golden.columns if c.startswith("output__")]
    if not input_cols or not output_cols:
        return {"ok": False, "error": "golden.parquet 缺少输入或输出列"}
    preds = model.predict(golden[input_cols])
    if len(output_cols) == 1:
        pred_frame = pd.DataFrame({output_cols[0]: preds})
    else:
        pred_frame = pd.DataFrame(
            preds, columns=output_cols
        )
    max_rel = 0.0
    worst: str | None = None
    for col in output_cols:
        expected = golden[col].to_numpy(dtype="float64")
        actual = pred_frame[col].to_numpy(dtype="float64")
        for i, (a, e) in enumerate(zip(actual, expected)):
            err = abs(float(a) - float(e))
            rel = err / max(abs(float(e)), GOLDEN_ABS_TOL)
            if rel > max_rel:
                max_rel, worst = rel, f"{col}[{i}]"
            if err > GOLDEN_ABS_TOL + GOLDEN_REL_TOL * abs(float(e)):
                return {
                    "ok": False,
                    "n": int(len(golden)),
                    "max_rel_error": max_rel,
                    "error": (
                        f"golden 比对超差: {col}[{i}] "
                        f"actual={float(a)!r} expected={float(e)!r}"
                        f"（容差 rel={GOLDEN_REL_TOL} abs={GOLDEN_ABS_TOL}）"
                    ),
                }
    return {
        "ok": True,
        "n": int(len(golden)),
        "max_rel_error": max_rel,
        "worst": worst,
    }


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print(json.dumps({"ok": False, "error": "用法: _smoke <package_dir>"}))
        return 2
    try:
        result = run(Path(argv[1]))
    except Exception:
        traceback.print_exc()
        result = {"ok": False,
                  "error": traceback.format_exc(limit=3)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
