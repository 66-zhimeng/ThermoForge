"""Experiment Runner（research-loop.md §5、implementation-notes.md §7）。

按 Experiment 契约（contracts/experiment）执行实验：

- **子进程隔离**（§7.1）：`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `MKL_NUM_THREADS` / `PYTHONHASHSEED` 由父进程预置进子进程环境，
  在 numpy/sklearn 被 import 之前生效。
- **种子清单**（§7.2）：random / np.random.default_rng / sklearn
  random_state / xgboost seed+nthread 逐一设置并记录；未声明种子报
  TFX-902。
- **环境指纹**（conventions §5.3）：`environment_lock` 不符报 TFX-901；
  CPU 型号、核数、线程变量记录但不纳入哈希。
- **复现为机器判定**（§7.3）：同机同 environment_lock 重跑要求指标
  bit-exact；跨 CPU 相对误差 ≤ 1e-9；超出报 TFX-901。
- 记录：数据集 revision + view_hash、代码版本（git rev）、环境指纹、
  超参、stdout/stderr/退出状态/时长；产出模型制品、指标、图表数据
  （per-surface 预测序列）、预测结果、验证报告、结构化实验报告。
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from thermoforge_core.canonical import sha256_hex
from thermoforge_core.contracts.experiment import Experiment
from thermoforge_core.fingerprint import current_platform_tag, environment_lock

from .errors import ResearchError
from .ledger import ResearchLedger
from .splits import DEFAULT_EMBARGO_SECONDS

# §7.1：必须在子进程 import numpy 之前生效，故由父进程预置
THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

REPRO_TOLERANCE_CROSS_CPU = 1e-9  # §7.3 [草案]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_environment_lock() -> tuple[str, dict[str, Any]]:
    """当前环境指纹（conventions §5.3）。

    返回 (environment_lock, environment_doc)。包条目 sha256 取
    `name==version` 的摘要作为锁内身份（分发包内容哈希成本高，
    版本锁定由 uv.lock 保证）。CPU 型号、核数、线程变量记录但不入哈希。
    """
    packages = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if not name:
            continue
        packages.append({
            "name": name,
            "version": dist.version,
            "sha256": sha256_hex(f"{name}=={dist.version}"),
        })
    lock = environment_lock(platform.python_version(),
                            current_platform_tag(), packages)
    doc = {
        "environment_lock": lock,
        "python": platform.python_version(),
        "platform": current_platform_tag(),
        "packages": sorted(packages, key=lambda p: p["name"]),
        # §5.3：以下记录但不纳入哈希
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in sorted(THREAD_ENV_VARS)},
    }
    return lock, doc


def _git_rev(cwd: Path) -> str | None:
    """代码版本（research-loop §5）；非 git 环境返回 None。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _write_json(path: Path, doc: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(doc, fp, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        fp.write("\n")
    os.replace(tmp, path)


def run_experiment(
    experiment: Experiment,
    *,
    research_root: str | Path,
    vault_root: str | Path,
    ledger: ResearchLedger | None = None,
    view_definition: Mapping[str, Any] | None = None,
    purge_seconds: float = 0.0,
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
    y_floor: float | None = None,
    actor: str = "experiment-runner",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """执行一个实验，返回结构化实验报告（同时写 report.json）。

    失败不抛异常：报告 `status: "failed"` + `error_code`，由 Ledger/调用方
    判定（DD-13）。TFX-904 等契约性违规仍抛 `ResearchError`。
    """
    research_root = Path(research_root)
    exp_id = experiment.experiment_id
    # 必须绝对：子进程的 cwd 就是 exp_dir，相对路径会在子进程内被二次解析，
    # 拼成 `<exp_dir>/<relative_root>/experiments/<exp_id>/...` 而找不到目录。
    exp_dir = (research_root / "experiments" / exp_id).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    if view_definition is None:
        if ledger is None:
            raise ValueError("未提供 view_definition 时必须提供 ledger")
        view_definition = ledger.get(experiment.dataset_view)["definition"]

    lock, env_doc = current_environment_lock()
    spec = {
        "experiment": experiment.model_dump(by_alias=True, mode="json"),
        "view_definition": dict(view_definition),
        "vault_root": str(vault_root),
        "view_cache_root": str(research_root / "view_cache"),
        "experiment_dir": str(exp_dir),
        "purge_seconds": purge_seconds,
        "embargo_seconds": embargo_seconds,
        "y_floor": y_floor,
        "code_version": _git_rev(Path.cwd()),
        "parent_environment_lock": lock,
    }
    _write_json(exp_dir / "spec.json", spec)
    # 清除上次运行的残留结果，避免子进程早夭时读到陈旧状态
    for stale in ("child_result.json", "metrics.json"):
        stale_path = exp_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    if ledger is not None:
        ledger.transition(
            exp_id, "running", reason="runner 启动子进程", actor=actor,
            inputs=[f"experiments/{exp_id}/spec.json"],
        )

    env = dict(os.environ)
    env.update(THREAD_ENV_VARS)
    env["PYTHONHASHSEED"] = str(experiment.runtime.random_seed)

    started = time.monotonic()
    started_at = _utcnow()
    proc = subprocess.run(
        [sys.executable, "-m", "thermoforge_research._child", "spec.json"],
        cwd=exp_dir, env=env, capture_output=True, text=True,
        timeout=timeout_seconds, encoding="utf-8", errors="replace",
    )
    duration = time.monotonic() - started

    (exp_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8",
                                        newline="\n")
    (exp_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8",
                                        newline="\n")

    child_result: dict[str, Any] = {}
    result_path = exp_dir / "child_result.json"
    if result_path.exists():
        with open(result_path, encoding="utf-8") as fp:
            child_result = json.load(fp)

    status = "completed" if (
        proc.returncode == 0 and child_result.get("status") == "completed"
    ) else "failed"
    error_code = child_result.get("error_code")
    error_message = child_result.get("error") or (
        proc.stderr.strip().splitlines()[-1] if proc.returncode != 0
        and proc.stderr.strip() else None
    )

    report: dict[str, Any] = {
        "experiment_id": exp_id,
        "goal_id": experiment.goal_id,
        "hypothesis_id": experiment.hypothesis_id,
        "dataset_view": experiment.dataset_view,
        "status": status,
        "error_code": error_code,
        "error": error_message,
        "started_at": started_at,
        "duration_seconds": duration,
        "exit_status": proc.returncode,
        "code_version": spec["code_version"],
        "environment_lock": lock,
        "random_seed": experiment.runtime.random_seed,
        "artifacts": child_result.get("artifacts", {}),
        "metrics": child_result.get("metrics"),
        "physics": child_result.get("physics"),
        "rolling_cv": child_result.get("rolling_cv"),
        "conclusion": child_result.get("conclusion"),
        "failure_reason": error_message if status == "failed" else None,
        "next_questions": child_result.get("next_questions", []),
    }
    _write_json(exp_dir / "report.json", report)

    if ledger is not None:
        ledger.transition(
            exp_id, status,
            reason=("实验完成" if status == "completed"
                    else f"实验失败: {error_code or 'CHILD_ERROR'}"),
            actor=actor,
            inputs=[f"experiments/{exp_id}/spec.json"],
            outputs=[f"experiments/{exp_id}/report.json"],
            error=error_message if status == "failed" else None,
        )
    return report


# ---------------------------------------------------------------- 复现判定


def _flatten_metrics(doc: Any, prefix: str = "") -> dict[str, float | None]:
    """把 metrics.json 中的数值叶子展平为 {路径: 值}（None 原样保留）。"""
    flat: dict[str, float | None] = {}
    if isinstance(doc, dict):
        for key, value in doc.items():
            flat.update(_flatten_metrics(value, f"{prefix}{key}."))
    elif isinstance(doc, (int, float)) and not isinstance(doc, bool):
        flat[prefix[:-1]] = float(doc)
    elif doc is None:
        flat[prefix[:-1]] = None
    return flat


def verify_reproducibility(
    metrics_a: Mapping[str, Any],
    metrics_b: Mapping[str, Any],
    *,
    cross_cpu: bool = False,
) -> None:
    """复现的机器判定（§7.3）：同机 bit-exact，跨 CPU 相对误差 ≤ 1e-9。

    超出容差报 TFX-901。`metrics_a/b` 为两次运行的 metrics.json 内容。
    """
    rel_tol = REPRO_TOLERANCE_CROSS_CPU if cross_cpu else 0.0
    flat_a = _flatten_metrics(metrics_a)
    flat_b = _flatten_metrics(metrics_b)
    if set(flat_a) != set(flat_b):
        raise ResearchError(
            "TFX-901",
            f"两次运行的指标结构不一致: "
            f"{sorted(set(flat_a) ^ set(flat_b))[:5]}",
        )
    for key in sorted(flat_a):
        va, vb = flat_a[key], flat_b[key]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            raise ResearchError("TFX-901", f"指标 {key} 一次未定义一次有值")
        scale = max(abs(va), abs(vb), 1e-300)
        if abs(va - vb) / scale > rel_tol:
            raise ResearchError(
                "TFX-901",
                f"指标 {key} 复现超差: {va!r} vs {vb!r}"
                f"（容差 rel_tol={rel_tol}，§7.3）",
            )
