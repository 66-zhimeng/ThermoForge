"""Experiment Runner 测试（research-loop.md §5、implementation-notes §7）。

- 子进程隔离执行：预置线程环境变量 + PYTHONHASHSEED（§7.1）。
- 种子清单：未声明种子报 TFX-902（§7.2）。
- 环境指纹不符报 TFX-901（conventions §5.3）。
- 复现为机器判定：同一实验定义跑两次，指标 bit-exact（§7.3）。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from thermoforge_research.errors import ResearchError
from thermoforge_research.ledger import ResearchLedger
from thermoforge_research.runner import run_experiment, verify_reproducibility

from phase2_helpers import (
    FEATURES,
    TARGET,
    build_chiller_vault,
    make_experiment,
    view_definition,
)

ACTOR = "test-runner"


@pytest.fixture()
def env(tmp_path):
    vault, ref = build_chiller_vault(tmp_path)
    ledger = ResearchLedger(tmp_path / "research")
    ledger.create_goal("g", actor=ACTOR)
    ledger.register_view(view_definition(ref), actor=ACTOR)
    ledger.create_hypothesis("RG-0001", "h", actor=ACTOR)
    return tmp_path, vault, ref, ledger


def _register_and_run(env, exp_id: str, **kwargs):
    tmp_path, _, _, ledger = env
    experiment = make_experiment(exp_id, **kwargs)
    ledger.register_experiment(
        experiment.model_dump(by_alias=True, mode="json"), actor=ACTOR)
    report = run_experiment(
        experiment, research_root=tmp_path / "research",
        vault_root=tmp_path / "vault", ledger=ledger, actor=ACTOR,
        purge_seconds=0.0, embargo_seconds=2700.0)
    return experiment, report


def test_run_experiment_produces_artifacts(env, tmp_path):
    exp_id = "EXP-0001"
    experiment, report = _register_and_run(env, exp_id)
    assert report["status"] == "completed", report.get("error")
    assert report["exit_status"] == 0
    assert report["duration_seconds"] > 0

    exp_dir = tmp_path / "research" / "experiments" / exp_id
    for name in ("report.json", "metrics.json", "split.json",
                 "split_profile.json",
                 "seed_manifest.json", "environment.json",
                 "predictions.parquet", "physics_report.json",
                 "stdout.log", "stderr.log", "model/model.json"):
        assert (exp_dir / name).exists(), name

    # 边界时间点记录进制品，不只是比例（§4.1）
    split = json.loads((exp_dir / "split.json").read_text(encoding="utf-8"))
    for key in ("t0", "b1", "b2", "t_end", "purge_seconds",
                "embargo_seconds", "view_hash"):
        assert key in split, key
    assert split["dataset"].endswith("@rev_0001")

    # 训练/验证/测试各段的自变量+目标分布画像（回答"验证/测试集是否
    # 落在训练集见过的工况范围之外"，不只是切分的时间边界）
    split_profile = json.loads(
        (exp_dir / "split_profile.json").read_text(encoding="utf-8"))
    assert split_profile["reference"] == "train"
    subsets = split_profile["subsets"]
    assert set(subsets) == {"train", "validate", "test"}
    for name in ("train", "validate", "test"):
        assert subsets[name]["n_samples"] > 0
        variables = subsets[name]["variables"]
        assert set(variables) == {*FEATURES, TARGET}
        for entry in variables.values():
            assert entry["min"] <= entry["quantiles"]["p50"] <= entry["max"]
    for name in ("validate", "test"):
        # 越界占比只对非参照段计算，train 自身没有这个字段
        any_key = next(iter(subsets[name]["variables"].values()))
        assert "out_of_train_range_fraction" in any_key
    assert "out_of_train_range_fraction" not in next(
        iter(subsets["train"]["variables"].values()))

    # 种子清单逐一记录（§7.2）
    seeds = json.loads((exp_dir / "seed_manifest.json").read_text(
        encoding="utf-8"))
    entries = seeds["entries"]
    assert entries["random.seed"] == entries["numpy.default_rng"] == 20260808
    assert entries["xgboost.nthread"] == 1
    assert seeds["thread_env"]["OMP_NUM_THREADS"] == "1"  # 子进程预置（§7.1）

    # 指标：micro + per-object + 三个评估面口径
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    surfaces = metrics["surfaces"]
    assert surfaces["validate"]["n_samples"] > 0
    assert surfaces["A"]["n_samples"] > 0
    rmse = surfaces["A"]["metrics"]["RMSE"]
    assert rmse > 0
    assert set(surfaces["A"]["per_object"]) >= {"CH-01"}

    # Ledger 状态已进入 completed（§2 留痕）
    assert ledger_status(env, exp_id) == "completed"


def ledger_status(env, exp_id: str) -> str:
    return env[3].get(exp_id)["status"]


def test_reproducibility_bit_exact_same_machine(env, tmp_path):
    """同一实验定义跑两次：同机同 environment_lock 必须 bit-exact（§7.3）。"""
    _, report_a = _register_and_run(env, "EXP-0001")
    _, report_b = _register_and_run(env, "EXP-0002")
    assert report_a["status"] == report_b["status"] == "completed"
    verify_reproducibility(report_a["metrics"], report_b["metrics"])
    # 显式断言 bit-exact：逐值完全相等，不是近似
    flat_a = json.dumps(report_a["metrics"], sort_keys=True)
    flat_b = json.dumps(report_b["metrics"], sort_keys=True)
    assert flat_a == flat_b


def test_environment_lock_mismatch_tfx901(env):
    _, report = _register_and_run(env, "EXP-0001",
                                  environment_lock="0" * 64)
    assert report["status"] == "failed"
    assert report["error_code"] == "TFX-901"
    assert ledger_status(env, "EXP-0001") == "failed"


def test_seed_missing_tfx902(env, tmp_path):
    experiment, report = _register_and_run(env, "EXP-0001")
    assert report["status"] == "completed"
    exp_dir = tmp_path / "research" / "experiments" / "EXP-0001"
    spec_path = exp_dir / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["experiment"]["runtime"]["random_seed"] = None
    spec_path.write_text(json.dumps(spec, ensure_ascii=False),
                         encoding="utf-8", newline="\n")
    proc = subprocess.run(
        [sys.executable, "-m", "thermoforge_research._child", "spec.json"],
        cwd=exp_dir, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 2
    child = json.loads((exp_dir / "child_result.json").read_text(
        encoding="utf-8"))
    assert child["error_code"] == "TFX-902"


def test_verify_reproducibility_detects_drift():
    metrics_a = {"surfaces": {"A": {"metrics": {"RMSE": 1.0}}}}
    metrics_b = {"surfaces": {"A": {"metrics": {"RMSE": 1.0 + 1e-8}}}}
    with pytest.raises(ResearchError) as excinfo:
        verify_reproducibility(metrics_a, metrics_b)  # 同机要求 bit-exact
    assert excinfo.value.code == "TFX-901"
    # 跨 CPU 容差 1e-9：1e-8 的相对偏差同样超差
    with pytest.raises(ResearchError):
        verify_reproducibility(metrics_a, metrics_b, cross_cpu=True)
    # 容差内通过
    same = {"surfaces": {"A": {"metrics": {"RMSE": 1.0}}}}
    verify_reproducibility(metrics_a, same)


def test_relative_research_root_does_not_double_experiment_path(tmp_path, monkeypatch):
    """相对 research_root 不得让子进程把实验目录拼两遍。

    子进程的 cwd 就是 exp_dir。若 spec 里的 `experiment_dir` 是相对路径，
    子进程会把它再解析一次，得到 `<exp_dir>/<relative_root>/...` —— 目录不存在，
    实验在写 seed_manifest.json 时就崩（实测 EXP-0005）。
    """
    from thermoforge_research.tools import ToolContext

    monkeypatch.chdir(tmp_path)
    (tmp_path / "vault").mkdir()
    (tmp_path / "research").mkdir()

    ctx = ToolContext(vault_root="vault", research_root="research",
                      models_root="models")
    for root in (ctx.vault_root, ctx.research_root, ctx.models_root):
        assert root.is_absolute(), f"根目录必须是绝对路径: {root}"
    assert ctx.research_root == (tmp_path / "research").resolve()
