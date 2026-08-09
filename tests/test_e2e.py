"""端到端最小闭环（implementation-notes §13.6）。

fixture 数据走完：导入 → Vault → View 物化 → Ledger 登记 →
实验（残差混合 + 留一设备）→ 指标 → 物理验证 → 结构化报告 → 发现入库。
"""

from __future__ import annotations

import json

import pandas as pd

from thermoforge_data.views import materialize_view
from thermoforge_research.ledger import ResearchLedger
from thermoforge_research.runner import run_experiment

from phase2_helpers import (
    OBJECTS,
    build_chiller_vault,
    make_experiment,
    view_definition,
)

ACTOR = "e2e"


def test_end_to_end_minimal_loop(tmp_path):
    # 导入 → Vault（Phase 1 链路）
    vault, ref = build_chiller_vault(tmp_path)
    assert ref == "TEST02_RESEARCH@rev_0001"

    # View 物化（长表 + 缓存）
    view_def = view_definition(ref)
    mv = materialize_view(vault, view_def, tmp_path / "research" / "view_cache")
    assert mv.table.num_rows > 0
    assert set(OBJECTS) <= set(mv.table.column("object_id").to_pylist())

    # Ledger：goal → hypothesis → view → experiment（留一设备：CH-02）
    ledger = ResearchLedger(tmp_path / "research")
    goal = ledger.create_goal("冷水机输入功率模型", actor=ACTOR)
    view = ledger.register_view(view_def, actor=ACTOR)
    hyp = ledger.create_hypothesis(
        goal["id"], "残差混合在未见设备上保持精度", actor=ACTOR)
    exp_id = ledger.allocator.allocate("EXP-")
    experiment = make_experiment(
        exp_id, goal_id=goal["id"], hypothesis_id=hyp["id"],
        view_id=view["id"], category="hybrid", holdout=["CH-02"])
    ledger.register_experiment(
        experiment.model_dump(by_alias=True, mode="json"), actor=ACTOR)

    # 实验执行（子进程隔离）
    report = run_experiment(
        experiment, research_root=tmp_path / "research",
        vault_root=tmp_path / "vault", ledger=ledger, actor=ACTOR,
        purge_seconds=0.0, embargo_seconds=2700.0)
    assert report["status"] == "completed", report.get("error")

    # 三维测试面（§4.3）：A 已见×未来 / B 未见×已见 / C 未见×未来
    surfaces = report["metrics"]["surfaces"]
    assert surfaces["A"]["n_samples"] > 0
    assert surfaces["B"]["n_samples"] > 0
    assert surfaces["C"]["n_samples"] > 0
    assert surfaces["A"]["metrics"]["CVRMSE"] is not None
    assert "NMBE" in surfaces["C"]["metrics"]  # DD-14 必报偏差指标
    # DD-07 配套：残差混合必须同时报告物理主干单独指标与残差占比
    assert "physics_only" in report["metrics"]
    assert report["metrics"]["residual_share"]["A"] >= 0.0

    # 物理验证报告：硬约束 + 总体口径
    physics = report["physics"]
    assert physics is not None
    assert "power_within_rated" in physics["hard_constraints"]
    assert physics["overall_rate"] >= 0.0

    # 预测结果制品可读且行数与指标样本数一致
    preds = pd.read_parquet(
        tmp_path / "research" / "experiments" / exp_id / "predictions.parquet")
    assert set(preds["surface"]) == {"validate", "A", "B", "C"}
    assert len(preds) == sum(s["n_samples"] for s in surfaces.values())

    # 结构化实验报告 + Ledger 终态
    assert report["conclusion"]["dataset_revision"] == ref
    assert report["conclusion"]["view_hash"] == mv.view_hash
    assert ledger.get(exp_id)["status"] == "completed"

    # 失败/成功的证据回流：发现引用实验（§8）
    finding = ledger.create_finding(
        "残差混合在面 C 上 CVRMSE 可接受", actor=ACTOR,
        supported_by=[exp_id], hypothesis_id=hyp["id"])
    assert [e["id"] for e in ledger.experiments_supporting(finding["id"])] == [
        exp_id]
    model = ledger.register_model(
        "residual-hybrid-v1", exp_id, actor=ACTOR,
        metrics=surfaces["C"]["metrics"],
        artifact_path=f"experiments/{exp_id}/model/")
    assert model["status"] == "candidate"
    progress = ledger.goal_progress(goal["id"])
    assert progress["experiments"]["by_status"] == {"completed": 1}
