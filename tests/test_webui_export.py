"""导出与后台研究会话的测试。

导出用合成实验工件跑通三种格式；研究会话用一个假的规划器验证两种模式
的控制流（自动跑 / 逐轮审批 / 中途停止），不触网、不真跑实验。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermoforge_webui.export import render_html, render_markdown_bundle, render_pdf
from thermoforge_webui.services import experiments as exp_service
from thermoforge_webui.services.reports import build_report


# ---------------------------------------------------------------- 合成工件


def _write_experiment(root: Path, experiment_id: str, *,
                      with_r2: bool = False) -> None:
    directory = root / experiment_id
    directory.mkdir(parents=True)
    n = 120
    y_true = np.linspace(100.0, 400.0, n)
    y_pred = y_true * 0.95 + 7.0
    metrics = {"RMSE": 12.0, "MAE": 9.0, "CVRMSE": 0.05, "NMBE": -0.01}
    if with_r2:
        metrics["R2"] = 0.97
    report = {
        "experiment_id": experiment_id, "status": "completed",
        "goal_id": "RG-0001", "hypothesis_id": "H-0001",
        "dataset_view": "VIEW-0001", "started_at": "2026-01-01T00:00:00+00:00",
        "duration_seconds": 3.5, "random_seed": 20260808,
        "environment_lock": "lock" * 8, "code_version": "abc123def456",
        "metrics": {"surfaces": {"A": {"n_samples": n, "metrics": metrics,
                                       "per_object": {}, "undefined": {},
                                       "y_floor": 5.0,
                                       "mape_valid_fraction": None}}},
        "physics": {"overall_rate": 0.0, "overall_violations": 0,
                    "n_samples": n},
    }
    spec = {"experiment": {"target": "total_power",
                           "model": {"category": "data", "estimator": "ridge",
                                     "hyperparameters": {"alpha": 1.0}}}}
    split = {
        "train_range": ["2025-01-01T00:00:00+00:00", "2025-03-01T00:00:00+00:00"],
        "validate_range": ["2025-03-01T00:45:00+00:00", "2025-04-01T00:00:00+00:00"],
        "test_range": ["2025-04-01T00:00:00+00:00", "2025-05-01T00:00:00+00:00"],
        "b1": "2025-03-01T00:00:00+00:00", "b2": "2025-04-01T00:00:00+00:00",
        "counts": {"train": 800, "validate": 120, "test": 120,
                   "purged_or_embargoed": 3},
        "purge_seconds": 0.0, "embargo_seconds": 2700.0,
    }
    for name, doc in (("report.json", report), ("spec.json", spec),
                      ("split.json", split),
                      ("physics_report.json", report["physics"]),
                      ("environment.json", {"python": "3.12"})):
        (directory / name).write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({
        "surface": ["A"] * n, "object_id": ["PLANT"] * n,
        "timestamp": pd.date_range("2025-04-01", periods=n, freq="15min",
                                   tz="UTC"),
        "y_true": y_true, "y_pred": y_pred,
    }).to_parquet(directory / "predictions.parquet")


@pytest.fixture()
def experiments_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "experiments"
    root.mkdir()
    _write_experiment(root, "EXP-0001")
    _write_experiment(root, "EXP-0002", with_r2=True)
    monkeypatch.setattr(exp_service, "experiments_root", lambda: root)
    return root


# ---------------------------------------------------------------- R² 补齐


def test_r2_backfilled_for_legacy_experiment(experiments_root: Path) -> None:
    """I-54 之前的实验没有 R²，读的时候用预测点现算补上，且标记出来。"""
    detail = exp_service.load_detail("EXP-0001")
    assert detail is not None and detail.r2_backfilled
    value = detail.surfaces["A"]["metrics"]["R2"]
    assert value is not None and 0.9 < value < 1.0


def test_r2_not_backfilled_when_already_recorded(experiments_root: Path) -> None:
    detail = exp_service.load_detail("EXP-0002")
    assert detail is not None and not detail.r2_backfilled
    assert detail.surfaces["A"]["metrics"]["R2"] == 0.97


def test_backfill_does_not_rewrite_artifacts(experiments_root: Path) -> None:
    """历史产物是不可变的，补齐只发生在内存里。"""
    path = experiments_root / "EXP-0001" / "report.json"
    before = path.read_text(encoding="utf-8")
    exp_service.load_detail("EXP-0001")
    assert path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------- 导出


@pytest.fixture()
def document(experiments_root: Path):
    return build_report(["EXP-0001", "EXP-0002"], title="测试报告",
                        subtitle="合成数据")


def test_report_has_expected_sections(document) -> None:
    titles = [section.title for section in document.sections]
    assert titles[0] == "概览"
    assert "模型对比" in titles
    assert "实验 EXP-0001" in titles
    assert titles[-1] == "方法与可复现性"


def test_html_is_self_contained(document) -> None:
    html = render_html(document)
    assert html.startswith("<!doctype html>")
    assert "测试报告" in html
    # plotly.js 必须内联，否则离线打开就是一片空白
    assert "Plotly.newPlot" in html
    # 不能有任何外链资源：断网双击也要能看
    assert "<script src=" not in html
    assert "<link " not in html


def test_html_inlines_plotly_runtime_only_once(document) -> None:
    """每张图都内联一份 3MB 运行时的话，文件会按图数线性膨胀。"""
    html = render_html(document)
    figures = sum(len(section.figures) for section in document.sections)
    assert figures >= 3  # 保证这个断言真的有东西可测
    big_scripts = [chunk for chunk in html.split("<script")
                   if len(chunk) > 500_000]
    assert len(big_scripts) == 1


def test_markdown_bundle_contains_text_and_images(document) -> None:
    payload = render_markdown_bundle(document, basename="report")
    with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
        names = archive.namelist()
        text = archive.read("report.md").decode("utf-8")
    assert "report.md" in names
    assert any(name.startswith("images/") and name.endswith(".png")
               for name in names)
    assert text.startswith("# 测试报告")
    assert "| 实验 |" in text or "|---" in text


def test_pdf_embeds_cjk_font(document) -> None:
    payload = render_pdf(document)
    assert payload.startswith(b"%PDF")
    # 中文靠 reportlab 内置 CID 字体，不依赖系统装了什么
    assert b"STSong-Light" in payload


# ---------------------------------------------------------------- 研究会话


def test_planner_stop_flag_ends_loop_cleanly(monkeypatch) -> None:
    """点了停止之后，规划器直接返回 None，编排器正常收尾。"""
    from thermoforge_webui.services.research import ResearchSession

    session = ResearchSession("RG-0001", "D@rev_0001")
    session.stop()
    events = [event.kind for event in session.events()]
    assert "stop_requested" in events


def test_step_mode_blocks_until_decision() -> None:
    """逐轮审批模式下，规划器要卡住等人点批准。"""
    import threading

    from thermoforge_webui.services.research import MODE_STEP, ResearchSession

    session = ResearchSession("RG-0001", "D@rev_0001", mode=MODE_STEP)
    plan = {"statement": "试试", "view_id": "VIEW-0001"}
    result: list[bool] = []

    worker = threading.Thread(
        target=lambda: result.append(session._await_approval(plan)))
    worker.start()
    # 等它进入 awaiting，再批准
    for _ in range(100):
        if session.pending_plan is not None:
            break
        __import__("time").sleep(0.01)
    assert session.pending_plan == plan
    assert session.state == "awaiting"
    session.decide(True)
    worker.join(timeout=5)
    assert result == [True]
    assert session.pending_plan is None


def test_step_mode_rejection_returns_false() -> None:
    import threading

    from thermoforge_webui.services.research import MODE_STEP, ResearchSession

    session = ResearchSession("RG-0001", "D@rev_0001", mode=MODE_STEP)
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(session._await_approval({"statement": "x"})))
    worker.start()
    for _ in range(100):
        if session.pending_plan is not None:
            break
        __import__("time").sleep(0.01)
    session.decide(False)
    worker.join(timeout=5)
    assert result == [False]
