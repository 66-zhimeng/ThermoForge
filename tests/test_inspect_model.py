"""模型结构解析（inspect_model）与报告侧渲染的测试。

全部用手写的最小工件：六种 format 各自一个目录，验证解析结果；再用合成
Section 验证 extra_tables 在 HTML / Markdown / PDF 三种导出里都不炸。
不触网、不真跑实验（hybrid 的 booster 是现场训的 50 行小树）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from thermoforge_models.baseline import MODEL_FORMAT as LINEAR_FORMAT
from thermoforge_models.hybrid import MODEL_FORMAT as HYBRID_FORMAT
from thermoforge_models.identification import GN_FORMAT
from thermoforge_models.physics import MODEL_FORMAT as PHYSICS_V1_FORMAT
from thermoforge_webui.charts import series, static
from thermoforge_webui.export import render_html, render_markdown_bundle, render_pdf
from thermoforge_webui.services import inspect_model
from thermoforge_webui.services.reports import (
    ReportDocument,
    Section,
    _params_frame,
)


# ---------------------------------------------------------------- 最小工件


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _linear_dir(root: Path) -> Path:
    directory = root / "linear"
    directory.mkdir()
    _write_json(directory / "model.json", {
        "format": LINEAR_FORMAT,
        "method": "ridge",
        "alpha": 1.0,
        "intercept": 512.3,
        "coefficients": {"t_chws": -12.5, "t_cws": 30.1, "plr": 200.0},
        "scaler": {"t_chws": {"mean": 7.0, "std": 1.5}},
    })
    return directory


def _gordon_ng_dir(root: Path) -> Path:
    directory = root / "gn"
    directory.mkdir()
    _write_json(directory / "model.json", {
        "format": GN_FORMAT,
        "parameters": {"r_evaporator": 0.0012, "r_condenser": 0.0008,
                       "delta_s_internal": 0.45},
        "inputs": {"q_e": "cooling_load", "t_ci": "cond_in_temp"},
        "n_train": 3000,
        "q_floor": 50.0,
    })
    return directory


def _physics_v1_dir(root: Path) -> Path:
    directory = root / "physics"
    directory.mkdir()
    _write_json(directory / "model.json", {"format": PHYSICS_V1_FORMAT})
    (directory / "params.yaml").write_text(yaml.safe_dump({
        "format": PHYSICS_V1_FORMAT,
        "equations": ["Q = m·Cp·ΔT", "P = Q / COP"],
        "inputs": {"t_chws": "chilled_supply_temp"},
        "parameters": {
            "cp": {"value": 4.186, "unit": "kJ/(kg·K)", "note": "定压比热"},
            "cop_coefficients": {
                "c0": {"value": 2.1, "bounds": [0.0, 10.0]},
                "c1": {"value": 0.05},
            },
        },
        "identification": {"method": "least_squares", "n_samples": 2000,
                           "power_rel_rmse": 0.08},
    }, allow_unicode=True), encoding="utf-8")
    return directory


def _hybrid_dir(root: Path) -> Path:
    directory = root / "hybrid"
    directory.mkdir()
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    booster = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(
        x, x[:, 0] * 3 + rng.normal(scale=0.1, size=50)).get_booster()
    booster.save_model(directory / "booster.json")
    _write_json(directory / "base_model.json", {
        "format": GN_FORMAT,
        "parameters": {"r_evaporator": 0.001, "r_condenser": 0.001,
                       "delta_s_internal": 0.4},
        "inputs": {"q_e": "cooling_load"},
    })
    _write_json(directory / "model.json", {
        "format": HYBRID_FORMAT,
        "base_format": GN_FORMAT,
        "xgb_params": {"n_estimators": 5, "max_depth": 2},
        "monotone_constraints": {"q_e": 1},
        "seed": 42,
        "scaler": {"q_e": {"mean": 100.0, "std": 20.0}},
    })
    return directory


# ---------------------------------------------------------------- 解析


def test_missing_directory_returns_none(tmp_path: Path) -> None:
    assert inspect_model.load_model_doc(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inspect_model.load_model_doc(empty) is None


def test_unknown_format_falls_back(tmp_path: Path) -> None:
    directory = tmp_path / "mystery"
    directory.mkdir()
    _write_json(directory / "model.json", {"format": "acme.v9"})
    doc = inspect_model.load_model_doc(directory)
    assert doc is not None and doc.kind == "unknown"
    assert doc.raw["format"] == "acme.v9"


def test_linear_parsed(tmp_path: Path) -> None:
    doc = inspect_model.load_model_doc(_linear_dir(tmp_path))
    assert doc is not None and doc.kind == "linear"
    assert doc.coefficients == {"t_chws": -12.5, "t_cws": 30.1, "plr": 200.0}
    assert doc.intercept == 512.3
    assert "α=" in doc.summary
    assert any(p.name == "alpha" for p in doc.params)
    assert doc.equations and doc.substituted and doc.latex
    assert "200" in doc.substituted[0]


def test_gordon_ng_parsed(tmp_path: Path) -> None:
    doc = inspect_model.load_model_doc(_gordon_ng_dir(tmp_path))
    assert doc is not None and doc.kind == "gordon_ng"
    names = [p.name for p in doc.params]
    assert names == ["r_evaporator", "r_condenser", "delta_s_internal"]
    assert doc.params[0].unit == "K/kW"
    assert len(doc.equations) == 4 and len(doc.latex) == 4
    assert any("3000" in note for note in doc.notes)


def test_physics_v1_parsed_with_nested_params(tmp_path: Path) -> None:
    doc = inspect_model.load_model_doc(_physics_v1_dir(tmp_path))
    assert doc is not None and doc.kind == "physics_v1"
    names = [p.name for p in doc.params]
    # 嵌套段 cop_coefficients 要带前缀展开
    assert "cp" in names and "cop_coefficients.c0" in names
    c0 = next(p for p in doc.params if p.name == "cop_coefficients.c0")
    assert c0.bounds == (0.0, 10.0)
    assert any("最小二乘" in note for note in doc.notes)
    assert any("COP = 2.1" in line for line in doc.substituted)


def test_hybrid_parsed_with_live_booster(tmp_path: Path) -> None:
    doc = inspect_model.load_model_doc(_hybrid_dir(tmp_path))
    assert doc is not None and doc.kind == "hybrid"
    assert doc.base is not None and doc.base.kind == "gordon_ng"
    assert doc.xgb["n_trees"] == 5
    # 现场训的 booster 能算出 gain 占比，且加起来约 100%
    importance = doc.xgb["importance"]
    assert importance and abs(sum(importance.values()) - 100.0) < 1e-6
    nodes, edges = inspect_model.hybrid_flow_spec(doc)
    assert len(nodes) == 5 and (0, 1) in edges and (4, 2) in edges


# ---------------------------------------------------------------- 图数据


def test_prepare_coef_bars_orders_importance() -> None:
    bars = series.prepare_coef_bars({"a": 10.0, "b": 60.0, "c": 30.0},
                                    unit="%")
    assert bars is not None
    assert bars.labels == ["b", "c", "a"]  # unit="%" 降序
    assert bars.values == [60.0, 30.0, 10.0]


def test_prepare_coef_bars_keeps_feature_order() -> None:
    bars = series.prepare_coef_bars({"a": 1.0, "b": 9.0})
    assert bars is not None and bars.labels == ["a", "b"]  # 非 % 保持传入顺序
    assert series.prepare_coef_bars({}) is None


def test_static_pngs_are_real_pngs() -> None:
    bars = series.prepare_coef_bars({"a": 1.0, "b": -2.0})
    assert static.coef_bars_png(bars).startswith(b"\x89PNG")
    nodes, edges = [(0.0, 1.0, "输入"), (1.0, 1.0, "输出")], [(0, 1)]
    assert static.structure_flow_png(nodes, edges).startswith(b"\x89PNG")


# ---------------------------------------------------------------- 报告渲染


def _document_with_structure() -> ReportDocument:
    frame = pd.DataFrame([{"参数": "r_evaporator", "取值": 0.0012,
                           "单位": "K/kW", "合法范围": "—", "备注": "蒸发器热阻"}])
    section = Section(
        title="实验 EXP-9001",
        paragraphs=["建模方式：测试。",
                    "```text\nP = Q_c - Q_e\n```"],
        extra_tables=[("模型参数（辨识取值与合法范围）", frame)])
    return ReportDocument(title="结构测试", subtitle="",
                          generated_at="2026-01-01", sections=[section],
                          footer="尾")


def test_extra_tables_render_in_all_formats() -> None:
    document = _document_with_structure()
    html = render_html(document)
    assert "r_evaporator" in html and "模型参数" in html
    assert "<pre>" in html and "```" not in html

    payload = render_markdown_bundle(document, basename="report")
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read("report.md").decode("utf-8")
    assert "r_evaporator" in text and "```text" in text

    pdf = render_pdf(document)
    assert pdf.startswith(b"%PDF")


def test_params_frame_shape(tmp_path: Path) -> None:
    doc = inspect_model.load_model_doc(_gordon_ng_dir(tmp_path))
    frame = _params_frame(doc)
    assert list(frame.columns) == ["参数", "取值", "单位", "合法范围", "备注"]
    assert len(frame) == 3
