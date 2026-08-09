"""模型注册、状态机与发布门禁（model-package §5/§8、conventions §7.9）。

- 同版本内容变化 → TFM-1006；同内容重复注册幂等；
- 非法状态转换拒绝；
- 硬性验收不满足 → TFM-1003（不得发布）；
- 签名与 TFOM 不兼容 → TFM-1001；
- 延迟超标 → TFM-1004（口径 batch=1/预热/采样）；
- 冒烟在新解释器子进程执行 + golden 容差比对（TFM-1005）；
- 回滚 production A→B→A；无上一版本 → TFM-1007。
"""

from __future__ import annotations

import pytest

from thermoforge_data.importer import default_registry
from thermoforge_runtime.errors import ModelRegistryError
from thermoforge_runtime.package import write_checksums
from thermoforge_runtime.registry import ModelRegistry

from phase34_helpers import build_test_package

FAST = {"latency_warmup": 5, "latency_samples": 30}


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models")


def _register(registry, tmp_path, version, **kw):
    pkg = build_test_package(tmp_path / f"pkg_{version}", version=version, **kw)
    return pkg, registry.register(pkg, actor="test")


def test_register_idempotent_and_version_conflict(registry, tmp_path):
    pkg, entry = _register(registry, tmp_path, "1.0.0")
    assert entry["status"] == "candidate"
    again = registry.register(pkg, actor="test")  # 同内容：幂等
    assert again["content_id"] == entry["content_id"]

    # 同版本号内容变化 → TFM-1006
    (pkg / "README.md").write_text("changed\n", encoding="utf-8", newline="\n")
    write_checksums(pkg)
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.register(pkg, actor="test")
    assert excinfo.value.code == "TFM-1006"


def test_illegal_transitions(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    # 沿链前进允许（candidate → validated），跳级前进亦由 can_transition 定义
    registry.transition("chiller-power", "1.0.0", "validated",
                        reason="正常", actor="test")
    with pytest.raises(ValueError):
        registry.transition("chiller-power", "1.0.0", "candidate",
                            reason="回退", actor="test")
    with pytest.raises(ValueError):
        registry.transition("chiller-power", "1.0.0", "validated",
                            reason="原地停留", actor="test")
    with pytest.raises(ValueError):
        registry.transition("chiller-power", "1.0.0", "bogus",
                            reason="非法状态", actor="test")


def test_publish_happy_path_with_smoke_and_latency(registry, tmp_path):
    """发布门禁全过：完整性/签名/验收/延迟 p99/冷加载冒烟/回滚记录。"""
    _register(registry, tmp_path, "1.0.0")
    result = registry.publish(
        "chiller-power", "1.0.0", actor="test",
        acceptance={"cvrmse_max": 0.5, "mape_max": 0.5,
                    "nmbe_abs_max": 0.5, "physics_violation_rate_max": 0.001,
                    "inference_latency_ms_max": 50.0},
        tfom_registry=default_registry(), run_smoke=True, **FAST)
    assert result["status"] == "production"
    gate_names = [g["name"] for g in result["gates"]]
    assert gate_names == ["integrity", "signature_tfom", "acceptance",
                          "latency", "smoke", "rollback_recorded"]
    latency = result["gates"][3]["measurement"]
    assert latency["batch"] == 1 and latency["warmup"] == 5
    assert latency["samples"] == 30
    assert latency["p99_ms"] >= latency["p50_ms"] >= 0
    assert registry.current_production("chiller-power") == "1.0.0"
    assert registry.get_status("chiller-power", "1.0.0") == "production"


def test_publish_acceptance_not_met_tfm1003(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.publish("chiller-power", "1.0.0", actor="test",
                         acceptance={"cvrmse_max": 1e-9}, run_smoke=False)
    assert excinfo.value.code == "TFM-1003"
    # 拒绝后仍为 candidate，门禁明细已留痕
    assert registry.get_status("chiller-power", "1.0.0") == "candidate"
    gates = registry.last_gate_results("chiller-power", "1.0.0")
    assert gates[-1]["ok"] is False and "TFM-1003" in gates[-1]["detail"]


def test_publish_signature_incompatible_tfm1001(registry, tmp_path):
    pkg = build_test_package(tmp_path / "pkg_bad", version="1.0.0")
    # 重建签名不兼容的包：直接改 signature.yaml 后重算校验和
    import yaml

    with open(pkg / "signature.yaml", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    doc["inputs"][0]["unit"] = "kW"
    with open(pkg / "signature.yaml", "w", encoding="utf-8", newline="\n") as fp:
        yaml.safe_dump(doc, fp, allow_unicode=True, sort_keys=True)
    write_checksums(pkg)
    registry.register(pkg, actor="test")
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.publish("chiller-power", "1.0.0", actor="test",
                         tfom_registry=default_registry(), run_smoke=False)
    assert excinfo.value.code == "TFM-1001"


def test_publish_latency_exceeded_tfm1004(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.publish("chiller-power", "1.0.0", actor="test",
                         acceptance={"inference_latency_ms_max": 1e-9},
                         run_smoke=False, **FAST)
    assert excinfo.value.code == "TFM-1004"


def test_publish_requires_candidate(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    registry.publish("chiller-power", "1.0.0", actor="test",
                     acceptance={"cvrmse_max": 0.5}, run_smoke=False, **FAST)
    with pytest.raises(ValueError):
        registry.publish("chiller-power", "1.0.0", actor="test",
                         run_smoke=False)


def test_rollback_a_b_a(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    registry.publish("chiller-power", "1.0.0", actor="test",
                     acceptance={"cvrmse_max": 0.5}, run_smoke=False, **FAST)
    _register(registry, tmp_path, "1.1.0")
    registry.publish("chiller-power", "1.1.0", actor="test",
                     acceptance={"cvrmse_max": 0.5}, run_smoke=False, **FAST)
    assert registry.current_production("chiller-power") == "1.1.0"
    assert registry.get_status("chiller-power", "1.0.0") == "deprecated"

    out = registry.rollback("chiller-power", actor="ops", reason="线上漂移")
    assert out["production"] == "1.0.0"
    assert registry.current_production("chiller-power") == "1.0.0"
    assert registry.get_status("chiller-power", "1.1.0") == "deprecated"
    # 回滚留痕：deprecated → production 的受控例外
    entry = registry._load_index("chiller-power")["versions"]["1.0.0"]
    assert entry["history"][-1]["to"] == "production"
    assert "回滚" in entry["history"][-1]["reason"]


def test_rollback_without_previous_tfm1007(registry, tmp_path):
    _register(registry, tmp_path, "1.0.0")
    registry.publish("chiller-power", "1.0.0", actor="test",
                     acceptance={"cvrmse_max": 0.5}, run_smoke=False, **FAST)
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.rollback("chiller-power", actor="ops")
    assert excinfo.value.code == "TFM-1007"


def test_rollback_without_production_tfm1007(registry):
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.rollback("ghost-model", actor="ops")
    assert excinfo.value.code == "TFM-1007"


def test_smoke_gate_detects_golden_mismatch(registry, tmp_path):
    """篡改 golden 输出（重算校验和）→ 冷加载子进程比对失败 TFM-1005。"""
    import pandas as pd

    pkg, _ = _register(registry, tmp_path, "1.0.0")
    registered = registry.package_dir("chiller-power", "1.0.0")
    golden_path = registered / "golden.parquet"
    golden = pd.read_parquet(golden_path)
    out_col = [c for c in golden.columns if c.startswith("output__")][0]
    golden[out_col] = golden[out_col] * 1.05  # 超容差
    golden.to_parquet(golden_path, compression="zstd", index=False)
    write_checksums(registered)
    with pytest.raises(ModelRegistryError) as excinfo:
        registry.publish("chiller-power", "1.0.0", actor="test",
                         acceptance={"cvrmse_max": 0.5}, run_smoke=True,
                         **FAST)
    assert excinfo.value.code == "TFM-1005"
