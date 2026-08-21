"""模型注册与发布门禁（model-package.md §5/§7/§8、conventions.md §7.9）。

目录结构::

    models/
    └── <model_id>/
        ├── registry.json      # 版本状态与回滚链（单写者，原子替换）
        └── <version>/         # 完整模型包（verify_package 通过后复制进来）

- 版本状态机（§5）：candidate → validated → approved → production →
  deprecated → retired，只允许沿链前进；**回滚是唯一受控例外**
  （deprecated → production，仅 `rollback()` 发起并留痕）。
- 同版本内容不可变：重复注册同一版本时比对 checksums，一致则幂等返回，
  不一致报 **TFM-1006**。
- 发布门禁（§8）：文件齐全+校验和（TFM-1002）、签名与 TFOM 兼容
  （TFM-1001）、硬性验收条件（TFM-1003）、推理延迟 p99（TFM-1004，
  batch=1、预热 100 次、测 1000 次）、冷加载冒烟（TFM-1005，新解释器
  子进程 + golden 容差比对）、回滚版本已记录。
- 回滚到上一生产版本；无则报 **TFM-1007**。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from thermoforge_core.canonical import sha256_hex
from thermoforge_core.contracts.model_package import (
    MODEL_STATUSES,
    can_transition,
)
from thermoforge_core.units import UnitError, conversion_factor

from .artifact import load_model_artifact
from .errors import ModelRegistryError
from .inference import measure_latency
from .package import load_signature, verify_package

# 延迟口径（implementation-notes §9.3）：batch=1，预热 100 次，测 1000 次
LATENCY_WARMUP = 100
LATENCY_SAMPLES = 1000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, doc: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(doc, fp, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        fp.write("\n")
    os.replace(tmp, path)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in str(text).split("."))


def _version_compat_ok(compat: str, version: str) -> bool:
    """最小解析 ">=1.0,<2.0" 形式的兼容范围（conventions §6 [草案]）。"""
    v = _version_tuple(version)
    for clause in str(compat).split(","):
        clause = clause.strip()
        for op in (">=", "<=", "==", ">", "<"):
            if clause.startswith(op):
                bound = _version_tuple(clause[len(op):])
                n = max(len(v), len(bound))
                vv = v + (0,) * (n - len(v))
                bb = bound + (0,) * (n - len(bound))
                if op == ">=" and not vv >= bb:
                    return False
                if op == "<=" and not vv <= bb:
                    return False
                if op == "==" and not vv == bb:
                    return False
                if op == ">" and not vv > bb:
                    return False
                if op == "<" and not vv < bb:
                    return False
                break
        else:
            raise ValueError(f"无法解析的兼容范围子句: {clause!r}")
    return True


class ModelRegistry:
    """文件系统模型注册表（单写者）。

    用法::

        registry = ModelRegistry(Path("models"))
        registry.register(package_dir, actor="agent")
        registry.publish("chiller-power", "1.0.0", acceptance={...})
        registry.rollback("chiller-power", actor="ops")
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 索引

    def _model_dir(self, model_id: str) -> Path:
        return self.root / model_id

    def _index_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "registry.json"

    def _load_index(self, model_id: str) -> dict[str, Any]:
        path = self._index_path(model_id)
        if not path.exists():
            return {"model_id": model_id, "production": None,
                    "production_history": [], "versions": {}}
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    def _save_index(self, index: Mapping[str, Any]) -> None:
        _write_json_atomic(self._index_path(str(index["model_id"])), index)

    def package_dir(self, model_id: str, version: str) -> Path:
        return self._model_dir(model_id) / version

    def _entry(self, index: dict[str, Any], version: str) -> dict[str, Any]:
        try:
            return index["versions"][version]
        except KeyError:
            raise ValueError(
                f"版本未注册: {index['model_id']}@{version}"
            ) from None

    # ---------------------------------------------------------------- 注册

    def register(self, package_dir: str | Path, *, actor: str) -> dict[str, Any]:
        """校验并注册模型包为 candidate（TFM-1002 / TFM-1006）。"""
        package_dir = Path(package_dir)
        verify_package(package_dir)  # TFM-1002
        with open(package_dir / "model.yaml", encoding="utf-8") as fp:
            import yaml

            meta = yaml.safe_load(fp)
        model_id, version = str(meta["model_id"]), str(meta["version"])
        with open(package_dir / "checksums.json", encoding="utf-8") as fp:
            checksums = json.load(fp)["files"]
        content_id = sha256_hex(json.dumps(checksums, sort_keys=True))

        dest = self.package_dir(model_id, version)
        index = self._load_index(model_id)
        if dest.exists():
            existing = self._entry(index, version)
            if existing["content_id"] != content_id:
                raise ModelRegistryError(
                    "TFM-1006",
                    f"同版本号内容发生变化: {model_id}@{version}（§5：内容"
                    "变化必须生成新版本）",
                )
            return existing  # 幂等：相同内容重复注册直接返回

        self._model_dir(model_id).mkdir(parents=True, exist_ok=True)
        shutil.copytree(package_dir, dest)
        entry = {
            "version": version,
            "status": "candidate",
            "content_id": content_id,
            "rollback_version": None,
            "last_gate_results": [],
            "history": [{
                "from": None, "to": "candidate", "reason": "注册模型包",
                "actor": actor, "at": _utcnow(),
            }],
        }
        index["versions"][version] = entry
        self._save_index(index)
        return entry

    # ---------------------------------------------------------------- 状态机

    def get_status(self, model_id: str, version: str) -> str:
        return str(self._entry(self._load_index(model_id), version)["status"])

    def transition(
        self,
        model_id: str,
        version: str,
        to_status: str,
        *,
        reason: str,
        actor: str,
        _rollback: bool = False,
    ) -> dict[str, Any]:
        """状态转换（§5）。`_rollback` 仅供 rollback() 使用：deprecated →
        production 是回滚的受控例外，必须留痕。"""
        if to_status not in MODEL_STATUSES:
            raise ValueError(f"非法模型状态: {to_status!r}（允许 {MODEL_STATUSES}）")
        index = self._load_index(model_id)
        entry = self._entry(index, version)
        from_status = entry["status"]
        if not can_transition(from_status, to_status) and not (
            _rollback and from_status == "deprecated" and to_status == "production"
        ):
            raise ValueError(
                f"非法状态转换: {from_status} → {to_status}"
                "（§5：只允许沿链前进；回滚请用 rollback()）"
            )
        entry["history"].append({
            "from": from_status, "to": to_status, "reason": reason,
            "actor": actor, "at": _utcnow(),
        })
        entry["status"] = to_status
        if to_status == "production":
            previous = index["production"]
            if previous and previous != version:
                entry["rollback_version"] = previous
            index["production"] = version
            index["production_history"].append(version)
        self._save_index(index)
        return entry

    def current_production(self, model_id: str) -> str | None:
        return self._load_index(model_id)["production"]

    def last_gate_results(
        self, model_id: str, version: str
    ) -> list[dict[str, Any]]:
        """最近一次发布门禁的明细（发布失败时供工具层回流诊断）。"""
        try:
            entry = self._entry(self._load_index(model_id), version)
        except ValueError:
            return []
        return list(entry.get("last_gate_results") or [])

    # ---------------------------------------------------------------- 发布门禁

    def publish(
        self,
        model_id: str,
        version: str,
        *,
        actor: str,
        acceptance: Mapping[str, Any] | None = None,
        tfom_registry: Any = None,
        latency_warmup: int = LATENCY_WARMUP,
        latency_samples: int = LATENCY_SAMPLES,
        run_smoke: bool = True,
    ) -> dict[str, Any]:
        """执行发布门禁（§8）。任一硬门禁失败抛对应 TFM-10xx 错误。"""
        pkg = self.package_dir(model_id, version)
        index = self._load_index(model_id)
        entry = self._entry(index, version)
        if entry["status"] != "candidate":
            raise ValueError(
                f"只有 candidate 可进入发布门禁，当前状态: {entry['status']}"
            )
        # 回滚目标在门禁前锁定（当前生产版本），门禁记录它已就位
        entry["rollback_version"] = index["production"]

        gates: list[dict[str, Any]] = []
        try:
            gates.append(self._gate_integrity(pkg))
            gates.append(self._gate_signature(pkg, tfom_registry))
            gates.append(self._gate_acceptance(pkg, acceptance))
            gates.append(self._gate_latency(pkg, acceptance,
                                            latency_warmup, latency_samples))
            if run_smoke:
                gates.append(self._gate_smoke(pkg))
            gates.append(self._gate_rollback_recorded(index, entry))
        except ModelRegistryError as exc:
            gates.append({"name": "failed", "ok": False, "detail": str(exc)})
            entry["last_gate_results"] = gates
            self._save_index(index)
            raise

        entry["last_gate_results"] = gates
        self._save_index(index)
        # 门禁全过：candidate → validated → approved → production
        for to_status, why in (
            ("validated", "发布门禁全部通过（§8）"),
            ("approved", "机器判定批准（DD-13）"),
            ("production", "发布为生产版本"),
        ):
            self.transition(model_id, version, to_status, reason=why, actor=actor)
        index = self._load_index(model_id)
        previous = [
            v for v in index["versions"]
            if v != version
            and index["versions"][v]["status"] == "production"
        ]
        for old in previous:  # 理论上一个 model_id 同时只有一个 production
            self.transition(model_id, old, "deprecated",
                            reason=f"被新版本 {version} 取代", actor=actor)
        return {"model_id": model_id, "version": version, "status": "production",
                "gates": gates}

    # ---------------------------------------------------------------- 各门禁

    @staticmethod
    def _gate_integrity(pkg: Path) -> dict[str, Any]:
        result = verify_package(pkg)  # TFM-1002
        return {"name": "integrity", "ok": True,
                "detail": f"文件齐全且校验和一致（{result['files']} 个文件）"}

    @staticmethod
    def _gate_signature(pkg: Path, tfom_registry: Any) -> dict[str, Any]:
        """签名与 TFOM 兼容（TFM-1001）：对象模型存在、属性齐全、单位兼容。"""
        signature, _, _ = load_signature(pkg)
        if tfom_registry is None:
            return {"name": "signature_tfom", "ok": True,
                    "detail": "未提供 TFOM 注册表，跳过兼容检查"}
        model = tfom_registry.get(signature.object_model)
        if model is None:
            raise ModelRegistryError(
                "TFM-1001", f"对象模型未注册: {signature.object_model}"
            )
        compat = signature.object_model_compat
        if compat and not _version_compat_ok(compat, model.version):
            raise ModelRegistryError(
                "TFM-1001",
                f"TFOM 版本 {model.version} 不满足签名兼容范围 {compat!r}",
            )
        for port in (*signature.inputs, *signature.outputs):
            prop = model.properties.get(port.property_code)
            if prop is None:
                raise ModelRegistryError(
                    "TFM-1001",
                    f"签名属性不属于 {signature.object_model}: {port.property_code}",
                )
            try:
                conversion_factor(port.unit, prop.unit)
            except UnitError as exc:
                raise ModelRegistryError(
                    "TFM-1001",
                    f"签名单位与 TFOM 不兼容: {port.property_code} "
                    f"{port.unit} vs {prop.unit}（{exc}）",
                ) from exc
        return {"name": "signature_tfom", "ok": True,
                "detail": f"签名与 {signature.object_model} 兼容"}

    @staticmethod
    def _primary_surface(metrics: Mapping[str, Any],
                         evaluated_on: str = "auto") -> dict[str, Any]:
        """发布口径。默认优先面 C（未见×未来），其次面 A，再次 validate（§4.3）。

        `evaluated_on` 由 Goal 的 acceptance 指定，可钉死到某一个面，或钉到
        `rolling_cv`（折间均值）。钉死时**不回退**：判据面拿不到数就该发布
        失败，悄悄换一个面等于把门槛判在了另一件事上。
        """
        metrics = metrics or {}
        if evaluated_on == "rolling_cv":
            return dict(metrics.get("rolling_cv") or {})
        surfaces = metrics.get("surfaces", {})
        order = (("C", "A", "validate") if evaluated_on in ("auto", "", None)
                 else (evaluated_on,))
        for name in order:
            surf = surfaces.get(name) or {}
            if surf.get("n_samples"):
                return dict(surf)
        return {}

    def _gate_acceptance(
        self, pkg: Path, acceptance: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """硬性验收条件（TFM-1003）。硬性条件失败时不得发布（§7）。"""
        if not acceptance:
            return {"name": "acceptance", "ok": True,
                    "detail": "未声明验收条件，跳过"}
        with open(pkg / "metrics.json", encoding="utf-8") as fp:
            metrics = json.load(fp)
        with open(pkg / "validation.json", encoding="utf-8") as fp:
            validation = json.load(fp)
        evaluated_on = str(acceptance.get("evaluated_on") or "auto")
        surf = self._primary_surface(metrics, evaluated_on)
        values = dict(surf.get("metrics") or {})
        physics = (validation.get("physics") or {})
        if physics.get("overall_rate") is not None:
            values["physics_violation_rate"] = physics["overall_rate"]

        if evaluated_on != "auto" and not values:
            raise ModelRegistryError(
                "TFM-1003",
                f"验收判据面 {evaluated_on!r} 在模型包里没有指标 —— "
                "实验是否启用了对应切分？（rolling_cv 需 validation.rolling_cv.enabled）",
            )

        unmet: list[str] = []
        checks = (
            ("cvrmse_max", "CVRMSE", lambda v, lim: v <= lim),
            ("r2_min", "R2", lambda v, lim: v >= lim),
            ("mape_max", "MAPE", lambda v, lim: v <= lim),
            ("nmbe_abs_max", "NMBE", lambda v, lim: abs(v) <= lim),
            ("physics_violation_rate_max", "physics_violation_rate",
             lambda v, lim: v <= lim),
        )
        for key, metric_name, ok_fn in checks:
            limit = acceptance.get(key)
            if limit is None:
                continue
            value = values.get(metric_name)
            if value is None or not ok_fn(value, float(limit)):
                unmet.append(f"{metric_name}={value!r} 未满足 {key}={limit}")
        if unmet:
            raise ModelRegistryError(
                "TFM-1003", "未满足硬性验收条件: " + "; ".join(unmet)
            )
        return {"name": "acceptance", "ok": True,
                "detail": f"硬性验收条件全部满足（判据面 {evaluated_on}，"
                          f"{len(values)} 项指标）"}

    @staticmethod
    def _gate_latency(
        pkg: Path,
        acceptance: Mapping[str, Any] | None,
        warmup: int,
        samples: int,
    ) -> dict[str, Any]:
        """推理延迟 p99（TFM-1004；口径 batch=1、预热 100、测 1000，§9.3）。"""
        limit = (acceptance or {}).get("inference_latency_ms_max")
        if limit is None:
            return {"name": "latency", "ok": True,
                    "detail": "未声明延迟上限，跳过"}
        result = measure_latency(pkg, warmup=warmup, samples=samples)
        if result["p99_ms"] > float(limit):
            raise ModelRegistryError(
                "TFM-1004",
                f"推理延迟 p99={result['p99_ms']:.3f} ms 超标 "
                f"（上限 {limit} ms，口径 batch=1/预热 {warmup}/测 {samples}）",
            )
        return {"name": "latency", "ok": True,
                "detail": f"p99={result['p99_ms']:.3f} ms <= {limit} ms",
                "measurement": result}

    @staticmethod
    def _gate_smoke(pkg: Path) -> dict[str, Any]:
        """冷加载冒烟（TFM-1005）：新解释器子进程 + golden 容差比对。

        子进程以隔离模式（-I）启动，cwd 为空临时目录，仅挂载模型包目录
        （implementation-notes §8.3）。
        """
        with tempfile.TemporaryDirectory(prefix="tf_smoke_") as cold_cwd:
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "thermoforge_runtime._smoke",
                 str(pkg)],
                cwd=cold_cwd, capture_output=True, text=True, timeout=180,
                encoding="utf-8", errors="replace",
            )
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {"ok": False, "error": proc.stderr.strip()[-500:]}
        if proc.returncode != 0 or not result.get("ok"):
            raise ModelRegistryError(
                "TFM-1005",
                f"冷加载冒烟失败: {result.get('error') or proc.stderr.strip()[-300:]}",
            )
        return {"name": "smoke", "ok": True,
                "detail": f"冷加载 + golden 比对通过（n={result.get('n')}, "
                          f"max_rel_error={result.get('max_rel_error'):.3g}）"}

    @staticmethod
    def _gate_rollback_recorded(
        index: Mapping[str, Any], entry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """回滚版本与兼容策略已记录（§8）。首个生产版本允许无回滚目标。"""
        current = index.get("production")
        if current and entry.get("rollback_version") != current:
            raise ModelRegistryError(
                "TFM-1007",
                f"回滚版本未正确记录: 当前生产 {current}，"
                f"记录 {entry.get('rollback_version')}",
            )
        detail = (f"回滚目标: {entry['rollback_version']}"
                  if entry.get("rollback_version")
                  else "首个生产版本，无回滚目标（已记录）")
        return {"name": "rollback_recorded", "ok": True, "detail": detail}

    # ---------------------------------------------------------------- 回滚

    def rollback(self, model_id: str, *, actor: str,
                 reason: str = "回滚到上一生产版本") -> dict[str, Any]:
        """回滚到上一生产版本（§7）。无上一生产版本报 TFM-1007。"""
        index = self._load_index(model_id)
        current = index["production"]
        if current is None:
            raise ModelRegistryError(
                "TFM-1007", f"{model_id} 当前无生产版本，无法回滚"
            )
        entry = self._entry(index, current)
        target = entry.get("rollback_version")
        if target is None:
            history = index["production_history"]
            previous = [v for v in history if v != current]
            target = previous[-1] if previous else None
        if target is None or target not in index["versions"]:
            raise ModelRegistryError(
                "TFM-1007",
                f"{model_id} 无可回滚的上一生产版本（当前 {current}）",
            )
        self.transition(model_id, current, "deprecated",
                        reason=f"回滚：{reason}", actor=actor)
        restored = self.transition(
            model_id, target, "production",
            reason=f"回滚恢复：{reason}", actor=actor, _rollback=True,
        )
        return {"model_id": model_id, "rolled_back_from": current,
                "production": target, "entry": restored}
