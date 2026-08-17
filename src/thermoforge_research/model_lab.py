"""模型实验室（model_lab）：agent 起草模型代码 → 静态扫描 → 子进程结构校验
→ 版本化入库 → 人工审批 → 实验引用（design-decisions DD-02 方案 C 落地）。

与预处理规则集（thermoforge_data.preprocess）同构的「提议-审批-落库」
工作流，区别在于提议物是**代码**而非参数：

- 源码与元数据版本化存放：`<root>/model_lab/<name>.v<N>.py` +
  `<name>.v<N>.yaml`；content_hash 只覆盖源码，审批留痕不改指纹；
  同一 (name, version) 内容不可变（TFML-004）。
- 静态扫描（scan_source）是**粗筛不是安全边界**：import 白名单 +
  危险调用黑名单，挡住明显出格写法；真正的闸门是人工审批
  （TFML-005 非 human 不得审批；TFML-006 未批准不得用于实验）。
- 结构校验在子进程里跑（_lab_check.py）：接口齐全 → 合成数据
  fit/predict → save/load 往返逐位一致 → 同种子重训逐位一致。
- 实验运行时由 runner 把已批准源码快照进实验目录（spec.lab_module），
  发布时随 artifact/ 打包为 lab_source.py——实验与模型包都不依赖
  实验室存储即可复现。

错误码：TFML- 前缀。conventions.md §7 错误码表冻结，本模块自带小注册表
（同 preprocess.py 的做法，issues I-53）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from thermoforge_data.preprocess import ApprovalRecord
from thermoforge_models.lab import MODEL_FORMAT_PREFIX

# ---------------------------------------------------------------- 错误码（模型实验室域）

TFML_SOURCE_VIOLATION = "TFML-001"     # 静态扫描不通过
TFML_VALIDATION_FAILED = "TFML-002"    # 子进程结构校验失败
TFML_MODULE_NOT_FOUND = "TFML-003"     # 模块/版本不存在
TFML_VERSION_CONFLICT = "TFML-004"     # 同版本内容变更（不可变约束）
TFML_APPROVAL_FORBIDDEN = "TFML-005"   # 非 human 审批
TFML_NOT_APPROVED = "TFML-006"         # 未批准模块用于实验
TFML_BAD_REF = "TFML-007"              # 引用格式非法

LAB_STATUSES = ("proposed", "approved", "deprecated")

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_SOURCE_BYTES = 64 * 1024
VALIDATE_TIMEOUT_SECONDS = 120.0


class LabError(RuntimeError):
    """模型实验室错误，携带 TFML-xxx 错误码（机器可判定）。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_lab_ref(ref: str) -> tuple[str, int | None]:
    """解析 `name` / `name@vN` 引用 → (name, version|None)。"""
    name, sep, ver = str(ref).strip().partition("@v")
    if not NAME_PATTERN.match(name):
        raise LabError(
            TFML_BAD_REF,
            f"实验室模块名非法: {name!r}（{NAME_PATTERN.pattern}）",
        )
    if not sep:
        return name, None
    if not ver.isdigit() or int(ver) < 1:
        raise LabError(TFML_BAD_REF, f"实验室引用版本非法: {ref!r}")
    return name, int(ver)


# ---------------------------------------------------------------- 静态扫描（粗筛）

# import 白名单（根模块）。os/pathlib/tempfile/json 是 save/load 的正当需求；
# 网络与进程类模块一律不在名单内。
ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "math", "json", "typing", "dataclasses",
    "os", "pathlib", "tempfile",
    "numpy", "pandas", "sklearn", "scipy", "xgboost",
    "thermoforge_models", "thermoforge_core",
})

# bare 调用黑名单（绕过白名单的动态逃逸口）
BANNED_CALL_NAMES = frozenset({
    "eval", "exec", "__import__", "compile", "globals", "locals", "vars",
    "breakpoint", "input",
})

# 危险属性黑名单（无论挂在哪个对象上）
BANNED_ATTRS = frozenset({
    "system", "popen", "getoutput", "getstatusoutput",
    "spawnl", "spawnlp", "spawnv", "execl", "execle", "execlp",
    "execv", "execve", "execvp", "kill", "killpg",
    "remove", "unlink", "rmdir", "removedirs", "rmtree",
    "urlopen", "urlretrieve", "socket", "connect", "create_connection",
    "Popen", "check_output", "check_call",
    "getattr", "setattr", "delattr",
})

# 允许访问的 dunder（其余 dunder 一律拒：__globals__/__subclasses__ 等
# 都是把黑名单变成摆设的逃生梯）
ALLOWED_DUNDERS = frozenset({
    "__init__", "__name__", "__doc__", "__all__", "__file__", "__version__",
})


def scan_source(source: str) -> list[str]:
    """AST 粗筛：返回违规清单（空 = 通过）。每条违规带行号，便于修订。"""
    violations: list[str] = []
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"源码超过 {MAX_SOURCE_BYTES} 字节上限"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"第 {exc.lineno} 行: 语法错误 {exc.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(
                        f"第 {node.lineno} 行: import {alias.name} 不在白名单"
                        f"（允许 {sorted(ALLOWED_IMPORT_ROOTS)}）")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"第 {node.lineno} 行: 不允许相对 import")
                continue
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(
                    f"第 {node.lineno} 行: from {node.module} 不在白名单")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALL_NAMES:
                violations.append(
                    f"第 {node.lineno} 行: 禁止调用 {func.id}()")
            elif isinstance(func, ast.Attribute):
                if func.attr in BANNED_ATTRS:
                    violations.append(
                        f"第 {node.lineno} 行: 禁止调用 .{func.attr}()")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                violations.append(
                    f"第 {node.lineno} 行: 禁止访问 dunder 属性 .{node.attr}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id not in ALLOWED_DUNDERS:
                violations.append(
                    f"第 {node.lineno} 行: 禁止引用 dunder 名称 {node.id}")
    return violations


# ---------------------------------------------------------------- 子进程结构校验


def validate_module(
    source_path: str | Path,
    work_dir: str | Path,
    *,
    timeout_seconds: float = VALIDATE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """在子进程里对候选模块跑结构校验，返回报告 dict（ok/checks/…）。

    子进程 = 隔离边界：候选代码的 import 与 fit 副作用不污染调用方进程。
    报告落 `<work_dir>/lab_check.json`；超时/异常退出也返回 ok=False 报告。
    """
    from .runner import THREAD_ENV_VARS  # 局部 import：避免与 runner 循环依赖

    source_path = Path(source_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(THREAD_ENV_VARS)
    env["PYTHONHASHSEED"] = "0"
    report_path = work_dir / "lab_check.json"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "thermoforge_research._lab_check",
             str(source_path), str(work_dir)],
            cwd=work_dir, env=env, capture_output=True, text=True,
            timeout=timeout_seconds, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "checks": [], "error": (
            f"结构校验超过 {timeout_seconds}s 被杀掉（时间预算也是护栏）")}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as fp:
            return json.load(fp)
    return {"ok": False, "checks": [], "error": (
        f"校验子进程异常退出（exit {proc.returncode}）: "
        f"{proc.stderr.strip()[-300:]}")}


# ---------------------------------------------------------------- 版本化存储


def _content_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class LabStore:
    """实验室模块的版本化存储（`<root>/model_lab/<name>.v<N>.py|.yaml`）。

    源码不可变：同 (name, version) 只允许写入相同内容哈希；
    审批/弃用只改 .yaml 元数据，不动 .py。
    """

    def __init__(self, root: str | Path):
        self.dir = Path(root) / "model_lab"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- 路径与读写

    def _src_path(self, name: str, version: int) -> Path:
        return self.dir / f"{name}.v{version}.py"

    def _meta_path(self, name: str, version: int) -> Path:
        return self.dir / f"{name}.v{version}.yaml"

    def _write_meta(self, meta: dict[str, Any]) -> None:
        path = self._meta_path(meta["name"], meta["version"])
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            yaml.safe_dump(meta, fp, allow_unicode=True, sort_keys=True)
        os.replace(tmp, path)

    def _read_meta(self, name: str, version: int) -> dict[str, Any]:
        path = self._meta_path(name, version)
        if not path.exists():
            raise LabError(
                TFML_MODULE_NOT_FOUND, f"实验室模块不存在: {name}@v{version}")
        with open(path, encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    # ---- 查询

    def latest_version(self, name: str) -> int:
        versions = [
            int(p.stem.rsplit(".v", 1)[1])
            for p in self.dir.glob(f"{name}.v*.yaml")
        ]
        if not versions:
            raise LabError(TFML_MODULE_NOT_FOUND, f"实验室模块不存在: {name}")
        return max(versions)

    def get(self, name: str, version: int | None = None) -> dict[str, Any]:
        """元数据 + 源码。"""
        version = version or self.latest_version(name)
        meta = self._read_meta(name, version)
        with open(self._src_path(name, version), encoding="utf-8") as fp:
            source = fp.read()
        if _content_hash(source) != meta["content_hash"]:
            raise LabError(
                TFML_VERSION_CONFLICT,
                f"{name}@v{version} 源码与登记哈希不符（存储被篡改）",
            )
        return {**meta, "source": source}

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.dir.glob("*.v*.yaml")):
            with open(path, encoding="utf-8") as fp:
                meta = yaml.safe_load(fp)
            out.append({
                "name": meta["name"],
                "version": meta["version"],
                "ref": f"{meta['name']}@v{meta['version']}",
                "status": meta["status"],
                "content_hash": meta["content_hash"],
                "description": meta.get("description"),
                "validation_ok": (meta.get("validation") or {}).get("ok"),
            })
        return out

    # ---- 变更

    def submit(
        self,
        name: str,
        source: str,
        *,
        description: str | None = None,
        proposer: str = "agent",
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """入库为新版本（幂等：与最新版内容一致则直接返回该版）。"""
        if not NAME_PATTERN.match(name):
            raise LabError(
                TFML_BAD_REF,
                f"实验室模块名非法: {name!r}（{NAME_PATTERN.pattern}）",
            )
        digest = _content_hash(source)
        try:
            latest = self.latest_version(name)
            meta = self._read_meta(name, latest)
            if meta["content_hash"] == digest:
                return {**meta, "source": source}  # 幂等重交
        except LabError as exc:
            if exc.code != TFML_MODULE_NOT_FOUND:
                raise
            latest = 0
        version = latest + 1
        src_path = self._src_path(name, version)
        if src_path.exists():
            with open(src_path, encoding="utf-8") as fp:
                if _content_hash(fp.read()) != digest:
                    raise LabError(
                        TFML_VERSION_CONFLICT,
                        f"{name}@v{version} 已存在且内容不同（不可变约束）",
                    )
        else:
            fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp_",
                                       suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
                fp.write(source)
            os.replace(tmp, src_path)
        meta = {
            "name": name,
            "version": version,
            "content_hash": digest,
            "status": "proposed",
            "proposer": proposer,
            "description": description,
            "created_at": _utcnow(),
            "approvals": [],
            "validation": dict(validation) if validation else None,
        }
        self._write_meta(meta)
        return {**meta, "source": source}

    def approve(
        self,
        name: str,
        version: int | None = None,
        *,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """审批（actor 的 human 约束由工具层强制）。仅改元数据。"""
        version = version or self.latest_version(name)
        record = self.get(name, version)
        meta = {k: v for k, v in record.items() if k != "source"}
        if meta["status"] != "approved":
            meta["status"] = "approved"
        meta["approvals"] = [
            *meta.get("approvals", []),
            ApprovalRecord(actor=actor, action="approve", at=_utcnow(),
                           note=note).model_dump(mode="json"),
        ]
        self._write_meta(meta)
        return {**meta, "source": record["source"]}

    def require_approved(
        self, name: str, version: int | None = None
    ) -> dict[str, Any]:
        """实验/发布门禁：仅 approved 可用，否则 TFML-006。"""
        record = self.get(name, version)
        if record["status"] != "approved":
            raise LabError(
                TFML_NOT_APPROVED,
                f"实验室模块未批准: {name}@v{record['version']}"
                f"（当前 {record['status']}，需 actor=human 经 "
                "tf_lab_approve 审批）",
            )
        return record
