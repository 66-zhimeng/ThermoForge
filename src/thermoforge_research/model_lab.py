"""模型实验室（model_lab）：agent 起草模型代码 → 静态扫描 → 子进程结构校验
→ 版本化入库 → 直接参与实验（design-decisions DD-02 方案 C 落地）。

**这条通路是自治的：没有人工审批环节。** agent 写模型、跑真实数据、读
指标、改模型、再跑，闭环由它自己完成。与预处理规则集
（thermoforge_data.preprocess，改的是数据、必须人批）的区别正在这里——
实验室改的是**模型假设**，而假设本来就该由证据淘汰，不该由人排队放行。

- 源码与元数据版本化存放：`<root>/model_lab/<name>.v<N>.py` +
  `<name>.v<N>.yaml`；content_hash 只覆盖源码，审计留痕不改指纹；
  同一 (name, version) 内容不可变（TFML-004）。改模型 = 提交新版本。
- 进实验的门禁是**机器判定**，不是人：结构校验必须通过
  （status=validated），且未被停用（TFML-006）。
- 静态扫描（scan_source）是执行前的唯一自动关卡：import 白名单 +
  危险调用/dunder 黑名单 + 体积上限。它是粗筛，挡住明显出格的写法与
  已知的逃逸口（远程反序列化、动态库加载、数据集下载），但不是内核级
  沙箱——候选代码是在本机子进程里以当前用户身份执行的。
- 结构校验在子进程里跑（_lab_check.py）：接口齐全 → 合成数据
  fit/predict → save/load 往返逐位一致 → 同种子重训逐位一致。
- 实验运行时由 runner 把源码快照进实验目录（spec.lab_module），
  发布时随 artifact/ 打包为 lab_source.py——实验与模型包都不依赖
  实验室存储即可复现。
- `deprecate()` 是事后否决而非事前放行：停用只挡住后续引用，不阻塞
  任何一轮循环，人不在关键路径上。

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
# TFML-005 已退役：原「非 human 审批」，实验室通路取消人工审批后不再使用；
# 号段保留不复用，免得历史信封里的码换了意思。
TFML_NOT_RUNNABLE = "TFML-006"         # 模块不可用于实验（未过校验 / 已停用）
TFML_BAD_REF = "TFML-007"              # 引用格式非法

# validated = 五连检通过，可直接进实验；unvalidated = 绕过工具层直接入库
# （只在测试/人工摆放时出现）；deprecated = 事后停用，禁止再被引用。
LAB_STATUSES = ("validated", "unvalidated", "deprecated")

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

# 白名单根模块下的例外子模块：这些是绕过根白名单的现成逃逸口
# （数据集下载 = 网络出站；ctypeslib/f2py = 加载任意本地库）。
DENIED_IMPORT_PREFIXES = (
    "sklearn.datasets", "numpy.ctypeslib", "numpy.f2py", "numpy.distutils",
    "scipy.datasets", "pandas.io.clipboard",
)

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
    # pickle 家族：反序列化即任意代码执行（implementation-notes §8.1），
    # 且 pandas 的读取口能直接吃 URL。注意不能拦 `loads`——json.loads 是
    # load_model 的正当写法，而 pickle 根本 import 不进来。
    "read_pickle", "to_pickle",
    # 加载任意本地动态库 / 联网下载数据集
    "load_library", "fetch_openml", "fetch_species_distributions",
})

# 允许访问的 dunder（其余 dunder 一律拒：__globals__/__subclasses__ 等
# 都是把黑名单变成摆设的逃生梯）
ALLOWED_DUNDERS = frozenset({
    "__init__", "__name__", "__doc__", "__all__", "__file__", "__version__",
})


def _check_import(module: str, lineno: int, *, kind: str = "import") -> list[str]:
    """单条 import 的白名单 + 子模块黑名单判定。"""
    root = module.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        return [f"第 {lineno} 行: {kind} {module} 不在白名单"
                f"（允许 {sorted(ALLOWED_IMPORT_ROOTS)}）"]
    if any(module == p or module.startswith(p + ".")
           for p in DENIED_IMPORT_PREFIXES):
        return [f"第 {lineno} 行: {kind} {module} 是白名单下的禁用子模块"
                f"（{list(DENIED_IMPORT_PREFIXES)}）"]
    return []


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
                violations.extend(_check_import(alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"第 {node.lineno} 行: 不允许相对 import")
                continue
            module = node.module or ""
            bad = _check_import(module, node.lineno, kind="from")
            violations.extend(bad)
            if not bad:
                # `from sklearn import datasets` 的禁用子模块藏在 names 里
                for alias in node.names:
                    violations.extend(_check_import(
                        f"{module}.{alias.name}", node.lineno, kind="from"))
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


def _audit_trail(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """审计留痕，兼容取消审批前写下的 `approvals` 字段名。"""
    return list(meta.get("audit") or meta.get("approvals") or [])


def _normalize_status(meta: Mapping[str, Any]) -> str:
    """把取消审批前入库的旧状态映射到当前状态机（读时归一，不改盘上文件）。

    旧 `approved`/`proposed` 都不再表达「能否运行」——能否运行只看结构
    校验，于是一律按 validation.ok 重判；`deprecated` 保持停用。
    """
    status = str(meta.get("status") or "")
    if status in ("validated", "unvalidated", "deprecated"):
        return status
    return "validated" if (meta.get("validation") or {}).get("ok") \
        else "unvalidated"


class LabStore:
    """实验室模块的版本化存储（`<root>/model_lab/<name>.v<N>.py|.yaml`）。

    源码不可变：同 (name, version) 只允许写入相同内容哈希；
    停用只改 .yaml 元数据，不动 .py。改模型一律走新版本，
    于是「第 N 版效果如何」在 Ledger 里是可追溯的证据链。
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
            meta = yaml.safe_load(fp)
        meta["status"] = _normalize_status(meta)
        return meta

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
            status = _normalize_status(meta)
            out.append({
                "name": meta["name"],
                "version": meta["version"],
                "ref": f"{meta['name']}@v{meta['version']}",
                "status": status,
                "runnable": status == "validated",
                "content_hash": meta["content_hash"],
                "description": meta.get("description"),
                "validation_ok": (meta.get("validation") or {}).get("ok"),
                # 模块声明要吃哪些角色列。五连检只能拿模块自己声明的名字造
                # 合成数据，检不出「声明的列在任何视图里都不存在」——所以这
                # 份声明必须外露，让规划器侧拿真实视图对一遍（见 planner.py
                # `_lab_view_error`）。
                "input_roles": list(
                    (meta.get("validation") or {}).get("input_roles") or []),
            })
        return out

    def runnable_refs(self) -> list[str]:
        """可直接进实验的模块引用（规划上下文与提示词只该看见这些）。"""
        return [m["ref"] for m in self.list() if m["runnable"]]

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
            # 结构校验通过即可用——门禁是机器判定，不等人
            "status": "validated" if (validation or {}).get("ok")
                      else "unvalidated",
            "proposer": proposer,
            "description": description,
            "created_at": _utcnow(),
            "audit": [],
            "validation": dict(validation) if validation else None,
        }
        self._write_meta(meta)
        return {**meta, "source": source}

    def deprecate(
        self,
        name: str,
        version: int | None = None,
        *,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """停用：事后否决，之后不得再被实验引用。仅改元数据。

        这是实验室通路上唯一的否决口，且**不在关键路径上**——不停用什么
        也不会挡住 agent 的任何一轮循环。agent 自己也能用它清掉走不通的
        方案（留痕记的是谁停的）。已跑完的实验不受影响：源码快照已冻结
        在实验目录里，历史结论仍可复现。
        """
        version = version or self.latest_version(name)
        record = self.get(name, version)
        meta = {k: v for k, v in record.items() if k != "source"}
        meta["status"] = "deprecated"
        meta["audit"] = [
            *_audit_trail(meta),
            ApprovalRecord(actor=actor, action="deprecate", at=_utcnow(),
                           note=note).model_dump(mode="json"),
        ]
        meta.pop("approvals", None)  # 旧版元数据的字段名，读进来后统一为 audit
        self._write_meta(meta)
        return {**meta, "source": record["source"]}

    def require_runnable(
        self, name: str, version: int | None = None
    ) -> dict[str, Any]:
        """实验门禁（机器判定）：结构校验通过且未停用，否则 TFML-006。"""
        record = self.get(name, version)
        status = record["status"]
        if status == "validated":
            return record
        reason = ("已被停用" if status == "deprecated"
                  else f"未通过结构校验（status={status}）")
        raise LabError(
            TFML_NOT_RUNNABLE,
            f"实验室模块不可用于实验: {name}@v{record['version']}——{reason}。"
            "改模型请用 tf_lab_submit 提交新版本（通过五连检即可直接引用）",
        )
