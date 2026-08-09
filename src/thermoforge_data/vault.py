"""Data Vault（data-contract.md §6、conventions.md §5.2、implementation-notes §10）。

目录结构::

    vault/datasets/<dataset_id>/rev_NNNN/
    ├── source/<原始文件名>          # 写后设只读
    ├── canonical/{data,variables,objects,parameters}.parquet   # zstd
    ├── manifest.json / quality.json / profile.json
    ├── lineage.json / fingerprint.json

- `content_sha256` 相同复用已有 revision（即使 `source_sha256` 不同）；
  不同则新建 `rev_NNNN`（4 位），不可覆盖（TFV-701）。
- 所有写入「临时文件 + os.replace」原子完成；读取校验指纹（TFV-702）。
- DuckDB 元数据索引（datasets/revisions/variables），单写者文件锁（§10.2）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from thermoforge_core.canonical import canonical_json
from thermoforge_core.errors import Diagnostic, Level
from thermoforge_core.fingerprint import Column, content_sha256

from .importer import ImportResult
from .profile import build_profile

if sys.platform == "win32":
    import msvcrt

    def _lock(fp) -> None:
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(fp) -> None:
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fp) -> None:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def _unlock(fp) -> None:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class VaultError(RuntimeError):
    """Data Vault 错误，携带 conventions.md §7.8 的错误码。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RevisionInfo:
    dataset_id: str
    revision: str
    content_sha256: str
    source_sha256: str
    path: Path

    @property
    def ref(self) -> str:
        return f"{self.dataset_id}@{self.revision}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, obj: Any) -> None:
    """原子写 JSON 文本：encoding=utf-8、newline=\\n（§10.3）。"""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(canonical_json(obj))
            fp.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _table_fingerprint(table: pa.Table, variables: Mapping[str, str]) -> str:
    """从 Arrow 表计算 content_sha256（§5.2）。

    `variables`：variable_id → dtype（float/integer/boolean/string）。
    """
    ts = table.column("timestamp").to_pylist()
    columns = [
        Column(variable_id=name, dtype=variables[name],
               values=table.column(name).to_pylist())
        for name in table.column_names
        if name != "timestamp"
    ]
    return content_sha256(ts, columns)


class DataVault:
    """单写者 Data Vault。

    用法::

        vault = DataVault(Path("vault"))
        ref = vault.store(import_result)          # -> "DC01_2026_CHILLER@rev_0001"
        table = vault.load_data(ref)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.datasets_dir = self.root / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / "vault.lock"
        self._db_path = self.root / "metadata.duckdb"
        with self._locked():
            self._with_db(lambda con: self._init_schema(con))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with open(self._lock_path, "a+b") as fp:
            _lock(fp)
            try:
                yield
            finally:
                _unlock(fp)

    def _with_db(self, fn):
        con = duckdb.connect(str(self._db_path))
        try:
            return fn(con)
        finally:
            con.close()

    @staticmethod
    def _init_schema(con) -> None:
        con.execute(
            "CREATE TABLE IF NOT EXISTS datasets ("
            "dataset_id VARCHAR PRIMARY KEY, created_at VARCHAR)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS revisions ("
            "dataset_id VARCHAR, revision VARCHAR, "
            "content_sha256 VARCHAR, source_sha256 VARCHAR, "
            "created_at VARCHAR, rows BIGINT, "
            "time_start VARCHAR, time_end VARCHAR, "
            "PRIMARY KEY (dataset_id, revision))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS variables ("
            "dataset_id VARCHAR, revision VARCHAR, variable_id VARCHAR, "
            "object_id VARCHAR, property_code VARCHAR, unit VARCHAR, "
            "dtype VARCHAR, role VARCHAR, "
            "PRIMARY KEY (dataset_id, revision, variable_id))"
        )

    # ---------------------------------------------------------------- store

    def store(
        self,
        result: ImportResult,
        *,
        source_path: str | Path | None = None,
        lineage: Mapping[str, Any] | None = None,
    ) -> str:
        """把一次成功导入落为不可变 revision，返回 `dataset_id@rev_NNNN`。

        `content_sha256` 相同则复用已有 revision（不重复写入）。
        """
        if not result.ok or result.dataset is None or result.table is None:
            raise VaultError("TFV-701", "存在 ERROR 级诊断的导入不得写入 vault")
        dataset = result.dataset
        dataset_id = dataset.manifest.dataset_id
        dtypes = {v.variable_id: v.dtype for v in dataset.variables}
        content_hash = _table_fingerprint(result.table, dtypes)
        src = Path(source_path) if source_path else result.source_path
        source_hash = _sha256_file(src) if src else None

        with self._locked():
            existing = self._with_db(
                lambda con: con.execute(
                    "SELECT revision, source_sha256 FROM revisions "
                    "WHERE dataset_id = ? AND content_sha256 = ?",
                    [dataset_id, content_hash],
                ).fetchall()
            )
            if existing:
                # 内容去重：复用 revision（conventions §5.2）
                return f"{dataset_id}@{existing[0][0]}"

            revision = self._next_revision(dataset_id)
            rev_dir = self.datasets_dir / dataset_id / revision
            if rev_dir.exists():
                raise VaultError(
                    "TFV-701", f"revision 已存在，不得覆盖: {rev_dir}"
                )
            (rev_dir / "source").mkdir(parents=True)
            (rev_dir / "canonical").mkdir()

            # source：原始文件永久只读（§10.1）
            stored_source_name = None
            if src:
                stored_source_name = src.name
                dst = rev_dir / "source" / src.name
                shutil.copyfile(src, dst)
                os.chmod(dst, 0o444)

            # canonical parquet（zstd）
            file_hashes: dict[str, str] = {}
            canonical_tables = {
                "data.parquet": result.table,
                "variables.parquet": _variables_table(dataset),
                "objects.parquet": _objects_table(dataset),
                "parameters.parquet": _parameters_table(dataset),
            }
            for name, table in canonical_tables.items():
                path = rev_dir / "canonical" / name
                _write_parquet_atomic(table, path)
                file_hashes[f"canonical/{name}"] = _sha256_file(path)

            now = datetime.now(timezone.utc).isoformat()
            ts = result.table.column("timestamp").to_pylist()
            fingerprints = {
                "source_sha256": source_hash,
                "content_sha256": content_hash,
                "file_sha256": file_hashes,
            }
            manifest_doc = json.loads(
                dataset.manifest.model_dump_json())
            quality_doc = {
                "dataset_id": dataset_id,
                "revision": revision,
                "imported_at": now,
                "importer_version": result.importer_version,
                "degradations": result.degradations,
                "diagnostics": [
                    {"code": d.code, "level": d.level.value,
                     "location": d.location, "count": d.count,
                     "message": d.message}
                    for d in result.diagnostics
                ],
            }
            profile_doc = build_profile(
                dataset, result.table, result.diagnostics,
                fingerprints=fingerprints,
                importer_version=result.importer_version,
                degradations=result.degradations,
            )
            lineage_doc = {
                "dataset_id": dataset_id,
                "revision": revision,
                "created_at": now,
                "source_file": str(src) if src else None,
                "stored_source": f"source/{stored_source_name}"
                if stored_source_name else None,
                "importer_version": result.importer_version,
                **(dict(lineage) if lineage else {}),
            }
            fingerprint_doc = {
                "dataset_id": dataset_id,
                "revision": revision,
                **fingerprints,
            }
            for name, doc in (
                ("manifest.json", manifest_doc),
                ("quality.json", quality_doc),
                ("profile.json", profile_doc),
                ("lineage.json", lineage_doc),
                ("fingerprint.json", fingerprint_doc),
            ):
                _write_json_atomic(rev_dir / name, doc)

            def _index(con) -> None:
                con.execute(
                    "INSERT OR IGNORE INTO datasets VALUES (?, ?)",
                    [dataset_id, now],
                )
                con.execute(
                    "INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [dataset_id, revision, content_hash, source_hash, now,
                     len(ts),
                     ts[0].isoformat() if ts else None,
                     ts[-1].isoformat() if ts else None],
                )
                con.executemany(
                    "INSERT INTO variables VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [[dataset_id, revision, v.variable_id, v.object_id,
                      v.property_code, v.unit, v.dtype, v.role]
                     for v in dataset.variables],
                )

            self._with_db(_index)
            return f"{dataset_id}@{revision}"

    def _next_revision(self, dataset_id: str) -> str:
        rows = self._with_db(
            lambda con: con.execute(
                "SELECT revision FROM revisions WHERE dataset_id = ?",
                [dataset_id],
            ).fetchall()
        )
        numbers = [int(r[0].split("_", 1)[1]) for r in rows]
        return f"rev_{(max(numbers) + 1) if numbers else 1:04d}"

    # ---------------------------------------------------------------- load

    def resolve(self, ref: str) -> RevisionInfo:
        """解析 `dataset_id@rev_NNNN`，不存在报 TFV-703。"""
        try:
            dataset_id, revision = ref.split("@", 1)
        except ValueError:
            raise VaultError("TFV-703", f"非法版本引用: {ref!r}") from None
        rev_dir = self.datasets_dir / dataset_id / revision
        fp_path = rev_dir / "fingerprint.json"
        if not rev_dir.is_dir() or not fp_path.exists():
            raise VaultError("TFV-703", f"数据版本不存在: {ref}")
        with open(fp_path, encoding="utf-8") as fp:
            fp_doc = json.load(fp)
        return RevisionInfo(
            dataset_id=dataset_id,
            revision=revision,
            content_sha256=fp_doc["content_sha256"],
            source_sha256=fp_doc.get("source_sha256") or "",
            path=rev_dir,
        )

    def _verify_files(self, info: RevisionInfo) -> None:
        """读取时校验落盘文件指纹（TFV-702）。"""
        with open(info.path / "fingerprint.json", encoding="utf-8") as fp:
            fp_doc = json.load(fp)
        for rel, expected in fp_doc.get("file_sha256", {}).items():
            path = info.path / rel
            if not path.exists() or _sha256_file(path) != expected:
                raise VaultError(
                    "TFV-702", f"落盘内容与记录指纹不符: {path}"
                )

    def load_data(self, ref: str) -> pa.Table:
        """加载 canonical/data.parquet（先校验文件指纹）。"""
        info = self.resolve(ref)
        self._verify_files(info)
        return pq.read_table(info.path / "canonical" / "data.parquet")

    def load_variables(self, ref: str) -> list[dict[str, Any]]:
        info = self.resolve(ref)
        table = pq.read_table(info.path / "canonical" / "variables.parquet")
        return table.to_pylist()

    def load_parameters(self, ref: str) -> list[dict[str, Any]]:
        info = self.resolve(ref)
        table = pq.read_table(info.path / "canonical" / "parameters.parquet")
        return table.to_pylist()

    def load_objects(self, ref: str) -> list[dict[str, Any]]:
        info = self.resolve(ref)
        table = pq.read_table(info.path / "canonical" / "objects.parquet")
        return table.to_pylist()

    def verify_source(self, ref: str) -> None:
        """读取原始文件时校验 source_sha256（TFV-702，§10.1）。"""
        info = self.resolve(ref)
        with open(info.path / "lineage.json", encoding="utf-8") as fp:
            lineage = json.load(fp)
        stored = lineage.get("stored_source")
        if not stored:
            return
        path = info.path / stored
        if not path.exists() or _sha256_file(path) != info.source_sha256:
            raise VaultError("TFV-702", f"原始文件指纹不符: {path}")

    def list_datasets(self) -> list[dict[str, Any]]:
        """列出全部数据集及其 revision 摘要（工具层 tf_dataset_list 用）。"""
        rows = self._with_db(
            lambda con: con.execute(
                "SELECT dataset_id, revision, content_sha256, rows, "
                "time_start, time_end, created_at FROM revisions "
                "ORDER BY dataset_id, revision"
            ).fetchall()
        )
        datasets: dict[str, dict[str, Any]] = {}
        for dataset_id, revision, content_hash, n_rows, t0, t1, created in rows:
            entry = datasets.setdefault(dataset_id, {
                "dataset_id": dataset_id, "revisions": [],
            })
            entry["revisions"].append({
                "revision": revision,
                "ref": f"{dataset_id}@{revision}",
                "content_sha256": content_hash,
                "rows": n_rows,
                "time_range": [t0, t1],
                "created_at": created,
            })
        return [datasets[k] for k in sorted(datasets)]

    def list_revisions(self, dataset_id: str) -> list[str]:
        return [
            r[0]
            for r in self._with_db(
                lambda con: con.execute(
                    "SELECT revision FROM revisions WHERE dataset_id = ? "
                    "ORDER BY revision",
                    [dataset_id],
                ).fetchall()
            )
        ]


# ---------------------------------------------------------------- canonical 表


def _variables_table(dataset) -> pa.Table:
    rows = [
        {
            "variable_id": v.variable_id,
            "object_id": v.object_id,
            "property_code": v.property_code,
            "unit": v.unit,
            "dtype": v.dtype,
            "role": v.role,
            "source_kind": v.source_kind,
            "nullable": v.nullable,
            "min_value": v.min_value,
            "max_value": v.max_value,
            "sample_period": v.sample_period,
            "aggregation": v.aggregation,
        }
        for v in dataset.variables
    ]
    return pa.Table.from_pylist(rows)


def _objects_table(dataset) -> pa.Table:
    rows = [
        {
            "object_id": o.object_id,
            "object_model_id": o.object_model_id,
            "object_name": o.object_name,
            "parent_id": o.parent_id,
            "system_id": o.system_id,
        }
        for o in dataset.objects
    ]
    return pa.Table.from_pylist(rows)


def _parameters_table(dataset) -> pa.Table:
    schema = pa.schema([
        ("object_id", pa.string()),
        ("parameter_code", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
    ])
    rows = [
        {
            "object_id": p.object_id,
            "parameter_code": p.parameter_code,
            "value": p.value,
            "unit": p.unit,
        }
        for p in dataset.parameters
    ]
    return pa.Table.from_pylist(rows, schema=schema)
