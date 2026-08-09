"""顺序 ID 分配器（conventions.md §1.3）。

前缀 RG- / H- / EXP- / F- / D- / M- / VIEW-，四位零填充十进制序号，
超过 9999 后自然扩展位数，不重置、不复用、不回收。

分配由单一写者串行完成：计数器文件 + 排他文件锁 + 原子替换（os.replace）。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

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

ID_PREFIXES: tuple[str, ...] = ("RG-", "H-", "EXP-", "F-", "D-", "M-", "VIEW-")

_ID_PATTERN = {prefix: f"^{prefix}[0-9]{{4,}}$" for prefix in ID_PREFIXES}


class IdAllocationError(RuntimeError):
    """顺序 ID 分配失败。"""


class IdAllocator:
    """基于计数器文件的顺序 ID 分配器（单写者）。

    用法::

        allocator = IdAllocator(Path("research/ids"))
        exp_id = allocator.allocate("EXP-")   # -> "EXP-0001"
    """

    def __init__(self, directory: str | os.PathLike[str]):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._dir / "id_counters.json"
        self._lock_path = self._dir / "id_counters.lock"

    @staticmethod
    def _validate_prefix(prefix: str) -> None:
        if prefix not in ID_PREFIXES:
            raise IdAllocationError(f"未登记的 ID 前缀: {prefix!r}（允许 {ID_PREFIXES}）")

    def _read_counters(self) -> dict[str, int]:
        if not self._counter_path.exists():
            return {}
        with open(self._counter_path, encoding="utf-8", newline="") as fp:
            data = json.load(fp)
        return {k: int(v) for k, v in data.items()}

    def _write_counters_atomic(self, counters: dict[str, int]) -> None:
        # 临时文件 + os.replace：同卷内原子替换（implementation-notes §10.1）
        fd, tmp_name = tempfile.mkstemp(
            dir=self._dir, prefix=".id_counters.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
                json.dump(counters, fp, ensure_ascii=False, sort_keys=True)
                fp.write("\n")
            os.replace(tmp_name, self._counter_path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def allocate(self, prefix: str) -> str:
        """分配下一个 ID，如 `RG-0001`。"""
        self._validate_prefix(prefix)
        with open(self._lock_path, "a+b") as lock_fp:
            _lock(lock_fp)
            try:
                counters = self._read_counters()
                next_value = counters.get(prefix, 0) + 1
                counters[prefix] = next_value
                self._write_counters_atomic(counters)
            finally:
                _unlock(lock_fp)
        return f"{prefix}{next_value:04d}"

    def peek(self, prefix: str) -> int:
        """返回该前缀已分配到的序号（未分配过为 0）。"""
        self._validate_prefix(prefix)
        return self._read_counters().get(prefix, 0)
