"""ThermoForge 核心契约库（Phase 0：契约定稿）。

实现 docs/conventions.md 的规范性内容：命名、单位、时间、哈希、
顺序 ID、错误码；并提供五份契约的 pydantic 模型（contracts 子包）。
"""

from . import canonical, errors, fingerprint, ids, naming, timeutil, units

__all__ = ["canonical", "errors", "fingerprint", "ids", "naming", "timeutil", "units"]
