"""运行时层错误（conventions.md §7.9 的 TFM-10xx）。

与 `thermoforge_research.errors.ResearchError` 同构：异常携带注册表中的
错误码，机器可判定（DD-13），不依赖自由文本解析。
"""

from __future__ import annotations

from thermoforge_core.errors import ERROR_REGISTRY


class ModelRegistryError(RuntimeError):
    """模型注册/发布错误，携带 conventions.md §7.9 的 TFM-10xx 错误码。"""

    def __init__(self, code: str, message: str):
        if code not in ERROR_REGISTRY:
            raise ValueError(f"未登记的错误码: {code!r}")
        self.code = code
        super().__init__(f"[{code}] {message}")
