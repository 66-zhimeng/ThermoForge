"""五份契约的 pydantic v2 模型（Phase 0）。

- `tfom`：物模型（TFOM v1）
- `tfdc`：TFDC-XLSX 各表 record 结构与整册容器
- `research_goal`：Research Goal（research-loop.md §1，DD-14 纳入 NMBE）
- `experiment`：Experiment Contract（research-loop.md §5）
- `model_package`：模型包（model-package.md §2–5）

每个模块的 `SET_SEMANTIC_FIELDS` 声明哈希前需按字典序排序的数组字段
（conventions.md §5.1 规则 6），并写入 JSON Schema 的 `x-tf-set-semantics`。
JSON Schema 由 `thermoforge_core.contracts.export` 导出到顶层 `contracts/` 目录。
"""

from .experiment import Experiment
from .model_package import ModelPackage
from .research_goal import ResearchGoal
from .tfdc import TfdcDataset
from .tfom import ObjectModel

__all__ = ["Experiment", "ModelPackage", "ObjectModel", "ResearchGoal", "TfdcDataset"]
