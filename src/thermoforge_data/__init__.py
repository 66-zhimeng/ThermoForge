"""ThermoForge 数据层（Phase 1：数据底座）。

- `importer`：TFDC-XLSX 导入器与验证器（ZIP 结构检查 + 全量语义/质量校验）。
- `vault`：不可变 Data Vault（revision、content 指纹去重、DuckDB 索引）。
- `profile`：数据画像与质量报告（data-contract §8）。
- `views`：Dataset View 定义、哈希与物化缓存（data-contract §7）。
- `legacy`：旧格式 11 表多级表头工作簿适配器（risks R8）。
- `preprocess`：预处理规则库与执行器（I-49、DD-02 方案 C；Agent 只提参数，
  执行是预先注册的确定性变换，审批门禁 + 全程留痕）。
"""

from . import importer, legacy, preprocess, profile, vault, views

__all__ = ["importer", "legacy", "preprocess", "profile", "vault", "views"]
