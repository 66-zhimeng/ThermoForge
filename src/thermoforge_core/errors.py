"""结构化诊断与错误码注册表（conventions.md §7）。

所有诊断必须返回结构化对象（§7.1），不得只返回自由文本。
注册表是 §7 错误码表的唯一实现来源，`tests/test_errors.py`
会与 conventions.md 原文逐条比对，防止两边漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Level(str, Enum):
    """诊断级别（conventions.md §7.1）。"""

    ERROR = "ERROR"  # 违反硬性契约：中止，不产生 revision
    REJECT = "REJECT"  # 局部数据不可用：剔除相关行/列，其余继续
    WARN = "WARN"  # 可疑但可继续：记入质量报告，需人工确认


@dataclass(frozen=True)
class Diagnostic:
    """结构化诊断对象（conventions.md §7.1）。

    同类问题必须聚合计数（`count`），不得逐行刷屏。
    """

    code: str
    level: Level
    message: str
    location: str | None = None  # sheet / 行 / 列 / variable_id
    count: int = 1

    def __post_init__(self) -> None:
        if self.code not in ERROR_REGISTRY:
            raise ValueError(f"未登记的错误码: {self.code!r}")
        if self.count < 1:
            raise ValueError("count 必须 >= 1")


@dataclass(frozen=True)
class ErrorDef:
    """错误码注册表条目。"""

    code: str
    name: str
    level: str  # 主级别；个别代码依上下文为 WARN/ERROR 双级别，见 trigger 说明
    trigger: str  # 触发条件简述


def _e(code: str, name: str, level: str, trigger: str) -> ErrorDef:
    return ErrorDef(code=code, name=name, level=level, trigger=trigger)


# conventions.md §7 全量注册表（61 条）
ERROR_REGISTRY: dict[str, ErrorDef] = {e.code: e for e in [
    # §7.2 结构与格式 TFDC-1xx
    _e("TFDC-101", "FILE_FORMAT_UNSUPPORTED", "ERROR", "非 .xlsx（含 .xls、.xlsm、.csv）"),
    _e("TFDC-102", "MACRO_PRESENT", "ERROR", "工作簿含宏或 VBA 工程"),
    _e("TFDC-103", "SHEET_MISSING", "ERROR", "缺少 manifest/objects/variables/data"),
    _e("TFDC-104", "SHEET_UNKNOWN", "WARN", "存在契约未定义的工作表"),
    _e("TFDC-105", "MERGED_CELL_PRESENT", "ERROR", "存在合并单元格"),
    _e("TFDC-106", "HEADER_EMPTY", "ERROR", "表头存在空列名"),
    _e("TFDC-107", "HEADER_DUPLICATED", "ERROR", "表头列名重复"),
    _e("TFDC-108", "FORMULA_PRESENT", "ERROR", "data 中存在公式"),
    _e("TFDC-109", "EXTERNAL_LINK_PRESENT", "ERROR", "存在外部工作簿引用"),
    _e("TFDC-110", "SHEET_SIZE_EXCEEDED", "ERROR", "超出 1,048,576 行或 16,384 列"),
    # §7.3 清单 TFDC-2xx
    _e("TFDC-201", "MANIFEST_KEY_MISSING", "ERROR", "缺少必需 key"),
    _e("TFDC-202", "MANIFEST_VALUE_INVALID", "ERROR", "值格式非法（含 1904 日期系统）"),
    _e("TFDC-203", "CONTRACT_VERSION_UNSUPPORTED", "ERROR", "MAJOR 超出支持范围"),
    _e("TFDC-204", "TIMEZONE_UNKNOWN", "ERROR", "非 IANA 时区名"),
    _e("TFDC-205", "RESOLUTION_INVALID", "ERROR", "time_resolution 不符合格式"),
    # §7.4 引用与语义 TFDC-3xx
    _e("TFDC-301", "OBJECT_UNKNOWN", "ERROR", "引用 objects 中不存在的对象"),
    _e("TFDC-302", "OBJECT_MODEL_UNKNOWN", "ERROR", "object_model_id 未在 TFOM Registry 注册"),
    _e("TFDC-303", "PROPERTY_UNKNOWN", "ERROR", "property_code 不属于对应 TFOM"),
    _e("TFDC-304", "VARIABLE_ID_MALFORMED", "ERROR", "不符合 §1.1 正则"),
    _e("TFDC-305", "VARIABLE_UNDECLARED", "ERROR", "data 列未在 variables 声明"),
    _e("TFDC-306", "VARIABLE_COLUMN_MISSING", "WARN", "variables 已声明但 data 无对应列；nullable=false 时为 ERROR"),
    _e("TFDC-307", "VARIABLE_DUPLICATED", "ERROR", "variable_id 重复声明"),
    _e("TFDC-308", "RELATION_INVALID", "ERROR", "relations 引用非法或端口不匹配"),
    _e("TFDC-309", "OBJECT_HIERARCHY_CYCLE", "ERROR", "parent_id 构成环"),
    # §7.5 单位与类型 TFDC-4xx
    _e("TFDC-401", "UNIT_UNKNOWN", "ERROR", "单位字符串不在规范表或别名表"),
    _e("TFDC-402", "UNIT_INCOMPATIBLE", "ERROR", "与 TFOM 声明的量纲不一致"),
    _e("TFDC-403", "UNIT_CONVERSION_UNREGISTERED", "ERROR", "量纲一致但无登记换算规则"),
    _e("TFDC-404", "DTYPE_MISMATCH", "REJECT", "实际值类型与声明 dtype 不符"),
    # §7.6 时间 TFDC-5xx
    _e("TFDC-501", "TIMESTAMP_UNPARSEABLE", "ERROR", "无法解析，或落在 DST 跳变的不存在时刻"),
    _e("TFDC-502", "TIMESTAMP_NAIVE", "ERROR", "无时区信息且未授权按 manifest 时区降级"),
    _e("TFDC-503", "TIMESTAMP_DUPLICATED", "WARN / ERROR", "重复时间戳；值不一致时为 ERROR"),
    _e("TFDC-504", "TIMESTAMP_OUT_OF_ORDER", "WARN", "时间戳非单调递增"),
    _e("TFDC-505", "TIMESTAMP_GAP", "WARN", "存在超过 N 倍采样周期的空洞"),
    _e("TFDC-506", "RESOLUTION_DRIFT", "WARN", "实际采样间隔偏离声明值"),
    _e("TFDC-507", "TIMESTAMP_AMBIGUOUS", "ERROR", "DST 折返导致的重复本地时刻"),
    # §7.7 数值与质量 TFDC-6xx
    _e("TFDC-601", "RANGE_VIOLATION", "REJECT", "超出 TFOM min_value/max_value"),
    _e("TFDC-602", "VALUE_NOT_FINITE", "ERROR", "inf / -inf，或 nullable=false 时为空"),
    _e("TFDC-603", "SENTINEL_VALUE_DETECTED", "ERROR", "检测到哨兵值"),
    _e("TFDC-604", "DATA_CONSISTENCY_ERROR", "ERROR", "派生量与 TFOM expression 不一致超出容差"),
    _e("TFDC-605", "CONSTANT_SERIES", "WARN", "变量全程恒定，疑似死点"),
    _e("TFDC-606", "DISTRIBUTION_IMPLAUSIBLE", "WARN", "分布与声明单位不符（如比率类超出量级）"),
    _e("TFDC-607", "MISSING_RATE_HIGH", "WARN", "缺失率超阈值"),
    # §7.8 数据仓与视图 TFV-7xx / TFV-8xx
    _e("TFV-701", "REVISION_IMMUTABLE_VIOLATION", "ERROR", "试图修改已有 revision"),
    _e("TFV-702", "FINGERPRINT_MISMATCH", "ERROR", "落盘内容与记录指纹不符"),
    _e("TFV-703", "DATASET_REVISION_NOT_FOUND", "ERROR", "引用不存在的数据版本"),
    _e("TFV-801", "VIEW_HASH_MISMATCH", "ERROR", "缓存物化结果与 View 定义哈希不符"),
    _e("TFV-802", "VIEW_FEATURE_UNAVAILABLE", "ERROR", "View 请求的变量在该数据版本中不存在"),
    _e("TFV-803", "VIEW_EMPTY_RESULT", "ERROR", "过滤后无样本"),
    # §7.9 实验与发布 TFX-9xx / TFM-10xx
    _e("TFX-901", "ENVIRONMENT_LOCK_MISMATCH", "ERROR", "运行环境与实验声明不符"),
    _e("TFX-902", "SEED_MISSING", "ERROR", "未声明随机种子"),
    _e("TFX-903", "SPLIT_LEAKAGE_DETECTED", "ERROR", "训练集与测试集在时间或设备维度重叠"),
    _e("TFX-904", "EXPERIMENT_IMMUTABLE_VIOLATION", "ERROR", "试图修改已完成实验"),
    _e("TFX-905", "METRIC_UNDEFINED", "ERROR", "指标无法计算（如 MAPE 有效样本不足）"),
    _e("TFX-906", "BUDGET_EXCEEDED", "WARN", "超出实验数、时长或计算预算（触发停止）"),
    _e("TFM-1001", "SIGNATURE_INCOMPATIBLE", "ERROR", "模型签名与目标 TFOM 版本不兼容"),
    _e("TFM-1002", "CHECKSUM_MISMATCH", "ERROR", "模型包文件校验和不符"),
    _e("TFM-1003", "ACCEPTANCE_NOT_MET", "ERROR", "未满足硬性验收条件"),
    _e("TFM-1004", "LATENCY_EXCEEDED", "ERROR", "推理延迟超标"),
    _e("TFM-1005", "SMOKE_TEST_FAILED", "ERROR", "冷环境加载或最小推理失败"),
    _e("TFM-1006", "VERSION_CONFLICT", "ERROR", "同版本号内容发生变化"),
    _e("TFM-1007", "ROLLBACK_TARGET_MISSING", "ERROR", "无可回滚的上一生产版本"),
]}

assert len(ERROR_REGISTRY) == 61, "错误码注册表条目数与 conventions.md §7 不一致"
