# 工程约定

本文定义实现阶段必须共同遵守的规范性细节：命名、单位、时间、哈希、版本和错误码。这些细节在设计文档中被引用但未展开，若不先行统一，不同模块会各自选择不兼容的做法。

> 本文中「必须 / 不得」为契约硬性要求，变更需提升契约版本；「应」为强烈建议，偏离需在实验或模型包中记录理由；「可」为可选。
>
> 标注 **[草案]** 的条目是首次提出的具体取值，尚未经过真实数据验证，实现前建议评审。

## 1. 命名与标识符

### 1.1 字符集与格式

| 对象 | 规则 | 正则 |
|---|---|---|
| `property_code` | `lower_snake_case`，仅 ASCII 小写字母、数字、下划线，不以数字开头 | `^[a-z][a-z0-9_]{0,63}$` |
| `object_id` | ASCII 大小写字母、数字、连字符、下划线，**不得包含 `.`** | `^[A-Za-z0-9_-]{1,64}$` |
| `variable_id` | `object_id` + `.` + `property_code` | `^[A-Za-z0-9_-]{1,64}\.[a-z][a-z0-9_]{0,63}$` |
| `object_model_id` | `name.vN` | `^[a-z][a-z0-9_]*\.v[0-9]+$` |
| `dataset_id` | ASCII 大写字母、数字、下划线 | `^[A-Z][A-Z0-9_]{0,63}$` |

`object_id` 禁止包含 `.` 是硬性要求：否则 `variable_id` 无法从右侧第一个 `.` 唯一切分。解析时必须按**第一个** `.` 切分，而非最后一个（`property_code` 不含 `.`，但 `object_id` 若违规包含 `.` 会导致静默错配）。

### 1.2 大小写

`property_code` 强制小写。`object_id` 与 `variable_id` **大小写敏感**，比较时不得做 casefold。

> **实现约束**：`variable_id` 不得直接用作文件名或目录名。Windows 与 macOS 默认文件系统大小写不敏感，`CH-01.x` 与 `ch-01.x` 会互相覆盖。落盘时必须使用 `sha256(variable_id)` 前 16 位或统一转义，真实名称存于元数据。

### 1.3 顺序 ID

`RG-`、`H-`、`EXP-`、`F-`、`D-`、`M-`、`VIEW-` 使用四位零填充十进制序号，超过 9999 后自然扩展位数（`RG-10000`），不重置、不复用、不回收。前缀含义见 [术语表 §4](./glossary.md)。

序号分配必须由单一写者串行完成（计数器文件 + 排他锁 + 原子替换）。若未来引入并发写入，应改用时间有序 ID（如 ULID）并保留可读别名，不得依赖「读取最大值 + 1」。

## 2. 单位

### 2.1 规范单位表示

单位字符串采用 **UCUM**（Unified Code for Units of Measure）区分大小写形式。文档中已使用的 `Cel`、`kW`、`m3/h` 均符合该规范。

| 物理量 | 规范单位 | UCUM 写法 | 允许的输入别名 **[草案]** |
|---|---|---|---|
| 温度 | 摄氏度 | `Cel` | `℃`、`°C`、`C`、`degC` |
| 温差 | 开尔文 | `K` | `Cel`（见下方说明） |
| 功率 | 千瓦 | `kW` | `KW`、`kw` |
| 电能 | 千瓦时 | `kW.h` | `kWh`、`kwh` |
| 体积流量 | 立方米每小时 | `m3/h` | `m³/h`、`CMH`、`m3/hr` |
| 质量流量 | 千克每秒 | `kg/s` | — |
| 压力 | 千帕 | `kPa` | `KPA` |
| 压差 | 千帕 | `kPa` | `kpa` |
| 相对湿度 | 百分比 | `%` | `%RH`、`RH` |
| 比率（PLR、开度） | 无量纲 | `1` | `-`、`ratio` |
| 频率 | 赫兹 | `Hz` | — |
| 时间 | 秒 | `s` | — |

别名映射必须是登记在册的确定性规则表；导入器**不得**对未登记的单位字符串做模糊匹配，应报 `TFDC-401`。

### 2.2 温度与温差

摄氏度与开尔文的换算是仿射的（`K = Cel + 273.15`），温差的换算是线性的（`ΔK = ΔCel`）。二者不能共用同一转换函数。

因此 TFOM 必须区分 `role`/量纲语义：`supply_return_temp_diff` 的单位是 `K` 而非 `Cel`，即使数值上等价。当前 [data-contract.md §3](./data-contract.md) 的示例把温差标为 `Cel`——实现时应统一为 `K`，否则任何自动单位换算都会引入 273.15 的偏移。

> **[草案] 建议**：TFOM 属性增加 `quantity_kind` 字段（`temperature` / `temperature_difference` / `power` / …），换算依据 `quantity_kind` 而非单位字符串。

### 2.3 百分比与比率

`%` 与 `1`（无量纲比率）的换算系数为 100。这是 HVAC 数据最常见的静默错误来源：相对湿度、阀门开度、PLR、变频器频率百分比在不同 BMS 中分别以 `0–1` 和 `0–100` 导出，两者都能通过「非负」检查。

规则：

- TFOM 必须为每个比率类属性显式声明单位（`%` 或 `1`）及 `min_value` / `max_value`。
- 导入器必须做**分布合理性检查**：声明为 `1` 但 95 分位 > 1.5，或声明为 `%` 但最大值 ≤ 1.0，报警告 `TFDC-606`，由人工确认后再导入。
- 转换只在导入边界执行一次，Dataset View 和模型内部只见规范单位。

### 2.4 换算精度

换算使用 IEEE-754 double，**不得**在换算后做四舍五入或有效数字截断。原始值与换算值均需保留时，派生列另立 `variable_id`，不覆盖原列。

## 3. 时间与时区

### 3.1 Excel 的时区陷阱

这是导入器最容易出错的地方，必须显式处理：

**xlsx 格式不存储时区。** Excel 的日期时间以「1900 日期系统的浮点序列号 + 显示格式」存储，`openpyxl` / `pandas` 读出的是 naive `datetime`。即使单元格显示为 `2026-01-01T00:00:00+08:00`，只要它被识别为日期类型，偏移量就已经丢失。

因此 TFDC-XLSX 规定：

- `data.timestamp` 列**必须**为**文本格式**单元格（单元格格式设为「文本」，或值以 `'` 前缀录入），内容为带偏移量的 ISO 8601 字符串。
- 导入器读取时必须以 `data_only=True` 且按原始类型判断：若该列被解析为 `datetime` 类型，说明它是 Excel 序列号而非文本，此时**不得**猜测时区，应报 `TFDC-502`，除非 `manifest.timezone` 已声明且用户显式允许按该时区解释（导入报告中必须记录这一降级行为）。
- 禁止 1904 日期系统的工作簿（macOS 旧版 Excel 默认），检测到时报 `TFDC-202`。

### 3.2 内部表示

- 规范存储为 **UTC 的 `timestamp[us, tz=UTC]`**（Arrow/Parquet）。
- `manifest.timezone` 必须是 IANA 时区名（`Asia/Shanghai`），不得使用 `CST`、`UTC+8`、`GMT+8` 等缩写或固定偏移——缩写有歧义（`CST` 同时指中国、美国中部、古巴），固定偏移无法表达夏令时。
- 展示、按小时/日聚合、季节划分时才转回站点本地时区。**跨日/跨月聚合必须在本地时区进行**，否则中国以外站点的「日」边界会错位。

### 3.3 夏令时

`Asia/Shanghai` 自 1991 年起无夏令时，但契约不得依赖这一点。对存在 DST 的站点：

- 春季跳变（不存在的本地时刻）：报 `TFDC-501`。
- 秋季折返（重复的本地时刻）：若时间戳带偏移量可无歧义解析；若为 naive 降级解释，报 `TFDC-507` 并中止。

### 3.4 采样与重采样

- `manifest.time_resolution` 使用 ISO 8601 duration 或 `<数字><单位>` 简写（`60s`、`5min`、`1h`），二者择一并在实现中固定 **[草案：采用简写，正则 `^[0-9]+(s|min|h|d)$`]**。
- 重采样区间约定：**左闭右开，标签取左边界**（`label='left'`, `closed='left'`）。即 `00:05` 的 5min 桶覆盖 `[00:05:00, 00:10:00)`。此约定必须在 Dataset View 物化和在线特征计算中保持一致，否则离线训练与在线推理存在半个窗口的错位。
- 桶内有效样本数低于 `min_count`（**[草案：默认为期望样本数的 50%]**）时，该桶输出为空值而非部分聚合结果，并计入质量报告。
- `aggregation` 允许值 **[草案]**：`mean`、`sum`、`min`、`max`、`first`、`last`、`median`。累积量（电能表读数）必须使用 `last` 后差分，不得使用 `mean`。

### 3.5 重复与乱序

| 情况 | 处理 |
|---|---|
| 完全重复的时间戳且各列值一致 | 去重保留一条，记入质量报告（警告 `TFDC-503`） |
| 重复时间戳但值不一致 | 错误 `TFDC-503`，中止导入。**不得**自动取均值或保留首条 |
| 时间戳乱序 | 警告 `TFDC-504`，按时间排序后继续 |
| 时间空洞 | 警告 `TFDC-505`，不填充。填充策略属于 Dataset View 的职责 |

原始导入层**不做任何插值和填充**。这是「原始数据只读」原则在时间维度上的具体落实。

## 4. 数据类型与缺失值

### 4.1 类型映射

| TFDC `dtype` | Arrow / Parquet | pandas | 说明 |
|---|---|---|---|
| `float` | `double` | `float64` | 缺失为 `null`，不是 `NaN` |
| `integer` | `int64` | `Int64`（可空） | **不得**使用 numpy `int64`，其无法表达缺失 |
| `boolean` | `bool` | `boolean`（可空） | 只接受 `TRUE`/`FALSE`，不接受 `1`/`0`/`是`/`否` |
| `string` | `string` | `string` | 空串与缺失不等价，必须区分 |
| `timestamp` | `timestamp[us, tz=UTC]` | `datetime64[us, UTC]` | 仅用于 `data.timestamp` |

### 4.2 缺失值

- Excel 中缺失**只能**表示为空单元格。
- `NULL`、`N/A`、`NA`、`--`、`-`、`#N/A`、`9999`、`-9999`、`99999`、`32767`、`65535` 视为哨兵值，报 `TFDC-603`。哨兵值列表可按站点扩展，但必须显式登记，不得静默接受。
- 空白字符串（含全角空格）在数值列中等同缺失，在字符串列中报 `TFDC-603`。
- **NaN 与 null 必须区分**：导入层不产生 NaN；计算层若产生 NaN（如 `0/0`），属于缺陷，应触发实验失败而非静默传播。

### 4.3 数值规范化

- 所有浮点数按 IEEE-754 double 存储。
- `-0.0` 规范化为 `0.0`。
- 禁止 `inf` / `-inf` 进入 Data Vault，检测到报 `TFDC-602`。
- NaN 在哈希计算中必须规范化为固定字节序列（见 §5.2），否则不同来源的 NaN 位模式会产生不同指纹。

## 5. 规范化与哈希

所有哈希算法统一为 **SHA-256**，以小写十六进制表示；在 ID 和路径中显示时取前 16 位（64 bit），完整值存于元数据。前 16 位仅用于展示和目录命名，**比较和校验必须使用完整值**。

### 5.1 结构化定义的规范化（Canonical JSON）

用于 Dataset View、Research Goal、Experiment 等 YAML/JSON 定义的哈希。规则 **[草案]**：

1. YAML 解析为数据结构后序列化为 JSON，UTF-8 编码，不转义非 ASCII（`ensure_ascii=False`）。
2. 对象键按 Unicode 码点升序排序。
3. 分隔符无空格：`,` 与 `:`。
4. 浮点数使用最短往返表示（Python `repr` 语义）；整数值的浮点数写为整数形式（`5.0` → `5`）需**禁止**，以免 `5` 与 `5.0` 混淆——统一保留类型。
5. 值为 `null` 的键与键缺失等价，序列化前一律删除。
6. 数组默认**保留顺序**；仅对契约中显式声明为集合语义的字段（如 `objects`、`features`、`metrics`）按字典序排序后再哈希。该字段清单必须写进 Schema，不得由实现自行判断。
7. 排除不影响语义的字段：`description`、`name_zh`、`name_en`、`created_at`、`author`、`comment`、`tags`。

```text
view_hash = sha256(dataset_revision_id + "\n" + canonical_json(view_definition))
```

`view_hash` 必须包含数据集 revision，否则同一 View 定义在不同数据版本上会命中同一份缓存。

### 5.2 数据内容指纹

Excel 文件是 ZIP 容器，内部包含 `docProps/core.xml` 的修改时间戳。**同一份数据用 Excel 打开再保存，字节哈希必然改变**，因此不能只用文件字节哈希做去重。

TFDC 定义两个独立指纹：

| 指纹 | 计算对象 | 用途 |
|---|---|---|
| `source_sha256` | 原始文件字节 | 溯源、确认原件未被篡改 |
| `content_sha256` | 规范化后的逻辑数据内容 | 去重、判断是否需要新建 revision |

`content_sha256` 计算规则 **[草案]**：

1. 列按 `variable_id` 字典序排序；行按 `timestamp` 升序排序。
2. 每列计算列摘要：`sha256(variable_id ‖ 0x00 ‖ dtype ‖ 0x00 ‖ 值字节流)`。
3. 值字节流按行拼接，小端序：`double` 8 字节，`int64` 8 字节，`bool` 1 字节，`string` 为 `长度(uint32) ‖ UTF-8 字节`。
4. **缺失值统一编码为 8 字节 `0xFF` 重复序列**（对 string 编码为长度 `0xFFFFFFFF`），不写入任何 NaN 位模式。
5. `content_sha256 = sha256(所有列摘要按列序拼接)`。

规则 4 是关键：NaN 有多种位模式（quiet/signaling、不同 payload），直接哈希浮点字节会导致逻辑相同的数据产生不同指纹。

去重语义：`content_sha256` 相同则复用已有 revision，即使 `source_sha256` 不同（对应「同一份数据被重新保存」）；`content_sha256` 不同必须新建 revision，**不得**覆盖。

> **不要**用 Parquet 文件的字节哈希作为内容指纹。Parquet 的压缩、行组大小、字典编码、写入库版本和统计信息都会影响字节，逻辑相同的数据可产生不同文件。

### 5.3 环境指纹

```text
environment_lock = sha256(canonical_json({
  python: "3.12.x",
  platform: "win_amd64" | "linux_x86_64" | ...,
  packages: [{name, version, sha256}, ...]  // 按 name 排序
}))
```

必须同时记录（但不纳入哈希）：CPU 型号、逻辑核数、BLAS 实现与线程数环境变量。跨平台复现时指纹必然不同，此时指标比对采用放宽容差（见 [implementation-notes.md](./implementation-notes.md)）。

## 6. 版本与兼容

| 对象 | 版本形式 | 递增规则 |
|---|---|---|
| 契约（TFDC / TFOM Schema） | `MAJOR.MINOR` | 不兼容改动升 MAJOR；新增可选字段升 MINOR |
| 物模型 `object_model_id` | `name.vN` | 属性语义、单位或约束变化即升 N；**不允许**原地修改 |
| 数据集 revision | `rev_NNNN` | 内容变化即新建，不可变 |
| 模型 | 语义化 `MAJOR.MINOR.PATCH` | 见 [model-package.md §5](./model-package.md) |

模型包必须声明它兼容的物模型版本范围，而非单一版本：

```yaml
object_model: chiller.v1
object_model_compat: ">=1.0,<2.0"   # [草案] 建议新增字段
```

导入器必须拒绝 `contract_version` 的 MAJOR 高于自身支持范围的文件（`TFDC-203`）；MINOR 更高时可读取，但必须警告存在未识别字段。

## 7. 错误码

### 7.1 级别

| 级别 | 含义 | 行为 |
|---|---|---|
| `ERROR` | 违反硬性契约 | 中止，不产生 revision |
| `REJECT` | 局部数据不可用 | 剔除相关行/列并记录，其余继续；剔除比例超阈值升级为 `ERROR` |
| `WARN` | 可疑但可继续 | 记入质量报告，需人工确认 |

所有诊断必须返回结构化对象，包含 `code`、`level`、`message`、`location`（sheet / 行 / 列 / `variable_id`）和 `count`。**不得**只返回自由文本。同类问题必须聚合计数，不得逐行刷屏。

### 7.2 结构与格式 `TFDC-1xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-101 | `FILE_FORMAT_UNSUPPORTED` | 非 `.xlsx`（含 `.xls`、`.xlsm`、`.csv`） | ERROR |
| TFDC-102 | `MACRO_PRESENT` | 工作簿含宏或 VBA 工程 | ERROR |
| TFDC-103 | `SHEET_MISSING` | 缺少 `manifest`/`objects`/`variables`/`data` | ERROR |
| TFDC-104 | `SHEET_UNKNOWN` | 存在契约未定义的工作表 | WARN |
| TFDC-105 | `MERGED_CELL_PRESENT` | 存在合并单元格 | ERROR |
| TFDC-106 | `HEADER_EMPTY` | 表头存在空列名 | ERROR |
| TFDC-107 | `HEADER_DUPLICATED` | 表头列名重复 | ERROR |
| TFDC-108 | `FORMULA_PRESENT` | `data` 中存在公式 | ERROR |
| TFDC-109 | `EXTERNAL_LINK_PRESENT` | 存在外部工作簿引用 | ERROR |
| TFDC-110 | `SHEET_SIZE_EXCEEDED` | 超出 1,048,576 行或 16,384 列 | ERROR |

### 7.3 清单 `TFDC-2xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-201 | `MANIFEST_KEY_MISSING` | 缺少必需 key | ERROR |
| TFDC-202 | `MANIFEST_VALUE_INVALID` | 值格式非法（含 1904 日期系统） | ERROR |
| TFDC-203 | `CONTRACT_VERSION_UNSUPPORTED` | MAJOR 超出支持范围 | ERROR |
| TFDC-204 | `TIMEZONE_UNKNOWN` | 非 IANA 时区名 | ERROR |
| TFDC-205 | `RESOLUTION_INVALID` | `time_resolution` 不符合格式 | ERROR |

### 7.4 引用与语义 `TFDC-3xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-301 | `OBJECT_UNKNOWN` | 引用 `objects` 中不存在的对象 | ERROR |
| TFDC-302 | `OBJECT_MODEL_UNKNOWN` | `object_model_id` 未在 TFOM Registry 注册 | ERROR |
| TFDC-303 | `PROPERTY_UNKNOWN` | `property_code` 不属于对应 TFOM | ERROR |
| TFDC-304 | `VARIABLE_ID_MALFORMED` | 不符合 §1.1 正则 | ERROR |
| TFDC-305 | `VARIABLE_UNDECLARED` | `data` 列未在 `variables` 声明 | ERROR |
| TFDC-306 | `VARIABLE_COLUMN_MISSING` | `variables` 已声明但 `data` 无对应列 | WARN（`nullable=false` 时为 ERROR） |
| TFDC-307 | `VARIABLE_DUPLICATED` | `variable_id` 重复声明 | ERROR |
| TFDC-308 | `RELATION_INVALID` | `relations` 引用非法或端口不匹配 | ERROR |
| TFDC-309 | `OBJECT_HIERARCHY_CYCLE` | `parent_id` 构成环 | ERROR |

### 7.5 单位与类型 `TFDC-4xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-401 | `UNIT_UNKNOWN` | 单位字符串不在规范表或别名表 | ERROR |
| TFDC-402 | `UNIT_INCOMPATIBLE` | 与 TFOM 声明的量纲不一致 | ERROR |
| TFDC-403 | `UNIT_CONVERSION_UNREGISTERED` | 量纲一致但无登记换算规则 | ERROR |
| TFDC-404 | `DTYPE_MISMATCH` | 实际值类型与声明 `dtype` 不符 | REJECT |

### 7.6 时间 `TFDC-5xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-501 | `TIMESTAMP_UNPARSEABLE` | 无法解析，或落在 DST 跳变的不存在时刻 | ERROR |
| TFDC-502 | `TIMESTAMP_NAIVE` | 无时区信息且未授权按 manifest 时区降级 | ERROR |
| TFDC-503 | `TIMESTAMP_DUPLICATED` | 重复时间戳（值不一致时为 ERROR） | WARN / ERROR |
| TFDC-504 | `TIMESTAMP_OUT_OF_ORDER` | 时间戳非单调递增 | WARN |
| TFDC-505 | `TIMESTAMP_GAP` | 存在超过 N 倍采样周期的空洞 | WARN |
| TFDC-506 | `RESOLUTION_DRIFT` | 实际采样间隔偏离声明值 | WARN |
| TFDC-507 | `TIMESTAMP_AMBIGUOUS` | DST 折返导致的重复本地时刻 | ERROR |

### 7.7 数值与质量 `TFDC-6xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFDC-601 | `RANGE_VIOLATION` | 超出 TFOM `min_value`/`max_value` | REJECT |
| TFDC-602 | `VALUE_NOT_FINITE` | `inf` / `-inf`，或 `nullable=false` 时为空 | ERROR |
| TFDC-603 | `SENTINEL_VALUE_DETECTED` | 检测到哨兵值 | ERROR |
| TFDC-604 | `DATA_CONSISTENCY_ERROR` | 派生量与 TFOM `expression` 不一致超出容差 | ERROR |
| TFDC-605 | `CONSTANT_SERIES` | 变量全程恒定，疑似死点 | WARN |
| TFDC-606 | `DISTRIBUTION_IMPLAUSIBLE` | 分布与声明单位不符（如比率类超出量级） | WARN |
| TFDC-607 | `MISSING_RATE_HIGH` | 缺失率超阈值 | WARN |

### 7.8 数据仓与视图 `TFV-7xx` / `TFV-8xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFV-701 | `REVISION_IMMUTABLE_VIOLATION` | 试图修改已有 revision | ERROR |
| TFV-702 | `FINGERPRINT_MISMATCH` | 落盘内容与记录指纹不符 | ERROR |
| TFV-703 | `DATASET_REVISION_NOT_FOUND` | 引用不存在的数据版本 | ERROR |
| TFV-801 | `VIEW_HASH_MISMATCH` | 缓存物化结果与 View 定义哈希不符 | ERROR |
| TFV-802 | `VIEW_FEATURE_UNAVAILABLE` | View 请求的变量在该数据版本中不存在 | ERROR |
| TFV-803 | `VIEW_EMPTY_RESULT` | 过滤后无样本 | ERROR |

### 7.9 实验与发布 `TFX-9xx` / `TFM-10xx`

| 代码 | 名称 | 触发条件 | 级别 |
|---|---|---|---|
| TFX-901 | `ENVIRONMENT_LOCK_MISMATCH` | 运行环境与实验声明不符 | ERROR |
| TFX-902 | `SEED_MISSING` | 未声明随机种子 | ERROR |
| TFX-903 | `SPLIT_LEAKAGE_DETECTED` | 训练集与测试集在时间或设备维度重叠 | ERROR |
| TFX-904 | `EXPERIMENT_IMMUTABLE_VIOLATION` | 试图修改已完成实验 | ERROR |
| TFX-905 | `METRIC_UNDEFINED` | 指标无法计算（如 MAPE 有效样本不足） | ERROR |
| TFX-906 | `BUDGET_EXCEEDED` | 超出实验数、时长或计算预算 | WARN（触发停止） |
| TFM-1001 | `SIGNATURE_INCOMPATIBLE` | 模型签名与目标 TFOM 版本不兼容 | ERROR |
| TFM-1002 | `CHECKSUM_MISMATCH` | 模型包文件校验和不符 | ERROR |
| TFM-1003 | `ACCEPTANCE_NOT_MET` | 未满足硬性验收条件 | ERROR |
| TFM-1004 | `LATENCY_EXCEEDED` | 推理延迟超标 | ERROR |
| TFM-1005 | `SMOKE_TEST_FAILED` | 冷环境加载或最小推理失败 | ERROR |
| TFM-1006 | `VERSION_CONFLICT` | 同版本号内容发生变化 | ERROR |
| TFM-1007 | `ROLLBACK_TARGET_MISSING` | 无可回滚的上一生产版本 | ERROR |

## 8. 待决策

以下条目影响实现但尚未定稿，建议在 Phase 0 结束前明确：

| # | 议题 | 备选 |
|---:|---|---|
| 1 | TFOM 是否引入 `quantity_kind` 字段 | 引入（换算更安全） / 仅靠单位字符串推断 |
| 2 | `time_resolution` 格式 | 简写 `60s` / ISO 8601 `PT60S` |
| 3 | 温差单位统一为 `K` | 是（需修订 data-contract 示例） / 保持 `Cel` 并特殊处理 |
| 4 | 数据集与实验的存储后端 | 文件系统 + DuckDB（一期） / 直接上 PostgreSQL |
| 5 | 超宽数据（变量数 > 16383）的 Excel 表达 | 多个 `data_N` 表 / 强制长表格式 / 改用 TFDC-Parquet |
| 6 | 集合语义字段清单（影响哈希稳定性） | 需逐 Schema 明确 |
| 7 | 质量阈值默认值（缺失率、剔除比例、桶最小样本数） | 需用真实数据标定 |

## 相关文档

- [实现细则与已知陷阱](./implementation-notes.md) — 本文约定在代码中的落地方式与边界情况
- [TFDC 数据契约](./data-contract.md) — 本文展开的契约主体
- [开放议题](./open-questions.md) — 本文 §8 的待决策项在此有更完整的判据和时点建议
- [术语表](./glossary.md) · [文档总览](./README.md)

> **注意**：项目当前处于[方案讨论阶段](./scope.md)，本文的约定在方案定型前仍可能变动，标注 **[草案]** 的取值尤其如此。
