"""模型实验室模块的公开协议与辅助函数（thermoforge_research.model_lab 的配套）。

实验室模块是 agent 起草、人审批后进入实验的单文件 Python 模块
（design-decisions DD-02 方案 C 的落地）。本文件是**唯一**建议 lab 模块
import 的项目内依赖（`thermoforge_models` 根包在静态扫描白名单内）。

## 模块协议

一个实验室模块（`<name>.py`）必须定义::

    MODEL_FORMAT = "thermoforge.lab.<name>.v1"   # 前缀固定，见 MODEL_FORMAT_PREFIX
    INPUT_ROLES = ["cooling_load", "t_evap_out"]  # 物理式角色名；[] = 通用模型

    def build_model(hyperparameters: Mapping, seed: int):
        '''按契约超参与种子构造模型对象。hyperparameters 是契约原文
        （标量字典，含 "lab" 引用与 "inputs" 映射字符串）。'''

    def load_model(directory):
        '''从 save() 产物目录重建模型（发布包冷加载走这里）。'''

模型对象必须提供::

    fit(df: pandas.DataFrame, y) -> self   # df 是视图行（含全部列），自行选列
    predict(df: pandas.DataFrame) -> 等长数值序列
    save(directory) -> None                # 必须写 model.json，且含
                                           # "format" == 本模块 MODEL_FORMAT

## 纪律

- **确定性是硬要求**：同种子同数据重跑必须逐位一致（runner §7.3 的机器
  判定对 lab 模型同样生效）。随机源一律用传入的 seed（如
  `numpy.random.default_rng(seed)`），禁止 `time` / 全局 `random`。
- save/load 用 JSON/YAML 等明文格式，**禁止 pickle**（implementation-notes
  §8.1：跨版本不可加载 + 任意代码执行）。
- 结构校验（`model_lab.validate_module`）会合成一份以角色名为列名的数据
  跑 fit/predict/save/load/确定性五连检，通不过不得入库。
"""

from __future__ import annotations

from typing import Any, Mapping

MODEL_FORMAT_PREFIX = "thermoforge.lab."

# 结构校验合成的样本规模（行列量小，保证校验快；列值为正，
# 兼容 1/x、ln 类物理项）
SYNTHETIC_ROWS = 240
SYNTHETIC_GENERIC_COLUMNS = ("f1", "f2", "f3", "f4")
SYNTHETIC_DISTRACTOR = "zzz_distractor"


def parse_inputs(raw: Any) -> dict[str, str] | None:
    """解析 `inputs` 角色→列名映射字符串：`"role=列名;role2=列名2"`。

    与 runner 侧 `thermoforge_research._child._parse_inputs` 同规则：
    分隔符 `;` 与 `,` 都接受；条目省略 `=` 视为同名映射。
    实验室模块在 `build_model` 里用它把角色名映射到视图列名::

        inputs = parse_inputs(hyperparameters.get("inputs"))
        col = (inputs or {}).get("cooling_load", "cooling_load")
    """
    if not raw:
        return None
    out: dict[str, str] = {}
    for chunk in str(raw).replace(",", ";").split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, sep, col = item.partition("=")
        key, col = key.strip(), col.strip()
        out[key] = col if (sep and col) else key
    return out or None


def require_finite(predictions: Any, *, context: str = "predict") -> None:
    """自检辅助：预测必须等长、全有限（校验门禁同款检查）。

    模块可在自己的 `predict` 末尾调用它尽早暴露数值发散；
    不调用也不影响校验（校验器会独立检查）。
    """
    import numpy as np

    arr = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{context} 产生 NaN/inf")
