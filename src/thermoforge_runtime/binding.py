"""部署绑定（model-package.md §3/§6、data-contract.md 两层命名 DD-11）。

模型签名用通用 `property_code` 表达输入输出；部署到具体对象时由
`DeploymentBinding` 解析为 `variable_id`：

- 默认约定：`object_id.property_code`（如 `CH-01.evap_chw_flow`）。
- 显式 bindings：`property_code → variable_id` 的覆盖映射，来自 TFDC
  `bindings` 表（`BindingRecord`）在部署侧的投影——现场点位到
  `variable_id` 的转换发生在 Adapter 边界，模型包内只见到
  `variable_id`（model-package §6）。

模型内部不得依赖 BACnet / OPC UA / MQTT 地址（model-package §3）。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from thermoforge_core.contracts.tfdc import BindingRecord
from thermoforge_core.naming import (
    validate_object_id,
    validate_property_code,
    validate_variable_id,
    variable_id_for,
)


class DeploymentBinding:
    """一个已部署对象的 property_code ↔ variable_id 绑定。

    用法::

        binding = DeploymentBinding("CH-01")
        binding.variable_id_for("evap_chw_flow")   # -> "CH-01.evap_chw_flow"
        props = binding.resolve({"CH-01.evap_chw_flow": 521.4})

    显式 bindings（property_code → variable_id）覆盖默认约定，用于
    跨设备取数或现场命名与约定不一致的场景::

        DeploymentBinding("CH-01", {"outdoor_wet_bulb_temp": "MT-01.wet_bulb"})
    """

    def __init__(
        self,
        object_id: str,
        bindings: Mapping[str, str] | None = None,
    ):
        self.object_id = validate_object_id(object_id)
        self._prop_to_var: dict[str, str] = {}
        for prop, var in (bindings or {}).items():
            self._prop_to_var[validate_property_code(prop)] = (
                validate_variable_id(var)
            )
        self._var_to_prop: dict[str, str] = {
            v: k for k, v in self._prop_to_var.items()
        }
        if len(self._var_to_prop) != len(self._prop_to_var):
            raise ValueError("bindings 中不同的 property_code 映射到了同一 variable_id")

    @classmethod
    def from_binding_records(
        cls,
        object_id: str,
        records: Iterable[BindingRecord],
    ) -> "DeploymentBinding":
        """从 TFDC `bindings` 表记录构造（取 variable_id 末段作 property_code）。

        只选取属于 `object_id` 的记录；显式映射仅在 variable_id 不符合
        默认约定 `object_id.property_code` 时产生。
        """
        explicit: dict[str, str] = {}
        for rec in records:
            obj, _, prop = rec.variable_id.partition(".")
            if obj != object_id or not prop:
                continue
            if rec.variable_id != variable_id_for(object_id, prop):
                explicit[prop] = rec.variable_id
        return cls(object_id, explicit)

    # ---------------------------------------------------------------- 解析

    def variable_id_for(self, property_code: str) -> str:
        """签名 property_code → 线上 variable_id（model-package §3）。"""
        validate_property_code(property_code)
        return self._prop_to_var.get(
            property_code, variable_id_for(self.object_id, property_code)
        )

    def resolve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """线上消息 values（variable_id → 值）解析为 property_code → 值。

        未绑定到本对象的 variable_id 被忽略（调用方按签名 required
        判定缺失）；解析结果交给模型，模型内部只见 property_code。
        """
        out: dict[str, Any] = {}
        for var, value in values.items():
            prop = self._var_to_prop.get(str(var))
            if prop is None:
                obj, _, tail = str(var).partition(".")
                if obj == self.object_id and tail:
                    prop = tail
            if prop is not None:
                out[prop] = value
        return out

    def as_dict(self) -> dict[str, Any]:
        """部署描述（可落盘）：对象与显式绑定。"""
        return {
            "object_id": self.object_id,
            "bindings": dict(self._prop_to_var),
        }
