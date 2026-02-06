"""Plugin for Decimal and complex number serialization."""

from __future__ import annotations

import decimal as decimal_mod
from typing import Any

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

_HANDLED_TYPES = (decimal_mod.Decimal, complex)

_TYPE_NAMES = {
    decimal_mod.Decimal: "decimal.Decimal",
    complex: "complex",
}


class DecimalPlugin:
    """Handles serialization/deserialization of Decimal and complex."""

    @property
    def name(self) -> str:
        return "decimal"

    @property
    def priority(self) -> int:
        return 102

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, _HANDLED_TYPES)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        type_name = _TYPE_NAMES.get(type(obj))
        if type_name is None:
            raise PluginError(f"Unexpected type: {type(obj).__name__}")

        if isinstance(obj, decimal_mod.Decimal):  # noqa: SIM108
            value = str(obj)
        else:
            # complex -> [real, imag]
            value = [obj.real, obj.imag]

        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: value}
        return value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) in _TYPE_NAMES.values()

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        type_name = data[TYPE_METADATA_KEY]
        value = data[VALUE_METADATA_KEY]

        match type_name:
            case "decimal.Decimal":
                if not isinstance(value, str):
                    raise PluginError(f"Expected string for Decimal, got {type(value).__name__}")
                return decimal_mod.Decimal(value)
            case "complex":
                if not isinstance(value, list) or len(value) != 2:  # noqa: PLR2004
                    raise PluginError(f"Expected [real, imag] for complex, got {value!r}")
                return complex(value[0], value[1])
            case _:
                raise PluginError(f"Unknown decimal type: {type_name}")
