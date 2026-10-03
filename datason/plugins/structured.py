"""Safe JSON normalization for structured stdlib values and binary data."""

from __future__ import annotations

import base64
import binascii
import dataclasses
from enum import Enum
from typing import Any

from .._errors import DeserializationError, PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class StructuredPlugin:
    """Normalize application structures without importing or constructing classes."""

    @property
    def name(self) -> str:
        return "structured"

    @property
    def priority(self) -> int:
        # Application plugins get the first opportunity to preserve their classes.
        return 10_000

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, bytes | bytearray | Enum) or (
            dataclasses.is_dataclass(obj) and not isinstance(obj, type)
        )

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        if isinstance(obj, bytes | bytearray):
            value = base64.b64encode(obj).decode("ascii")
            if ctx.config.include_type_hints:
                return {
                    TYPE_METADATA_KEY: "bytes" if isinstance(obj, bytes) else "bytearray",
                    VALUE_METADATA_KEY: value,
                }
            return value
        if isinstance(obj, Enum):
            return obj.value
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
        raise PluginError("Unsupported structured value")

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) in ("bytes", "bytearray")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        value = data.get(VALUE_METADATA_KEY)
        if not isinstance(value, str):
            raise DeserializationError("Binary payload must be a base64 string")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise DeserializationError("Invalid base64 binary payload") from exc
        return bytearray(decoded) if data[TYPE_METADATA_KEY] == "bytearray" else decoded
