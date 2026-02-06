"""Plugin for UUID serialization."""

from __future__ import annotations

import uuid as uuid_mod
from typing import Any

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class UUIDPlugin:
    """Handles serialization/deserialization of UUID objects."""

    @property
    def name(self) -> str:
        return "uuid"

    @property
    def priority(self) -> int:
        return 101

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, uuid_mod.UUID)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        value = str(obj)
        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: "uuid.UUID", VALUE_METADATA_KEY: value}
        return value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == "uuid.UUID"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        value = data[VALUE_METADATA_KEY]
        if not isinstance(value, str):
            raise PluginError(f"Expected string for UUID, got {type(value).__name__}")
        return uuid_mod.UUID(value)
