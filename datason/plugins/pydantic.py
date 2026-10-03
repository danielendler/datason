"""Optional Pydantic model normalization, without dynamic class reconstruction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .._errors import PluginError, SerializationError
from .._protocols import DeserializeContext, SerializeContext


class PydanticPlugin:
    """Export model fields using Pydantic's Python-mode serializer."""

    @property
    def name(self) -> str:
        return "pydantic"

    @property
    def priority(self) -> int:
        return 10_001

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, BaseModel)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        if not isinstance(obj, BaseModel):
            raise PluginError("Expected a Pydantic model")
        try:
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="python", by_alias=True)
            # Only used when v1 is installed; v2 marks this legacy API deprecated.
            legacy_dump = getattr(obj, "dict", None)
            if legacy_dump is None:
                raise SerializationError("Pydantic model has no supported field serializer")
            return legacy_dump(by_alias=True)
        except (ValueError, TypeError) as exc:
            raise SerializationError("Pydantic model could not be normalized") from exc

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return False

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        raise PluginError("Application models require explicit validation after loading")
