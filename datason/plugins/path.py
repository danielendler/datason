"""Plugin for pathlib.Path serialization."""

from __future__ import annotations

import pathlib
from typing import Any

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

_HANDLED_TYPES = (pathlib.PurePath,)


class PathPlugin:
    """Handles serialization/deserialization of Path objects."""

    @property
    def name(self) -> str:
        return "path"

    @property
    def priority(self) -> int:
        return 103

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, _HANDLED_TYPES)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        value = str(obj)
        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: "pathlib.Path", VALUE_METADATA_KEY: value}
        return value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == "pathlib.Path"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        value = data[VALUE_METADATA_KEY]
        if not isinstance(value, str):
            raise PluginError(f"Expected string for Path, got {type(value).__name__}")
        return pathlib.Path(value)
