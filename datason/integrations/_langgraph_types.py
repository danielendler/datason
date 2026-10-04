"""Reviewed codecs for LangGraph runtime records; enabled by the adapter only."""

from __future__ import annotations

import importlib
from dataclasses import fields
from typing import Any

from .._errors import DeserializationError, SerializationError
from .._protocols import DeserializeContext, SerializeContext
from .._registry import default_registry
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

_INTERRUPT = "langgraph.Interrupt.v1"
_SEND = "langgraph.Send.v1"


class LangGraphPlugin:
    """Reconstruct a closed set of installed framework classes, not dotted names."""

    name = "langgraph_runtime"
    priority = 400

    def __init__(self) -> None:
        module = importlib.import_module("langgraph.types")
        self._interrupt_type: Any = module.Interrupt
        self._send_type: Any = module.Send
        self._interrupt_fields = {field.name for field in fields(self._interrupt_type)}

    def can_handle(self, obj: Any) -> bool:
        return type(obj) in (self._interrupt_type, self._send_type)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        if type(obj) is self._interrupt_type:
            schema = getattr(obj, "response_schema", None)
            if schema is not None and not isinstance(schema, dict):
                raise SerializationError("Interrupt response_schema must be JSON Schema data")
            value = {"value": obj.value, "id": obj.id, "response_schema": schema}
            tag = _INTERRUPT
        else:
            if getattr(obj, "timeout", None) is not None:
                raise SerializationError("Send timeout policies require an explicit application codec")
            value, tag = {"node": obj.node, "arg": obj.arg}, _SEND
        return {TYPE_METADATA_KEY: tag, VALUE_METADATA_KEY: value} if ctx.config.include_type_hints else value

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) in (_INTERRUPT, _SEND)

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        from .._deserialize import _deserialize_recursive  # pyright: ignore[reportPrivateUsage]

        value = data[VALUE_METADATA_KEY]
        if not isinstance(value, dict):
            raise DeserializationError("LangGraph runtime record must contain a dictionary")
        child = ctx.child()
        if data[TYPE_METADATA_KEY] == _INTERRUPT:
            identifier, schema = value.get("id"), value.get("response_schema")
            if not isinstance(identifier, str) or "value" not in value:
                raise DeserializationError("Interrupt requires a string id and a value")
            if schema is not None and not isinstance(schema, dict):
                raise DeserializationError("Interrupt response_schema must be JSON Schema data")
            if schema is not None and "response_schema" not in self._interrupt_fields:
                raise DeserializationError("Installed LangGraph cannot restore this interrupt response schema")
            kwargs = {"response_schema": schema} if schema is not None else {}
            return self._interrupt_type(value=_deserialize_recursive(value["value"], child), id=identifier, **kwargs)
        if not isinstance(value.get("node"), str) or "arg" not in value:
            raise DeserializationError("Send requires a string node and an argument")
        return self._send_type(node=value["node"], arg=_deserialize_recursive(value["arg"], child))


def register_langgraph_types() -> None:
    """Enable optional runtime codecs once without making LangGraph required."""
    try:
        plugin = LangGraphPlugin()
    except ModuleNotFoundError as exc:
        if exc.name not in ("langgraph", "langgraph.types"):
            raise
        return
    default_registry.register_once(plugin)
