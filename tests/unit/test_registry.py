"""Tests for datason plugin registry."""

from __future__ import annotations

from typing import Any

from datason._config import SerializationConfig
from datason._protocols import DeserializeContext, SerializeContext
from datason._registry import PluginRegistry
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class _MockPlugin:
    """A mock plugin for testing."""

    def __init__(self, name: str, priority: int, handles_type: type):
        self._name = name
        self._priority = priority
        self._handles_type = handles_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, self._handles_type)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return {TYPE_METADATA_KEY: self._name, VALUE_METADATA_KEY: str(obj)}

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == self._name

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return data.get(VALUE_METADATA_KEY)


class TestPluginRegistry:
    """Test plugin registration and dispatch."""

    def test_register_and_count(self):
        reg = PluginRegistry()
        assert reg.plugin_count == 0
        reg.register(_MockPlugin("test", 100, complex))
        assert reg.plugin_count == 1

    def test_find_serializer(self):
        reg = PluginRegistry()
        reg.register(_MockPlugin("complex_handler", 100, complex))
        ctx = SerializeContext(config=SerializationConfig())
        result = reg.find_serializer(complex(1, 2), ctx)
        assert result is not None
        plugin, serialized = result
        assert plugin.name == "complex_handler"

    def test_find_serializer_returns_none_for_unknown(self):
        reg = PluginRegistry()
        ctx = SerializeContext(config=SerializationConfig())
        assert reg.find_serializer("just a string", ctx) is None

    def test_priority_ordering(self):
        reg = PluginRegistry()
        reg.register(_MockPlugin("low_priority", 200, complex))
        reg.register(_MockPlugin("high_priority", 100, complex))
        ctx = SerializeContext(config=SerializationConfig())
        result = reg.find_serializer(complex(1, 2), ctx)
        assert result is not None
        plugin, _ = result
        assert plugin.name == "high_priority"

    def test_find_deserializer(self):
        reg = PluginRegistry()
        reg.register(_MockPlugin("my_type", 100, complex))
        ctx = DeserializeContext(config=SerializationConfig())
        data = {TYPE_METADATA_KEY: "my_type", VALUE_METADATA_KEY: "(1+2j)"}
        result = reg.find_deserializer(data, ctx)
        assert result is not None
        _, deserialized = result
        assert deserialized == "(1+2j)"

    def test_find_deserializer_no_metadata(self):
        reg = PluginRegistry()
        reg.register(_MockPlugin("my_type", 100, complex))
        ctx = DeserializeContext(config=SerializationConfig())
        assert reg.find_deserializer({"key": "value"}, ctx) is None

    def test_clear(self):
        reg = PluginRegistry()
        reg.register(_MockPlugin("test", 100, complex))
        assert reg.plugin_count == 1
        reg.clear()
        assert reg.plugin_count == 0
