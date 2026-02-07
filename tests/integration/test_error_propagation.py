"""Integration tests for error propagation and plugin dispatch.

Tests plugin priority ordering, strict mode behavior, and config
scoping when errors occur.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import datason
from datason._config import SerializationConfig
from datason._errors import DeserializationError, SerializationError
from datason._protocols import DeserializeContext, SerializeContext
from datason._registry import PluginRegistry
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class _MockPlugin:
    """Test plugin for dispatch priority testing."""

    def __init__(self, name: str, priority: int, *, serialize_result: Any = None) -> None:
        self._name = name
        self._priority = priority
        self._serialize_result = serialize_result

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, complex)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        if self._serialize_result is not None:
            return self._serialize_result
        return {"plugin": self._name, "value": str(obj)}

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get(TYPE_METADATA_KEY) == f"mock.{self._name}"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return complex(data[VALUE_METADATA_KEY])


class TestPluginDispatchPriority:
    """Higher-priority plugin wins when two handle same type."""

    def test_higher_priority_wins(self) -> None:
        registry = PluginRegistry()
        high = _MockPlugin("high", 100, serialize_result={"source": "high"})
        low = _MockPlugin("low", 200, serialize_result={"source": "low"})
        registry.register(low)
        registry.register(high)

        ctx = SerializeContext(config=SerializationConfig())
        result = registry.find_serializer(complex(1, 2), ctx)
        assert result is not None
        plugin, serialized = result
        assert plugin.name == "high"
        assert serialized["source"] == "high"

    def test_registration_order_irrelevant(self) -> None:
        registry = PluginRegistry()
        high = _MockPlugin("high", 100, serialize_result={"source": "high"})
        low = _MockPlugin("low", 200, serialize_result={"source": "low"})
        # Register high first, then low — order shouldn't matter
        registry.register(high)
        registry.register(low)

        ctx = SerializeContext(config=SerializationConfig())
        result = registry.find_serializer(complex(1, 2), ctx)
        assert result is not None
        assert result[0].name == "high"


class TestStrictModeErrors:
    """Strict mode behavior for unknown type metadata."""

    def test_strict_mode_unknown_metadata_raises(self) -> None:
        # Build JSON with unknown type annotation
        data = {TYPE_METADATA_KEY: "totally.unknown", VALUE_METADATA_KEY: {"x": 1}}
        s = json.dumps(data)
        with pytest.raises(DeserializationError, match="No plugin registered"):
            datason.loads(s, strict=True)

    def test_non_strict_mode_returns_raw_dict(self) -> None:
        data = {TYPE_METADATA_KEY: "totally.unknown", VALUE_METADATA_KEY: {"x": 1}}
        s = json.dumps(data)
        result = datason.loads(s, strict=False)
        assert isinstance(result, dict)
        assert result[TYPE_METADATA_KEY] == "totally.unknown"

    def test_strict_mode_known_type_succeeds(self) -> None:
        import datetime as dt

        data = {"ts": dt.datetime(2024, 1, 1)}  # noqa: DTZ001
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s, strict=True)
        assert isinstance(result["ts"], dt.datetime)


class TestConfigScopingErrors:
    """Config context manager behavior when errors occur."""

    def test_exception_inside_config_resets(self) -> None:
        """Config resets even if an exception occurs inside the context."""

        class UnserializableObj:
            pass

        try:
            with datason.config(fallback_to_string=False, sort_keys=True):
                datason.dumps(UnserializableObj())
        except SerializationError:
            pass

        # After exception, default config should be active
        data = {"z": 1, "a": 2}
        s = datason.dumps(data)
        keys = list(json.loads(s).keys())
        assert keys == ["z", "a"]  # Default: unsorted

    def test_nested_config_exception_resets_outer(self) -> None:
        """Exception in inner scope resets to outer scope."""

        class UnserializableObj:
            pass

        with datason.config(sort_keys=True):
            try:
                with datason.config(fallback_to_string=False):
                    datason.dumps(UnserializableObj())
            except SerializationError:
                pass

            # Should still be in outer scope (sort_keys=True)
            data = {"z": 1, "a": 2}
            s = datason.dumps(data)
            keys = list(json.loads(s).keys())
            assert keys == ["a", "z"]


class TestSerializationErrorMessages:
    """Error messages provide useful information."""

    def test_unknown_type_message_includes_type_name(self) -> None:
        class MyCustomClass:
            pass

        with pytest.raises(SerializationError, match="MyCustomClass"):
            datason.dumps(MyCustomClass())

    def test_unknown_type_message_suggests_fallback(self) -> None:
        class MyCustomClass:
            pass

        with pytest.raises(SerializationError, match="fallback_to_string"):
            datason.dumps(MyCustomClass())
