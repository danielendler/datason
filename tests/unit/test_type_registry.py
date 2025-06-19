"""Unit tests for datason.type_registry module.

This test file covers the unified type registry system:
- TypeHandler abstract base class
- TypeRegistry central registry
- Global registry functions
- Serialization and deserialization workflows
"""

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from datason.type_registry import (
    TypeHandler,
    TypeRegistry,
    deserialize_with_registry,
    get_type_registry,
    register_type_handler,
    serialize_with_registry,
)


class ConcreteTypeHandler(TypeHandler):
    """Concrete implementation of TypeHandler for testing."""

    def __init__(self, type_name: str, target_type: type):
        self._type_name = type_name
        self._target_type = target_type

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, self._target_type)

    def serialize(self, obj: Any) -> Dict[str, Any]:
        return {"__datason_type__": self.type_name, "__datason_value__": str(obj)}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        value = data["__datason_value__"]
        if self._target_type is int:
            return int(value)
        elif self._target_type is float:
            return float(value)
        else:
            return value

    @property
    def type_name(self) -> str:
        return self._type_name


class FaultyTypeHandler(TypeHandler):
    """TypeHandler that raises exceptions for testing error handling."""

    def __init__(self, fail_on: str = "all"):
        self.fail_on = fail_on

    def can_handle(self, obj: Any) -> bool:
        if self.fail_on in ["all", "can_handle"]:
            raise RuntimeError("Simulated can_handle failure")
        return isinstance(obj, str)

    def serialize(self, obj: Any) -> Dict[str, Any]:
        if self.fail_on in ["all", "serialize"]:
            raise RuntimeError("Simulated serialize failure")
        return {"__datason_type__": "faulty", "__datason_value__": obj}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        if self.fail_on in ["all", "deserialize"]:
            raise RuntimeError("Simulated deserialize failure")
        return data["__datason_value__"]

    @property
    def type_name(self) -> str:
        if self.fail_on in ["all", "type_name"]:
            raise RuntimeError("Simulated type_name failure")
        return "faulty"


class TestTypeHandler:
    """Test the TypeHandler abstract base class."""

    def test_type_handler_is_abstract(self):
        """Test that TypeHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TypeHandler()  # Should fail - abstract class

    def test_concrete_type_handler_implementation(self):
        """Test concrete implementation of TypeHandler."""
        handler = ConcreteTypeHandler("test_int", int)

        # Test properties and methods
        assert handler.type_name == "test_int"
        assert handler.can_handle(42) is True
        assert handler.can_handle("string") is False

        # Test serialization
        result = handler.serialize(42)
        assert result["__datason_type__"] == "test_int"
        assert result["__datason_value__"] == "42"

        # Test deserialization
        data = {"__datason_type__": "test_int", "__datason_value__": "42"}
        deserialized = handler.deserialize(data)
        assert deserialized == 42
        assert isinstance(deserialized, int)

    def test_type_handler_with_different_types(self):
        """Test TypeHandler with various types."""
        # Float handler
        float_handler = ConcreteTypeHandler("test_float", float)
        assert float_handler.can_handle(3.14) is True
        assert float_handler.can_handle(42) is False

        serialized = float_handler.serialize(3.14)
        assert serialized["__datason_type__"] == "test_float"

        data = {"__datason_type__": "test_float", "__datason_value__": "3.14"}
        deserialized = float_handler.deserialize(data)
        assert deserialized == 3.14
        assert isinstance(deserialized, float)

        # String handler
        str_handler = ConcreteTypeHandler("test_str", str)
        assert str_handler.can_handle("hello") is True
        assert str_handler.can_handle(42) is False


class TestTypeRegistry:
    """Test the TypeRegistry central registry."""

    def setup_method(self):
        """Set up for each test method."""
        self.registry = TypeRegistry()

    def test_registry_initialization(self):
        """Test TypeRegistry initialization."""
        registry = TypeRegistry()
        assert len(registry._handlers) == 0
        assert registry.get_registered_types() == []

    def test_register_handler(self):
        """Test registering type handlers."""
        handler1 = ConcreteTypeHandler("int_handler", int)
        handler2 = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(handler1)
        assert len(self.registry._handlers) == 1

        self.registry.register_handler(handler2)
        assert len(self.registry._handlers) == 2

    def test_find_handler(self):
        """Test finding handlers by object."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        str_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(int_handler)
        self.registry.register_handler(str_handler)

        # Test finding handlers
        found_int = self.registry.find_handler(42)
        assert found_int is int_handler

        found_str = self.registry.find_handler("hello")
        assert found_str is str_handler

        # Test no handler found
        found_none = self.registry.find_handler([1, 2, 3])
        assert found_none is None

    def test_find_handler_with_faulty_handler(self):
        """Test find_handler with handlers that raise exceptions."""
        faulty_handler = FaultyTypeHandler("can_handle")
        good_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(faulty_handler)
        self.registry.register_handler(good_handler)

        # Should skip faulty handler and find good one
        found = self.registry.find_handler("hello")
        assert found is good_handler

    def test_find_handler_by_type_name(self):
        """Test finding handlers by type name."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        str_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(int_handler)
        self.registry.register_handler(str_handler)

        # Test finding by type name
        found_int = self.registry.find_handler_by_type_name("int_handler")
        assert found_int is int_handler

        found_str = self.registry.find_handler_by_type_name("str_handler")
        assert found_str is str_handler

        # Test no handler found
        found_none = self.registry.find_handler_by_type_name("nonexistent")
        assert found_none is None

    def test_find_handler_by_type_name_with_faulty_handler(self):
        """Test find_handler_by_type_name with handlers that raise exceptions."""
        faulty_handler = FaultyTypeHandler("type_name")
        good_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(faulty_handler)
        self.registry.register_handler(good_handler)

        # Should skip faulty handler and find good one
        found = self.registry.find_handler_by_type_name("str_handler")
        assert found is good_handler

    def test_serialize_with_handler(self):
        """Test serialization using registry."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        self.registry.register_handler(int_handler)

        # Test successful serialization
        result = self.registry.serialize(42)
        assert result == {"__datason_type__": "int_handler", "__datason_value__": "42"}

        # Test no handler found
        result = self.registry.serialize([1, 2, 3])
        assert result is None

    def test_deserialize_with_handler(self):
        """Test deserialization using registry."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        self.registry.register_handler(int_handler)

        # Test successful deserialization
        data = {"__datason_type__": "int_handler", "__datason_value__": "42"}
        result = self.registry.deserialize(data)
        assert result == 42
        assert isinstance(result, int)

        # Test no handler found - should return original data
        data = {"__datason_type__": "unknown_type", "__datason_value__": "42"}
        result = self.registry.deserialize(data)
        assert result == data

    def test_deserialize_invalid_data(self):
        """Test deserialization with invalid data."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        self.registry.register_handler(int_handler)

        # Test non-dict data
        result = self.registry.deserialize("not a dict")
        assert result == "not a dict"

        # Test dict without __datason_type__
        result = self.registry.deserialize({"key": "value"})
        assert result == {"key": "value"}

        # Test empty dict
        result = self.registry.deserialize({})
        assert result == {}

    def test_get_registered_types(self):
        """Test getting list of registered type names."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        str_handler = ConcreteTypeHandler("str_handler", str)

        # Empty registry
        assert self.registry.get_registered_types() == []

        # Add handlers
        self.registry.register_handler(int_handler)
        types = self.registry.get_registered_types()
        assert types == ["int_handler"]

        self.registry.register_handler(str_handler)
        types = self.registry.get_registered_types()
        assert set(types) == {"int_handler", "str_handler"}

    def test_get_registered_types_with_faulty_handler(self):
        """Test get_registered_types with handlers that raise exceptions."""
        faulty_handler = FaultyTypeHandler("type_name")
        good_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(faulty_handler)
        self.registry.register_handler(good_handler)

        # Should skip faulty handler
        types = self.registry.get_registered_types()
        assert types == ["str_handler"]

    def test_clear_handlers(self):
        """Test clearing all handlers."""
        int_handler = ConcreteTypeHandler("int_handler", int)
        str_handler = ConcreteTypeHandler("str_handler", str)

        self.registry.register_handler(int_handler)
        self.registry.register_handler(str_handler)
        assert len(self.registry._handlers) == 2

        self.registry.clear_handlers()
        assert len(self.registry._handlers) == 0
        assert self.registry.get_registered_types() == []

    def test_registry_handler_order(self):
        """Test that handlers are checked in registration order."""
        # Create two handlers that can both handle strings
        handler1 = Mock(spec=TypeHandler)
        handler1.can_handle.return_value = True
        handler1.serialize.return_value = {"type": "handler1"}

        handler2 = Mock(spec=TypeHandler)
        handler2.can_handle.return_value = True
        handler2.serialize.return_value = {"type": "handler2"}

        self.registry.register_handler(handler1)
        self.registry.register_handler(handler2)

        # First registered handler should be found first
        found = self.registry.find_handler("test")
        assert found is handler1

        # Only first handler should be called for serialization
        result = self.registry.serialize("test")
        assert result == {"type": "handler1"}

        # First handler should be called, second should not be called
        assert handler1.can_handle.call_count >= 1
        handler2.can_handle.assert_not_called()


class TestGlobalRegistry:
    """Test global registry functions."""

    def setup_method(self):
        """Set up for each test method."""
        # Clear the global registry before each test
        registry = get_type_registry()
        registry.clear_handlers()

    def teardown_method(self):
        """Clean up after each test method."""
        # Clear the global registry after each test
        registry = get_type_registry()
        registry.clear_handlers()

    def test_get_type_registry(self):
        """Test getting the global type registry."""
        registry1 = get_type_registry()
        registry2 = get_type_registry()

        # Should return the same instance
        assert registry1 is registry2
        assert isinstance(registry1, TypeRegistry)

    def test_register_type_handler(self):
        """Test registering handler with global registry."""
        handler = ConcreteTypeHandler("global_int", int)

        # Register with global function
        register_type_handler(handler)

        # Verify it's in the global registry
        registry = get_type_registry()
        types = registry.get_registered_types()
        assert "global_int" in types

        found = registry.find_handler_by_type_name("global_int")
        assert found is handler

    def test_serialize_with_registry(self):
        """Test global serialize function."""
        handler = ConcreteTypeHandler("global_int", int)
        register_type_handler(handler)

        # Test serialization
        result = serialize_with_registry(42)
        assert result == {"__datason_type__": "global_int", "__datason_value__": "42"}

        # Test no handler
        result = serialize_with_registry([1, 2, 3])
        assert result is None

    def test_deserialize_with_registry(self):
        """Test global deserialize function."""
        handler = ConcreteTypeHandler("global_int", int)
        register_type_handler(handler)

        # Test deserialization
        data = {"__datason_type__": "global_int", "__datason_value__": "42"}
        result = deserialize_with_registry(data)
        assert result == 42
        assert isinstance(result, int)

        # Test no handler
        data = {"__datason_type__": "unknown", "__datason_value__": "42"}
        result = deserialize_with_registry(data)
        assert result == data

    def test_global_registry_isolation_between_tests(self):
        """Test that tests don't interfere with each other."""
        # This test should start with empty registry
        registry = get_type_registry()
        assert registry.get_registered_types() == []

        # Add a handler
        handler = ConcreteTypeHandler("test_isolation", str)
        register_type_handler(handler)

        types = registry.get_registered_types()
        assert "test_isolation" in types


class TestRegistryIntegration:
    """Test integration scenarios with multiple handlers."""

    def setup_method(self):
        """Set up for each test method."""
        self.registry = TypeRegistry()

    def test_multiple_handlers_different_types(self):
        """Test registry with handlers for different types."""
        int_handler = ConcreteTypeHandler("int_type", int)
        str_handler = ConcreteTypeHandler("str_type", str)
        float_handler = ConcreteTypeHandler("float_type", float)

        self.registry.register_handler(int_handler)
        self.registry.register_handler(str_handler)
        self.registry.register_handler(float_handler)

        # Test serialization of different types
        int_result = self.registry.serialize(42)
        assert int_result["__datason_type__"] == "int_type"

        str_result = self.registry.serialize("hello")
        assert str_result["__datason_type__"] == "str_type"

        float_result = self.registry.serialize(3.14)
        assert float_result["__datason_type__"] == "float_type"

        # Test deserialization
        int_data = {"__datason_type__": "int_type", "__datason_value__": "42"}
        assert self.registry.deserialize(int_data) == 42

        str_data = {"__datason_type__": "str_type", "__datason_value__": "hello"}
        assert self.registry.deserialize(str_data) == "hello"

        float_data = {"__datason_type__": "float_type", "__datason_value__": "3.14"}
        assert self.registry.deserialize(float_data) == 3.14

    def test_handler_priority_order(self):
        """Test that first matching handler is used."""
        # Create two handlers for the same type but different names
        handler1 = ConcreteTypeHandler("first_int", int)
        handler2 = ConcreteTypeHandler("second_int", int)

        self.registry.register_handler(handler1)
        self.registry.register_handler(handler2)

        # First handler should be used
        result = self.registry.serialize(42)
        assert result["__datason_type__"] == "first_int"

        found = self.registry.find_handler(42)
        assert found is handler1

    def test_error_resilience(self):
        """Test that registry continues working despite faulty handlers."""
        faulty_handler = FaultyTypeHandler("all")
        good_handler = ConcreteTypeHandler("good_int", int)

        # Register faulty handler first
        self.registry.register_handler(faulty_handler)
        self.registry.register_handler(good_handler)

        # Should skip faulty and use good handler
        result = self.registry.serialize(42)
        assert result["__datason_type__"] == "good_int"

        # Should work for deserialization too
        data = {"__datason_type__": "good_int", "__datason_value__": "42"}
        result = self.registry.deserialize(data)
        assert result == 42

    def test_round_trip_serialization(self):
        """Test complete round-trip serialization/deserialization."""
        handler = ConcreteTypeHandler("round_trip", int)
        self.registry.register_handler(handler)

        original = 42

        # Serialize
        serialized = self.registry.serialize(original)
        assert isinstance(serialized, dict)
        assert "__datason_type__" in serialized
        assert "__datason_value__" in serialized

        # Deserialize
        deserialized = self.registry.deserialize(serialized)
        assert deserialized == original
        assert type(deserialized) is type(original)

    def test_empty_registry_behavior(self):
        """Test registry behavior when no handlers are registered."""
        # Empty registry should handle gracefully
        assert self.registry.find_handler(42) is None
        assert self.registry.find_handler_by_type_name("any") is None
        assert self.registry.serialize(42) is None

        # Should return original data for unknown types
        data = {"__datason_type__": "unknown", "__datason_value__": "42"}
        result = self.registry.deserialize(data)
        assert result == data

        # Should return non-dict data as-is
        assert self.registry.deserialize("not dict") == "not dict"
        assert self.registry.deserialize(42) == 42
        assert self.registry.deserialize([1, 2, 3]) == [1, 2, 3]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up for each test method."""
        self.registry = TypeRegistry()

    def test_none_values(self):
        """Test handling of None values."""
        # Registry methods should handle None gracefully
        assert self.registry.find_handler(None) is None
        assert self.registry.serialize(None) is None
        assert self.registry.deserialize(None) is None

    def test_malformed_data_deserialization(self):
        """Test deserialization with malformed data."""
        # Missing __datason_value__
        data = {"__datason_type__": "test_type"}
        result = self.registry.deserialize(data)
        assert result == data

        # Wrong type for __datason_type__
        data = {"__datason_type__": 123, "__datason_value__": "value"}
        result = self.registry.deserialize(data)
        assert result == data

    def test_handler_with_complex_data(self):
        """Test handler with complex nested data."""

        class ComplexHandler(TypeHandler):
            def can_handle(self, obj: Any) -> bool:
                return isinstance(obj, dict) and "special_marker" in obj

            def serialize(self, obj: Any) -> Dict[str, Any]:
                return {"__datason_type__": "complex", "__datason_value__": obj}

            def deserialize(self, data: Dict[str, Any]) -> Any:
                return data["__datason_value__"]

            @property
            def type_name(self) -> str:
                return "complex"

        handler = ComplexHandler()
        self.registry.register_handler(handler)

        # Test with complex nested object
        original = {"special_marker": True, "nested": {"deep": [1, 2, {"even_deeper": "value"}]}}

        serialized = self.registry.serialize(original)
        assert serialized["__datason_type__"] == "complex"

        deserialized = self.registry.deserialize(serialized)
        assert deserialized == original

    def test_duplicate_type_names(self):
        """Test behavior with duplicate type names."""
        handler1 = ConcreteTypeHandler("duplicate", int)
        handler2 = ConcreteTypeHandler("duplicate", str)

        self.registry.register_handler(handler1)
        self.registry.register_handler(handler2)

        # First handler with the name should be found
        found = self.registry.find_handler_by_type_name("duplicate")
        assert found is handler1

        # Both should be in the registry
        assert len(self.registry._handlers) == 2

    def test_handler_state_persistence(self):
        """Test that handler state is maintained."""

        class StatefulHandler(TypeHandler):
            def __init__(self):
                self.call_count = 0

            def can_handle(self, obj: Any) -> bool:
                self.call_count += 1
                return isinstance(obj, tuple)

            def serialize(self, obj: Any) -> Dict[str, Any]:
                return {
                    "__datason_type__": "stateful",
                    "__datason_value__": list(obj),
                    "__call_count__": self.call_count,
                }

            def deserialize(self, data: Dict[str, Any]) -> Any:
                return tuple(data["__datason_value__"])

            @property
            def type_name(self) -> str:
                return "stateful"

        handler = StatefulHandler()
        self.registry.register_handler(handler)

        # Test that state is maintained across calls
        self.registry.find_handler((1, 2, 3))
        assert handler.call_count == 1

        self.registry.find_handler((4, 5, 6))
        assert handler.call_count == 2

        result = self.registry.serialize((1, 2, 3))
        assert result["__call_count__"] == 3
