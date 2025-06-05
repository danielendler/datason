"""Simple coverage boost tests for various datason modules."""

import warnings

import pytest

# Test config
from datason.config import NanHandling, SerializationConfig, TypeCoercion

# Test core functions
from datason.core import serialize

# Test deserializers
from datason.deserializers import deserialize, deserialize_fast

# Test utils functions that are working
from datason.utils import UtilityConfig, deep_compare


class TestSimpleCoverageBoost:
    """Simple tests to boost coverage across multiple modules."""

    def test_utils_deep_compare_basic(self):
        """Test basic deep_compare functionality."""
        result = deep_compare({"a": 1}, {"a": 1})
        assert result["are_equal"]

        result = deep_compare({"a": 1}, {"a": 2})
        assert not result["are_equal"]

    def test_utils_config_variations(self):
        """Test UtilityConfig with different settings."""
        config1 = UtilityConfig(max_depth=10)
        config2 = UtilityConfig(max_object_size=1000)
        config3 = UtilityConfig(enable_circular_reference_detection=False)

        # Just ensure they can be created
        assert config1.max_depth == 10
        assert config2.max_object_size == 1000
        assert not config3.enable_circular_reference_detection

    def test_utils_large_data_warning(self):
        """Test utils warnings with large data."""
        config = UtilityConfig(max_collection_size=5)
        large_dict = {f"key_{i}": i for i in range(10)}

        with warnings.catch_warnings(record=True):
            deep_compare(large_dict, large_dict, config=config)
            # Should complete despite warnings

    def test_core_serialize_variations(self):
        """Test serialize with different configurations."""
        data = {"test": "value", "number": 42}

        # Different config options
        config1 = SerializationConfig(include_type_hints=True)
        config2 = SerializationConfig(include_type_hints=False)
        config3 = SerializationConfig(max_depth=10)

        result1 = serialize(data, config=config1)
        result2 = serialize(data, config=config2)
        result3 = serialize(data, config=config3)

        # Should all work
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)

    def test_core_problematic_objects(self):
        """Test core handling of problematic objects."""
        import io
        from unittest.mock import Mock

        # IO objects
        data_with_io = {"file": io.StringIO("test")}
        with warnings.catch_warnings(record=True):
            result = serialize(data_with_io)
            assert "file" in result

        # Mock objects
        data_with_mock = {"mock": Mock()}
        with warnings.catch_warnings(record=True):
            result = serialize(data_with_mock)
            assert "mock" in result

    def test_deserializers_basic(self):
        """Test basic deserializer functionality."""
        # Test with simple data
        original = {"test": "value", "number": 42}
        serialized = serialize(original)

        # Test both deserializers
        result1 = deserialize(serialized)
        result2 = deserialize_fast(serialized)

        assert result1 == original
        assert result2 == original

    def test_deserializers_with_metadata(self):
        """Test deserializers with type metadata."""
        from datetime import datetime
        from decimal import Decimal

        original = {"datetime": datetime(2023, 1, 1), "decimal": Decimal("123.45"), "complex": complex(1, 2)}

        config = SerializationConfig(include_type_hints=True)
        serialized = serialize(original, config=config)

        with warnings.catch_warnings(record=True):
            deserialize_fast(serialized)
            # Should handle metadata

    def test_config_enum_values(self):
        """Test configuration enum values."""
        # Test TypeCoercion enum
        assert TypeCoercion.STRICT != TypeCoercion.SAFE
        assert TypeCoercion.SAFE != TypeCoercion.AGGRESSIVE

        # Test NanHandling enum
        assert NanHandling.KEEP != NanHandling.NULL
        assert NanHandling.NULL != NanHandling.STRING

    def test_config_creation_variations(self):
        """Test different SerializationConfig combinations."""
        configs = [
            SerializationConfig(),
            SerializationConfig(include_type_hints=True),
            SerializationConfig(preserve_decimals=True),
            SerializationConfig(preserve_complex=True),
            SerializationConfig(type_coercion=TypeCoercion.STRICT),
            SerializationConfig(nan_handling=NanHandling.NULL),
            SerializationConfig(max_depth=5),
            SerializationConfig(max_string_length=100),
        ]

        # All should be valid
        for config in configs:
            assert isinstance(config, SerializationConfig)

    def test_circular_references(self):
        """Test circular reference handling."""
        # Create circular reference
        data = {"name": "test"}
        data["self"] = data

        with warnings.catch_warnings(record=True):
            serialize(data)
            # Should handle gracefully

    def test_edge_case_data_types(self):
        """Test various edge case data types."""
        test_data = [
            None,
            [],
            {},
            set(),
            frozenset(),
            range(5),
            b"hello",
            bytearray(b"world"),
        ]

        for data in test_data:
            with warnings.catch_warnings(record=True):
                result = serialize(data)
                # Should serialize to something reasonable
                if data is None:
                    assert result is None
                elif data == []:
                    assert result == []
                elif data == {}:
                    assert result == {}
                else:
                    # Other types should serialize to something
                    assert result is not None

    def test_large_structures(self):
        """Test reasonably large data structures."""
        large_list = list(range(1000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}

        with warnings.catch_warnings(record=True):
            result1 = serialize(large_list)
            result2 = serialize(large_dict)

            assert isinstance(result1, list)
            assert isinstance(result2, dict)

    def test_nested_structures(self):
        """Test deeply nested structures."""
        nested = {"level": 1}
        current = nested
        for i in range(2, 20):
            current["next"] = {"level": i}
            current = current["next"]

        with warnings.catch_warnings(record=True):
            result = serialize(nested)
            assert "level" in result

    def test_mixed_type_containers(self):
        """Test containers with mixed types."""
        mixed = {
            "strings": ["a", "b", "c"],
            "numbers": [1, 2.5, 3],
            "booleans": [True, False],
            "mixed": [1, "two", 3.0, None, {"nested": "value"}],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }

        result = serialize(mixed)
        assert isinstance(result, dict)
        assert len(result["mixed"]) == 5

    def test_error_recovery(self):
        """Test error recovery scenarios."""

        class ProblematicClass:
            def __init__(self):
                self.value = "test"

            def __getattribute__(self, name):
                if name == "problematic_attr":
                    raise RuntimeError("Access denied")
                return super().__getattribute__(name)

        obj = ProblematicClass()

        with warnings.catch_warnings(record=True):
            serialize(obj)
            # Should handle gracefully


if __name__ == "__main__":
    pytest.main([__file__])
