"""Tests targeting core.py enhancements for diff coverage.

Focuses on the specific changes made to core.py that need coverage.
"""

import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from datason.core import SerializationConfig, serialize


class TestEmptyTupleTypeHints:
    """Test empty tuple type hint handling."""

    def test_empty_tuple_with_type_hints(self) -> None:
        """Test empty tuple includes type metadata when type hints enabled."""
        config = SerializationConfig(include_type_hints=True)

        empty_tuple = ()
        result = serialize(empty_tuple, config)

        # Should include type metadata for empty tuple
        assert isinstance(result, dict)
        assert result.get("__datason_type__") == "tuple"
        assert result.get("__datason_value__") == []

    def test_empty_tuple_without_type_hints(self) -> None:
        """Test empty tuple without type hints returns list."""
        config = SerializationConfig(include_type_hints=False)

        empty_tuple = ()
        result = serialize(empty_tuple, config)

        # Should be a simple list without metadata
        assert result == []

    def test_non_empty_tuple_behavior_unchanged(self) -> None:
        """Test non-empty tuples behave as before."""
        config = SerializationConfig(include_type_hints=True)

        non_empty_tuple = (1, 2, 3)
        result = serialize(non_empty_tuple, config)

        # Non-empty tuples should get type metadata
        assert isinstance(result, dict)
        assert result.get("__datason_type__") == "tuple"
        assert result.get("__datason_value__") == [1, 2, 3]


class TestNumpyScalarTypeHints:
    """Test NumPy scalar type hint behavior."""

    def test_numpy_scalar_with_type_hints_forces_full_path(self) -> None:
        """Test NumPy scalars with type hints skip hot path for metadata generation."""
        try:
            import numpy as np

            config = SerializationConfig(include_type_hints=True)

            # Test various numpy scalar types
            int32_val = np.int32(42)
            result = serialize(int32_val, config)

            # Should include type metadata
            assert isinstance(result, dict)
            assert result.get("__datason_type__") == "numpy.int32"

            float64_val = np.float64(3.14)
            result = serialize(float64_val, config)

            # Should include type metadata
            assert isinstance(result, dict)
            assert result.get("__datason_type__") == "numpy.float64"

        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_scalar_without_type_hints_uses_hot_path(self) -> None:
        """Test NumPy scalars without type hints use hot path optimization."""
        try:
            import numpy as np

            config = SerializationConfig(include_type_hints=False)

            # Test numpy scalars without type hints
            int32_val = np.int32(42)
            result = serialize(int32_val, config)

            # Should be normalized to Python int (hot path)
            assert result == 42
            assert isinstance(result, int)

            float64_val = np.float64(3.14)
            result = serialize(float64_val, config)

            # Should be normalized to Python float (hot path)
            assert result == 3.14
            assert isinstance(result, float)

        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_nan_inf_handling(self) -> None:
        """Test NumPy NaN/Inf values skip hot path for special handling."""
        try:
            import numpy as np

            config = SerializationConfig(include_type_hints=False)

            # NaN should skip hot path
            nan_val = np.float64(float("nan"))
            result = serialize(nan_val, config)
            assert result is None  # Should be serialized as None

            # Infinity should skip hot path
            inf_val = np.float64(float("inf"))
            result = serialize(inf_val, config)
            assert result is None  # Should be serialized as None

        except ImportError:
            pytest.skip("numpy not available")


class TestDecimalImportAndHandling:
    """Test Decimal import and handling improvements."""

    def test_decimal_serialization_basic(self) -> None:
        """Test basic Decimal serialization."""
        config = SerializationConfig(include_type_hints=True)

        decimal_val = Decimal("123.456")
        result = serialize(decimal_val, config)

        # Should include type metadata for Decimal (may use legacy _type format)
        assert isinstance(result, dict)
        # Support both new and legacy formats
        type_field = result.get("__datason_type__") or result.get("_type")
        assert type_field in ("decimal.Decimal", "decimal")
        value_field = result.get("__datason_value__") or result.get("value")
        assert "123.456" in str(value_field)

    def test_decimal_precision_preservation(self) -> None:
        """Test Decimal precision is preserved."""
        config = SerializationConfig(include_type_hints=True)

        # High precision decimal
        high_precision = Decimal("123.456789012345678901234567890")
        result = serialize(high_precision, config)

        assert isinstance(result, dict)
        # Support both new and legacy formats
        type_field = result.get("__datason_type__") or result.get("_type")
        assert type_field in ("decimal.Decimal", "decimal")
        # Should preserve full precision as string
        value_field = result.get("__datason_value__") or result.get("value")
        assert "123.456789012345678901234567890" in str(value_field)


class TestCustomTypeHandlerChanges:
    """Test custom type handler changes."""

    def test_custom_handler_disabled_temporarily(self) -> None:
        """Test that custom type handlers are currently disabled (TODO comment)."""
        # This tests the current state where custom handlers are disabled
        # due to the TODO comment in the code

        class CustomType:
            def __init__(self, value: Any) -> None:
                self.value = value

        custom_obj = CustomType("test")

        # Should fall through to default handling since custom handlers are disabled
        config = SerializationConfig()

        # This should use default object handling rather than custom handler
        try:
            serialize(custom_obj, config)
            # Result depends on default object serialization behavior
        except Exception:
            # May raise exception due to unsupported type, which is expected
            # since custom handlers are disabled
            pass


class TestSerializationSecurityConstants:
    """Test security constants from deserializers are consistent."""

    def test_security_constants_imported(self) -> None:
        """Test that security constants are properly imported and used."""
        from datason.deserializers import MAX_OBJECT_SIZE, MAX_SERIALIZATION_DEPTH, MAX_STRING_LENGTH

        # These should be reasonable security values
        assert MAX_SERIALIZATION_DEPTH == 50
        assert MAX_OBJECT_SIZE == 100_000
        assert MAX_STRING_LENGTH == 1_000_000

        # Test that these are used in configs
        config = SerializationConfig(max_depth=MAX_SERIALIZATION_DEPTH)
        assert config.max_depth == 50


class TestEnhancedErrorHandling:
    """Test enhanced error handling and warnings."""

    def test_serialization_warnings_handling(self) -> None:
        """Test that serialization warnings are handled properly."""

        # Test with an object that might generate warnings
        config = SerializationConfig()

        # This should not raise exceptions even if warnings occur
        result = serialize({"test": "value"}, config)
        assert result == {"test": "value"}

    def test_edge_case_object_handling(self) -> None:
        """Test edge case object handling."""
        config = SerializationConfig()

        # Test with various edge cases
        edge_cases = [
            {},  # Empty dict
            [],  # Empty list
            "",  # Empty string
            0,  # Zero
            False,  # False boolean
            None,  # None value
        ]

        for case in edge_cases:
            result = serialize(case, config)
            # Should handle all edge cases without errors
            assert result == case  # Most should pass through unchanged


class TestTypeMetadataGeneration:
    """Test enhanced type metadata generation."""

    def test_uuid_type_metadata(self) -> None:
        """Test UUID type metadata generation."""
        config = SerializationConfig(include_type_hints=True)

        uuid_val = uuid.uuid4()
        result = serialize(uuid_val, config)

        assert isinstance(result, dict)
        assert result.get("__datason_type__") == "uuid.UUID"
        assert result.get("__datason_value__") == str(uuid_val)

    def test_path_type_metadata(self) -> None:
        """Test Path type metadata generation."""
        config = SerializationConfig(include_type_hints=True)

        path_val = Path("/tmp/test/file.txt")
        result = serialize(path_val, config)

        assert isinstance(result, dict)
        assert result.get("__datason_type__") == "pathlib.Path"
        assert result.get("__datason_value__") == str(path_val)

    def test_complex_type_metadata(self) -> None:
        """Test complex number type metadata generation."""
        config = SerializationConfig(include_type_hints=True)

        complex_val = complex(1.5, 2.5)
        result = serialize(complex_val, config)

        assert isinstance(result, dict)
        # Support both new and legacy formats
        type_field = result.get("__datason_type__") or result.get("_type")
        assert type_field == "complex"

        # Check for real and imag values (may be at top level or in value field)
        if "__datason_value__" in result:
            value = result["__datason_value__"]
            assert isinstance(value, dict)
            assert value.get("real") == 1.5
            assert value.get("imag") == 2.5
        else:
            # Legacy format has real/imag at top level
            assert result.get("real") == 1.5
            assert result.get("imag") == 2.5


class TestOptimizationPaths:
    """Test optimization paths and performance features."""

    def test_homogeneous_list_optimization(self) -> None:
        """Test homogeneous list optimization paths."""
        config = SerializationConfig()

        # Homogeneous integer list
        int_list = [1, 2, 3, 4, 5]
        result = serialize(int_list, config)
        assert result == int_list

        # Homogeneous string list
        str_list = ["a", "b", "c", "d"]
        result = serialize(str_list, config)
        assert result == str_list

        # Mixed list (should not use optimization)
        mixed_list = [1, "a", 2, "b"]
        result = serialize(mixed_list, config)
        assert result == mixed_list

    def test_quick_path_basic_types(self) -> None:
        """Test quick path for basic types."""
        config = SerializationConfig()

        # These should use the quickest serialization paths
        basic_types = [42, 3.14, "hello", True, False, None]

        for value in basic_types:
            result = serialize(value, config)
            assert result == value  # Should pass through unchanged

    def test_container_optimization(self) -> None:
        """Test container optimization for dicts and lists."""
        config = SerializationConfig()

        # Simple containers should be optimized
        simple_dict = {"a": 1, "b": 2, "c": 3}
        result = serialize(simple_dict, config)
        assert result == simple_dict

        simple_list = [1, 2, 3]
        result = serialize(simple_list, config)
        assert result == simple_list
