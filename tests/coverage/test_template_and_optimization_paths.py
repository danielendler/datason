"""Template deserializer and optimization paths coverage tests.

Targets remaining uncovered lines in template processing and performance optimizations.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from datason.deserializers import (
    TemplateDeserializationError,
    TemplateDeserializer,
    _clear_deserialization_caches,
    _get_pooled_dict,
    _get_pooled_list,
    _return_dict_to_pool,
    _return_list_to_pool,
    create_ml_round_trip_template,
    deserialize_fast,
    deserialize_with_template,
    infer_template_from_data,
)


class TestTemplateDeserializerCore:
    """Test core TemplateDeserializer functionality."""

    def test_template_deserializer_initialization(self):
        """Test TemplateDeserializer initialization and template analysis."""
        template = {
            "name": "John Doe",
            "age": 30,
            "active": True,
            "score": 95.5,
        }

        deserializer = TemplateDeserializer(template, strict=True, fallback_auto_detect=True)
        assert deserializer.strict is True
        assert deserializer.fallback_auto_detect is True
        assert deserializer.template == template

    def test_template_deserializer_with_datetime_template(self):
        """Test TemplateDeserializer with datetime template."""
        template = {
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
            "event": "test_event",
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "timestamp": "2023-06-15T14:30:00",
            "event": "user_login",
        }

        result = deserializer.deserialize(data)
        assert isinstance(result["timestamp"], datetime)
        assert result["event"] == "user_login"

    def test_template_deserializer_with_uuid_template(self):
        """Test TemplateDeserializer with UUID template."""
        template = {
            "id": UUID("12345678-1234-5678-9012-123456789abc"),
            "name": "test",
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "id": "87654321-4321-8765-2109-876543210def",
            "name": "actual_name",
        }

        result = deserializer.deserialize(data)
        assert isinstance(result["id"], UUID)
        assert result["name"] == "actual_name"

    def test_template_deserializer_with_complex_template(self):
        """Test TemplateDeserializer with complex number template."""
        template = {
            "impedance": complex(1, 2),
            "frequency": 100.0,
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "impedance": "3+4j",
            "frequency": 200.0,
        }

        result = deserializer.deserialize(data)
        assert isinstance(result["impedance"], complex)
        assert result["impedance"] == complex(3, 4)

    def test_template_deserializer_with_decimal_template(self):
        """Test TemplateDeserializer with Decimal template."""
        template = {
            "price": Decimal("10.99"),
            "currency": "USD",
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "price": "25.50",
            "currency": "EUR",
        }

        result = deserializer.deserialize(data)
        assert isinstance(result["price"], Decimal)
        assert result["price"] == Decimal("25.50")

    def test_template_deserializer_with_path_template(self):
        """Test TemplateDeserializer with Path template."""
        template = {
            "config_path": Path("/etc/config"),
            "debug": False,
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "config_path": "/home/user/config",
            "debug": True,
        }

        result = deserializer.deserialize(data)
        assert isinstance(result["config_path"], Path)
        assert result["config_path"] == Path("/home/user/config")

    def test_template_deserializer_list_processing(self):
        """Test TemplateDeserializer with list templates."""
        template = [
            {
                "id": 1,
                "timestamp": datetime(2023, 1, 1),
            }
        ]

        deserializer = TemplateDeserializer(template)

        data = [
            {"id": 10, "timestamp": "2023-06-15T14:30:00"},
            {"id": 20, "timestamp": "2023-06-16T09:15:00"},
        ]

        result = deserializer.deserialize(data)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0]["timestamp"], datetime)
        assert isinstance(result[1]["timestamp"], datetime)

    def test_template_deserializer_pandas_dataframe(self):
        """Test TemplateDeserializer with pandas DataFrame template."""
        pd = pytest.importorskip("pandas")

        template_df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4.0, 5.0, 6.0],
                "C": ["x", "y", "z"],
            }
        )

        deserializer = TemplateDeserializer(template_df)

        # Test with records format
        data = [
            {"A": 10, "B": 1.1, "C": "a"},
            {"A": 20, "B": 2.2, "C": "b"},
        ]

        result = deserializer.deserialize(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)

    def test_template_deserializer_pandas_series(self):
        """Test TemplateDeserializer with pandas Series template."""
        pd = pytest.importorskip("pandas")

        template_series = pd.Series([1, 2, 3], name="values")

        deserializer = TemplateDeserializer(template_series)

        data = {"0": 10, "1": 20, "2": 30}

        result = deserializer.deserialize(data)
        assert isinstance(result, pd.Series)
        assert result.name == "values"

    def test_template_deserializer_numpy_array(self):
        """Test TemplateDeserializer with NumPy array template."""
        np = pytest.importorskip("numpy")

        template_array = np.array([[1, 2], [3, 4]])

        deserializer = TemplateDeserializer(template_array)

        data = [[10, 20], [30, 40]]

        result = deserializer.deserialize(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_template_deserializer_numpy_scalar(self):
        """Test TemplateDeserializer with NumPy scalar template."""
        np = pytest.importorskip("numpy")

        template_scalar = np.int32(42)

        deserializer = TemplateDeserializer(template_scalar)

        data = 123

        result = deserializer.deserialize(data)
        assert isinstance(result, np.int32)
        assert result == 123

    def test_template_deserializer_torch_tensor(self):
        """Test TemplateDeserializer with PyTorch tensor template."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        template_tensor = torch.tensor([1.0, 2.0, 3.0])

        deserializer = TemplateDeserializer(template_tensor)

        data = [10.0, 20.0, 30.0]

        result = deserializer.deserialize(data)
        assert torch.is_tensor(result)
        assert result.tolist() == [10.0, 20.0, 30.0]

    def test_template_deserializer_strict_mode(self):
        """Test TemplateDeserializer strict mode behavior."""
        template = {
            "required_field": "string",
            "numeric_field": 42,
        }

        deserializer = TemplateDeserializer(template, strict=True)

        # Missing field should be handled
        data = {"required_field": "value"}  # Missing numeric_field

        deserializer.deserialize(data)
        # Should still work but may have different behavior

    def test_template_deserializer_fallback_auto_detect(self):
        """Test TemplateDeserializer fallback auto-detection."""
        template = {"known_field": "string"}

        deserializer = TemplateDeserializer(template, strict=False, fallback_auto_detect=True)

        data = {
            "known_field": "value",
            "unknown_field": "2023-01-01T12:00:00",  # Should be auto-detected as datetime
        }

        result = deserializer.deserialize(data)
        assert result["known_field"] == "value"
        # unknown_field might be auto-detected as datetime

    def test_template_deserializer_type_coercion(self):
        """Test TemplateDeserializer type coercion."""
        template = {
            "integer_field": 42,
            "float_field": 3.14,
            "string_field": "hello",
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "integer_field": "123",  # String that should be coerced to int
            "float_field": "2.71",  # String that should be coerced to float
            "string_field": 999,  # Number that should be coerced to string
        }

        deserializer.deserialize(data)
        # Should attempt type coercion based on template


class TestTemplateInferenceAndUtilities:
    """Test template inference and utility functions."""

    def test_infer_template_from_simple_data(self):
        """Test template inference from simple data structures."""
        data = {
            "name": "John",
            "age": 30,
            "active": True,
            "score": 95.5,
        }

        template = infer_template_from_data(data)
        assert isinstance(template, dict)
        # Template should capture the structure and types

    def test_infer_template_from_list_of_dicts(self):
        """Test template inference from list of dictionaries."""
        data = [
            {"id": 1, "name": "Alice", "score": 85.5},
            {"id": 2, "name": "Bob", "score": 92.0},
            {"id": 3, "name": "Charlie", "score": 78.5},
        ]

        infer_template_from_data(data, max_samples=2)
        # Should infer a template that captures the common structure

    def test_infer_template_from_nested_data(self):
        """Test template inference from nested data structures."""
        data = {
            "user": {
                "profile": {
                    "name": "John",
                    "email": "john@example.com",
                },
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                },
            },
            "metadata": {
                "created": "2023-01-01T12:00:00",
                "version": 1,
            },
        }

        template = infer_template_from_data(data)
        assert isinstance(template, dict)

    def test_create_ml_round_trip_template(self):
        """Test ML object round-trip template creation."""
        # Test with a simple object
        simple_obj = {"model_type": "test", "parameters": [1, 2, 3]}

        create_ml_round_trip_template(simple_obj)
        # Should create a template suitable for ML object serialization

    def test_create_ml_round_trip_template_with_numpy(self):
        """Test ML round-trip template with NumPy arrays."""
        np = pytest.importorskip("numpy")

        ml_obj = {
            "weights": np.array([[1, 2], [3, 4]]),
            "bias": np.array([0.1, 0.2]),
            "config": {"learning_rate": 0.01},
        }

        create_ml_round_trip_template(ml_obj)
        # Should handle NumPy arrays in the template

    def test_deserialize_with_template_function(self):
        """Test the standalone deserialize_with_template function."""
        template = {
            "timestamp": datetime(2023, 1, 1),
            "value": 42,
        }

        data = {
            "timestamp": "2023-06-15T14:30:00",
            "value": 123,
        }

        result = deserialize_with_template(data, template)
        assert isinstance(result["timestamp"], datetime)
        assert result["value"] == 123


class TestPerformanceOptimizations:
    """Test performance optimization paths."""

    def test_object_pooling_dict(self):
        """Test dictionary object pooling."""
        # Get a pooled dict
        pooled_dict1 = _get_pooled_dict()
        assert isinstance(pooled_dict1, dict)
        assert len(pooled_dict1) == 0

        # Use it
        pooled_dict1["test"] = "value"

        # Return it to pool
        _return_dict_to_pool(pooled_dict1)

        # Get another - might be the same recycled object
        pooled_dict2 = _get_pooled_dict()
        assert isinstance(pooled_dict2, dict)

    def test_object_pooling_list(self):
        """Test list object pooling."""
        # Get a pooled list
        pooled_list1 = _get_pooled_list()
        assert isinstance(pooled_list1, list)
        assert len(pooled_list1) == 0

        # Use it
        pooled_list1.extend([1, 2, 3])

        # Return it to pool
        _return_list_to_pool(pooled_list1)

        # Get another - might be the same recycled object
        pooled_list2 = _get_pooled_list()
        assert isinstance(pooled_list2, list)

    def test_cache_clearing(self):
        """Test deserialization cache clearing."""
        # Clear all caches - should not raise errors
        _clear_deserialization_caches()

        # Should be safe to call multiple times
        _clear_deserialization_caches()
        _clear_deserialization_caches()

    def test_deserialize_fast_with_config(self):
        """Test fast deserialization with config."""
        # Test with a config
        from datason.config import SerializationConfig

        config = SerializationConfig()

        data = {
            "string": "hello",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        result = deserialize_fast(data, config=config)
        assert result == data

    def test_deserialize_fast_with_none_config(self):
        """Test fast deserialization without config."""
        data = {
            "simple": "data",
            "nested": {"key": "value"},
        }

        result = deserialize_fast(data, config=None)
        assert result == data

    def test_deserialize_fast_complex_nested(self):
        """Test fast deserialization with complex nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "metadata": {
                            "created": "2023-01-01",
                            "version": 1,
                        },
                    }
                }
            },
            "arrays": [[1, 2], [3, 4], [5, 6]],
            "mixed": [
                {"type": "A", "value": 1},
                {"type": "B", "value": 2},
            ],
        }

        result = deserialize_fast(data)
        assert isinstance(result, dict)
        assert result["level1"]["level2"]["level3"]["data"] == [1, 2, 3]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in template processing."""

    def test_template_deserializer_error_handling(self):
        """Test TemplateDeserializer error handling."""
        template = {"field": "string"}

        deserializer = TemplateDeserializer(template)

        # Test with various problematic inputs
        problematic_inputs = [
            None,
            [],
            "",
            123,
            {"different": "structure"},
        ]

        for problematic_input in problematic_inputs:
            # Should handle gracefully without crashing
            deserializer.deserialize(problematic_input)
            # Result should be something reasonable

    def test_template_deserializationerror(self):
        """Test TemplateDeserializationError exception."""
        # Create and raise the error
        error = TemplateDeserializationError("Test error message")
        assert str(error) == "Test error message"

        # Should be a proper exception
        assert isinstance(error, Exception)

    def test_infer_template_with_empty_data(self):
        """Test template inference with empty or minimal data."""
        # Empty data
        infer_template_from_data([])
        # Should handle gracefully

        # Single item
        infer_template_from_data([{"single": "item"}])
        # Should work with minimal data

    def test_infer_template_with_inconsistent_data(self):
        """Test template inference with inconsistent data."""
        data = [
            {"field1": "string", "field2": 123},
            {"field1": 456, "field3": True},  # Different structure
            {"field2": "another_string"},  # Different fields
        ]

        infer_template_from_data(data)
        # Should handle inconsistent structures gracefully

    def test_template_deserializer_with_missing_libraries(self):
        """Test TemplateDeserializer when libraries are missing."""
        # Mock missing pandas
        with patch("datason.deserializers.pd", None):
            try:
                import pandas as pd

                template_df = pd.DataFrame({"A": [1, 2, 3]})
                deserializer = TemplateDeserializer(template_df)

                data = [{"A": 10}, {"A": 20}]
                deserializer.deserialize(data)
                # Should handle missing pandas gracefully
            except ImportError:
                # If pandas isn't available at all, that's fine too
                pass

    def test_deserialize_fast_security_limits(self):
        """Test deserialize_fast with security constraints."""
        # Test depth limit (if applicable)
        deep_data = {"level": 1}
        current = deep_data
        for i in range(2, 55):  # Create deep nesting
            current["next"] = {"level": i}
            current = current["next"]

        # Should handle deep nesting appropriately
        try:
            result = deserialize_fast(deep_data)
            # If it succeeds, the data should be processed
            assert isinstance(result, dict)
        except Exception as e:
            # If it fails due to security limits, that's expected
            assert "depth" in str(e).lower() or "security" in str(e).lower()

    def test_template_with_custom_types(self):
        """Test template handling with custom/unknown types."""

        # Create a custom class
        class CustomType:
            def __init__(self, value):
                self.value = value

        template = {
            "custom": CustomType("template_value"),
            "normal": "string",
        }

        deserializer = TemplateDeserializer(template)

        data = {
            "custom": "data_value",
            "normal": "normal_value",
        }

        # Should handle custom types gracefully
        result = deserializer.deserialize(data)
        assert result["normal"] == "normal_value"
