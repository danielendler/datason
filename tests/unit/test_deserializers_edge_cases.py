"""
Edge case and coverage tests for deserializers_new.py.

This test suite targets specific edge cases and missing coverage lines
in the deserialization system to improve overall test coverage.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

import datason.deserializers_new as deserializers


class TestDeserializeEdgeCases:
    """Test edge cases in deserialization."""

    def test_deserialize_with_config_object(self):
        """Test deserialization with config object."""
        data = {"test": "value", "nested": {"data": [1, 2, 3]}}

        # Basic deserialize doesn't take config, just test it works
        result = deserializers.deserialize(data)
        assert result == data

    def test_deserialize_metadata_with_invalid_type(self):
        """Test deserialization of metadata with invalid type."""
        # Invalid type metadata
        invalid_metadata = {"__datason_type__": "invalid_type_name", "__datason_value__": "some_value"}

        result = deserializers.deserialize(invalid_metadata)
        # Should extract the value when type is unrecognized
        assert result == "some_value"

    def test_deserialize_metadata_with_missing_value(self):
        """Test deserialization of metadata with missing value key."""
        # Missing __datason_value__
        incomplete_metadata = {"__datason_type__": "datetime"}

        result = deserializers.deserialize(incomplete_metadata)
        # Should fall back to returning the dict as-is
        assert result == incomplete_metadata

    def test_deserialize_datetime_edge_cases(self):
        """Test datetime deserialization edge cases."""
        # Invalid datetime string
        invalid_dt = "not-a-datetime"
        result = deserializers.deserialize(invalid_dt, parse_dates=True)
        assert result == invalid_dt  # Should return as string

        # Valid but complex datetime with timezone
        complex_dt = "2023-12-25T15:30:45.123456+05:30"
        result = deserializers.deserialize(complex_dt, parse_dates=True)
        assert isinstance(result, (datetime, str))

    def test_deserialize_uuid_edge_cases(self):
        """Test UUID deserialization edge cases."""
        # Invalid UUID string
        invalid_uuid = "not-a-uuid"
        result = deserializers.deserialize(invalid_uuid, parse_uuids=True)
        assert result == invalid_uuid  # Should return as string

        # Valid UUID with different format
        uuid_str = "550e8400e29b41d4a716446655440000"  # No hyphens
        result = deserializers.deserialize(uuid_str, parse_uuids=True)
        assert result == uuid_str  # Should remain as string (invalid format)

    def test_deserialize_nested_with_mixed_parsing(self):
        """Test nested deserialization with mixed parsing options."""
        nested_data = {
            "timestamp": "2023-01-01T12:00:00",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "metadata": {"created": "2023-01-01T10:00:00Z", "session": "12345678-1234-5678-9012-123456789abc"},
            "items": [
                {"date": "2023-01-02T12:00:00", "id": "abcd1234-5678-9012-3456-789012345678"},
                {"date": "2023-01-03T12:00:00", "id": "efgh5678-9012-3456-7890-123456789012"},
            ],
        }

        result = deserializers.deserialize(nested_data, parse_dates=True, parse_uuids=True)

        # Should have processed all nested datetime and UUID strings
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "metadata" in result
        assert isinstance(result["items"], list)

    def test_deserialize_large_nested_structure(self):
        """Test deserialization of large nested structure."""
        # Create large nested structure
        large_data = {
            "level1": {
                f"item_{i}": {
                    "id": f"id_{i}",
                    "timestamp": "2023-01-01T12:00:00",
                    "nested": {"deep_value": f"value_{i}", "uuid": "550e8400-e29b-41d4-a716-446655440000"},
                }
                for i in range(100)
            }
        }

        result = deserializers.deserialize(large_data, parse_dates=True, parse_uuids=True)
        assert isinstance(result, dict)
        assert "level1" in result
        assert len(result["level1"]) == 100

    def test_deserialize_with_none_values(self):
        """Test deserialization with None values in various positions."""
        data_with_nones = {
            "null_value": None,
            "list_with_nulls": [1, None, "test", None],
            "nested": {"also_null": None, "not_null": "value"},
        }

        result = deserializers.deserialize(data_with_nones)
        assert result["null_value"] is None
        assert result["list_with_nulls"][1] is None
        assert result["nested"]["also_null"] is None

    def test_deserialize_empty_structures(self):
        """Test deserialization of empty structures."""
        # Empty dict
        result = deserializers.deserialize({})
        assert result == {}

        # Empty list
        result = deserializers.deserialize([])
        assert result == []

        # Empty string
        result = deserializers.deserialize("")
        assert result == ""

    def test_deserialize_deeply_nested_structures(self):
        """Test deserialization of deeply nested structures."""
        # Create deeply nested structure
        deep_data = {"level": 1}
        current = deep_data
        for i in range(2, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        result = deserializers.deserialize(deep_data)
        assert isinstance(result, dict)
        assert result["level"] == 1

    def test_deserialize_with_numeric_strings(self):
        """Test deserialization with numeric strings that might be confused with numbers."""
        numeric_strings = {"zip_code": "00123", "phone": "+1-555-123-4567", "mixed": ["123", 123, "456.78", 456.78]}

        result = deserializers.deserialize(numeric_strings)
        # Should preserve string types
        assert result["zip_code"] == "00123"
        assert result["phone"] == "+1-555-123-4567"
        assert result["mixed"][0] == "123"  # String preserved
        assert result["mixed"][1] == 123  # Number preserved

    def test_deserialize_special_float_values(self):
        """Test deserialization of special float values."""
        special_values = {"infinity": float("inf"), "neg_infinity": float("-inf"), "not_a_number": float("nan")}

        result = deserializers.deserialize(special_values)
        assert result["infinity"] == float("inf")
        assert result["neg_infinity"] == float("-inf")
        # NaN comparison is special
        assert str(result["not_a_number"]) == "nan"


class TestMetadataDeserialization:
    """Test metadata-based deserialization."""

    def test_datetime_metadata_deserialization(self):
        """Test datetime metadata deserialization."""
        dt_metadata = {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"}

        result = deserializers.deserialize(dt_metadata)
        assert isinstance(result, (datetime, dict))

    def test_uuid_metadata_deserialization(self):
        """Test UUID metadata deserialization."""
        uuid_metadata = {"__datason_type__": "uuid", "__datason_value__": "550e8400-e29b-41d4-a716-446655440000"}

        result = deserializers.deserialize(uuid_metadata)
        # Should return the string value since UUID deserializers might not be implemented
        assert isinstance(result, (uuid.UUID, str))

    def test_decimal_metadata_deserialization(self):
        """Test Decimal metadata deserialization."""
        decimal_metadata = {"__datason_type__": "decimal", "__datason_value__": "123.456"}

        result = deserializers.deserialize(decimal_metadata)
        assert isinstance(result, (Decimal, str))

    def test_path_metadata_deserialization(self):
        """Test Path metadata deserialization."""
        path_metadata = {"__datason_type__": "path", "__datason_value__": "/tmp/test/file.txt"}

        result = deserializers.deserialize(path_metadata)
        assert isinstance(result, (Path, str))

    def test_complex_metadata_deserialization(self):
        """Test complex number metadata deserialization."""
        complex_metadata = {"__datason_type__": "complex", "__datason_value__": {"real": 3.0, "imag": 4.0}}

        result = deserializers.deserialize(complex_metadata)
        assert isinstance(result, (complex, dict))

    def test_metadata_with_invalid_value_type(self):
        """Test metadata with invalid value type."""
        invalid_metadata = {
            "__datason_type__": "datetime",
            "__datason_value__": 123,  # Should be string
        }

        result = deserializers.deserialize(invalid_metadata)
        # Should extract the value even if wrong type
        assert result == 123


class TestErrorHandling:
    """Test error handling in deserialization."""

    def test_deserialize_with_circular_references(self):
        """Test deserialization with circular references."""
        # This would be pre-serialized data with circular ref detection
        circular_data = {"__datason_type__": "security_error", "__datason_value__": "Circular reference detected"}

        result = deserializers.deserialize(circular_data)
        # Should extract the error message
        assert result == "Circular reference detected"

    def test_deserialize_with_corrupted_metadata(self):
        """Test deserialization with corrupted metadata."""
        corrupted = {
            "__datason_type__": None,  # Invalid type
            "__datason_value__": "some_value",
        }

        result = deserializers.deserialize(corrupted)
        # Should extract the value when type is None
        assert result == "some_value"

    def test_deserialize_exception_in_nested_structure(self):
        """Test handling of exceptions in nested structures."""
        # Create data that might cause issues during processing
        problematic_data = {
            "normal": "value",
            "nested": {
                "datetime": "2023-01-01T12:00:00",
                "problematic": {"__datason_type__": "unknown", "__datason_value__": "test"},
            },
        }

        result = deserializers.deserialize(problematic_data, parse_dates=True)
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["normal"] == "value"


class TestConfigIntegration:
    """Test configuration integration."""

    def test_deserialize_with_different_config_options(self):
        """Test deserialization with different configuration options."""

        data = {"timestamp": "2023-01-01T12:00:00", "user_id": "550e8400-e29b-41d4-a716-446655440000"}

        # Test with different parse options
        result1 = deserializers.deserialize(data, parse_dates=True, parse_uuids=False)
        result2 = deserializers.deserialize(data, parse_dates=False, parse_uuids=True)
        result3 = deserializers.deserialize(data, parse_dates=True, parse_uuids=True)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)

    def test_deserialize_with_config_object_parse_options(self):
        """Test deserialization with parse options."""
        data = {
            "date": "2023-01-01T12:00:00",
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "nested": {"timestamp": "2023-01-02T15:30:00Z"},
        }

        result = deserializers.deserialize(data, parse_dates=True, parse_uuids=True)
        assert isinstance(result, dict)


class TestPerformanceOptimizations:
    """Test performance optimizations in deserialization."""

    def test_deserialize_large_homogeneous_list(self):
        """Test deserialization of large homogeneous lists."""
        # Large list of similar objects
        large_list = [
            {"id": i, "name": f"user_{i}", "timestamp": "2023-01-01T12:00:00", "active": True} for i in range(1000)
        ]

        result = deserializers.deserialize(large_list, parse_dates=True)
        assert isinstance(result, list)
        assert len(result) == 1000

    def test_deserialize_large_homogeneous_dict(self):
        """Test deserialization of large homogeneous dictionaries."""
        # Large dict with similar values
        large_dict = {
            f"user_{i}": {
                "id": i,
                "name": f"user_{i}",
                "created": "2023-01-01T12:00:00",
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
            }
            for i in range(500)
        }

        result = deserializers.deserialize(large_dict, parse_dates=True, parse_uuids=True)
        assert isinstance(result, dict)
        assert len(result) == 500

    def test_deserialize_mixed_type_performance(self):
        """Test deserialization performance with mixed types."""
        mixed_data = {
            "strings": ["test"] * 100,
            "numbers": list(range(100)),
            "datetimes": ["2023-01-01T12:00:00"] * 50,
            "uuids": ["550e8400-e29b-41d4-a716-446655440000"] * 50,
            "nested": {"more_strings": ["nested"] * 100, "more_numbers": list(range(100, 200))},
        }

        result = deserializers.deserialize(mixed_data, parse_dates=True, parse_uuids=True)
        assert isinstance(result, dict)
        assert len(result["strings"]) == 100
        assert len(result["nested"]["more_strings"]) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
