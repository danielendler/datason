"""Exception flows and edge cases tests - Batch 3.

Targets remaining missing lines in deserializers.py focusing on security,
performance optimizations, and complex data structures.
"""

import pytest

from datason.deserializers import (
    DeserializationSecurityError,
    TemplateDeserializationError,
    _clear_deserialization_caches,
    _get_cached_string_pattern,
    _get_pooled_dict,
    _is_homogeneous_basic_types,
    _return_dict_to_pool,
    auto_deserialize,
    deserialize,
    deserialize_fast,
    parse_datetime_string,
    parse_uuid_string,
)


class TestSecurityConstraints:
    """Test security constraint enforcement."""

    def test_deserialization_security_errors(self):
        """Test various security errors are raised properly."""
        # Test maximum depth exceeded
        deeply_nested = {"level": 1}
        current = deeply_nested
        for i in range(2, 60):  # Create deeply nested structure
            current["next"] = {"level": i}
            current = current["next"]

        with pytest.raises(DeserializationSecurityError, match="Maximum depth exceeded"):
            deserialize_fast(deeply_nested)

    def test_large_object_security_check(self):
        """Test security checks on large objects."""
        # Create a very large dictionary
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100000)}

        # Should trigger security checks
        with pytest.raises(DeserializationSecurityError, match="Object too large"):
            deserialize_fast(large_dict)

    def test_string_length_security_check(self):
        """Test security checks on very long strings."""
        # Create an extremely long string
        very_long_string = "x" * 2_000_000  # 2MB string

        data = {"long_string": very_long_string}

        # Should trigger security checks
        with pytest.raises(DeserializationSecurityError, match="String too long"):
            deserialize_fast(data)


class TestCachingMechanisms:
    """Test caching mechanisms for parsed objects."""

    def test_cached_string_pattern_storage_and_retrieval(self):
        """Test string pattern caching mechanisms."""
        test_string = "2023-01-01T12:00:00"

        # Test cache miss first
        cached_miss = _get_cached_string_pattern("not_in_cache")
        assert cached_miss is None

        # Test with datetime-like string
        cached = _get_cached_string_pattern(test_string)
        # Should return either None or a pattern type
        assert cached is None or isinstance(cached, str)

    def test_object_pooling_mechanisms(self):
        """Test object pooling for performance optimization."""
        # Test dict pooling
        pooled_dict = _get_pooled_dict()
        assert isinstance(pooled_dict, dict)
        assert len(pooled_dict) == 0  # Should be empty

        # Return to pool
        pooled_dict["test"] = "value"
        _return_dict_to_pool(pooled_dict)

        # Test cache clearing
        _clear_deserialization_caches()
        # Should not raise any errors


class TestStringParsingFallbacks:
    """Test string parsing edge cases and fallbacks."""

    def test_parse_datetime_string_edge_cases(self):
        """Test datetime string parsing with various edge cases."""
        # Test None input
        assert parse_datetime_string(None) is None

        # Test non-string input
        assert parse_datetime_string(123) is None
        assert parse_datetime_string([]) is None

        # Test invalid datetime strings
        assert parse_datetime_string("invalid datetime") is None
        assert parse_datetime_string("{not-a-datetime}") is None

        # Test empty string
        assert parse_datetime_string("") is None

    def test_parse_uuid_string_edge_cases(self):
        """Test UUID parsing with edge cases."""
        # Test None input
        assert parse_uuid_string(None) is None

        # Test invalid UUID strings
        assert parse_uuid_string("invalid-uuid") is None
        assert parse_uuid_string("12345") is None

        # Test empty string
        assert parse_uuid_string("") is None


class TestBasicTypeDetection:
    """Test basic type detection for optimization."""

    def test_is_homogeneous_basic_types_edge_cases(self):
        """Test basic homogeneous type detection with edge cases."""
        # Test empty list
        assert _is_homogeneous_basic_types([])

        # Test single element
        assert _is_homogeneous_basic_types([1])

        # Test None values mixed with other types
        assert not _is_homogeneous_basic_types([1, None, 2])

        # Test with different numeric types
        assert not _is_homogeneous_basic_types([1, 1.5, 2])

        # Test with very large lists
        large_homogeneous = [1] * 1000  # Smaller to avoid timeout
        assert _is_homogeneous_basic_types(large_homogeneous)


class TestLargeDataProcessing:
    """Test large data processing optimizations."""

    def test_auto_deserialize_with_large_data(self):
        """Test auto-deserialization with large data structures."""
        # Create moderately large data to test processing
        large_data = {
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(100)},
            "nested": {"data": [{"id": i, "value": i * 2} for i in range(100)]},
        }

        # Should handle large data gracefully
        result = auto_deserialize(large_data)
        assert isinstance(result, dict)
        assert len(result["large_list"]) == 1000
        assert len(result["large_dict"]) == 100


class TestDataStructureEdgeCases:
    """Test edge cases with complex data structures."""

    def test_deserialize_circular_references(self):
        """Test handling of circular references in data structures."""
        # Create a structure that would cause infinite recursion if not handled
        data = {"self": None}
        data["self"] = data  # Circular reference

        # This should be handled gracefully (either by detection or limits)
        with pytest.raises(DeserializationSecurityError):
            deserialize_fast(data)

    def test_deserialize_very_wide_objects(self):
        """Test deserialization of objects with many keys."""
        # Create an object with many keys
        wide_object = {f"key_{i}": i for i in range(10000)}

        # Should handle or reject based on security constraints
        try:
            result = deserialize_fast(wide_object)
            # If successful, should preserve structure
            assert len(result) == len(wide_object)
        except DeserializationSecurityError:
            # Expected for very wide objects due to security limits
            pass

    def test_deserialize_mixed_containers(self):
        """Test deserialization of complex mixed containers."""
        complex_data = {
            "lists": [[1, 2, 3], ["a", "b", "c"], [True, False, None]],
            "dicts": [{"a": 1}, {"b": 2}, {"c": {"nested": True}}],
            "mixed": [1, "string", {"dict": "value"}, [1, 2, 3]],
            "empty_containers": [[], {}, None],
        }

        result = deserialize(complex_data)
        assert isinstance(result, dict)
        assert len(result["lists"]) == 3
        assert len(result["dicts"]) == 3


class TestTemplateDeserializerEdgeCases:
    """Test TemplateDeserializer edge cases."""

    def test_template_deserializer_with_invalid_template(self):
        """Test TemplateDeserializer with invalid template data."""
        from datason.deserializers import TemplateDeserializer

        # Test with invalid template structure
        invalid_templates = [
            None,
            "not_a_dict",
            123,
            [],
            {"invalid": "structure"},
        ]

        for invalid_template in invalid_templates:
            deserializer = TemplateDeserializer()

            # Should handle invalid templates gracefully
            try:
                result = deserializer.deserialize({"data": "test"}, template=invalid_template)
                # If it doesn't raise an error, it should return something reasonable
                assert result is not None
            except (TemplateDeserializationError, TypeError, ValueError):
                # These exceptions are acceptable for invalid templates
                pass

    def test_template_deserializer_with_mismatched_data(self):
        """Test TemplateDeserializer with data that doesn't match template."""
        from datason.deserializers import TemplateDeserializer

        template = {
            "expected_key": "string",
            "expected_number": 42,
        }

        mismatched_data = {
            "wrong_key": "value",
            "different_structure": [1, 2, 3],
        }

        deserializer = TemplateDeserializer()

        # Should handle mismatched data gracefully
        result = deserializer.deserialize(mismatched_data, template=template)
        # Should return the data even if it doesn't match template
        assert result is not None


class TestAutoDeserializeAggressiveMode:
    """Test aggressive mode in auto_deserialize."""

    def test_auto_deserialize_aggressive_with_edge_cases(self):
        """Test aggressive auto-deserialization with edge cases."""
        # Test with data that might trigger pandas detection in aggressive mode
        tabular_like_data = {"columns": ["A", "B", "C"], "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "index": [0, 1, 2]}

        # Test both modes
        result_normal = auto_deserialize(tabular_like_data, aggressive=False)
        result_aggressive = auto_deserialize(tabular_like_data, aggressive=True)

        # Both should return valid results
        assert result_normal is not None
        assert result_aggressive is not None

        # In aggressive mode, might get converted to DataFrame if pandas available
        # In normal mode, should remain as dict
        assert isinstance(result_normal, dict)

    def test_auto_deserialize_with_complex_nested_structure(self):
        """Test auto-deserialization with complex nested structures."""
        complex_structure = {
            "metadata": {
                "timestamp": "2023-01-01T12:00:00",
                "uuid": "12345678-1234-5678-9012-123456789abc",
            },
            "data": {
                "records": [
                    {"id": 1, "date": "2023-01-01", "value": 100.5},
                    {"id": 2, "date": "2023-01-02", "value": 200.5},
                ],
                "summary": {
                    "total": 2,
                    "avg_value": 150.5,
                },
            },
        }

        result = auto_deserialize(complex_structure, aggressive=True)

        # Should process nested structures appropriately
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result
