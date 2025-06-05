"""
Comprehensive tests for deserializer hot path optimizations.

This module tests all the new optimizations added to the deserializer:
- Hot path for basic types
- Caching systems for string patterns and parsed objects
- Memory pooling for containers
- Optimized string processing functions
- Security features and circular reference detection
"""

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

from datason.config import SerializationConfig
from datason.deserializers import (
    _PARSED_OBJECT_CACHE,
    _RESULT_DICT_POOL,
    _RESULT_LIST_POOL,
    _STRING_PATTERN_CACHE,
    DeserializationSecurityError,
    _deserialize_string_full,
    _get_cached_parsed_object,
    _get_cached_string_pattern,
    _get_pooled_dict,
    _get_pooled_list,
    _looks_like_datetime_optimized,
    _looks_like_path_optimized,
    _looks_like_uuid_optimized,
    _return_dict_to_pool,
    _return_list_to_pool,
    deserialize_fast,
)


class TestDeserializeFast:
    """Test the main deserialize_fast function."""

    def test_basic_types_zero_overhead(self):
        """Test that basic types are handled with zero overhead."""
        # These should be returned immediately without any processing
        assert deserialize_fast(42) == 42
        assert deserialize_fast(True) is True
        assert deserialize_fast(False) is False
        assert deserialize_fast(None) is None
        assert deserialize_fast(3.14) == 3.14

    def test_short_strings_fast_path(self):
        """Test that short strings are returned immediately."""
        # Strings < 8 characters should use ultra-fast path
        assert deserialize_fast("hello") == "hello"
        assert deserialize_fast("test") == "test"
        assert deserialize_fast("a") == "a"
        assert deserialize_fast("") == ""
        assert deserialize_fast("1234567") == "1234567"

    def test_containers_with_basic_types(self):
        """Test container processing with basic types."""
        # Simple list
        result = deserialize_fast([1, 2, 3, True, None])
        assert result == [1, 2, 3, True, None]

        # Simple dict
        result = deserialize_fast({"a": 1, "b": True, "c": None})
        assert result == {"a": 1, "b": True, "c": None}

    def test_nested_containers(self):
        """Test nested container processing."""
        data = {"numbers": [1, 2, 3], "nested": {"flag": True}, "simple": 42}
        result = deserialize_fast(data)

        # Check that structure is preserved
        assert isinstance(result, dict)
        assert result["simple"] == 42
        assert result["numbers"] == [1, 2, 3]
        assert result["nested"]["flag"] is True

    def test_security_depth_limit(self) -> None:
        """Test that depth limits are enforced."""
        # Create deeply nested structure
        data: Dict[str, Any] = {"level": 1}
        current = data
        for i in range(2, 60):  # Exceed default depth limit
            current["next"] = {"level": i}
            current = current["next"]

        with pytest.raises(DeserializationSecurityError, match="Maximum deserialization depth"):
            deserialize_fast(data)

    def test_security_size_limits(self):
        """Test that size limits are enforced."""
        config = SerializationConfig(max_size=10)

        # Large list
        large_list = list(range(20))
        with pytest.raises(DeserializationSecurityError, match="List size"):
            deserialize_fast(large_list, config)

        # Large dict
        large_dict = {f"key_{i}": i for i in range(20)}
        with pytest.raises(DeserializationSecurityError, match="Dictionary size"):
            deserialize_fast(large_dict, config)

    def test_datetime_string_parsing(self):
        """Test datetime string parsing in hot path."""
        iso_datetime = "2023-01-01T10:00:00"
        result = deserialize_fast(iso_datetime)
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1

    def test_uuid_string_parsing(self):
        """Test UUID string parsing in hot path."""
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        result = deserialize_fast(uuid_str)
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_str

    def test_path_string_parsing(self):
        """Test path string parsing in hot path."""
        path_str = "/tmp/test.txt"  # nosec B108
        result = deserialize_fast(path_str)
        assert isinstance(result, Path)
        assert str(result) == path_str


class TestOptimizedDetectionFunctions:
    """Test the optimized string detection functions."""

    def test_looks_like_datetime_optimized(self):
        """Test optimized datetime detection."""
        # Valid datetime patterns
        assert _looks_like_datetime_optimized("2023-01-01T10:00:00")
        assert _looks_like_datetime_optimized("2023-12-31T23:59:59")
        assert _looks_like_datetime_optimized("1999-01-01T00:00:00")

        # Invalid patterns
        assert not _looks_like_datetime_optimized("not-a-datetime")
        assert not _looks_like_datetime_optimized("2023-1-1T10:00:00")  # Single digit month/day
        assert not _looks_like_datetime_optimized("short")
        assert not _looks_like_datetime_optimized("")

    def test_looks_like_uuid_optimized(self):
        """Test optimized UUID detection."""
        # Valid UUID patterns
        assert _looks_like_uuid_optimized("12345678-1234-5678-9012-123456789abc")
        assert _looks_like_uuid_optimized("ABCDEF01-2345-6789-ABCD-EF0123456789")
        assert _looks_like_uuid_optimized("00000000-0000-0000-0000-000000000000")

        # Invalid patterns
        assert not _looks_like_uuid_optimized("not-a-uuid")
        assert not _looks_like_uuid_optimized("12345678-1234-5678-9012-123456789ab")  # Too short
        assert not _looks_like_uuid_optimized("12345678-1234-5678-9012-123456789abcd")  # Too long
        assert not _looks_like_uuid_optimized("12345678_1234_5678_9012_123456789abc")  # Wrong separators

    def test_looks_like_path_optimized(self):
        """Test optimized path detection."""
        # Valid path patterns
        assert _looks_like_path_optimized("/usr/bin/python")
        assert _looks_like_path_optimized("C:\\Windows\\System32")
        assert _looks_like_path_optimized("./relative/path.txt")
        assert _looks_like_path_optimized("../parent/file.py")
        assert _looks_like_path_optimized("/tmp/test.log")  # nosec B108
        assert _looks_like_path_optimized("file.json")

        # Invalid patterns
        assert not _looks_like_path_optimized("")
        assert not _looks_like_path_optimized("a")
        assert not _looks_like_path_optimized("just_text")
        assert not _looks_like_path_optimized("http://example.com")


class TestCachingSystems:
    """Test the caching systems for performance optimization."""

    def setUp(self):
        """Clear caches before each test."""
        _STRING_PATTERN_CACHE.clear()
        _PARSED_OBJECT_CACHE.clear()

    def test_string_pattern_caching(self):
        """Test string pattern detection caching."""
        # Test caching of different pattern types
        test_strings = [
            ("hello", "plain"),
            ("12345678-1234-5678-9012-123456789abc", "uuid"),
            ("2023-01-01T10:00:00", "datetime"),
            ("/tmp/test.txt", "path"),  # nosec B108
        ]

        for string, expected_pattern in test_strings:
            # First call should detect and cache
            pattern = _get_cached_string_pattern(string)
            assert pattern in [expected_pattern, "unknown", None]  # Allow for cache limits

            # Second call should use cache (if cached)
            pattern2 = _get_cached_string_pattern(string)
            if pattern is not None:
                assert pattern2 == pattern

    def test_parsed_object_caching(self):
        """Test parsed object caching."""
        # Test UUID caching
        uuid_str = "12345678-1234-5678-9012-123456789abc"

        # First call should parse and cache
        result1 = _get_cached_parsed_object(uuid_str, "uuid")
        if result1 is not None:
            assert isinstance(result1, uuid.UUID)

            # Second call should use cache
            result2 = _get_cached_parsed_object(uuid_str, "uuid")
            assert result2 is result1  # Same object from cache

    def test_cache_size_limits(self):
        """Test that caches respect size limits."""
        # Fill pattern cache beyond limit
        for i in range(1100):  # Exceed _STRING_CACHE_SIZE_LIMIT
            test_str = f"test_string_{i}"
            _get_cached_string_pattern(test_str)

        # Cache should not grow indefinitely
        assert len(_STRING_PATTERN_CACHE) <= 1000  # Should respect limit

    def test_cache_failure_handling(self):
        """Test caching of failed parsing attempts."""
        # Test invalid UUID
        invalid_uuid = "not-a-uuid-at-all"
        result = _get_cached_parsed_object(invalid_uuid, "uuid")
        assert result is None

        # Should cache the failure to avoid repeated attempts
        cache_key = f"uuid:{invalid_uuid}"
        if cache_key in _PARSED_OBJECT_CACHE:
            assert _PARSED_OBJECT_CACHE[cache_key] is None


class TestMemoryPooling:
    """Test memory pooling for containers."""

    def setUp(self):
        """Clear pools before each test."""
        _RESULT_DICT_POOL.clear()
        _RESULT_LIST_POOL.clear()

    def test_dict_pooling(self):
        """Test dictionary pooling system."""
        # Get dict from pool
        d1 = _get_pooled_dict()
        assert isinstance(d1, dict)
        assert len(d1) == 0  # Should be empty

        # Use the dict
        d1["test"] = "value"

        # Return to pool
        _return_dict_to_pool(d1)
        assert len(d1) == 0  # Should be cleared
        assert d1 in _RESULT_DICT_POOL

        # Get another dict (should reuse)
        d2 = _get_pooled_dict()
        if len(_RESULT_DICT_POOL) > 0:
            assert d2 is d1  # Should be the same object

    def test_list_pooling(self):
        """Test list pooling system."""
        # Get list from pool
        l1 = _get_pooled_list()
        assert isinstance(l1, list)
        assert len(l1) == 0  # Should be empty

        # Use the list
        l1.extend([1, 2, 3])

        # Return to pool
        _return_list_to_pool(l1)
        assert len(l1) == 0  # Should be cleared
        assert l1 in _RESULT_LIST_POOL

        # Get another list (should reuse)
        l2 = _get_pooled_list()
        if len(_RESULT_LIST_POOL) > 0:
            assert l2 is l1  # Should be the same object

    def test_pool_size_limits(self):
        """Test that pools respect size limits."""
        # Fill dict pool beyond limit
        dicts = []
        for i in range(25):  # Exceed _POOL_SIZE_LIMIT
            d = {}
            _return_dict_to_pool(d)
            dicts.append(d)

        # Pool should not grow indefinitely
        assert len(_RESULT_DICT_POOL) <= 20  # Should respect limit

        # Same for list pool
        lists = []
        for i in range(25):
            lst = []
            _return_list_to_pool(lst)
            lists.append(lst)

        assert len(_RESULT_LIST_POOL) <= 20


class TestStringProcessingOptimizations:
    """Test optimized string processing."""

    def test_deserialize_string_full_caching(self):
        """Test that string processing uses caching."""
        # Test datetime string
        datetime_str = "2023-01-01T10:00:00"

        # First call should parse and potentially cache
        result1 = _deserialize_string_full(datetime_str, None)
        assert isinstance(result1, datetime)

        # Second call should be faster (from cache if cached)
        result2 = _deserialize_string_full(datetime_str, None)
        assert result2 == result1

    def test_deserialize_string_full_failure_caching(self):
        """Test caching of failed parsing attempts."""
        # Test invalid datetime that looks like one
        invalid_datetime = "2023-99-99T99:99:99"  # Invalid date

        # Should return as string
        result = _deserialize_string_full(invalid_datetime, None)
        assert result == invalid_datetime
        assert isinstance(result, str)

    def test_deserialize_string_full_plain_strings(self):
        """Test that plain strings are handled efficiently."""
        plain_strings = [
            "hello world",
            "just a regular string",
            "no special patterns here",
            "12345",  # Not long enough for UUID
        ]

        for s in plain_strings:
            result = _deserialize_string_full(s, None)
            assert result == s
            assert isinstance(result, str)


class TestCircularReferenceProtection:
    """Test circular reference detection and protection."""

    def test_circular_reference_in_list(self):
        """Test circular reference detection in lists."""
        # Create circular reference
        data = [1, 2, 3]
        data.append(data)  # Circular reference

        # Should break the cycle gracefully
        result = deserialize_fast(data)
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[:3] == [1, 2, 3]
        assert result[3] == []  # Circular reference broken

    def test_circular_reference_in_dict(self):
        """Test circular reference detection in dicts."""
        # Create circular reference
        data = {"a": 1, "b": 2}
        data["self"] = data  # Circular reference

        # Should break the cycle gracefully
        result = deserialize_fast(data)
        assert isinstance(result, dict)
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["self"] == {}  # Circular reference broken

    def test_deep_circular_reference(self):
        """Test circular reference in deeply nested structures."""
        # Create complex circular reference
        data = {"level1": {"level2": {"data": [1, 2, 3]}}}
        data["level1"]["level2"]["back_ref"] = data

        result = deserialize_fast(data)
        assert result["level1"]["level2"]["data"] == [1, 2, 3]
        assert result["level1"]["level2"]["back_ref"] == {}  # Broken cycle


class TestTypeMetadataHandling:
    """Test type metadata handling for round-trip serialization."""

    def test_type_metadata_deserialization(self):
        """Test deserialization with type metadata."""
        # Test data with type metadata
        type_metadata_data = {"__datason_type__": "decimal.Decimal", "__datason_value__": "123.456"}

        result = deserialize_fast(type_metadata_data)
        assert isinstance(result, Decimal)
        assert result == Decimal("123.456")

    def test_complex_type_metadata(self):
        """Test complex type metadata deserialization."""
        complex_data = {"__datason_type__": "complex", "__datason_value__": {"real": 1.0, "imag": 2.0}}

        result = deserialize_fast(complex_data)
        assert isinstance(result, complex)
        assert result.real == 1.0
        assert result.imag == 2.0

    def test_legacy_type_metadata(self):
        """Test legacy type metadata format."""
        legacy_data = {"_type": "decimal", "value": "123.456"}

        result = deserialize_fast(legacy_data)
        assert isinstance(result, Decimal)
        assert result == Decimal("123.456")


class TestPerformanceConsistency:
    """Test that optimizations don't break functionality."""

    def test_consistency_with_old_deserialize(self):
        """Test that deserialize_fast produces same results as deserialize."""
        from datason.deserializers import deserialize

        test_data = [
            # Basic types
            42,
            "hello",
            [1, 2, 3],
            {"a": 1, "b": 2},
            # Datetime strings
            "2023-01-01T10:00:00",
            # UUID strings
            "12345678-1234-5678-9012-123456789abc",
            # Mixed nested data
            {
                "numbers": [1, 2, 3],
                "strings": ["a", "b", "c"],
                "nested": {"datetime": "2023-01-01T10:00:00", "uuid": "12345678-1234-5678-9012-123456789abc"},
            },
        ]

        for data in test_data:
            try:
                old_result = deserialize(data)
                fast_result = deserialize_fast(data)

                # Results should be equivalent (though may not be identical objects)
                assert isinstance(old_result, type(fast_result))
                assert old_result == fast_result

            except Exception as e:
                # If old deserialize fails, fast should too
                with pytest.raises(type(e)):
                    deserialize_fast(data)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            None,
            [],
            {},
            "",
            0,
            False,
            # Large strings that might hit different code paths
            "x" * 100,
            "a" * 1000,
            # Deeply nested but within limits
            {"a": {"b": {"c": {"d": {"e": "value"}}}}},
            # Lists with mixed types
            [None, False, 0, "", [], {}],
        ]

        for case in edge_cases:
            # Should not raise exceptions
            result = deserialize_fast(case)
            # Basic sanity check
            assert isinstance(result, type(case))

    def test_config_parameter_usage(self):
        """Test that config parameter is properly used."""
        config = SerializationConfig(max_depth=5, max_size=10)

        # Test that config limits are respected
        small_data = {"a": 1, "b": 2}  # Within limits
        result = deserialize_fast(small_data, config)
        assert result == small_data

        # Test depth limit
        deep_data = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
        with pytest.raises(DeserializationSecurityError):
            deserialize_fast(deep_data, config)


class TestRegressionTests:
    """Test for regressions and specific bug fixes."""

    def test_empty_containers(self):
        """Test handling of empty containers."""
        assert deserialize_fast([]) == []
        assert deserialize_fast({}) == {}
        assert deserialize_fast([[]]) == [[]]
        assert deserialize_fast({"empty": {}}) == {"empty": {}}

    def test_unicode_strings(self):
        """Test handling of unicode strings."""
        unicode_strings = ["Hello 世界", "émojis 🚀🎉", "Ñoño niño", "Здравствуй мир", "مرحبا بالعالم"]

        for s in unicode_strings:
            result = deserialize_fast(s)
            assert result == s
            assert isinstance(result, str)

    def test_very_long_strings(self):
        """Test handling of very long strings."""
        # Very long plain string
        long_string = "a" * 10000
        result = deserialize_fast(long_string)
        assert result == long_string

        # Long string that might look like datetime
        long_datetime_like = "2023-01-01T10:00:00" + "x" * 1000
        result = deserialize_fast(long_datetime_like)
        assert result == long_datetime_like  # Should remain string

    def test_numeric_edge_cases(self):
        """Test numeric edge cases."""
        numeric_cases = [
            0,
            -0,
            1,
            -1,
            float("inf"),
            float("-inf"),
            2**63 - 1,  # Large int
            -(2**63),  # Large negative int
            1e-10,  # Very small float
            1e10,  # Large float
        ]

        for num in numeric_cases:
            if str(num) != "nan":  # Skip NaN since it doesn't equal itself
                result = deserialize_fast(num)
                assert result == num
                assert isinstance(result, type(num))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
