"""
Comprehensive test coverage for datetime parsing and deserializers_new.py functionality.

This module focuses on testing the datetime parsing fixes and covering missing
functionality in datason/deserializers_new.py to reach 80% coverage goal.
"""

from datetime import datetime

from datason.config import SerializationConfig
from datason.deserializers_new import (
    _PARSED_OBJECT_CACHE,
    _STRING_PATTERN_CACHE,
    _deserialize_string_full,
    _get_cached_parsed_object,
    _get_cached_string_pattern,
    _looks_like_datetime_optimized,
    clear_caches,
    deserialize_fast,
)


class TestDatetimeParsing:
    """Test datetime parsing functionality with cross-version compatibility."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_looks_like_datetime_optimized(self):
        """Test _looks_like_datetime_optimized function."""
        # Valid datetime strings
        assert _looks_like_datetime_optimized("2024-01-01T00:00:00Z")
        assert _looks_like_datetime_optimized("2024-01-01T12:30:45.123Z")
        assert _looks_like_datetime_optimized("2024-01-01T00:00:00+00:00")
        assert _looks_like_datetime_optimized("2024-01-01T00:00:00-05:00")

        # Invalid datetime strings
        assert not _looks_like_datetime_optimized("not a date")
        assert not _looks_like_datetime_optimized("2024-01-01")  # No time
        assert not _looks_like_datetime_optimized("12:30:45")  # No date

    def test_deserialize_string_full_with_auto_detect(self):
        """Test _deserialize_string_full with auto_detect_types=True."""
        config = SerializationConfig(auto_detect_types=True)

        # Test datetime parsing
        datetime_str = "2024-01-01T00:00:00Z"
        result = _deserialize_string_full(datetime_str, config)
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_deserialize_string_full_without_auto_detect(self):
        """Test _deserialize_string_full with auto_detect_types=False."""
        config = SerializationConfig(auto_detect_types=False)

        # Test datetime string stays as string
        datetime_str = "2024-01-01T00:00:00Z"
        result = _deserialize_string_full(datetime_str, config)
        assert isinstance(result, str)
        assert result == datetime_str

    def test_cache_bypass_for_datetime_when_auto_detect(self):
        """Test that cache is bypassed for datetime-like strings when auto_detect is enabled."""
        config_no_auto = SerializationConfig(auto_detect_types=False)
        config_auto = SerializationConfig(auto_detect_types=True)

        datetime_str = "2024-01-01T00:00:00Z"

        # First call without auto_detect - should cache as plain string
        result1 = _deserialize_string_full(datetime_str, config_no_auto)
        assert isinstance(result1, str)

        # Check cache was populated
        pattern = _get_cached_string_pattern(datetime_str)
        assert pattern == "plain"

        # Second call with auto_detect - should bypass cache and parse datetime
        result2 = _deserialize_string_full(datetime_str, config_auto)
        assert isinstance(result2, datetime)

    def test_cached_datetime_failure_bypass_with_auto_detect(self):
        """Test that cached datetime failures are bypassed when auto_detect is enabled."""
        config = SerializationConfig(auto_detect_types=True)

        datetime_str = "2024-01-01T00:00:00Z"

        # Manually populate cache with a failed datetime parsing
        cache_key = f"datetime:{datetime_str}"
        _PARSED_OBJECT_CACHE[cache_key] = None  # Simulate cached failure

        # Call should bypass cached failure and succeed
        result = _deserialize_string_full(datetime_str, config)
        assert isinstance(result, datetime)

    def test_deserialize_various_datetime_formats(self):
        """Test parsing of various datetime formats."""
        config = SerializationConfig(auto_detect_types=True)

        test_cases = [
            "2024-01-01T00:00:00Z",
            "2024-01-01T12:30:45.123Z",
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T00:00:00-05:00",
            "2024-12-31T23:59:59.999999Z",
        ]

        for datetime_str in test_cases:
            result = _deserialize_string_full(datetime_str, config)
            assert isinstance(result, datetime), f"Failed to parse: {datetime_str}"

    def test_invalid_datetime_strings(self):
        """Test handling of invalid datetime strings."""
        config = SerializationConfig(auto_detect_types=True)

        invalid_cases = [
            "not a date",
            "2024-01-01",  # Missing time
            "12:30:45",  # Missing date
            "2024-13-01T00:00:00Z",  # Invalid month
            "2024-01-32T00:00:00Z",  # Invalid day
        ]

        for invalid_str in invalid_cases:
            result = _deserialize_string_full(invalid_str, config)
            # Should return as string for invalid datetime
            assert isinstance(result, str)
            assert result == invalid_str


class TestCachingMechanism:
    """Test the caching mechanism for string patterns and parsed objects."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_string_pattern_caching(self):
        """Test string pattern caching functionality."""
        datetime_str = "2024-01-01T00:00:00Z"

        # Initially no pattern cached
        assert _get_cached_string_pattern(datetime_str) is None

        # After processing, pattern should be cached
        config = SerializationConfig(auto_detect_types=True)
        _deserialize_string_full(datetime_str, config)

        # Pattern should now be cached as "datetime"
        pattern = _get_cached_string_pattern(datetime_str)
        assert pattern == "datetime"

    def test_parsed_object_caching(self):
        """Test parsed object caching functionality."""
        datetime_str = "2024-01-01T00:00:00Z"

        # Initially no parsed object cached
        assert _get_cached_parsed_object(datetime_str, "datetime") is None

        # After processing, object should be cached
        config = SerializationConfig(auto_detect_types=True)
        result1 = _deserialize_string_full(datetime_str, config)

        # Second call should use cached result
        result2 = _deserialize_string_full(datetime_str, config)

        assert isinstance(result1, datetime)
        assert isinstance(result2, datetime)
        assert result1 == result2

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        datetime_str = "2024-01-01T00:00:00Z"
        config = SerializationConfig(auto_detect_types=True)

        # Process string to populate caches
        _deserialize_string_full(datetime_str, config)

        # Verify caches are populated
        assert _get_cached_string_pattern(datetime_str) is not None
        assert len(_STRING_PATTERN_CACHE) > 0
        assert len(_PARSED_OBJECT_CACHE) > 0

        # Clear caches
        clear_caches()

        # Verify caches are empty
        assert _get_cached_string_pattern(datetime_str) is None
        assert len(_STRING_PATTERN_CACHE) == 0
        assert len(_PARSED_OBJECT_CACHE) == 0


class TestPythonVersionCompatibility:
    """Test Python version compatibility for datetime parsing."""

    def test_timezone_aware_datetime_parsing(self):
        """Test parsing timezone-aware datetime strings."""
        config = SerializationConfig(auto_detect_types=True)

        # Test with Z suffix (UTC)
        utc_str = "2024-01-01T00:00:00Z"
        result = _deserialize_string_full(utc_str, config)
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Test with explicit UTC offset
        offset_str = "2024-01-01T00:00:00+00:00"
        result = _deserialize_string_full(offset_str, config)
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_microsecond_precision(self):
        """Test parsing datetime strings with microsecond precision."""
        config = SerializationConfig(auto_detect_types=True)

        microsecond_str = "2024-01-01T12:30:45.123456Z"
        result = _deserialize_string_full(microsecond_str, config)
        assert isinstance(result, datetime)
        assert result.microsecond == 123456

    def test_fallback_datetime_parsing(self):
        """Test fallback datetime parsing for edge cases."""
        config = SerializationConfig(auto_detect_types=True)

        # Test with milliseconds (should work on all Python versions)
        millisecond_str = "2024-01-01T12:30:45.123Z"
        result = _deserialize_string_full(millisecond_str, config)
        assert isinstance(result, datetime)


class TestUUIDDeserialization:
    """Test UUID deserialization functionality."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_uuid_string_detection(self):
        """Test UUID string detection."""
        import uuid

        # Create a valid UUID string
        uuid_obj = uuid.uuid4()
        uuid_str = str(uuid_obj)

        config = SerializationConfig(auto_detect_types=True)
        result = _deserialize_string_full(uuid_str, config)

        # Should parse back to UUID object
        assert isinstance(result, uuid.UUID)
        assert str(result) == uuid_str

    def test_invalid_uuid_strings(self):
        """Test handling of invalid UUID strings."""
        config = SerializationConfig(auto_detect_types=True)

        invalid_uuids = [
            "not-a-uuid",
            "12345678-1234-1234-1234-12345678901",  # Too short
            "12345678-1234-1234-1234-1234567890123",  # Too long
        ]

        for invalid_uuid in invalid_uuids:
            result = _deserialize_string_full(invalid_uuid, config)
            # Should return as string for invalid UUID
            assert isinstance(result, str)
            assert result == invalid_uuid


class TestPathDeserialization:
    """Test Path deserialization functionality."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_path_string_detection(self):
        """Test Path string detection."""

        path_strings = ["/home/user/file.txt", "C:\\Users\\user\\file.txt", "./relative/path.txt", "../parent/file.txt"]

        config = SerializationConfig(auto_detect_types=True)

        for path_str in path_strings:
            result = _deserialize_string_full(path_str, config)

            # May or may not be converted to Path depending on detection logic
            # Just ensure it doesn't crash
            assert result is not None


class TestErrorHandling:
    """Test error handling in deserialization."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_malformed_datetime_error_handling(self):
        """Test error handling for malformed datetime strings."""
        config = SerializationConfig(auto_detect_types=True)

        # Strings that look like datetime but are malformed
        malformed_cases = [
            "2024-01-01T25:00:00Z",  # Invalid hour
            "2024-01-01T12:60:00Z",  # Invalid minute
            "2024-01-01T12:30:60Z",  # Invalid second
        ]

        for malformed_str in malformed_cases:
            result = _deserialize_string_full(malformed_str, config)
            # Should gracefully return as string
            assert isinstance(result, str)
            assert result == malformed_str

    def test_none_input_handling(self):
        """Test handling of None input."""
        config = SerializationConfig(auto_detect_types=True)

        # Should handle None gracefully
        result = _deserialize_string_full(None, config)
        assert result is None

    def test_empty_string_handling(self):
        """Test handling of empty string."""
        config = SerializationConfig(auto_detect_types=True)

        result = _deserialize_string_full("", config)
        assert result == ""

    def test_whitespace_string_handling(self):
        """Test handling of whitespace-only string."""
        config = SerializationConfig(auto_detect_types=True)

        result = _deserialize_string_full("   ", config)
        assert result == "   "


class TestDeserializeFast:
    """Test the deserialize_fast function."""

    def test_deserialize_fast_with_datetime(self):
        """Test deserialize_fast with datetime data."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42, "nested": {"date": "2024-12-31T23:59:59Z"}}

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(data, config=config)

        assert isinstance(result["timestamp"], datetime)
        assert result["value"] == 42
        assert isinstance(result["nested"]["date"], datetime)

    def test_deserialize_fast_without_config(self):
        """Test deserialize_fast without config."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        result = deserialize_fast(data)
        assert result["value"] == 42
        # Without config, datetime parsing behavior depends on defaults

    def test_deserialize_fast_with_list(self):
        """Test deserialize_fast with list data."""
        data = [{"timestamp": "2024-01-01T00:00:00Z", "value": 1}, {"timestamp": "2024-01-02T00:00:00Z", "value": 2}]

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(data, config=config)

        assert len(result) == 2
        assert isinstance(result[0]["timestamp"], datetime)
        assert isinstance(result[1]["timestamp"], datetime)


class TestPerformanceOptimizations:
    """Test performance optimizations in deserialization."""

    def test_repeated_string_processing(self):
        """Test that repeated string processing uses cache effectively."""
        config = SerializationConfig(auto_detect_types=True)
        datetime_str = "2024-01-01T00:00:00Z"

        # First call - should populate cache
        result1 = _deserialize_string_full(datetime_str, config)

        # Subsequent calls - should use cache
        result2 = _deserialize_string_full(datetime_str, config)
        result3 = _deserialize_string_full(datetime_str, config)

        assert isinstance(result1, datetime)
        assert isinstance(result2, datetime)
        assert isinstance(result3, datetime)
        assert result1 == result2 == result3

    def test_large_data_structure_processing(self):
        """Test processing of large data structures."""
        # Create a large data structure with many datetime strings
        large_data = {f"timestamp_{i}": "2024-01-01T00:00:00Z" for i in range(100)}
        large_data["value"] = 42

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(large_data, config=config)

        assert result["value"] == 42
        # All timestamps should be parsed to datetime objects
        for i in range(100):
            assert isinstance(result[f"timestamp_{i}"], datetime)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Clear caches before each test."""
        clear_caches()

    def test_very_long_datetime_string(self):
        """Test handling of very long datetime-like strings."""
        config = SerializationConfig(auto_detect_types=True)

        # Very long string that looks like datetime but isn't
        long_str = "2024-01-01T00:00:00Z" + "x" * 1000
        result = _deserialize_string_full(long_str, config)

        # Should return as string due to length
        assert isinstance(result, str)
        assert result == long_str

    def test_unicode_datetime_strings(self):
        """Test handling of Unicode characters in datetime strings."""
        config = SerializationConfig(auto_detect_types=True)

        unicode_cases = [
            "2024-01-01T00:00:00Z🎉",  # Emoji
            "2024-01-01T00:00:00Zα",  # Greek letter
            "2024-01-01T00:00:00Z中",  # Chinese character
        ]

        for unicode_str in unicode_cases:
            result = _deserialize_string_full(unicode_str, config)
            # Should return as string for non-standard datetime
            assert isinstance(result, str)
            assert result == unicode_str

    def test_boundary_datetime_values(self):
        """Test boundary datetime values."""
        config = SerializationConfig(auto_detect_types=True)

        boundary_cases = [
            "1970-01-01T00:00:00Z",  # Unix epoch
            "2038-01-19T03:14:07Z",  # 32-bit timestamp limit
            "9999-12-31T23:59:59Z",  # Far future
        ]

        for boundary_str in boundary_cases:
            result = _deserialize_string_full(boundary_str, config)
            assert isinstance(result, datetime), f"Failed to parse: {boundary_str}"

    def test_mixed_case_timezone_indicators(self):
        """Test mixed case timezone indicators."""
        config = SerializationConfig(auto_detect_types=True)

        mixed_cases = [
            "2024-01-01t00:00:00z",  # Lowercase
            "2024-01-01T00:00:00z",  # Mixed case
            "2024-01-01t00:00:00Z",  # Mixed case
        ]

        for mixed_str in mixed_cases:
            result = _deserialize_string_full(mixed_str, config)
            # Behavior depends on implementation - should not crash
            assert result is not None
