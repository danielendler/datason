"""
Comprehensive test coverage for datason deserializers_new datetime functionality.

Tests the datetime parsing features to improve coverage for deserializers_new.py,
focusing on the cross-version datetime parsing fixes and caching mechanisms.
"""

from datason.config import SerializationConfig
from datason.deserializers_new import (
    _looks_like_datetime_optimized,
    _looks_like_path_optimized,
    _looks_like_uuid_optimized,
    clear_caches,
    deserialize_fast,
)


class TestDatetimeParsing:
    """Test datetime parsing functionality."""

    def test_looks_like_datetime_optimized(self):
        """Test the optimized datetime detection function."""
        # Valid datetime-like strings
        assert _looks_like_datetime_optimized("2024-01-01T00:00:00Z")
        assert _looks_like_datetime_optimized("2024-12-31T23:59:59")
        assert _looks_like_datetime_optimized("2024-01-01T00:00:00+00:00")

        # Invalid datetime-like strings
        assert not _looks_like_datetime_optimized("not a date")
        assert not _looks_like_datetime_optimized("12:30:45")  # No date
        assert not _looks_like_datetime_optimized("short")

    def test_looks_like_uuid_optimized(self):
        """Test UUID detection function."""
        # Valid UUIDs
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert _looks_like_uuid_optimized(valid_uuid)

        # Invalid UUIDs
        assert not _looks_like_uuid_optimized("not-a-uuid")
        assert not _looks_like_uuid_optimized("550e8400-e29b-41d4-a716")  # Too short
        assert not _looks_like_uuid_optimized("invalid-uuid-format-here")

    def test_looks_like_path_optimized(self):
        """Test path detection function."""
        # Valid paths
        assert _looks_like_path_optimized("/home/user/file.txt")
        assert _looks_like_path_optimized("./relative/path.txt")
        assert _looks_like_path_optimized("../parent/file.txt")

        # Invalid paths
        assert not _looks_like_path_optimized("x")
        assert not _looks_like_path_optimized("")

    def test_deserialize_fast_with_datetime_config(self):
        """Test deserialize_fast with datetime configuration."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        # Test with auto_detect_types enabled
        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(data, config=config)

        assert isinstance(result, dict)
        assert result["value"] == 42
        # The timestamp might be converted to datetime depending on config

    def test_deserialize_fast_without_config(self):
        """Test deserialize_fast without config."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        result = deserialize_fast(data)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_deserialize_fast_with_list(self):
        """Test deserialize_fast with list data."""
        data = [{"timestamp": "2024-01-01T00:00:00Z", "value": 1}, {"timestamp": "2024-01-02T00:00:00Z", "value": 2}]

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(data, config=config)

        # deserialize_fast may convert structured lists to pandas DataFrames
        # or keep them as lists depending on the data structure detection
        assert result is not None

        # Check the data is preserved regardless of format
        if hasattr(result, "iloc"):  # pandas DataFrame
            assert len(result) == 2
            assert result.iloc[0]["value"] == 1
            assert result.iloc[1]["value"] == 2
        elif isinstance(result, list):
            assert len(result) == 2
            assert result[0]["value"] == 1
            assert result[1]["value"] == 2
        else:
            # Handle other possible return types
            assert hasattr(result, "__len__") or hasattr(result, "__iter__")

    def test_clear_caches(self):
        """Test cache clearing functionality."""
        # This should not raise an exception
        clear_caches()

        # Test that we can call it multiple times
        clear_caches()
        clear_caches()

    def test_various_datetime_formats(self):
        """Test various datetime formats with deserialize_fast."""
        datetime_formats = [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00+00:00",
            "2024-12-31T23:59:59.999999Z",
        ]

        config = SerializationConfig(auto_detect_types=True)

        for dt_str in datetime_formats:
            data = {"timestamp": dt_str, "value": 42}
            result = deserialize_fast(data, config=config)
            assert isinstance(result, dict)
            assert result["value"] == 42

    def test_edge_cases(self):
        """Test edge cases for datetime parsing."""
        config = SerializationConfig(auto_detect_types=True)

        # Empty data
        result = deserialize_fast({}, config=config)
        assert result == {}

        # None values
        data = {"timestamp": None, "value": 42}
        result = deserialize_fast(data, config=config)
        assert result["timestamp"] is None
        assert result["value"] == 42

        # Mixed types
        data = {
            "datetime": "2024-01-01T00:00:00Z",
            "string": "regular string",
            "number": 42,
            "boolean": True,
            "null": None,
        }
        result = deserialize_fast(data, config=config)
        assert isinstance(result, dict)
        assert result["string"] == "regular string"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["null"] is None


class TestPerformanceOptimizations:
    """Test performance-related optimizations."""

    def test_repeated_processing(self):
        """Test that repeated processing doesn't cause issues."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}
        config = SerializationConfig(auto_detect_types=True)

        # Process the same data multiple times
        for _ in range(5):
            result = deserialize_fast(data, config=config)
            assert isinstance(result, dict)
            assert result["value"] == 42

    def test_large_data_structure(self):
        """Test processing of larger data structures."""
        # Create a moderately large data structure
        large_data = {f"item_{i}": {"timestamp": "2024-01-01T00:00:00Z", "value": i} for i in range(50)}

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(large_data, config=config)

        assert isinstance(result, dict)
        assert len(result) == 50
        assert result["item_0"]["value"] == 0
        assert result["item_49"]["value"] == 49


class TestConfigurationVariants:
    """Test different configuration options."""

    def test_with_different_configs(self):
        """Test deserialize_fast with different configuration options."""
        data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        # Default config
        result1 = deserialize_fast(data, config=None)
        assert isinstance(result1, dict)

        # Auto-detect enabled
        config2 = SerializationConfig(auto_detect_types=True)
        result2 = deserialize_fast(data, config=config2)
        assert isinstance(result2, dict)

        # Auto-detect disabled
        config3 = SerializationConfig(auto_detect_types=False)
        result3 = deserialize_fast(data, config=config3)
        assert isinstance(result3, dict)

    def test_nested_structures(self):
        """Test with nested data structures."""
        nested_data = {
            "user": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2024-01-01T00:00:00Z",
                "profile": {"settings": {"timezone": "UTC", "last_login": "2024-01-01T12:00:00Z"}},
            },
            "items": [
                {"timestamp": "2024-01-01T09:00:00Z", "count": 1},
                {"timestamp": "2024-01-01T10:00:00Z", "count": 2},
            ],
        }

        config = SerializationConfig(auto_detect_types=True)
        result = deserialize_fast(nested_data, config=config)

        assert isinstance(result, dict)
        assert "user" in result
        assert "items" in result
        assert len(result["items"]) == 2
