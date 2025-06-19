"""Test coverage boost for datason API module.

Tests specific API features and edge cases to improve test coverage.
Focus on testing methods with proper parameters and lazy imports.
"""

import tempfile

import pytest

import datason.api as api_module
from datason.api import deserialize_modern, get_api_info, help_api, serialize_modern, suppress_deprecation_warnings


class TestAPIFeatures:
    """Test API utility functions and features."""

    def test_suppress_deprecation_warnings(self):
        """Test deprecation warnings suppression."""
        # Test enabling suppression
        suppress_deprecation_warnings(True)
        # Test disabling suppression
        suppress_deprecation_warnings(False)
        # Test default behavior
        suppress_deprecation_warnings()

    def test_help_api(self):
        """Test help_api() function."""
        result = help_api()
        assert isinstance(result, dict)
        assert "serialization" in result
        assert "deserialization" in result
        assert "recommendations" in result

    def test_get_api_info(self):
        """Test get_api_info() function."""
        result = get_api_info()
        assert isinstance(result, dict)
        assert "api_version" in result
        assert "dump_functions" in result
        assert "features" in result

    def test_serialize_modern(self):
        """Test serialize_modern() function."""
        test_data = {"value": 42}
        result = serialize_modern(test_data)
        assert isinstance(result, dict)

    def test_deserialize_modern(self):
        """Test deserialize_modern() function."""
        test_data = {"value": 42}
        # First serialize, then deserialize
        serialized = serialize_modern(test_data)
        result = deserialize_modern(serialized)
        assert result == test_data

    def test_serialize_with_modes(self):
        """Test serialize() function with different modes."""
        test_data = {"value": 42, "name": "test"}

        # Test ML mode
        result_ml = api_module.serialize(test_data, ml_mode=True)
        assert isinstance(result_ml, dict)

        # Test API mode
        result_api = api_module.serialize(test_data, api_mode=True)
        assert isinstance(result_api, dict)

        # Test secure mode
        result_secure = api_module.serialize(test_data, secure=True)
        assert isinstance(result_secure, dict)

        # Test chunked mode - returns ChunkedSerializationResult
        result_chunked = api_module.serialize(test_data, chunked=True, chunk_size=100)
        # ChunkedSerializationResult has chunks and metadata
        assert hasattr(result_chunked, "chunks")
        assert hasattr(result_chunked, "metadata")

        # Test fast mode
        result_fast = api_module.serialize(test_data, fast_mode=True)
        assert isinstance(result_fast, dict)

    def test_dump_with_file_path(self):
        """Test dump() function with file path instead of file object."""
        test_data = {"value": 42}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            file_path = f.name

        # Test dump with file path (should use save_ml)
        api_module.dump(test_data, file_path)

        # Verify file was created
        import os

        assert os.path.exists(file_path)

        # Clean up
        os.unlink(file_path)

    def test_file_operations_with_compression(self):
        """Test file operations with compressed files."""
        # Test .gz file detection - defaults to jsonl unless .json extension detected
        assert api_module._detect_file_format("file.json.gz") == "json"
        assert api_module._detect_file_format("file.jsonl.gz") == "jsonl"
        # .pkl.gz defaults to jsonl (not pickle format supported)
        assert api_module._detect_file_format("file.pkl.gz") == "jsonl"

    def test_error_conditions(self):
        """Test error handling in API functions."""
        # Test with None data
        result = api_module.serialize(None)
        assert result is None

        # Test empty data
        result = api_module.serialize({})
        assert result == {}

        # Test with invalid file format detection
        result = api_module._detect_file_format("unknown.xyz")
        # Should return default or handle gracefully
        assert isinstance(result, str)


class TestConfigurationMethods:
    """Test configuration-related API methods."""

    def test_lazy_imports(self):
        """Test that imports work correctly when called."""
        # These should work without raising import errors
        try:
            from datason.config import SerializationConfig

            config = SerializationConfig()
            assert config is not None
        except ImportError:
            pytest.skip("Config module not available")

    def test_api_with_config(self):
        """Test API methods with configuration objects."""
        try:
            from datason.config import SerializationConfig

            config = SerializationConfig(auto_detect_types=True)
            test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

            # Test serialize with config
            result = api_module.serialize(test_data, config=config)
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("Config module not available")

    def test_file_format_detection_edge_cases(self):
        """Test file format detection with various edge cases."""
        # Test different extensions - note .pkl defaults to jsonl, not pickle
        assert api_module._detect_file_format("data.json") == "json"
        assert api_module._detect_file_format("data.jsonl") == "jsonl"
        assert api_module._detect_file_format("data.pkl") == "jsonl"  # Defaults to jsonl

        # Test with paths
        assert api_module._detect_file_format("/path/to/data.json") == "json"
        assert api_module._detect_file_format("./relative/data.jsonl") == "jsonl"

        # Test case sensitive - uppercase defaults to jsonl
        assert api_module._detect_file_format("DATA.JSON") == "jsonl"  # Case sensitive, defaults to jsonl

    def test_get_version_and_info(self):
        """Test get_version and get_info functions from datason.__init__."""
        import datason

        # Test get_version function
        version = datason.get_version()
        assert isinstance(version, str)
        assert len(version) > 0

        # Test get_info function
        info = datason.get_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "author" in info
        assert "email" in info
        assert "description" in info
        assert "config_available" in info
        assert "cache_system" in info

        # Verify the version matches
        assert info["version"] == version

        # Verify config_available is boolean
        assert isinstance(info["config_available"], bool)

        # Verify cache_system is a string
        assert isinstance(info["cache_system"], str)
        assert info["cache_system"] in ["configurable", "legacy"]
