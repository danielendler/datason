"""
Focused test coverage to boost datason/api.py to 80% coverage.

This module targets specific missing coverage areas in the API.
"""

import tempfile

import pytest

import datason.api as api_module
from datason.api import deserialize_modern, get_api_info, help_api, serialize_modern, suppress_deprecation_warnings


class TestAPIFeatures:
    """Test API features for coverage boost."""

    def test_suppress_deprecation_warnings(self):
        """Test suppress_deprecation_warnings function."""
        # Test enabling suppression
        suppress_deprecation_warnings(True)

        # Test disabling suppression
        suppress_deprecation_warnings(False)

        # Test default behavior
        suppress_deprecation_warnings()

    def test_help_api(self):
        """Test help_api function."""
        result = help_api()
        assert isinstance(result, dict)
        assert "serialization" in result or "examples" in result

    def test_get_api_info(self):
        """Test get_api_info function."""
        result = get_api_info()
        assert isinstance(result, dict)
        assert "api_version" in result or "dump_functions" in result

    def test_serialize_modern(self):
        """Test serialize_modern function."""
        test_data = {"value": 42, "name": "test"}

        result = serialize_modern(test_data)
        assert result is not None

    def test_deserialize_modern(self):
        """Test deserialize_modern function."""
        test_data = {"value": 42, "name": "test"}

        result = deserialize_modern(test_data)
        assert result is not None

    def test_dump_with_kwargs(self):
        """Test dump with various kwargs."""
        test_data = {"value": 42}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            # Test with DataSON-specific kwargs
            api_module.dump(test_data, f, ml_mode=True)

            f.seek(0)
            content = f.read()
            assert "value" in content

    def test_load_basic_functionality(self):
        """Test load_basic function."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        result = api_module.load_basic(test_data)
        assert result["value"] == 42

    def test_load_typed_functionality(self):
        """Test load_typed function."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        result = api_module.load_typed(test_data)
        assert result["value"] == 42

    def test_load_perfect_functionality(self):
        """Test load_perfect function."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}
        template = {"timestamp": "", "value": 0}

        result = api_module.load_perfect(test_data, template)
        assert result["value"] == 42

    def test_file_operations_with_compression(self):
        """Test file operations with compressed files."""
        # Test .gz file detection
        assert api_module._detect_file_format("file.json.gz") == "json"
        assert api_module._detect_file_format("file.jsonl.gz") == "jsonl"
        assert api_module._detect_file_format("file.pkl.gz") == "pickle"

    def test_error_conditions(self):
        """Test various error conditions for coverage."""
        # Test invalid serialize modes
        with pytest.raises(ValueError):
            api_module.serialize({"test": 1}, ml_mode=True, api_mode=True, fast_mode=True)

    def test_edge_case_file_formats(self):
        """Test edge case file format detection."""
        # Test various extensions
        assert api_module._detect_file_format("test.jsonl.gz") == "jsonl"
        assert api_module._detect_file_format("test.unknown") in ["json", "jsonl"]
