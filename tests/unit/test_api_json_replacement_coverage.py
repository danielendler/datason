"""
Comprehensive test coverage for JSON drop-in replacement functionality and API features.

This module focuses on testing the JSON compatibility APIs and covering missing
functionality in datason/api.py to reach 80% coverage goal.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import datason
import datason.json
from datason.api import (
    _detect_file_format,
    _load_from_file,
    _save_to_file,
    dump,
    dump_api,
    dump_chunked,
    dump_fast,
    dump_json,
    dump_ml,
    dump_secure,
    dumps,
    load,
    load_basic_file,
    load_json,
    load_perfect_file,
    load_smart_file,
    loads,
    loads_json,
    save_api,
    save_chunked,
    save_ml,
    save_secure,
    serialize,
    stream_dump,
    stream_save_ml,
)


class TestJSONDropInReplacement:
    """Test the perfect JSON module drop-in replacement functionality."""

    def test_datason_json_loads_compatibility(self):
        """Test datason.json.loads() has exact same behavior as json.loads()."""
        test_data = '{"timestamp": "2024-01-01T00:00:00Z", "value": 42, "null": null}'

        # Standard json behavior
        json_result = json.loads(test_data)

        # DataSON compatibility behavior
        datason_result = datason.json.loads(test_data)

        # Should be identical
        assert json_result == datason_result
        assert isinstance(json_result["timestamp"], str) and isinstance(datason_result["timestamp"], str)
        assert json_result["timestamp"] == "2024-01-01T00:00:00Z"

    def test_datason_json_dumps_compatibility(self):
        """Test datason.json.dumps() has exact same behavior as json.dumps()."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42, "null": None}

        # Standard json behavior
        json_result = json.dumps(test_data)

        # DataSON compatibility behavior
        datason_result = datason.json.dumps(test_data)

        # Should be identical
        assert json_result == datason_result
        assert isinstance(json_result, str)
        assert isinstance(datason_result, str)

    def test_enhanced_loads_vs_compatibility(self):
        """Test enhanced loads() vs compatibility loads_json()."""
        test_data = '{"timestamp": "2024-01-01T00:00:00Z", "value": 42}'

        # Enhanced mode - should parse datetime
        enhanced_result = datason.loads(test_data)
        assert isinstance(enhanced_result["timestamp"], datetime)

        # Compatibility mode - should keep as string
        compat_result = loads_json(test_data)
        assert isinstance(compat_result["timestamp"], str)
        assert compat_result["timestamp"] == "2024-01-01T00:00:00Z"


class TestFileOperations:
    """Test file operation functions to improve API coverage."""

    def test_dump_with_file_object(self):
        """Test dump() function with file-like object."""
        test_data = {"timestamp": datetime.now(), "value": 42}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            # Test with file object
            dump(test_data, f)

            # Read back and verify
            f.seek(0)
            content = f.read()
            assert "timestamp" in content
            assert "value" in content

    def test_dump_with_file_path(self):
        """Test dump() function with file path (uses save_ml)."""
        test_data = {"timestamp": datetime.now(), "value": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        # Test with file path
        dump(test_data, file_path)

        # Verify file exists and has content
        assert Path(file_path).exists()
        with open(file_path) as f:
            content = f.read()
            assert len(content) > 0

    def test_dump_json_with_all_parameters(self):
        """Test dump_json() with all JSON parameters."""
        test_data = {"b": 2, "a": 1, "value": 42}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            dump_json(
                test_data,
                f,
                indent=2,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                skipkeys=False,
                check_circular=True,
                allow_nan=True,
            )

            f.seek(0)
            content = f.read()
            assert '"a":' in content or '"a": ' in content  # Depends on separators
            assert '"b":' in content or '"b": ' in content
            assert '"value":' in content or '"value": ' in content

    def test_load_with_file_object(self):
        """Test load() function with file-like object."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        # Create file with test data
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(test_data, f)
            f.seek(0)

            # Test load with file object
            result = load(f)
            assert result["value"] == 42
            # Should have enhanced datetime parsing
            assert isinstance(result["timestamp"], datetime)

    def test_load_with_file_path(self):
        """Test load() function with file path."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        # Test load with file path
        result = load(file_path)
        assert result["value"] == 42

    def test_load_json_compatibility(self):
        """Test load_json() for stdlib compatibility."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(test_data, f)
            f.seek(0)

            # Test load_json - should maintain string
            result = load_json(f)
            assert result["value"] == 42
            assert isinstance(result["timestamp"], str)

    def test_serialize_mode_conflicts(self):
        """Test serialize() with conflicting modes raises error."""
        test_data = {"value": 42}

        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            serialize(test_data, ml_mode=True, api_mode=True)

    def test_serialize_with_secure_mode(self):
        """Test serialize() with secure=True."""
        test_data = {"email": "user@example.com", "value": 42}

        result = serialize(test_data, secure=True)
        assert isinstance(result, dict)
        # Secure mode may redact data and wrap in structure
        assert "data" in result or "email" in result or "value" in result

    def test_dump_secure_with_pii(self):
        """Test dump_secure() with PII data."""
        test_data = {"name": "John Doe", "email": "john@example.com", "ssn": "123-45-6789", "value": 42}

        result = dump_secure(test_data, redact_pii=True)
        assert isinstance(result, dict)
        # Secure mode may redact data and wrap in structure
        assert "data" in result or "name" in result or "value" in result or "email" in result

    def test_detect_file_format_json(self):
        """Test _detect_file_format() with .json extension."""
        result = _detect_file_format("test.json")
        assert result == "json"

    def test_detect_file_format_jsonl(self):
        """Test _detect_file_format() with .jsonl extension."""
        result = _detect_file_format("test.jsonl")
        assert result == "jsonl"


class TestSerializationModes:
    """Test different serialization modes to improve coverage."""

    def test_serialize_with_ml_mode(self):
        """Test serialize() with ml_mode."""
        test_data = {"value": 42, "array": [1, 2, 3]}

        result = serialize(test_data, ml_mode=True)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_serialize_with_api_mode(self):
        """Test serialize() with api_mode."""
        test_data = {"value": 42, "timestamp": datetime.now()}

        result = serialize(test_data, api_mode=True)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_serialize_with_fast_mode(self):
        """Test serialize() with fast_mode."""
        test_data = {"value": 42, "data": [1, 2, 3]}

        result = serialize(test_data, fast_mode=True)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_serialize_with_chunked_mode(self):
        """Test serialize() with chunked=True."""
        test_data = {"value": 42, "large_data": list(range(100))}

        result = serialize(test_data, chunked=True, chunk_size=50)
        # Chunked mode may return special result object
        assert result is not None


class TestSpecializationFunctions:
    """Test specialized dump/load functions."""

    def test_dump_ml(self):
        """Test dump_ml() function."""
        test_data = {"model": "test_model", "weights": [1.0, 2.0, 3.0]}

        result = dump_ml(test_data)
        assert isinstance(result, dict)
        assert result["model"] == "test_model"

    def test_dump_api(self):
        """Test dump_api() function."""
        test_data = {"timestamp": datetime.now(), "value": 42}

        result = dump_api(test_data)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_dump_secure_with_custom_patterns(self):
        """Test dump_secure() with custom redaction patterns."""
        test_data = {"secret": "password123", "value": 42}

        result = dump_secure(test_data, redact_patterns=[r"password\d+"], redact_fields=["secret"])
        assert isinstance(result, dict)
        # Secure mode may redact data and wrap in structure
        assert "data" in result or "secret" in result or "value" in result

    def test_dump_fast(self):
        """Test dump_fast() function."""
        test_data = {"value": 42, "data": [1, 2, 3]}

        result = dump_fast(test_data)
        assert isinstance(result, dict)
        assert result["value"] == 42

    def test_dump_chunked(self):
        """Test dump_chunked() function."""
        test_data = {"value": 42, "large_list": list(range(1000))}

        result = dump_chunked(test_data, chunk_size=100)
        # Chunked mode may return special result object
        assert result is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_loads_with_invalid_json(self):
        """Test loads() with invalid JSON."""
        invalid_json = '{"incomplete": json'

        with pytest.raises(json.JSONDecodeError):
            loads(invalid_json)

    def test_loads_json_with_invalid_json(self):
        """Test loads_json() with invalid JSON."""
        invalid_json = '{"incomplete": json'

        with pytest.raises(json.JSONDecodeError):
            loads_json(invalid_json)

    def test_dump_with_invalid_file_object(self):
        """Test dump() error handling with invalid file object."""
        test_data = {"value": 42}

        # Test with object that doesn't have write method and isn't a file path
        fake_file = MagicMock()
        fake_file.__str__ = MagicMock(return_value="not_a_real_path")
        del fake_file.write  # Remove write method

        # Should raise an error when trying to use as file path
        with pytest.raises((AttributeError, FileNotFoundError)):
            dump(test_data, fake_file)

    def test_serialize_with_invalid_config_combination(self):
        """Test serialize() with invalid configuration."""
        test_data = {"value": 42}

        # Test with multiple conflicting modes
        with pytest.raises(ValueError):
            serialize(test_data, ml_mode=True, api_mode=True, fast_mode=True)


class TestFileFormatDetection:
    """Test file format detection functionality."""

    def test_detect_file_format_gz(self):
        """Test _detect_file_format() with .gz extension."""
        result = _detect_file_format("test.json.gz")
        assert result == "json"

        result = _detect_file_format("test.jsonl.gz")
        assert result == "jsonl"

    def test_detect_file_format_explicit(self):
        """Test _detect_file_format() with explicit format."""
        result = _detect_file_format("test.txt", format="json")
        assert result == "json"

    def test_detect_file_format_default(self):
        """Test _detect_file_format() with unknown extension."""
        result = _detect_file_format("test.txt")
        assert result in ["json", "jsonl"]  # Default may vary


class TestStreamingOperations:
    """Test streaming operations for coverage."""

    def test_stream_dump(self):
        """Test stream_dump() function."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        # Test stream_dump
        stream_context = stream_dump(file_path)
        assert stream_context is not None

    def test_stream_save_ml(self):
        """Test stream_save_ml() function."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        # Test stream_save_ml
        stream_context = stream_save_ml(file_path)
        assert stream_context is not None


class TestAdvancedFileOperations:
    """Test advanced file operations for coverage."""

    def test_save_to_file_json(self):
        """Test _save_to_file() with JSON format."""
        test_data = {"value": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        _save_to_file(test_data, file_path, format="json")

        # Verify file was created
        assert Path(file_path).exists()

    def test_save_to_file_jsonl(self):
        """Test _save_to_file() with JSONL format."""
        test_data = [{"value": 42}, {"value": 43}]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            file_path = f.name

        _save_to_file(test_data, file_path, format="jsonl")

        # Verify file was created
        assert Path(file_path).exists()

    def test_load_from_file_json(self):
        """Test _load_from_file() with JSON format."""
        test_data = {"value": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        results = list(_load_from_file(file_path, format="json"))
        assert len(results) == 1
        assert results[0]["value"] == 42

    def test_load_from_file_jsonl(self):
        """Test _load_from_file() with JSONL format."""
        test_data = [{"value": 42}, {"value": 43}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            file_path = f.name

        results = list(_load_from_file(file_path, format="jsonl"))
        assert len(results) == 2
        assert results[0]["value"] == 42
        assert results[1]["value"] == 43


class TestComprehensiveFileAPI:
    """Test comprehensive file API functions."""

    def test_save_ml_with_format(self):
        """Test save_ml() with explicit format."""
        test_data = {"model": "test", "weights": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        save_ml(test_data, file_path, format="json")
        assert Path(file_path).exists()

    def test_save_api_with_format(self):
        """Test save_api() with explicit format."""
        test_data = {"timestamp": datetime.now(), "value": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        save_api(test_data, file_path, format="json")
        assert Path(file_path).exists()

    def test_save_secure_with_format(self):
        """Test save_secure() with explicit format."""
        test_data = {"email": "user@example.com", "value": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        save_secure(test_data, file_path, format="json", redact_pii=True)
        assert Path(file_path).exists()

    def test_save_chunked_with_format(self):
        """Test save_chunked() with explicit format."""
        test_data = {"large_data": list(range(100)), "value": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = f.name

        save_chunked(test_data, file_path, chunk_size=50, format="json")
        assert Path(file_path).exists()

    def test_load_smart_file_with_format(self):
        """Test load_smart_file() with explicit format."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        results = list(load_smart_file(file_path, format="json"))
        assert len(results) == 1
        assert results[0]["value"] == 42

    def test_load_perfect_file_with_template(self):
        """Test load_perfect_file() with template."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}
        template = {"timestamp": datetime.now(), "value": 0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        results = list(load_perfect_file(file_path, template, format="json"))
        assert len(results) == 1
        assert results[0]["value"] == 42

    def test_load_basic_file_with_format(self):
        """Test load_basic_file() with explicit format."""
        test_data = {"value": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        results = list(load_basic_file(file_path, format="json"))
        assert len(results) == 1
        assert results[0]["value"] == 42


class TestCrossVersionCompatibility:
    """Test cross-version datetime parsing functionality."""

    def test_datetime_parsing_iso_format(self):
        """Test datetime parsing with various ISO formats."""
        test_cases = [
            '{"timestamp": "2024-01-01T00:00:00Z"}',
            '{"timestamp": "2024-01-01T12:30:45.123Z"}',
            '{"timestamp": "2024-01-01T00:00:00+00:00"}',
            '{"timestamp": "2024-01-01T00:00:00-05:00"}',
        ]

        for test_case in test_cases:
            result = loads(test_case)
            assert isinstance(result["timestamp"], datetime)

    def test_datetime_parsing_with_config(self):
        """Test datetime parsing with explicit config."""
        test_data = '{"timestamp": "2024-01-01T00:00:00Z", "value": 42}'

        from datason.config import SerializationConfig

        config = SerializationConfig(auto_detect_types=True)

        result = datason.api.load_smart(json.loads(test_data), config=config)
        assert isinstance(result["timestamp"], datetime)
        assert result["value"] == 42

    def test_mixed_datetime_formats(self):
        """Test handling of mixed datetime formats."""
        test_data = {
            "iso_datetime": "2024-01-01T00:00:00Z",
            "timestamp": 1640995200,  # Unix timestamp
            "regular_string": "not a date",
            "value": 42,
        }

        result = dumps(test_data)
        assert isinstance(result, dict)

        # When loading back with enhanced mode
        json_str = json.dumps(test_data)
        restored = loads(json_str)

        # ISO datetime should be parsed
        assert isinstance(restored["iso_datetime"], datetime)
        # Other values should remain as-is
        assert restored["timestamp"] == 1640995200
        assert restored["regular_string"] == "not a date"
        assert restored["value"] == 42
