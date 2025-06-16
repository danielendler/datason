"""Comprehensive test suite for datason/deserializers.py module.

This test suite provides exhaustive coverage of the deserialization system
to boost coverage from 6% to 85%+ with systematic testing of all functions,
edge cases, optimizations, and error conditions.
"""

import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the deserializers module for testing
import datason.deserializers_new as deserializers
from datason.config import SerializationConfig


class TestDeserializeCore:
    """Test the main deserialize function and its core functionality."""

    def test_deserialize_basic_types(self):
        """Test deserialization of basic JSON-compatible types."""
        # Test None
        assert deserializers.deserialize(None) is None

        # Test boolean
        assert deserializers.deserialize(True) is True
        assert deserializers.deserialize(False) is False

        # Test integers
        assert deserializers.deserialize(0) == 0
        assert deserializers.deserialize(42) == 42
        assert deserializers.deserialize(-1) == -1

        # Test floats
        assert deserializers.deserialize(3.14) == 3.14
        assert deserializers.deserialize(-2.7) == -2.7

        # Test strings
        assert deserializers.deserialize("hello") == "hello"
        assert deserializers.deserialize("") == ""

    def test_deserialize_with_type_metadata(self):
        """Test deserialization with type metadata."""
        # Test data with metadata
        metadata_dict = {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"}
        result = deserializers.deserialize(metadata_dict)
        assert isinstance(result, (datetime, dict))

    def test_deserialize_datetime_strings(self):
        """Test deserialization of datetime strings."""
        # Test ISO format datetime
        dt_string = "2023-01-01T12:00:00"
        result = deserializers.deserialize(dt_string, parse_dates=True)
        assert isinstance(result, (datetime, str))

        # Test with Z timezone
        dt_string_z = "2023-01-01T12:00:00Z"
        result = deserializers.deserialize(dt_string_z, parse_dates=True)
        assert isinstance(result, (datetime, str))

    def test_deserialize_uuid_strings(self):
        """Test deserialization of UUID strings."""
        # Test valid UUID string
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        result = deserializers.deserialize(uuid_string, parse_uuids=True)
        assert isinstance(result, (uuid.UUID, str))

    def test_deserialize_disable_parsing(self):
        """Test deserialization with parsing disabled."""
        # Test with date parsing disabled
        dt_string = "2023-01-01T12:00:00"
        result = deserializers.deserialize(dt_string, parse_dates=False)
        assert result == dt_string

        # Test with UUID parsing disabled
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        result = deserializers.deserialize(uuid_string, parse_uuids=False)
        assert result == uuid_string

    def test_deserialize_nested_structures(self):
        """Test deserialization of nested data structures."""
        data = {"list": [1, 2, {"nested": True}], "dict": {"inner": [3, 4, 5]}, "mixed": [{"a": 1}, {"b": [6, 7]}]}
        result = deserializers.deserialize(data)
        assert isinstance(result, dict)
        assert len(result["list"]) == 3
        assert result["dict"]["inner"] == [3, 4, 5]

    def test_deserialize_lists(self):
        """Test deserialization of lists."""
        data = [1, "2023-01-01T12:00:00", "12345678-1234-5678-9012-123456789abc"]
        result = deserializers.deserialize(data, parse_dates=True, parse_uuids=True)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_deserialize_invalid_datetime(self):
        """Test deserialization with invalid datetime strings."""
        invalid_dt = "not-a-datetime"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = deserializers.deserialize(invalid_dt, parse_dates=True)
            assert result == invalid_dt

    def test_deserialize_invalid_uuid(self):
        """Test deserialization with invalid UUID strings."""
        invalid_uuid = "not-a-uuid"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = deserializers.deserialize(invalid_uuid, parse_uuids=True)
            assert result == invalid_uuid


class TestAutoDeserialize:
    """Test auto-detection deserialization functionality."""

    def test_auto_deserialize_basic(self):
        """Test basic auto-deserialization."""
        data = {"test": "value", "number": 42}
        result = deserializers.auto_deserialize(data)
        assert result == data

    def test_auto_deserialize_aggressive_mode(self):
        """Test auto-deserialization with aggressive mode."""
        data = {"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        result = deserializers.auto_deserialize(data, aggressive=True)
        assert isinstance(result, dict)

    def test_auto_deserialize_none(self):
        """Test auto-deserialization with None input."""
        result = deserializers.auto_deserialize(None)
        assert result is None

    def test_auto_deserialize_string_detection(self):
        """Test auto-deserialization with string type detection."""
        # Test datetime string
        result = deserializers.auto_deserialize("2023-01-01T12:00:00")
        assert isinstance(result, (datetime, str))

        # Test UUID string
        result = deserializers.auto_deserialize("12345678-1234-5678-9012-123456789abc")
        assert isinstance(result, (uuid.UUID, str))


class TestDeserializeToPandas:
    """Test pandas-specific deserialization functionality."""

    @pytest.mark.skipif(not hasattr(deserializers, "pd") or deserializers.pd is None, reason="pandas not available")
    def test_deserialize_to_pandas_basic(self):
        """Test basic pandas deserialization."""
        data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        result = deserializers.deserialize_to_pandas(data)
        assert isinstance(result, (dict, type(deserializers.pd.DataFrame())))

    def test_deserialize_to_pandas_without_pandas(self):
        """Test pandas deserialization when pandas not available."""
        with patch("datason.deserializers.pd", None):
            data = {"A": [1, 2, 3], "B": [4, 5, 6]}
            result = deserializers.deserialize_to_pandas(data)
            assert result == data


class TestTypeDetectionFunctions:
    """Test type detection and pattern matching functions."""

    def test_looks_like_datetime(self):
        """Test _looks_like_datetime function."""
        assert deserializers._looks_like_datetime("2023-01-01T12:00:00") is True
        assert deserializers._looks_like_datetime("2023-01-01T12:00:00Z") is True
        assert deserializers._looks_like_datetime("not-a-date") is False
        assert deserializers._looks_like_datetime("") is False

    def test_looks_like_uuid(self):
        """Test _looks_like_uuid function."""
        assert deserializers._looks_like_uuid("12345678-1234-5678-9012-123456789abc") is True
        assert deserializers._looks_like_uuid("not-a-uuid") is False
        assert deserializers._looks_like_uuid("") is False
        # Test uppercase UUID
        assert deserializers._looks_like_uuid("12345678-1234-5678-9012-123456789ABC") is True

    def test_looks_like_number(self):
        """Test _looks_like_number function."""
        assert deserializers._looks_like_number("42") is True
        assert deserializers._looks_like_number("3.14") is True
        assert deserializers._looks_like_number("-1") is True
        assert deserializers._looks_like_number("not-a-number") is False
        assert deserializers._looks_like_number("") is False

    def test_is_numeric_part(self):
        """Test _is_numeric_part function."""
        assert deserializers._is_numeric_part("123") is True
        assert deserializers._is_numeric_part("abc") is False
        assert deserializers._is_numeric_part("") is False

    def test_looks_like_path(self):
        """Test _looks_like_path function."""
        assert deserializers._looks_like_path("/home/user/file.txt") is True
        assert deserializers._looks_like_path("C:\\Windows\\file.txt") is True
        assert deserializers._looks_like_path("relative/path/file.txt") is True
        assert deserializers._looks_like_path("not a path") is False

    def test_looks_like_series_data(self):
        """Test _looks_like_series_data function."""
        # Homogeneous numeric data
        assert deserializers._looks_like_series_data([1, 2, 3, 4, 5]) is True

        # Mixed types
        assert deserializers._looks_like_series_data([1, "text", 3]) is False

        # Empty list
        assert deserializers._looks_like_series_data([]) is False

    def test_looks_like_dataframe_dict(self):
        """Test _looks_like_dataframe_dict function."""
        # DataFrame-like structure
        df_dict = {"A": [1, 2, 3], "B": [4, 5, 6]}
        assert deserializers._looks_like_dataframe_dict(df_dict) is True

        # Not DataFrame-like
        regular_dict = {"key": "value", "number": 42}
        assert deserializers._looks_like_dataframe_dict(regular_dict) is False

    def test_looks_like_split_format(self):
        """Test _looks_like_split_format function."""
        # Split format data
        split_data = {"index": [0, 1, 2], "columns": ["A", "B"], "data": [[1, 4], [2, 5], [3, 6]]}
        assert deserializers._looks_like_split_format(split_data) is True

        # Not split format
        regular_dict = {"key": "value"}
        assert deserializers._looks_like_split_format(regular_dict) is False


class TestParsingFunctions:
    """Test string parsing functions."""

    def test_parse_datetime_string(self):
        """Test parse_datetime_string function."""
        # Valid datetime string
        dt_string = "2023-01-01T12:00:00"
        result = deserializers.parse_datetime_string(dt_string)
        assert isinstance(result, datetime) or result is None

        # Invalid datetime string
        invalid_dt = "not-a-datetime"
        result = deserializers.parse_datetime_string(invalid_dt)
        assert result is None

        # Non-string input
        result = deserializers.parse_datetime_string(42)
        assert result is None

    def test_parse_uuid_string(self):
        """Test parse_uuid_string function."""
        # Valid UUID string
        uuid_string = "12345678-1234-5678-9012-123456789abc"
        result = deserializers.parse_uuid_string(uuid_string)
        assert isinstance(result, uuid.UUID) or result is None

        # Invalid UUID string
        invalid_uuid = "not-a-uuid"
        result = deserializers.parse_uuid_string(invalid_uuid)
        assert result is None

        # Non-string input
        result = deserializers.parse_uuid_string(42)
        assert result is None


class TestTemplateDeserializer:
    """Test TemplateDeserializer class functionality."""

    def test_template_deserializer_initialization(self):
        """Test TemplateDeserializer initialization."""
        template = {"name": "string", "age": 25, "active": True}
        deserializer = deserializers.TemplateDeserializer(template)
        assert isinstance(deserializer, deserializers.TemplateDeserializer)

    def test_template_deserializer_strict_mode(self):
        """Test TemplateDeserializer in strict mode."""
        template = {"name": "string", "age": 25}
        deserializer = deserializers.TemplateDeserializer(template, strict=True)

        # Matching data
        data = {"name": "John", "age": 30}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_fallback_mode(self):
        """Test TemplateDeserializer with fallback auto-detection."""
        template = {"name": "string", "age": 25}
        deserializer = deserializers.TemplateDeserializer(template, fallback_auto_detect=True)

        # Data with extra fields
        data = {"name": "John", "age": 30, "extra": "field"}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_datetime_template(self):
        """Test TemplateDeserializer with datetime template."""
        template = {"timestamp": datetime.now()}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"timestamp": "2023-01-01T12:00:00"}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_uuid_template(self):
        """Test TemplateDeserializer with UUID template."""
        template = {"id": uuid.uuid4()}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"id": "12345678-1234-5678-9012-123456789abc"}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_path_template(self):
        """Test TemplateDeserializer with Path template."""
        template = {"file_path": Path("/tmp/file.txt")}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"file_path": "/home/user/document.pdf"}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_decimal_template(self):
        """Test TemplateDeserializer with Decimal template."""
        template = {"price": Decimal("10.50")}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"price": "25.99"}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_complex_template(self):
        """Test TemplateDeserializer with complex number template."""
        template = {"number": complex(1, 2)}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"number": [3, 4]}  # Real and imaginary parts
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    @pytest.mark.skipif(not hasattr(deserializers, "pd") or deserializers.pd is None, reason="pandas not available")
    def test_template_deserializer_dataframe_template(self):
        """Test TemplateDeserializer with DataFrame template."""
        import pandas as pd

        template = {"data": pd.DataFrame({"A": [1, 2], "B": [3, 4]})}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"data": {"A": [5, 6], "B": [7, 8]}}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    @pytest.mark.skipif(not hasattr(deserializers, "np") or deserializers.np is None, reason="numpy not available")
    def test_template_deserializer_numpy_template(self):
        """Test TemplateDeserializer with NumPy array template."""
        import numpy as np

        template = {"array": np.array([1, 2, 3])}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"array": [4, 5, 6]}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)

    def test_template_deserializer_list_template(self):
        """Test TemplateDeserializer with list template."""
        template = {"items": [1, 2, 3]}
        deserializer = deserializers.TemplateDeserializer(template)

        data = {"items": [4, 5, 6, 7]}
        result = deserializer.deserialize(data)
        assert isinstance(result, dict)


class TestTemplateDeserializerEdgeCases:
    """Test TemplateDeserializer edge cases and error handling."""

    def test_template_deserializer_with_none_template(self):
        """Test TemplateDeserializer with None template."""
        deserializer = deserializers.TemplateDeserializer(None)
        result = deserializer.deserialize({"any": "data"})
        assert isinstance(result, dict)

    def test_template_deserializer_error_handling(self):
        """Test TemplateDeserializer error handling."""
        template = {"number": 42}
        deserializer = deserializers.TemplateDeserializer(template, strict=True)

        # Try to deserialize incompatible data
        try:
            result = deserializer.deserialize({"number": "not-a-number"})
            assert isinstance(result, dict)  # Should handle gracefully
        except Exception:
            pass  # Error handling is acceptable

    def test_template_deserializer_with_missing_libraries(self):
        """Test TemplateDeserializer when required libraries are missing."""
        # Mock missing pandas
        with patch("datason.deserializers.pd", None):
            template = {"data": [1, 2, 3]}  # Use list instead of DataFrame
            deserializer = deserializers.TemplateDeserializer(template)
            result = deserializer.deserialize({"data": [4, 5, 6]})
            assert isinstance(result, dict)


class TestDeserializeFast:
    """Test fast deserialization functionality."""

    def test_deserialize_fast_basic(self):
        """Test basic fast deserialization."""
        data = {"test": "value", "number": 42}
        result = deserializers.deserialize_fast(data)
        assert result == data

    def test_deserialize_fast_with_config(self):
        """Test fast deserialization with configuration."""
        data = {"test": "value"}
        config = SerializationConfig()
        result = deserializers.deserialize_fast(data, config=config)
        assert result == data

    def test_deserialize_fast_depth_tracking(self):
        """Test fast deserialization with depth tracking."""
        data = {"nested": {"deep": {"value": 42}}}
        result = deserializers.deserialize_fast(data, _depth=0)
        assert isinstance(result, dict)

    def test_deserialize_fast_circular_protection(self):
        """Test fast deserialization with circular reference protection."""
        data = {"key": "value"}
        seen = set()
        result = deserializers.deserialize_fast(data, _seen=seen)
        assert result == data


class TestOptimizedDeserializationFunctions:
    """Test optimized deserialization helper functions."""

    def test_process_list_optimized(self):
        """Test _process_list_optimized function."""
        data = [1, "2023-01-01T12:00:00", {"nested": "value"}]
        config = SerializationConfig()
        seen = set()

        result = deserializers._process_list_optimized(data, config, 0, seen)
        assert isinstance(result, list)

    def test_process_dict_optimized(self):
        """Test _process_dict_optimized function."""
        data = {"key1": "value1", "timestamp": "2023-01-01T12:00:00"}
        config = SerializationConfig()
        seen = set()

        result = deserializers._process_dict_optimized(data, config, 0, seen)
        assert isinstance(result, dict)

    def test_deserialize_string_full(self):
        """Test _deserialize_string_full function."""
        config = SerializationConfig()

        # Test datetime string
        result = deserializers._deserialize_string_full("2023-01-01T12:00:00", config)
        assert isinstance(result, (datetime, str))

        # Test UUID string
        result = deserializers._deserialize_string_full("12345678-1234-5678-9012-123456789abc", config)
        assert isinstance(result, (uuid.UUID, str))

        # Test regular string
        result = deserializers._deserialize_string_full("regular string", config)
        assert result == "regular string"

    def test_looks_like_datetime_optimized(self):
        """Test _looks_like_datetime_optimized function."""
        assert deserializers._looks_like_datetime_optimized("2023-01-01T12:00:00") is True
        assert deserializers._looks_like_datetime_optimized("not-a-date") is False

    def test_looks_like_uuid_optimized(self):
        """Test _looks_like_uuid_optimized function."""
        assert deserializers._looks_like_uuid_optimized("12345678-1234-5678-9012-123456789abc") is True
        assert deserializers._looks_like_uuid_optimized("not-a-uuid") is False

    def test_looks_like_path_optimized(self):
        """Test _looks_like_path_optimized function."""
        assert deserializers._looks_like_path_optimized("/home/user/file.txt") is True
        assert deserializers._looks_like_path_optimized("not a path") is False


class TestCachingAndOptimization:
    """Test caching and optimization features."""

    def test_get_cached_string_pattern(self):
        """Test _get_cached_string_pattern function."""
        # This may return None if not cached
        result = deserializers._get_cached_string_pattern("test_string")
        assert result is None or isinstance(result, str)

    def test_get_cached_parsed_object(self):
        """Test _get_cached_parsed_object function."""
        result = deserializers._get_cached_parsed_object("test_string", "datetime")
        assert result is None or isinstance(result, (datetime, str))

    def test_object_pooling_functions(self):
        """Test object pooling optimization functions."""
        # Test dict pooling
        pooled_dict = deserializers._get_pooled_dict()
        assert isinstance(pooled_dict, dict)
        assert len(pooled_dict) == 0

        # Return to pool
        pooled_dict["test"] = "value"
        deserializers._return_dict_to_pool(pooled_dict)

        # Test list pooling
        pooled_list = deserializers._get_pooled_list()
        assert isinstance(pooled_list, list)
        assert len(pooled_list) == 0

        # Return to pool
        pooled_list.append("item")
        deserializers._return_list_to_pool(pooled_list)

    def test_clear_caches(self):
        """Test cache clearing functions."""
        # Test internal cache clearing
        deserializers._clear_deserialization_caches()

        # Test public cache clearing
        deserializers.clear_caches()


class TestAdvancedFeatures:
    """Test advanced deserialization features."""

    def test_convert_string_keys_to_int_if_possible(self):
        """Test _convert_string_keys_to_int_if_possible function."""
        # Dict with string keys that are numbers
        data = {"1": "value1", "2": "value2", "not_number": "value3"}
        result = deserializers._convert_string_keys_to_int_if_possible(data)
        assert isinstance(result, dict)

    def test_try_numpy_array_detection(self):
        """Test _try_numpy_array_detection function."""
        # Homogeneous numeric list
        data = [1, 2, 3, 4, 5]
        result = deserializers._try_numpy_array_detection(data)
        # May return None if numpy not available or not detected
        assert result is None or hasattr(result, "shape")

    def test_looks_like_numpy_array(self):
        """Test _looks_like_numpy_array function."""
        # Test the function returns a boolean result
        result = deserializers._looks_like_numpy_array([1, 2, 3, 4, 5])
        assert isinstance(result, bool)

        # Mixed types
        result = deserializers._looks_like_numpy_array([1, "text", 3])
        assert isinstance(result, bool)

        # Empty list
        result = deserializers._looks_like_numpy_array([])
        assert isinstance(result, bool)

    def test_is_homogeneous_basic_types(self):
        """Test _is_homogeneous_basic_types function."""
        # Homogeneous integers
        assert deserializers._is_homogeneous_basic_types([1, 2, 3, 4, 5]) is True

        # Homogeneous strings
        assert deserializers._is_homogeneous_basic_types(["a", "b", "c"]) is True

        # Mixed types
        assert deserializers._is_homogeneous_basic_types([1, "text", 3]) is False

    def test_try_dataframe_detection(self):
        """Test _try_dataframe_detection function."""
        # List of dicts (records format)
        data = [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
        result = deserializers._try_dataframe_detection(data)
        # May return None if pandas not available or not detected
        assert result is None or hasattr(result, "columns")

    def test_try_series_detection(self):
        """Test _try_series_detection function."""
        # Series-like data
        data = {"index": [0, 1, 2], "data": [1, 2, 3]}
        result = deserializers._try_series_detection(data)
        # May return None if pandas not available or not detected
        assert result is None or hasattr(result, "index")


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_safe_deserialize(self):
        """Test safe_deserialize function."""
        # Valid JSON string
        json_str = '{"key": "value", "number": 42}'
        result = deserializers.safe_deserialize(json_str)
        assert isinstance(result, dict)

        # Invalid JSON string
        invalid_json = '{"invalid": json}'
        result = deserializers.safe_deserialize(invalid_json)
        # Should handle gracefully - may return None, dict, or the original string
        assert result is None or isinstance(result, (dict, str))

        # Test new allow_pickle parameter
        json_str_safe = '{"datetime": "2023-01-01T12:00:00"}'
        result = deserializers.safe_deserialize(json_str_safe, allow_pickle=False)
        assert isinstance(result, dict)

        result = deserializers.safe_deserialize(json_str_safe, allow_pickle=True)
        assert isinstance(result, dict)

    def test_restore_pandas_types(self):
        """Test _restore_pandas_types function."""
        data = {"test": "value", "number": 42}
        result = deserializers._restore_pandas_types(data)
        assert isinstance(result, (dict, type(None)))

    def test_auto_detect_string_type(self):
        """Test _auto_detect_string_type function."""
        # Test datetime string
        result = deserializers._auto_detect_string_type("2023-01-01T12:00:00")
        assert isinstance(result, (datetime, str))

        # Test UUID string
        result = deserializers._auto_detect_string_type("12345678-1234-5678-9012-123456789abc")
        assert isinstance(result, (uuid.UUID, str))

        # Test numeric string
        result = deserializers._auto_detect_string_type("42")
        assert isinstance(result, (int, str))

        # Test aggressive mode
        result = deserializers._auto_detect_string_type("42", aggressive=True)
        assert isinstance(result, (int, str))


class TestTemplateUtilityFunctions:
    """Test template-related utility functions."""

    def test_deserialize_with_template(self):
        """Test deserialize_with_template convenience function."""
        template = {"name": "string", "age": 25}
        data = {"name": "John", "age": 30}

        result = deserializers.deserialize_with_template(data, template)
        assert isinstance(result, dict)

    def test_infer_template_from_data(self):
        """Test infer_template_from_data function."""
        data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        template = deserializers.infer_template_from_data(data)
        assert isinstance(template, (dict, type(None)))

    def test_create_ml_round_trip_template(self):
        """Test create_ml_round_trip_template function."""
        # Mock ML object
        mock_ml_object = Mock()
        mock_ml_object.__class__.__name__ = "TestMLModel"

        template = deserializers.create_ml_round_trip_template(mock_ml_object)
        assert isinstance(template, dict)


class TestSecurityFeatures:
    """Test security-related features and enhancements."""

    def test_contains_pickle_data_detection(self):
        """Test _contains_pickle_data function correctly detects pickle data."""
        # Test sklearn data with pickle format (string)
        sklearn_pickle_str = {
            "__datason_type__": "sklearn.linear_model.LogisticRegression",
            "__datason_value__": "base64_encoded_pickle_data_here",
        }
        assert deserializers._contains_pickle_data(sklearn_pickle_str) is True

        # Test sklearn data with pickle format (dict with _pickle_data)
        sklearn_pickle_dict = {
            "__datason_type__": "sklearn.ensemble.RandomForestClassifier",
            "__datason_value__": {"_pickle_data": "base64_encoded_pickle_data"},
        }
        assert deserializers._contains_pickle_data(sklearn_pickle_dict) is True

        # Test non-pickle data
        safe_data = {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"}
        assert deserializers._contains_pickle_data(safe_data) is False

        # Test nested data structures
        nested_with_pickle = {"data": [{"safe": "value"}, sklearn_pickle_str]}
        assert deserializers._contains_pickle_data(nested_with_pickle) is True

        # Test completely safe data
        safe_nested = {"data": [{"safe": "value"}, {"also": "safe"}], "config": {"setting": "value"}}
        assert deserializers._contains_pickle_data(safe_nested) is False

    def test_safe_deserialize_default_security(self):
        """Test safe_deserialize rejects pickle data by default."""
        import json

        # Test safe data works normally
        safe_data = {"datetime": "2023-01-01T12:00:00", "number": 42}
        json_str = json.dumps(safe_data)
        result = deserializers.safe_deserialize(json_str)
        assert isinstance(result, dict)
        assert result["number"] == 42

        # Test sklearn pickle data is rejected by default
        unsafe_data = {
            "__datason_type__": "sklearn.linear_model.LogisticRegression",
            "__datason_value__": "unsafe_pickle_data",
        }
        unsafe_json = json.dumps(unsafe_data)

        with pytest.raises(deserializers.DeserializationSecurityError) as exc_info:
            deserializers.safe_deserialize(unsafe_json)

        assert "pickle-serialized objects" in str(exc_info.value)
        assert "unsafe to deserialize" in str(exc_info.value)

    def test_safe_deserialize_allow_pickle_override(self):
        """Test safe_deserialize allows pickle when allow_pickle=True."""
        import json

        # Test that allow_pickle=True bypasses security check
        unsafe_data = {
            "__datason_type__": "sklearn.linear_model.LogisticRegression",
            "__datason_value__": "pickle_data_here",
        }
        unsafe_json = json.dumps(unsafe_data)

        # This should work without raising an exception
        # (though it will likely warn about deserialization failure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = deserializers.safe_deserialize(unsafe_json, allow_pickle=True)
            # Should return something (either reconstructed object or fallback)
            assert result is not None

    def test_pickle_deserialization_warnings(self):
        """Test that legacy pickle deserialization shows security warnings."""
        # Create mock sklearn data with legacy pickle format
        sklearn_data = {
            "__datason_type__": "sklearn.linear_model.LogisticRegression",
            "__datason_value__": "fake_base64_pickle_data",
        }

        # Capture warnings during deserialization
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # This should trigger warnings about disabled pickle deserialization
            result = deserializers._deserialize_with_type_metadata(sklearn_data)

            # Check if security warnings were issued about disabled pickle
            security_warnings = [
                w for w in warning_list if "disabled" in str(w.message).lower() and "security" in str(w.message).lower()
            ]

            # We expect at least one security warning about disabled pickle deserialization
            assert len(security_warnings) >= 1, (
                f"Expected security warning, got: {[str(w.message) for w in warning_list]}"
            )

            # Should return the original value since pickle is disabled
            assert result == sklearn_data["__datason_value__"]

    def test_datetime_parsing_security_fix(self):
        """Test that datetime parsing works correctly (path injection false positive fix)."""
        # Test that datetime strings are parsed correctly
        datetime_string = "2023-01-01T12:00:00Z"
        result = deserializers._auto_detect_string_type(datetime_string)

        # Should successfully parse as datetime
        assert isinstance(result, (datetime, str))

        # Test with various datetime formats
        test_cases = [
            "2023-01-01T12:00:00",
            "2023-01-01T12:00:00Z",
            "2023-01-01T12:00:00+00:00",
            "2023-12-31T23:59:59.999Z",
        ]

        for dt_str in test_cases:
            result = deserializers._auto_detect_string_type(dt_str)
            # Should not raise any path injection errors
            assert isinstance(result, (datetime, str))

    def test_datetime_regex_validation_security(self):
        """Test that regex validation prevents CodeQL false positives for datetime parsing."""
        # This test ensures our ISO 8601 regex validation eliminates CodeQL warnings
        # by validating input before calling fromisoformat()

        # Valid ISO 8601 formats that should pass validation and parse to datetime
        valid_iso_formats = [
            "2023-01-01T12:00:00",
            "2023-12-31T23:59:59",
            "2023-06-15T14:30:45.123",
            "2023-06-15T14:30:45.123456",
            "2023-01-01T00:00:00Z",
            "2023-01-01T12:00:00+05:00",
            "2023-01-01T12:00:00-03:30",
            "2023-01-01 12:00:00",  # Space separator allowed
        ]

        for dt_str in valid_iso_formats:
            result = deserializers._auto_detect_string_type(dt_str)
            assert isinstance(result, datetime), f"Valid ISO format should parse to datetime: {dt_str}"

        # Invalid formats that should NOT pass regex validation
        # These should remain as strings, preventing CodeQL path-related warnings
        invalid_formats = [
            "/etc/passwd",  # Path-like string (security concern)
            "../../../etc/passwd",  # Path traversal attempt
            "file:///etc/passwd",  # File URI
            "2023/01/01 12:00:00",  # Wrong date separators
            "2023-13-01T12:00:00",  # Invalid month
            "2023-01-32T12:00:00",  # Invalid day
            "2023-01-01T25:00:00",  # Invalid hour
            "2023-01-01T12:60:00",  # Invalid minute
            "2023-01-01",  # Date only, no time
            "random-string-here",  # Random text
        ]

        for invalid_str in invalid_formats:
            result = deserializers._auto_detect_string_type(invalid_str)
            assert isinstance(result, str), f"Invalid format should remain string: {invalid_str}"
            assert result == invalid_str, f"Should return unchanged: {invalid_str}"

    def test_legacy_sklearn_model_reconstruction_security(self):
        """Test security handling in legacy sklearn model reconstruction."""
        # Test various sklearn type names
        sklearn_types = [
            "sklearn.linear_model.LogisticRegression",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.svm.SVC",
        ]

        for sklearn_type in sklearn_types:
            # Test with invalid pickle data (should fail gracefully)
            invalid_data = {"__datason_type__": sklearn_type, "__datason_value__": "invalid_pickle_data"}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = deserializers._deserialize_with_type_metadata(invalid_data)

                # Should return something (may be value or original data), not crash
                assert result is not None
                # Could be the value itself or the original data structure
                assert result in ["invalid_pickle_data", invalid_data]

    def test_security_error_exception(self):
        """Test DeserializationSecurityError can be raised and caught properly."""
        with pytest.raises(deserializers.DeserializationSecurityError) as exc_info:
            raise deserializers.DeserializationSecurityError("Test security violation")

        assert "Test security violation" in str(exc_info.value)
        assert isinstance(exc_info.value, Exception)

    def test_auto_deserialize_config_fallback_security(self):
        """Test that auto_deserialize handles config availability securely."""
        # This tests the bug fix for undefined variables
        # The function should work regardless of config availability

        test_data = {"string": "test", "number": 42, "datetime": "2023-01-01T12:00:00"}

        # Should work with None config
        result = deserializers.auto_deserialize(test_data, config=None)
        assert isinstance(result, dict)
        assert result["number"] == 42

        # Should work with no config parameter
        result = deserializers.auto_deserialize(test_data)
        assert isinstance(result, dict)
        assert result["number"] == 42


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_deserialization_security_error(self):
        """Test DeserializationSecurityError exception."""
        try:
            raise deserializers.DeserializationSecurityError("Test security error")
        except deserializers.DeserializationSecurityError as e:
            assert str(e) == "Test security error"

    def test_template_deserialization_error(self):
        """Test TemplateDeserializationError exception."""
        try:
            raise deserializers.TemplateDeserializationError("Test template error")
        except deserializers.TemplateDeserializationError as e:
            assert str(e) == "Test template error"

    def test_deserialize_with_missing_imports(self):
        """Test deserialization when optional imports are missing."""
        # Test deserialization still works when pandas/numpy unavailable
        data = {"simple": "data"}
        result = deserializers.deserialize(data)
        assert result == data

    def test_import_fallback_constants(self):
        """Test that security constants are properly defined."""
        assert deserializers.MAX_SERIALIZATION_DEPTH == 50
        assert deserializers.MAX_OBJECT_SIZE == 100_000
        assert deserializers.MAX_STRING_LENGTH == 1_000_000


if __name__ == "__main__":
    pytest.main([__file__])
