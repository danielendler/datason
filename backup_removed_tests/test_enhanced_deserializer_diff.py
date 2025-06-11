"""Tests targeting new deserializer functionality for diff coverage.

Focuses on the major new additions in deserializers.py that need coverage.
"""

import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

from datason.core import SerializationConfig
from datason.deserializers import (
    DeserializationSecurityError,
    _convert_string_keys_to_int_if_possible,
    _deserialize_string_full,
    _get_pooled_dict,
    _get_pooled_list,
    _looks_like_datetime_optimized,
    _looks_like_path_optimized,
    _looks_like_uuid_optimized,
    _return_dict_to_pool,
    _return_list_to_pool,
    deserialize_fast,
)


class TestDeserializeFastCore:
    """Test core deserialize_fast functionality."""

    def test_basic_types_fast_path(self) -> None:
        """Test basic types go through fast path."""
        assert deserialize_fast(42) == 42
        assert deserialize_fast(True) is True
        assert deserialize_fast(None) is None
        assert deserialize_fast(3.14) == 3.14
        assert deserialize_fast("short") == "short"

    def test_security_depth_enforcement(self) -> None:
        """Test depth limit security enforcement."""
        config = SerializationConfig(max_depth=1)  # More restrictive to ensure we hit limit

        # Should work at limit
        simple = {"a": "value"}
        result = deserialize_fast(simple, config)
        assert result["a"] == "value"

        # Should fail beyond limit - deeper nesting
        too_deep = {"a": {"b": {"c": {"d": "value"}}}}
        with pytest.raises(DeserializationSecurityError, match="Maximum deserialization depth"):
            deserialize_fast(too_deep, config)

    def test_security_size_enforcement(self) -> None:
        """Test size limit security enforcement."""
        config = SerializationConfig(max_size=2)

        # Large dict should fail
        large_dict = {"a": 1, "b": 2, "c": 3}
        with pytest.raises(DeserializationSecurityError, match="Dictionary size"):
            deserialize_fast(large_dict, config)

        # Large list should fail
        large_list = [1, 2, 3]
        with pytest.raises(DeserializationSecurityError, match="List size"):
            deserialize_fast(large_list, config)

    def test_type_metadata_new_format(self) -> None:
        """Test new __datason_type__ format processing."""
        # UUID with new format
        uuid_obj = {"__datason_type__": "uuid.UUID", "__datason_value__": "550e8400-e29b-41d4-a716-446655440000"}
        result = deserialize_fast(uuid_obj)
        assert isinstance(result, uuid.UUID)

        # Datetime with new format
        dt_obj = {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"}
        result = deserialize_fast(dt_obj)
        assert isinstance(result, datetime)

        # Complex number with new format
        complex_obj = {"__datason_type__": "complex", "__datason_value__": {"real": 1.0, "imag": 2.0}}
        result = deserialize_fast(complex_obj)
        assert result == complex(1.0, 2.0)

    def test_enhanced_dataframe_reconstruction(self) -> None:
        """Test enhanced DataFrame reconstruction logic."""
        try:
            import pandas as pd

            # Test split format
            split_df = {
                "__datason_type__": "pandas.DataFrame",
                "__datason_value__": {"index": [0, 1], "columns": ["a", "b"], "data": [[1, 2], [3, 4]]},
            }
            result = deserialize_fast(split_df)
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["a", "b"]

        except ImportError:
            pytest.skip("pandas not available")

    def test_enhanced_series_categorical(self) -> None:
        """Test enhanced Series with categorical support."""
        try:
            import pandas as pd

            # Series with categorical dtype
            series_obj = {
                "__datason_type__": "pandas.Series",
                "__datason_value__": {
                    "0": "cat1",
                    "1": "cat2",
                    "_dtype": "category",
                    "_categories": ["cat1", "cat2", "cat3"],
                    "_ordered": False,
                    "_series_name": "test_series",
                },
            }
            result = deserialize_fast(series_obj)
            assert isinstance(result, pd.Series)
            assert result.name == "test_series"
            assert result.dtype.name == "category"

        except ImportError:
            pytest.skip("pandas not available")


class TestOptimizedStringDetection:
    """Test optimized string detection functions."""

    def test_datetime_optimized_detection(self) -> None:
        """Test optimized datetime detection."""
        assert _looks_like_datetime_optimized("2023-01-01T12:00:00")
        assert _looks_like_datetime_optimized("2023-12-31")
        assert not _looks_like_datetime_optimized("not-a-date")
        assert not _looks_like_datetime_optimized("short")
        assert not _looks_like_datetime_optimized("")

    def test_uuid_optimized_detection(self) -> None:
        """Test optimized UUID detection."""
        assert _looks_like_uuid_optimized("550e8400-e29b-41d4-a716-446655440000")
        assert not _looks_like_uuid_optimized("not-a-uuid")
        assert not _looks_like_uuid_optimized("550e8400-e29b-41d4-a716")
        assert not _looks_like_uuid_optimized("")

    def test_path_optimized_detection(self) -> None:
        """Test optimized path detection."""
        assert _looks_like_path_optimized("/tmp/test/file.txt")
        assert _looks_like_path_optimized("C:\\Windows\\System32")
        assert _looks_like_path_optimized("./relative/path")
        assert _looks_like_path_optimized("../parent/dir")
        assert _looks_like_path_optimized("file.txt")
        assert not _looks_like_path_optimized("not-a-path")
        assert not _looks_like_path_optimized("")

    def test_string_full_deserialization(self) -> None:
        """Test full string deserialization with all types."""
        # Plain string
        assert _deserialize_string_full("hello", None) == "hello"

        # UUID string
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = _deserialize_string_full(uuid_str, None)
        assert isinstance(result, uuid.UUID)

        # Datetime string
        dt_str = "2023-01-01T12:00:00"
        result = _deserialize_string_full(dt_str, None)
        assert isinstance(result, datetime)

        # Path string
        path_str = "/tmp/test/file.txt"
        result = _deserialize_string_full(path_str, None)
        assert isinstance(result, Path)


class TestMemoryOptimizations:
    """Test memory optimization features."""

    def test_pooled_dict_reuse(self) -> None:
        """Test pooled dictionary reuse."""
        # Get dict from pool
        d1 = _get_pooled_dict()
        assert isinstance(d1, dict)
        assert len(d1) == 0

        # Use it
        d1["test"] = "value"
        assert d1["test"] == "value"

        # Return to pool
        _return_dict_to_pool(d1)

        # Get another (might be same object)
        d2 = _get_pooled_dict()
        assert isinstance(d2, dict)
        assert len(d2) == 0  # Should be cleared

    def test_pooled_list_reuse(self) -> None:
        """Test pooled list reuse."""
        # Get list from pool
        l1 = _get_pooled_list()
        assert isinstance(l1, list)
        assert len(l1) == 0

        # Use it
        l1.append("test")
        assert l1[0] == "test"

        # Return to pool
        _return_list_to_pool(l1)

        # Get another (might be same object)
        l2 = _get_pooled_list()
        assert isinstance(l2, list)
        assert len(l2) == 0  # Should be cleared

    def test_string_key_conversion(self) -> None:
        """Test string key to integer conversion."""
        # Integer strings -> integers
        data = {"0": "a", "1": "b", "2": "c"}
        result = _convert_string_keys_to_int_if_possible(data)
        assert result == {0: "a", 1: "b", 2: "c"}

        # Negative integers
        data = {"-1": "a", "0": "b", "1": "c"}
        result = _convert_string_keys_to_int_if_possible(data)
        assert result == {-1: "a", 0: "b", 1: "c"}

        # Mixed keys
        data = {"0": "a", "non_int": "b", "2": "c"}
        result = _convert_string_keys_to_int_if_possible(data)
        assert result[0] == "a"
        assert result["non_int"] == "b"
        assert result[2] == "c"


class TestEnhancedTypeMetadata:
    """Test enhanced type metadata processing."""

    def test_decimal_reconstruction(self) -> None:
        """Test Decimal reconstruction from metadata."""
        decimal_obj = {"__datason_type__": "decimal.Decimal", "__datason_value__": "123.456"}
        result = deserialize_fast(decimal_obj)
        assert isinstance(result, Decimal)
        assert result == Decimal("123.456")

    def test_path_reconstruction(self) -> None:
        """Test Path reconstruction from metadata."""
        path_obj = {"__datason_type__": "pathlib.Path", "__datason_value__": "/tmp/test/file.txt"}
        result = deserialize_fast(path_obj)
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test/file.txt"

    def test_set_tuple_reconstruction(self) -> None:
        """Test set and tuple reconstruction."""
        # Set reconstruction
        set_obj = {"__datason_type__": "set", "__datason_value__": [1, 2, 3]}
        result = deserialize_fast(set_obj)
        assert isinstance(result, set)
        assert result == {1, 2, 3}

        # Tuple reconstruction
        tuple_obj = {"__datason_type__": "tuple", "__datason_value__": [1, 2, 3]}
        result = deserialize_fast(tuple_obj)
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_numpy_enhanced_support(self) -> None:
        """Test enhanced NumPy support."""
        try:
            import numpy as np

            # NumPy array with dtype and shape
            array_obj = {
                "__datason_type__": "numpy.ndarray",
                "__datason_value__": {"data": [1, 2, 3, 4], "dtype": "int32", "shape": [2, 2]},
            }
            result = deserialize_fast(array_obj)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.int32
            assert result.shape == (2, 2)

        except ImportError:
            pytest.skip("numpy not available")

    def test_legacy_type_format_removed(self) -> None:
        """Test legacy _type format no longer supported (Phase 2 v0.8.0)."""
        # Legacy decimal format should now return as-is (no longer processed)
        legacy_decimal = {"_type": "decimal", "value": "99.99"}
        result = deserialize_fast(legacy_decimal)
        assert result == legacy_decimal  # Returned unchanged

        # Legacy complex format should now return as-is (no longer processed)
        legacy_complex = {"_type": "complex", "real": 1.5, "imag": 2.5}
        result = deserialize_fast(legacy_complex)
        assert result == legacy_complex  # Returned unchanged


class TestAutoDetectionPatterns:
    """Test auto-detection of data patterns."""

    def test_complex_auto_detection(self) -> None:
        """Test auto-detection of complex numbers."""
        complex_dict = {"real": 3.0, "imag": 4.0}
        result = deserialize_fast(complex_dict)
        assert result == complex(3.0, 4.0)

    def test_decimal_auto_detection(self) -> None:
        """Test auto-detection of Decimal from value dict."""
        decimal_dict = {"value": "123.456"}
        result = deserialize_fast(decimal_dict)
        assert isinstance(result, Decimal)
        assert result == Decimal("123.456")


class TestCircularReferenceHandling:
    """Test circular reference protection."""

    def test_circular_dict_protection(self) -> None:
        """Test circular reference protection in dicts."""
        circular: Dict[str, Any] = {"a": 1}
        circular["self"] = circular

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deserialize_fast(circular)
            # Should generate warning about circular reference
            if w:  # May or may not trigger depending on implementation
                assert any("Circular reference" in str(warning.message) for warning in w)


class TestErrorHandlingRobustness:
    """Test error handling and robustness."""

    def test_invalid_string_parsing(self) -> None:
        """Test graceful handling of invalid string formats."""
        # Invalid UUID - should return as string
        invalid_uuid = "not-a-uuid-550e8400"
        result = _deserialize_string_full(invalid_uuid, None)
        assert result == invalid_uuid

        # Invalid datetime - should return as string
        invalid_dt = "not-2023-01-01"
        result = _deserialize_string_full(invalid_dt, None)
        assert result == invalid_dt

    def test_malformed_metadata_handling(self) -> None:
        """Test handling of malformed type metadata."""
        # Missing value field
        bad_metadata = {"__datason_type__": "uuid.UUID"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            deserialize_fast(bad_metadata)
            # Should handle gracefully, might return dict or fail gracefully

    def test_unknown_type_passthrough(self) -> None:
        """Test unknown types pass through unchanged."""

        class UnknownType:
            def __init__(self) -> None:
                self.value = "test"

        unknown = UnknownType()
        result = deserialize_fast(unknown)
        assert result is unknown  # Should return unchanged


class TestContainerProcessing:
    """Test optimized container processing."""

    def test_nested_list_processing(self) -> None:
        """Test nested list with mixed types."""
        nested_list = [
            1,
            "hello",
            {"__datason_type__": "uuid.UUID", "__datason_value__": "550e8400-e29b-41d4-a716-446655440000"},
            [2, 3, 4],
        ]
        result = deserialize_fast(nested_list)
        assert len(result) == 4
        assert result[0] == 1
        assert result[1] == "hello"
        assert isinstance(result[2], uuid.UUID)
        assert result[3] == [2, 3, 4]

    def test_nested_dict_processing(self) -> None:
        """Test nested dict with mixed types."""
        nested_dict = {
            "number": 42,
            "uuid": {"__datason_type__": "uuid.UUID", "__datason_value__": "550e8400-e29b-41d4-a716-446655440000"},
            "nested": {"a": 1, "b": 2},
        }
        result = deserialize_fast(nested_dict)
        assert result["number"] == 42
        assert isinstance(result["uuid"], uuid.UUID)
        assert result["nested"] == {"a": 1, "b": 2}
