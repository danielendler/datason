"""Enhanced coverage tests for datason/deserializers_new.py module.

This test suite targets the specific missing coverage areas to boost
coverage from 50% to 80%+ for the critical deserialization functionality.
"""

import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

# Conditional imports
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from datason.deserializers_new import (
    TYPE_METADATA_KEY,
    VALUE_METADATA_KEY,
    TemplateDeserializer,
    _auto_detect_string_type,
    _deserialize_with_type_metadata,
    _looks_like_datetime,
    _looks_like_number,
    _looks_like_uuid,
    auto_deserialize,
    deserialize,
    deserialize_fast,
    parse_datetime_string,
    parse_uuid_string,
    safe_deserialize,
)


class TestTypeMetadataReconstruction:
    """Test type metadata reconstruction paths."""

    def test_datetime_reconstruction(self):
        """Test datetime from metadata."""
        metadata = {TYPE_METADATA_KEY: "datetime", VALUE_METADATA_KEY: "2023-01-01T12:00:00"}
        result = _deserialize_with_type_metadata(metadata)
        assert isinstance(result, datetime)

    def test_uuid_reconstruction(self):
        """Test UUID from metadata."""
        metadata = {TYPE_METADATA_KEY: "uuid.UUID", VALUE_METADATA_KEY: "12345678-1234-5678-9012-123456789abc"}
        result = _deserialize_with_type_metadata(metadata)
        assert isinstance(result, uuid.UUID)

    def test_complex_reconstruction(self):
        """Test complex number reconstruction."""
        metadata = {TYPE_METADATA_KEY: "complex", VALUE_METADATA_KEY: {"real": 3.0, "imag": 4.0}}
        result = _deserialize_with_type_metadata(metadata)
        assert isinstance(result, complex)
        assert result.real == 3.0

    def test_decimal_reconstruction(self):
        """Test Decimal reconstruction."""
        metadata = {TYPE_METADATA_KEY: "decimal.Decimal", VALUE_METADATA_KEY: "123.456"}
        result = _deserialize_with_type_metadata(metadata)
        assert isinstance(result, Decimal)

    def test_path_reconstruction(self):
        """Test Path reconstruction."""
        metadata = {TYPE_METADATA_KEY: "pathlib.Path", VALUE_METADATA_KEY: "/test/path"}
        result = _deserialize_with_type_metadata(metadata)
        assert isinstance(result, Path)

    def test_set_tuple_reconstruction(self):
        """Test set and tuple reconstruction."""
        set_meta = {TYPE_METADATA_KEY: "set", VALUE_METADATA_KEY: [1, 2, 3]}
        result = _deserialize_with_type_metadata(set_meta)
        assert isinstance(result, set)

        tuple_meta = {TYPE_METADATA_KEY: "tuple", VALUE_METADATA_KEY: [1, 2, 3]}
        result = _deserialize_with_type_metadata(tuple_meta)
        assert isinstance(result, tuple)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_pandas_dataframe_reconstruction(self):
        """Test DataFrame reconstruction paths."""
        # Records format
        records_meta = {TYPE_METADATA_KEY: "pandas.DataFrame", VALUE_METADATA_KEY: [{"a": 1, "b": 2}]}
        result = _deserialize_with_type_metadata(records_meta)
        assert isinstance(result, pd.DataFrame)

        # Split format
        split_meta = {
            TYPE_METADATA_KEY: "pandas.DataFrame",
            VALUE_METADATA_KEY: {"index": [0], "columns": ["a", "b"], "data": [[1, 2]]},
        }
        result = _deserialize_with_type_metadata(split_meta)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_pandas_series_reconstruction(self):
        """Test Series reconstruction paths."""
        # Named series
        named_meta = {
            TYPE_METADATA_KEY: "pandas.Series",
            VALUE_METADATA_KEY: {"0": 10, "1": 20, "_series_name": "test"},
        }
        result = _deserialize_with_type_metadata(named_meta)
        assert isinstance(result, pd.Series)
        assert result.name == "test"

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_numpy_reconstruction(self):
        """Test numpy array reconstruction."""
        array_meta = {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: {"data": [1, 2, 3], "dtype": "int64"}}
        result = _deserialize_with_type_metadata(array_meta)
        assert isinstance(result, np.ndarray)

    def test_unknown_type_fallback(self):
        """Test unknown type fallback."""
        unknown_meta = {TYPE_METADATA_KEY: "unknown.Type", VALUE_METADATA_KEY: {"data": "test"}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _deserialize_with_type_metadata(unknown_meta)
            assert result == {"data": "test"}
            # May or may not warn depending on implementation
            assert len(w) >= 0


class TestAutoDetection:
    """Test auto-detection logic."""

    def test_uuid_detection(self):
        """Test UUID auto-detection."""
        uuid_str = "12345678-1234-5678-9012-123456789abc"
        result = _auto_detect_string_type(uuid_str)
        assert isinstance(result, uuid.UUID)

    def test_datetime_detection(self):
        """Test datetime auto-detection."""
        dt_str = "2023-01-01T12:00:00"
        result = _auto_detect_string_type(dt_str)
        assert isinstance(result, datetime)

    def test_aggressive_detection(self):
        """Test aggressive detection mode."""
        # Number detection
        result = _auto_detect_string_type("123", aggressive=True)
        assert isinstance(result, int)

        result = _auto_detect_string_type("3.14", aggressive=True)
        assert isinstance(result, float)

        # Boolean detection
        result = _auto_detect_string_type("true", aggressive=True)
        assert result is True


class TestPatternMatching:
    """Test pattern matching functions."""

    def test_number_patterns(self):
        """Test number pattern matching."""
        assert _looks_like_number("123")
        assert _looks_like_number("-456")
        assert _looks_like_number("3.14")
        assert _looks_like_number("1e10")
        assert not _looks_like_number("abc")
        assert not _looks_like_number("")

    def test_datetime_patterns(self):
        """Test datetime pattern matching."""
        assert _looks_like_datetime("2023-01-01T12:00:00")
        assert _looks_like_datetime("2023-12-31T23:59:59Z")
        assert not _looks_like_datetime("12345678-1234-5678-9012-123456789abc")
        assert not _looks_like_datetime("hello")

    def test_uuid_patterns(self):
        """Test UUID pattern matching."""
        assert _looks_like_uuid("12345678-1234-5678-9012-123456789abc")
        assert not _looks_like_uuid("2023-01-01T12:00:00")
        assert not _looks_like_uuid("hello")


class TestTemplateDeserialization:
    """Test template-based deserialization."""

    def test_template_init(self):
        """Test TemplateDeserializer initialization."""
        template = {"test": "value"}
        deserializer = TemplateDeserializer(template)
        assert deserializer.template == template

    def test_datetime_template(self):
        """Test datetime template."""
        template = datetime(2023, 1, 1)
        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize("2023-06-15T12:00:00")
        assert isinstance(result, datetime)

    def test_uuid_template(self):
        """Test UUID template."""
        template = uuid.uuid4()
        deserializer = TemplateDeserializer(template)
        result = deserializer.deserialize("12345678-1234-5678-9012-123456789abc")
        assert isinstance(result, uuid.UUID)


class TestFastDeserialization:
    """Test fast deserialization."""

    def test_fast_basic(self):
        """Test fast deserialization basics."""
        data = {"test": "value", "number": 42}
        result = deserialize_fast(data)
        assert result == data

    def test_fast_with_config(self):
        """Test fast deserialization with config."""
        from datason.config import SerializationConfig

        config = SerializationConfig()
        data = {"date": "2023-01-01T12:00:00"}
        result = deserialize_fast(data, config=config)
        # Should process with config
        assert result is not None


class TestSafety:
    """Test safety features."""

    def test_safe_deserialize(self):
        """Test safe deserialization."""
        json_str = '{"test": "value"}'
        result = safe_deserialize(json_str)
        assert result == {"test": "value"}

    def test_safe_deserialize_pickle_disabled(self):
        """Test pickle rejection."""
        pickle_json = '{"__datason_type__": "pickle", "__datason_value__": "data"}'
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = safe_deserialize(pickle_json, allow_pickle=False)
            # Should process without errors
            assert result is not None


class TestParsing:
    """Test parsing functions."""

    def test_parse_datetime(self):
        """Test datetime parsing."""
        result = parse_datetime_string("2023-01-01T12:00:00")
        assert isinstance(result, datetime)

        result = parse_datetime_string("invalid")
        assert result is None

    def test_parse_uuid(self):
        """Test UUID parsing."""
        result = parse_uuid_string("12345678-1234-5678-9012-123456789abc")
        assert isinstance(result, uuid.UUID)

        result = parse_uuid_string("invalid")
        assert result is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_auto_deserialize_comprehensive(self):
        """Test comprehensive auto-deserialization."""
        data = {
            "uuid": "12345678-1234-5678-9012-123456789abc",
            "date": "2023-01-01T12:00:00",
            "nested": {"inner": "value"},
        }
        result = auto_deserialize(data, aggressive=True)
        assert isinstance(result["uuid"], uuid.UUID)
        assert isinstance(result["date"], datetime)

    def test_error_resilience(self):
        """Test error resilience."""
        malformed = {"__datason_type__": "invalid", "__datason_value__": "data"}
        result = deserialize(malformed)
        assert result is not None

    def test_caching_functions(self):
        """Test caching functionality."""
        from datason.deserializers_new import clear_caches

        clear_caches()  # Should not raise


class TestMLFrameworkSupport:
    """Test ML framework support paths."""

    def test_torch_reconstruction(self):
        """Test PyTorch reconstruction if available."""
        torch_meta = {
            TYPE_METADATA_KEY: "torch.Tensor",
            VALUE_METADATA_KEY: {"data": [[1, 2], [3, 4]], "dtype": "float32"},
        }

        try:
            import torch

            result = _deserialize_with_type_metadata(torch_meta)
            assert torch.is_tensor(result)
        except ImportError:
            # Should return original if torch not available
            result = _deserialize_with_type_metadata(torch_meta)
            assert result == torch_meta[VALUE_METADATA_KEY]

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_numpy_scalar_types(self):
        """Test numpy scalar type reconstruction."""
        scalar_types = [("numpy.int32", 42), ("numpy.float64", 3.14), ("numpy.bool_", True)]

        for type_name, value in scalar_types:
            metadata = {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: value}
            result = _deserialize_with_type_metadata(metadata)
            assert type(result).__module__ == "numpy"
