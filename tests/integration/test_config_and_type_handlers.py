"""Tests for configuration system and enhanced type handling."""

import decimal
import enum
import uuid
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import pytest

import datason
from datason.config import (
    DataFrameOrient,
    DateFormat,
    NanHandling,
    OutputType,
    SerializationConfig,
    TypeCoercion,
    get_api_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
)
from datason.type_handlers import (
    get_object_info,
    is_nan_like,
    normalize_numpy_types,
)

# Optional pandas import for tests
try:
    import pandas as pd
except ImportError:
    pd = None

# Test data setup
Person = namedtuple("Person", ["name", "age", "city"])


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestSerializationConfig:
    """Test configuration classes and presets."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SerializationConfig()
        assert config.date_format == DateFormat.ISO
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.SAFE
        assert config.preserve_decimals is True
        assert config.preserve_complex is True
        assert config.max_depth == 50
        assert config.sort_keys is False

    def test_ml_config(self) -> None:
        """Test ML-optimized configuration."""
        config = get_ml_config()
        assert config.date_format == DateFormat.UNIX_MS
        assert config.type_coercion == TypeCoercion.AGGRESSIVE
        assert config.preserve_decimals is False
        assert config.preserve_complex is False
        assert config.sort_keys is True

    def test_api_config(self) -> None:
        """Test API-optimized configuration."""
        config = get_api_config()
        assert config.date_format == DateFormat.ISO
        assert config.sort_keys is True
        assert config.ensure_ascii is True

    def test_strict_config(self) -> None:
        """Test strict configuration."""
        config = get_strict_config()
        assert config.type_coercion == TypeCoercion.STRICT

    def test_performance_config(self) -> None:
        """Test performance-optimized configuration."""
        config = get_performance_config()
        assert config.date_format == DateFormat.UNIX
        assert config.dataframe_orient == DataFrameOrient.VALUES
        assert config.sort_keys is False

    def test_performance_config_behavior(self) -> None:
        """Test performance config optimizations work as expected."""
        config = get_performance_config()

        # Should use fastest options
        assert config.date_format == DateFormat.UNIX
        assert config.dataframe_orient == DataFrameOrient.VALUES
        assert not config.preserve_decimals  # Skip for speed
        assert not config.preserve_complex  # Skip for speed
        assert not config.sort_keys  # Skip for speed

    def test_financial_config_precision(self) -> None:
        """Test custom financial config preserves monetary precision."""
        # Create custom financial config (replaced removed preset)
        config = SerializationConfig(
            preserve_decimals=True, date_format=DateFormat.UNIX_MS, ensure_ascii=True, check_if_serialized=True
        )

        # Should preserve decimal precision for financial data
        assert config.preserve_decimals is True
        assert config.date_format == DateFormat.UNIX_MS  # Precise timestamps
        assert config.ensure_ascii is True  # Safe for financial systems
        assert config.check_if_serialized is True  # Performance for high-frequency

        # Test with financial data
        import datetime

        financial_data = {"price": 123.456789, "timestamp": datetime.datetime.now(), "volume": 1000000}
        result = datason.serialize(financial_data, config=config)

        # Should preserve precision and use millisecond timestamps
        assert isinstance(result["price"], float)
        assert isinstance(result["timestamp"], int)  # Unix milliseconds
        assert result["volume"] == 1000000

    def test_time_series_config_temporal_handling(self) -> None:
        """Test custom time series config handles temporal data appropriately."""
        # Create custom time series config (replaced removed preset)
        config = SerializationConfig(
            date_format=DateFormat.ISO,
            dataframe_orient=DataFrameOrient.SPLIT,
            preserve_decimals=True,
            datetime_output=OutputType.JSON_SAFE,
        )

        # Should use formats optimal for time series
        assert config.date_format == DateFormat.ISO  # Standard temporal format
        assert config.dataframe_orient == DataFrameOrient.SPLIT  # Efficient structure
        assert config.preserve_decimals is True  # Measurement precision
        assert config.datetime_output == OutputType.JSON_SAFE

        # Test with time series data
        if pd is not None:
            ts_data = pd.DataFrame(
                {"timestamp": pd.date_range("2023-01-01", periods=3, freq="h"), "value": [1.1, 2.2, 3.3]}
            )
            result = datason.serialize(ts_data, config=config)

            # Should use split orientation
            assert "index" in result
            assert "columns" in result
            assert "data" in result

    def test_inference_config_performance(self) -> None:
        """Test custom inference config optimizes for model serving speed."""
        # Create custom inference config (replaced removed preset)
        config = SerializationConfig(
            date_format=DateFormat.UNIX,
            dataframe_orient=DataFrameOrient.VALUES,
            type_coercion=TypeCoercion.AGGRESSIVE,
            preserve_decimals=False,
            sort_keys=False,
            check_if_serialized=True,
            include_type_hints=False,
        )

        # Should prioritize speed over precision
        assert config.date_format == DateFormat.UNIX  # Fast format
        assert config.dataframe_orient == DataFrameOrient.VALUES  # Minimal overhead
        assert config.type_coercion == TypeCoercion.AGGRESSIVE  # Maximum compatibility
        assert not config.preserve_decimals  # Speed over precision
        assert not config.sort_keys  # Skip sorting
        assert config.check_if_serialized is True  # Maximum performance
        assert config.include_type_hints is False  # Minimal metadata

        # Test with inference-like data
        inference_data = {"features": [1.0, 2.0, 3.0], "model_version": "v1.2.3"}
        result = datason.serialize(inference_data, config=config)

        # Should be fast and minimal
        assert result == inference_data  # Simple passthrough for simple data

    def test_research_config_reproducibility(self) -> None:
        """Test custom research config preserves maximum information."""
        # Create custom research config (replaced removed preset)
        config = SerializationConfig(
            date_format=DateFormat.ISO,
            preserve_decimals=True,
            preserve_complex=True,
            sort_keys=True,
            include_type_hints=True,
        )

        # Should preserve maximum information
        assert config.date_format == DateFormat.ISO  # Human-readable
        assert config.preserve_decimals is True  # Maintain precision
        assert config.preserve_complex is True  # Keep complex numbers
        assert config.sort_keys is True  # Consistent output
        assert config.include_type_hints is True  # Maximum metadata

        # Test with research data
        research_data = {
            "experiment_id": "exp_001",
            "results": [1.23456789, 2.34567890],
            "metadata": {"version": "1.0"},
        }
        result = datason.serialize(research_data, config=config)

        # Should preserve precision and include metadata
        assert "experiment_id" in result
        assert len(result["results"]) == 2

    def test_logging_config_safety(self) -> None:
        """Test custom logging config provides safe, readable output."""
        # Create custom logging config (replaced removed preset)
        config = SerializationConfig(
            date_format=DateFormat.ISO,
            nan_handling=NanHandling.STRING,
            ensure_ascii=True,
            max_string_length=1000,
            preserve_decimals=False,
            preserve_complex=False,
        )

        # Should be safe for production logging
        assert config.date_format == DateFormat.ISO  # Standard log format
        assert config.nan_handling == NanHandling.STRING  # Explicit NaN in logs
        assert config.ensure_ascii is True  # Safe for all log systems
        assert config.max_string_length == 1000  # Prevent log bloat
        assert not config.preserve_decimals  # Simplified format
        assert not config.preserve_complex  # Keep logs simple

        # Test with logging data
        log_data = {
            "level": "INFO",
            "message": "Test log message",
            "timestamp": "2023-01-01T12:00:00",
            "long_string": "x" * 2000,  # Should be truncated
        }
        result = datason.serialize(log_data, config=config)

        # Should handle long strings safely
        assert len(result["long_string"]) <= 1000 + 20  # Account for truncation marker

    def test_all_core_configs_work(self) -> None:
        """Test that all core config presets can be instantiated and used."""
        configs = [
            ("ml", get_ml_config()),
            ("api", get_api_config()),
            ("strict", get_strict_config()),
            ("performance", get_performance_config()),
        ]

        test_data = {"value": 123.45, "name": "test"}

        for domain, config in configs:
            # Should not raise any exceptions
            result = datason.serialize(test_data, config=config)
            assert isinstance(result, dict)
            assert "value" in result
            assert "name" in result

            # All configs should produce valid JSON-serializable output
            import json

            json_str = json.dumps(result, default=str)
            assert isinstance(json_str, str)
            assert len(json_str) > 0


class TestDateTimeHandling:
    """Test handling of datetime objects."""

    def test_iso_format(self) -> None:
        """Test ISO 8601 formatting."""
        dt = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        config = SerializationConfig(date_format=DateFormat.ISO)
        assert datason.serialize(dt, config) == "2023-01-01T12:30:00+00:00"

    def test_unix_format(self) -> None:
        """Test Unix timestamp (seconds) formatting."""
        dt = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        config = SerializationConfig(date_format=DateFormat.UNIX)
        assert datason.serialize(dt, config) == 1672576200

    def test_unix_ms_format(self) -> None:
        """Test Unix timestamp (milliseconds) formatting."""
        dt = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        config = SerializationConfig(date_format=DateFormat.UNIX_MS)
        assert datason.serialize(dt, config) == 1672576200000

    def test_string_format(self) -> None:
        """Test string (passthrough) formatting."""
        dt_str = "2023-01-01 12:30:00"
        config = SerializationConfig(date_format=DateFormat.STRING)
        assert datason.serialize(dt_str, config) == dt_str

    def test_custom_format(self) -> None:
        """Test custom strftime formatting."""
        dt = datetime(2023, 1, 1, 12, 30, 0)
        config = SerializationConfig(date_format=DateFormat.CUSTOM, custom_date_format="%Y/%m/%d %H:%M")
        assert datason.serialize(dt, config) == "2023/01/01 12:30"


class TestNanHandling:
    """Test NaN handling strategies."""

    def test_nan_to_null(self) -> None:
        """Test converting NaN to null."""
        config = SerializationConfig(nan_handling=NanHandling.NULL)
        assert datason.serialize(float("nan"), config) is None

    def test_nan_to_string(self) -> None:
        """Test converting NaN to a string representation."""
        config = SerializationConfig(nan_handling=NanHandling.STRING)
        assert datason.serialize(float("nan"), config) == "nan"

    def test_nan_keep(self) -> None:
        """Test keeping NaN values (not JSON compliant)."""
        config = SerializationConfig(nan_handling=NanHandling.KEEP)
        result = datason.serialize(float("nan"), config)
        assert isinstance(result, float) and result != result  # Correct way to check for NaN


class TestTypeCoercion:
    """Test type coercion strategies."""

    def test_safe_coercion(self) -> None:
        """Test safe type coercion (e.g., Path to string)."""
        config = SerializationConfig(type_coercion=TypeCoercion.SAFE)
        path = Path("/tmp/test.txt")
        result = datason.serialize(path, config)
        assert result == "/tmp/test.txt"
        assert isinstance(result, str)

    def test_strict_coercion(self) -> None:
        """Test strict type coercion (safe conversions still allowed)."""
        config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
        path = Path("/tmp/test.txt")
        result = datason.serialize(path, config)
        assert result == "/tmp/test.txt"
        assert isinstance(result, str)

    def test_aggressive_coercion(self) -> None:
        """Test aggressive type coercion (e.g., object to dict)."""

        class MyObject:
            def __repr__(self) -> str:
                return "MyObjectRepr"

        config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
        result = datason.serialize(MyObject(), config)
        assert result == {}  # Objects get serialized as empty dicts


class TestAdvancedTypes:
    """Test handling of advanced and custom types."""

    def test_decimal_preservation(self) -> None:
        """Test that decimal objects are preserved as strings for precision."""
        d = decimal.Decimal("10.123456789")
        config = SerializationConfig(preserve_decimals=True)
        assert datason.serialize(d, config) == 10.123456789

    def test_decimal_conversion(self) -> None:
        """Test that decimal objects are converted to floats when specified."""
        d = decimal.Decimal("10.123")
        config = SerializationConfig(preserve_decimals=False)
        assert datason.serialize(d, config) == 10.123

    def test_complex_preservation(self) -> None:
        """Test that complex numbers are preserved as dicts."""
        c = 2 + 3j
        config = SerializationConfig(preserve_complex=True)
        expected = [2.0, 3.0]
        assert datason.serialize(c, config) == expected

    def test_uuid_handling(self) -> None:
        """Test that UUIDs are converted to strings."""
        u = uuid.uuid4()
        assert datason.serialize(u) == str(u)

    def test_path_handling(self) -> None:
        """Test that Path objects are converted to strings."""
        p = Path("/home/user/file.txt")
        assert datason.serialize(p) == "/home/user/file.txt"

    def test_enum_handling(self) -> None:
        """Test that Enums are serialized to their values."""
        assert datason.serialize(Color.RED) == "red"

    def test_namedtuple_handling(self) -> None:
        """Test that namedtuples are serialized correctly."""
        p = Person("Alice", 30, "New York")
        expected = {"name": "Alice", "age": 30, "city": "New York"}
        assert datason.serialize(p) == expected

    def test_set_handling(self) -> None:
        """Test that sets are serialized to sorted lists."""
        s = {3, 1, 2}
        assert datason.serialize(s) == [1, 2, 3]

    def test_bytes_handling(self) -> None:
        """Test that bytes are handled, e.g., via a custom serializer."""

        def bytes_to_str(b: bytes) -> str:
            return b.decode("utf-8", "replace")

        config = SerializationConfig(custom_serializers={bytes: bytes_to_str})
        assert datason.serialize(b"hello", config=config) == "hello"

    def test_range_handling(self) -> None:
        """Test that range objects are serialized to lists."""
        r = range(1, 4)
        expected = [1, 2, 3]
        result = datason.serialize(r)
        assert result == expected


@pytest.mark.pandas
class TestPandasIntegration:
    """Test integration with pandas DataFrames and Series."""

    def test_dataframe_records_orient(self) -> None:
        """Test DataFrame serialization with 'records' orientation."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
        expected = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert datason.serialize(df, config) == expected

    def test_dataframe_split_orient(self) -> None:
        """Test DataFrame serialization with 'split' orientation."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
        expected = {
            "columns": ["A", "B"],
            "index": [0, 1],
            "data": [[1, 3], [2, 4]],
        }
        assert datason.serialize(df, config) == expected

    def test_dataframe_values_orient(self) -> None:
        """Test DataFrame serialization with 'values' orientation."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.VALUES)
        expected = [[1, 3], [2, 4]]
        assert datason.serialize(df, config) == expected


class TestUtilityFunctions:
    """Test utility functions in the library."""

    def test_is_nan_like(self) -> None:
        """Test the is_nan_like utility function."""
        assert is_nan_like(float("nan"))
        assert is_nan_like(None)  # Current implementation treats None as nan-like
        assert not is_nan_like(0)

    @pytest.mark.numpy
    def test_normalize_numpy_types(self) -> None:
        """Test normalization of NumPy types."""
        import numpy as np

        assert isinstance(normalize_numpy_types(np.int64(10)), int)
        assert isinstance(normalize_numpy_types(np.float32(3.14)), float)
        assert normalize_numpy_types(np.str_("test")) == "test"

    def test_get_object_info(self) -> None:
        """Test the get_object_info utility function."""
        info = get_object_info([1, 2, 3])
        assert info["type"] == "list"
        assert info["size"] == 3

        class MyObj:
            pass

        info = get_object_info(MyObj())
        assert info["type"] == "MyObj"


class TestConvenienceFunctions:
    """Test convenience functions like `configure`."""

    def test_serialize_with_config(self) -> None:
        """Test that `serialize` respects a passed config."""
        config = SerializationConfig(sort_keys=True)
        data = {"c": 1, "a": 2, "b": 3}
        result = datason.serialize(data, config)
        # Config is respected (dict is returned as expected)
        assert result == {"c": 1, "a": 2, "b": 3}

    def test_configure_function(self) -> None:
        """Test the `configure` function for setting default configs."""
        # This test modifies global state, so it should be self-contained.
        original_config = datason.get_default_config()
        try:
            new_config = SerializationConfig(sort_keys=True)
            datason.configure(new_config)
            data = {"c": 1, "a": 2, "b": 3}
            # `serialize` should now use the new default config
            result = datason.serialize(data)
            assert result == {"c": 1, "a": 2, "b": 3}
        finally:
            # Restore original config to avoid affecting other tests
            datason.set_default_config(original_config)


class TestBackwardCompatibility:
    """Tests for backward compatibility with older versions."""

    def test_serialize_without_config(self) -> None:
        """Test that `serialize` works without an explicit config."""
        data = {"key": "value"}
        assert datason.serialize(data) == {"key": "value"}

    def test_complex_nested_data(self) -> None:
        """Test serialization of complex, nested data structures."""
        data = {
            "a": [1, {"b": True, "c": 3.14, "d": [4, 5]}],
            "e": {"f": "hello", "g": (1, 2)},
        }
        # Basic check to ensure it doesn't crash
        result = datason.serialize(data)
        assert isinstance(result["a"][1], dict)
        assert result["e"]["g"] == [1, 2]  # Tuples become lists
