"""Tests for configuration system and enhanced type handling."""

import decimal
import enum
import uuid
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    TypeHandler,
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
    """Test configurable date/time serialization."""

    def test_iso_format(self):
        """Test ISO format (default)."""
        dt = datetime(2023, 12, 25, 10, 30, 45)
        config = SerializationConfig(date_format=DateFormat.ISO)
        result = datason.serialize(dt, config=config)
        assert result == "2023-12-25T10:30:45"

    def test_unix_format(self):
        """Test Unix timestamp format."""
        dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        config = SerializationConfig(date_format=DateFormat.UNIX)
        result = datason.serialize(dt, config=config)
        assert result == 1672531200.0

    def test_unix_ms_format(self):
        """Test Unix timestamp in milliseconds."""
        dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        config = SerializationConfig(date_format=DateFormat.UNIX_MS)
        result = datason.serialize(dt, config=config)
        assert result == 1672531200000

    def test_string_format(self):
        """Test string format."""
        dt = datetime(2023, 12, 25, 10, 30, 45)
        config = SerializationConfig(date_format=DateFormat.STRING)
        result = datason.serialize(dt, config=config)
        assert isinstance(result, str)
        assert "2023" in result

    def test_custom_format(self):
        """Test custom strftime format."""
        dt = datetime(2023, 12, 25, 10, 30, 45)
        config = SerializationConfig(date_format=DateFormat.CUSTOM, custom_date_format="%Y-%m-%d")
        result = datason.serialize(dt, config=config)
        assert result == "2023-12-25"


class TestNanHandling:
    """Test NaN value handling options."""

    def test_nan_to_null(self):
        """Test converting NaN to null."""
        data = [1, float("nan"), 3]
        config = SerializationConfig(nan_handling=NanHandling.NULL)
        result = datason.serialize(data, config=config)
        assert result == [1, None, 3]

    def test_nan_to_string(self):
        """Test converting NaN to string."""
        data = float("nan")
        config = SerializationConfig(nan_handling=NanHandling.STRING)
        result = datason.serialize(data, config=config)
        assert isinstance(result, str)

    def test_nan_keep(self):
        """Test keeping NaN as-is."""
        data = float("nan")
        config = SerializationConfig(nan_handling=NanHandling.KEEP)
        result = datason.serialize(data, config=config)
        assert result != result  # NaN != NaN


class TestTypeCoercion:
    """Test type coercion strategies."""

    def test_safe_coercion(self):
        """Test safe type coercion."""
        config = SerializationConfig(type_coercion=TypeCoercion.SAFE)
        handler = TypeHandler(config)

        # UUID should convert to string
        test_uuid = uuid.uuid4()
        result = handler.handle_uuid(test_uuid)
        assert isinstance(result, str)

    def test_strict_coercion(self):
        """Test strict type coercion."""
        config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
        handler = TypeHandler(config)

        # UUID should preserve details
        test_uuid = uuid.uuid4()
        result = handler.handle_uuid(test_uuid)
        assert isinstance(result, dict)
        assert result["_type"] == "uuid"
        assert "hex" in result

    def test_aggressive_coercion(self):
        """Test aggressive type coercion."""
        config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
        handler = TypeHandler(config)

        # Complex should convert to list
        result = handler.handle_complex(3 + 4j)
        assert result == [3.0, 4.0]


class TestAdvancedTypes:
    """Test handling of advanced Python types."""

    def test_decimal_preservation(self):
        """Test decimal preservation."""
        config = SerializationConfig(preserve_decimals=True)
        result = datason.serialize(decimal.Decimal("123.456"), config=config)
        assert isinstance(result, dict)
        assert result["_type"] == "decimal"
        assert result["value"] == "123.456"

    def test_decimal_conversion(self):
        """Test decimal conversion to float."""
        config = SerializationConfig(preserve_decimals=False)
        result = datason.serialize(decimal.Decimal("123.456"), config=config)
        assert isinstance(result, float)
        assert abs(result - 123.456) < 0.001

    def test_complex_preservation(self):
        """Test complex number preservation."""
        config = SerializationConfig(preserve_complex=True)
        result = datason.serialize(3 + 4j, config=config)
        assert isinstance(result, dict)
        assert result["_type"] == "complex"
        assert result["real"] == 3.0
        assert result["imag"] == 4.0

    def test_uuid_handling(self):
        """Test UUID handling."""
        test_uuid = uuid.uuid4()
        result = datason.serialize(test_uuid)
        assert isinstance(result, str)
        assert str(test_uuid) == result

    def test_path_handling(self):
        """Test pathlib.Path handling."""
        test_path = Path("/home/user/file.txt")
        result = datason.serialize(test_path)
        assert isinstance(result, str)
        assert result == str(test_path)

    def test_enum_handling(self):
        """Test enum handling."""
        result = datason.serialize(Color.RED)
        assert result == "red"

    def test_namedtuple_handling(self):
        """Test namedtuple handling."""
        person = Person("Alice", 30, "New York")
        result = datason.serialize(person)
        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["city"] == "New York"

    def test_set_handling(self):
        """Test set handling."""
        test_set = {3, 1, 4, 5}
        result = datason.serialize(test_set)
        assert isinstance(result, list)
        assert set(result) == {1, 3, 4, 5}  # Duplicates removed

    def test_bytes_handling(self):
        """Test bytes handling."""
        test_bytes = b"hello world"
        result = datason.serialize(test_bytes)
        assert result == "hello world"  # UTF-8 decodable

        # Test non-UTF-8 bytes
        test_bytes = b"\x80\x81\x82"
        result = datason.serialize(test_bytes)
        assert isinstance(result, str)  # Should be hex representation

    def test_range_handling(self):
        """Test range handling."""
        # Small range should expand
        result = datason.serialize(range(5))
        assert result == [0, 1, 2, 3, 4]

        # Large range should preserve structure
        large_range = range(0, 10000, 2)
        result = datason.serialize(large_range)
        assert isinstance(result, dict)
        assert result["_type"] == "range"


class TestPandasIntegration:
    """Test pandas DataFrame orientation options."""

    @pytest.mark.pandas
    def test_dataframe_records_orient(self):
        """Test DataFrame records orientation."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
        result = datason.serialize(df, config=config)
        expected = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert result == expected

    @pytest.mark.pandas
    def test_dataframe_split_orient(self):
        """Test DataFrame split orientation."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
        result = datason.serialize(df, config=config)
        assert "data" in result
        assert "columns" in result
        assert "index" in result

    @pytest.mark.pandas
    def test_dataframe_values_orient(self):
        """Test DataFrame values orientation."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        config = SerializationConfig(dataframe_orient=DataFrameOrient.VALUES)
        result = datason.serialize(df, config=config)
        assert result == [[1, 3], [2, 4]]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_nan_like(self):
        """Test NaN detection."""
        assert is_nan_like(None)
        assert is_nan_like(float("nan"))
        assert not is_nan_like(42)
        assert not is_nan_like("hello")

    @pytest.mark.numpy
    def test_normalize_numpy_types(self):
        """Test numpy type normalization."""
        np = pytest.importorskip("numpy")

        # Test various numpy types
        assert normalize_numpy_types(np.bool_(True)) is True
        assert normalize_numpy_types(np.int64(42)) == 42
        assert normalize_numpy_types(np.float64(3.14)) == 3.14
        assert normalize_numpy_types(np.str_("hello")) == "hello"

        # Test NaN handling
        assert normalize_numpy_types(np.float64("nan")) is None

    def test_get_object_info(self):
        """Test object information utility."""
        info = get_object_info([1, 2, 3])
        assert info["type"] == "list"
        assert info["size"] == 3
        assert "int" in info["sample_types"]

        info = get_object_info({"a": 1, "b": 2})
        assert info["type"] == "dict"
        assert info["size"] == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_serialize_with_config(self):
        """Test serialize_with_config function."""
        dt = datetime(2023, 1, 1)
        result = datason.serialize_with_config(dt, date_format="unix")
        assert isinstance(result, float)

    def test_configure_function(self):
        """Test global configuration."""
        # Save original config
        original = datason.get_default_config()

        try:
            # Set ML config
            datason.configure(get_ml_config())

            # Test that it's applied
            dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
            result = datason.serialize(dt)
            assert isinstance(result, int)  # Should be unix_ms format
        finally:
            # Restore original config
            datason.set_default_config(original)


class TestBackwardCompatibility:
    """Test that new features don't break existing code."""

    def test_serialize_without_config(self):
        """Test that serialize works without config parameter."""
        data = {"name": "test", "value": 42}
        result = datason.serialize(data)
        assert result == data

    def test_complex_nested_data(self):
        """Test complex nested data still works."""
        data = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "datetime": datetime.now(),
            "uuid": uuid.uuid4(),
            "set": {1, 2, 3},
        }
        result = datason.serialize(data)
        assert isinstance(result, dict)
        assert isinstance(result["list"], list)
        assert isinstance(result["dict"], dict)
        assert isinstance(result["datetime"], str)
        assert isinstance(result["uuid"], str)
        assert isinstance(result["set"], list)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_custom_serializer_failure(self):
        """Test graceful handling of custom serializer failures."""

        def failing_serializer(_obj: Any) -> None:  # Use _obj to indicate unused
            raise ValueError("Custom serializer failed")

        config = SerializationConfig(custom_serializers={str: failing_serializer})

        # Should fall back to default handling
        result = datason.serialize("test", config=config)
        assert result == "test"

    def test_security_limits(self):
        """Test that security limits are respected."""
        config = SerializationConfig(max_depth=0)  # Very low limit

        # Create deeply nested structure that should definitely exceed max_depth=0
        data = {"level1": {"level2": "too deep"}}

        with pytest.raises(datason.SecurityError):
            datason.serialize(data, config=config)

    def test_large_object_limits(self):
        """Test size limits."""
        config = SerializationConfig(max_size=10)

        # Create large list
        large_list = list(range(20))

        with pytest.raises(datason.SecurityError):
            datason.serialize(large_list, config=config)
