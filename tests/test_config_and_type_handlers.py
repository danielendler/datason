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

# Test data setup
Person = namedtuple("Person", ["name", "age", "city"])


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestSerializationConfig:
    """Test configuration classes and presets."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SerializationConfig()
        assert config.date_format == DateFormat.ISO
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.SAFE
        assert config.preserve_decimals is True
        assert config.preserve_complex is True
        assert config.max_depth == 1000
        assert config.sort_keys is False

    def test_ml_config(self):
        """Test ML-optimized configuration."""
        config = get_ml_config()
        assert config.date_format == DateFormat.UNIX_MS
        assert config.type_coercion == TypeCoercion.AGGRESSIVE
        assert config.preserve_decimals is False
        assert config.preserve_complex is False
        assert config.sort_keys is True

    def test_api_config(self):
        """Test API-optimized configuration."""
        config = get_api_config()
        assert config.date_format == DateFormat.ISO
        assert config.sort_keys is True
        assert config.ensure_ascii is True

    def test_strict_config(self):
        """Test strict configuration."""
        config = get_strict_config()
        assert config.type_coercion == TypeCoercion.STRICT

    def test_performance_config(self):
        """Test performance-optimized configuration."""
        config = get_performance_config()
        assert config.date_format == DateFormat.UNIX
        assert config.dataframe_orient == DataFrameOrient.VALUES
        assert config.sort_keys is False


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
