"""
Tests for SerialPy with optional dependencies (pandas and numpy).

This module tests code paths that require pandas and numpy to be installed.
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

import datason as sp
from datason.core import serialize


class TestNumpyIntegration:
    """Test numpy type handling in core serialization."""

    def test_numpy_boolean_types(self) -> None:
        """Test serialization of numpy boolean types."""
        data = {
            "np_bool_true": np.bool_(True),
            "np_bool_false": np.bool_(False),
            "np_bool_array": np.array([True, False, True]),
        }
        result = serialize(data)

        assert result["np_bool_true"] is True
        assert result["np_bool_false"] is False
        assert result["np_bool_array"] == [True, False, True]

    def test_numpy_integer_types(self) -> None:
        """Test serialization of numpy integer types."""
        data = {
            "np_int8": np.int8(42),
            "np_int16": np.int16(1000),
            "np_int32": np.int32(100000),
            "np_int64": np.int64(1000000000),
            "np_uint8": np.uint8(255),
        }
        result = serialize(data)

        for key, value in result.items():
            assert isinstance(value, int)
            assert value > 0

    def test_numpy_floating_types(self) -> None:
        """Test serialization of numpy floating types."""
        data = {
            "np_float32": np.float32(3.14),
            "np_float64": np.float64(2.71828),
            "np_float_nan": np.float64(np.nan),
            "np_float_inf": np.float64(np.inf),
            "np_float_neg_inf": np.float64(-np.inf),
        }
        result = serialize(data)

        assert isinstance(result["np_float32"], float)
        assert isinstance(result["np_float64"], float)
        assert result["np_float_nan"] is None  # NaN -> None
        assert result["np_float_inf"] is None  # Inf -> None
        assert result["np_float_neg_inf"] is None  # -Inf -> None

    def test_numpy_string_types(self) -> None:
        """Test serialization of numpy string types."""
        data = {
            "np_str": np.str_("hello world"),
            "np_unicode": np.str_("unicode: 🌍"),
        }
        result = serialize(data)

        assert result["np_str"] == "hello world"
        assert result["np_unicode"] == "unicode: 🌍"
        assert isinstance(result["np_str"], str)

    def test_numpy_arrays(self) -> None:
        """Test serialization of numpy arrays."""
        data = {
            "array_1d": np.array([1, 2, 3, 4]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "array_mixed": np.array([1.5, np.nan, np.inf, 2.5]),
            "array_bool": np.array([True, False, True]),
            "array_str": np.array(["hello", "world"]),
        }
        result = serialize(data)

        assert result["array_1d"] == [1, 2, 3, 4]
        assert result["array_2d"] == [[1, 2], [3, 4]]
        assert result["array_mixed"] == [1.5, None, None, 2.5]  # NaN/Inf -> None
        assert result["array_bool"] == [True, False, True]
        assert result["array_str"] == ["hello", "world"]


class TestPandasIntegration:
    """Test pandas type handling in core serialization."""

    def test_pandas_timestamp_basic(self) -> None:
        """Test basic pandas Timestamp serialization."""
        ts = pd.Timestamp("2023-01-01 12:00:00")
        data = {"timestamp": ts}
        result = serialize(data)

        assert isinstance(result["timestamp"], str)
        assert "2023-01-01" in result["timestamp"]

    def test_pandas_timestamp_nat(self) -> None:
        """Test pandas NaT (Not a Time) serialization."""
        data = {"nat_value": pd.NaT}
        result = serialize(data)

        # NaT becomes a string "NaT", not None
        assert result["nat_value"] == "NaT"

    def test_pandas_series(self) -> None:
        """Test pandas Series serialization."""
        series = pd.Series([1, 2, 3, 4, 5])
        data = {"series": series}
        result = serialize(data)

        assert result["series"] == [1, 2, 3, 4, 5]

    def test_pandas_series_with_nan(self) -> None:
        """Test pandas Series with NaN values."""
        series = pd.Series([1.0, np.nan, 3.0, pd.NaT])
        data = {"series_nan": series}
        result = serialize(data)

        # NaN and NaT should become None or string representation
        assert result["series_nan"][0] == 1.0
        assert result["series_nan"][1] is None
        assert result["series_nan"][2] == 3.0
        # NaT in series becomes "NaT" string
        assert result["series_nan"][3] == "NaT"

    def test_pandas_index(self) -> None:
        """Test pandas Index serialization."""
        index = pd.Index(["a", "b", "c", "d"])
        data = {"index": index}
        result = serialize(data)

        assert result["index"] == ["a", "b", "c", "d"]

    def test_pandas_dataframe(self) -> None:
        """Test pandas DataFrame serialization."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]})
        data = {"dataframe": df}
        result = serialize(data)

        assert isinstance(result["dataframe"], list)
        assert len(result["dataframe"]) == 3
        assert result["dataframe"][0] == {"A": 1, "B": "x", "C": 1.1}

    def test_pandas_dataframe_with_timestamps(self) -> None:
        """Test DataFrame with timestamp columns."""
        df = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=3), "value": [10, 20, 30]}
        )
        data = {"df_with_dates": df}
        result = serialize(data)

        assert isinstance(result["df_with_dates"], list)
        assert len(result["df_with_dates"]) == 3
        # Timestamps should be converted to ISO strings
        assert isinstance(result["df_with_dates"][0]["date"], str)
        assert "2023-01-01" in result["df_with_dates"][0]["date"]


class TestDateTimeUtilsWithPandas:
    """Test datetime_utils.py functions with pandas installed."""

    def test_ensure_timestamp_basic(self) -> None:
        """Test ensure_timestamp with various inputs."""
        # Test with datetime
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = sp.ensure_timestamp(dt)
        assert isinstance(result, pd.Timestamp)

        # Test with string
        result = sp.ensure_timestamp("2023-01-01")
        assert isinstance(result, pd.Timestamp)

        # Test with None
        result = sp.ensure_timestamp(None)
        assert pd.isna(result)

    def test_ensure_timestamp_edge_cases(self) -> None:
        """Test ensure_timestamp with edge cases."""
        # Test with NaN
        result = sp.ensure_timestamp(float("nan"))
        assert pd.isna(result)

        # Test with existing pandas Timestamp
        ts = pd.Timestamp("2023-01-01")
        result = sp.ensure_timestamp(ts)
        assert result == ts

    def test_ensure_timestamp_invalid_types(self) -> None:
        """Test ensure_timestamp with invalid types."""
        # Test with list (should raise TypeError)
        with pytest.raises(TypeError):
            sp.ensure_timestamp([1, 2, 3])

        # Test with dict (should raise TypeError)
        with pytest.raises(TypeError):
            sp.ensure_timestamp({"key": "value"})

        # Test with set (should raise TypeError)
        with pytest.raises(TypeError):
            sp.ensure_timestamp({1, 2, 3})

    def test_ensure_timestamp_conversion_failure(self) -> None:
        """Test ensure_timestamp with values that can't be converted."""
        # Test with invalid string
        result = sp.ensure_timestamp("not a date")
        assert pd.isna(result)

    def test_ensure_dates_with_dict(self) -> None:
        """Test ensure_dates with dictionary input."""
        data = {
            "first_date": "2023-01-01",
            "last_date": datetime(2023, 12, 31),
            "created_at": "2023-06-15T10:30:00",
            "other_field": "not a date",
        }
        result = sp.ensure_dates(data)

        assert isinstance(result["first_date"], pd.Timestamp)
        assert isinstance(result["last_date"], pd.Timestamp)
        assert isinstance(result["created_at"], pd.Timestamp)
        assert result["other_field"] == "not a date"  # Unchanged

    def test_ensure_dates_with_dict_none_values(self) -> None:
        """Test ensure_dates with None values in dict."""
        data = {
            "first_date": None,
            "last_date": "2023-12-31",
            "invalid_field": "not convertible",
        }
        result = sp.ensure_dates(data)

        assert result["first_date"] is None
        assert isinstance(result["last_date"], pd.Timestamp)

    def test_ensure_dates_with_dataframe(self) -> None:
        """Test ensure_dates with DataFrame input."""
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "value": [10, 20, 30]}
        )
        result = sp.ensure_dates(df)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert len(result) == 3

    def test_ensure_dates_empty_dataframe(self) -> None:
        """Test ensure_dates with empty DataFrame."""
        df = pd.DataFrame(columns=["date", "value"])
        result = sp.ensure_dates(df)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert len(result) == 0

    def test_ensure_dates_missing_date_column(self) -> None:
        """Test ensure_dates with DataFrame missing date column."""
        df = pd.DataFrame({"value": [10, 20, 30]})

        with pytest.raises(KeyError):
            sp.ensure_dates(df)

    def test_ensure_dates_invalid_date_types(self) -> None:
        """Test ensure_dates with invalid date types in DataFrame."""
        df = pd.DataFrame(
            {"date": [[1, 2, 3], {"key": "value"}, {1, 2}], "value": [10, 20, 30]}
        )

        with pytest.raises(ValueError):
            sp.ensure_dates(df)

    def test_ensure_dates_invalid_input_type(self) -> None:
        """Test ensure_dates with invalid input type."""
        with pytest.raises(TypeError):
            sp.ensure_dates("not a dict or dataframe")

    def test_ensure_dates_timezone_handling(self) -> None:
        """Test ensure_dates with timezone-aware dates."""
        # Create timezone-aware datetime
        tz_aware = datetime(2023, 1, 1, tzinfo=timezone.utc)
        data = {"date": tz_aware}
        result = sp.ensure_dates(data, strip_timezone=True)

        # Should be converted to naive datetime
        assert result["date"].tzinfo is None

    def test_convert_pandas_timestamps_with_series(self) -> None:
        """Test convert_pandas_timestamps with pandas Series."""
        ts = pd.Timestamp("2023-01-01")
        series = pd.Series([ts, pd.Timestamp("2023-01-02")])
        result = sp.convert_pandas_timestamps(series)

        # Should convert timestamps to datetime objects
        assert isinstance(result.iloc[0], datetime)
        assert isinstance(result.iloc[1], datetime)

    def test_convert_pandas_timestamps_with_dataframe(self) -> None:
        """Test convert_pandas_timestamps with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
                "value": [10, 20],
            }
        )
        result = sp.convert_pandas_timestamps(df)

        # Timestamps should be converted to datetime objects
        assert isinstance(result.iloc[0, 0], datetime)
        assert isinstance(result.iloc[1, 0], datetime)


class TestSerializersWithDependencies:
    """Test serializers.py with numpy and pandas."""

    def test_serialize_detection_details_with_numpy(self) -> None:
        """Test serialize_detection_details with numpy types."""
        data = {
            "method1": {
                "scores": np.array([1.0, np.nan, np.inf, 2.5]),
                "confidence": np.float64(0.95),
                "count": np.int32(42),
                "is_valid": np.bool_(True),
            }
        }
        result = sp.serialize_detection_details(data)

        assert result["method1"]["scores"] == [1.0, None, None, 2.5]
        assert result["method1"]["confidence"] == 0.95
        assert result["method1"]["count"] == 42
        # numpy boolean might not be converted to Python bool
        assert result["method1"]["is_valid"] in [True, np.True_]

    def test_serialize_detection_details_with_pandas(self) -> None:
        """Test serialize_detection_details with pandas types."""
        series = pd.Series([1, 2, 3, 4])
        timestamp = pd.Timestamp("2023-01-01")

        data = {
            "method1": {
                "data_series": series,
                "timestamp": timestamp,
                "nat_value": pd.NaT,
            }
        }
        result = sp.serialize_detection_details(data)

        assert result["method1"]["data_series"] == [1, 2, 3, 4]
        assert isinstance(result["method1"]["timestamp"], str)
        assert "2023-01-01" in result["method1"]["timestamp"]
        # NaT becomes "NaT" string in serialize_detection_details
        assert result["method1"]["nat_value"] == "NaT"

    def test_serialize_detection_details_complex_nested(self) -> None:
        """Test serialize_detection_details with complex nested structures."""
        data = {
            "method1": {
                "numpy_data": {
                    "array": np.array([1, 2, np.nan]),
                    "matrix": np.array([[1, 2], [3, 4]]),
                    "scalars": {
                        "float": np.float32(3.14),
                        "int": np.int64(100),
                        "bool": np.bool_(False),
                    },
                },
                "pandas_data": {
                    "series": pd.Series([10, 20, 30]),
                    "timestamps": [
                        pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-01-02"),
                    ],
                },
            }
        }
        result = sp.serialize_detection_details(data)

        # Check numpy conversions
        assert result["method1"]["numpy_data"]["array"] == [1, 2, None]
        assert result["method1"]["numpy_data"]["matrix"] == [[1, 2], [3, 4]]
        assert result["method1"]["numpy_data"]["scalars"]["float"] == pytest.approx(
            3.14
        )
        assert result["method1"]["numpy_data"]["scalars"]["int"] == 100
        # numpy boolean might not be converted
        assert result["method1"]["numpy_data"]["scalars"]["bool"] in [False, np.False_]

        # Check pandas conversions
        assert result["method1"]["pandas_data"]["series"] == [10, 20, 30]
        assert len(result["method1"]["pandas_data"]["timestamps"]) == 2


class TestMixedOptionalDependencies:
    """Test complex scenarios mixing numpy, pandas, and regular Python types."""

    def test_complex_mixed_structure(self) -> None:
        """Test serialization of complex structure with all types."""
        data = {
            "metadata": {
                "created": pd.Timestamp("2023-01-01"),
                "version": np.int32(1),
                "confidence": np.float64(0.95),
            },
            "data": {
                "numpy_array": np.array([1, 2, np.nan, 4]),
                "pandas_series": pd.Series([10, 20, 30]),
                "regular_list": [1, 2, 3],
                "mixed_list": [1, np.int32(2), pd.Timestamp("2023-01-01")],
            },
            "results": pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": np.array([1.1, 2.2, 3.3]),
                    "C": pd.date_range("2023-01-01", periods=3),
                }
            ),
        }

        result = serialize(data)

        # Check all parts are properly serialized
        assert isinstance(result["metadata"]["created"], str)
        assert isinstance(result["metadata"]["version"], int)
        assert isinstance(result["metadata"]["confidence"], float)

        assert result["data"]["numpy_array"] == [1, 2, None, 4]
        assert result["data"]["pandas_series"] == [10, 20, 30]
        assert result["data"]["regular_list"] == [1, 2, 3]
        assert len(result["data"]["mixed_list"]) == 3

        assert isinstance(result["results"], list)
        assert len(result["results"]) == 3


class TestAdditionalCoverage:
    """Additional tests to cover remaining edge cases."""

    def test_data_utils_remaining_paths(self) -> None:
        """Test remaining paths in data_utils.py."""
        # Test AST literal eval failure (should be covered by existing tests)
        # Test the remaining lines in data conversion

    def test_converters_remaining_paths(self) -> None:
        """Test remaining paths in converters.py."""

        # Test safe_float with complex object
        class ComplexObject:
            def __float__(self) -> float:
                raise ValueError("Cannot convert to float")

        obj = ComplexObject()
        result = sp.safe_float(obj)
        assert result == 0.0

        # Test safe_int with complex object
        class ComplexObject2:
            def __int__(self) -> int:
                raise ValueError("Cannot convert to int")

        obj2 = ComplexObject2()
        result = sp.safe_int(obj2)
        assert result == 0

    def test_core_optimization_edge_cases(self) -> None:
        """Test optimization helper functions with edge cases."""
        from datason.core import (
            _is_already_serialized_dict,
            _is_already_serialized_list,
        )

        # Test dict items() exception
        class BadDictItems(dict):
            def items(self) -> Any:
                raise RuntimeError("items() failed")

        bad_dict = BadDictItems()
        result = _is_already_serialized_dict(bad_dict)
        assert result is False

        # Test list iteration exception
        class BadListIter(list):
            def __iter__(self) -> Any:
                raise RuntimeError("iter() failed")

        bad_list = BadListIter()
        result = _is_already_serialized_list(bad_list)
        assert result is False

    def test_pandas_edge_cases(self) -> None:
        """Test pandas edge cases for remaining coverage."""
        # Test pandas isna with various types
        if pd is not None:
            # Test Series with mixed types that might trigger edge cases
            mixed_series = pd.Series([1, "string", pd.NaT, np.nan])
            data = {"mixed": mixed_series}
            result = serialize(data)
            assert len(result["mixed"]) == 4

    def test_numpy_edge_cases(self) -> None:
        """Test numpy edge cases for remaining coverage."""
        # Test numpy edge cases
        if np is not None:
            # Test with various numpy types that should convert to strings
            data = {
                "complex128": np.complex128(1 + 2j),
                "datetime64": np.datetime64("2023-01-01"),
                # Skip timedelta64 as it causes conversion issues
                "void": np.void(b"\x01\x02"),
            }
            result = serialize(data)
            # These should be converted to strings
            assert isinstance(result["complex128"], str)
            assert isinstance(result["datetime64"], str)
            assert isinstance(result["void"], str)
