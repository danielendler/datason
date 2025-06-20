"""Enhanced coverage tests for datason/datetime_utils.py module.

This test suite targets the specific missing coverage areas in datetime_utils.py to boost
coverage from 65% to 80%+. Focuses on date parsing edge cases, timezone handling,
error paths, and pandas type conversions.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# Try to import pandas for conditional tests
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

from datason.datetime_utils import (
    convert_pandas_timestamps,
    convert_pandas_timestamps_recursive,
    ensure_dates,
    ensure_timestamp,
    serialize_datetimes,
)


class TestConvertPandasTimestampsEdgeCases:
    """Test edge cases and missing lines in convert_pandas_timestamps."""

    def test_pandas_series_conversion_edge_cases(self):
        """Test lines 29 and 44-56 - pandas Series handling edge cases."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Test Series with mixed types containing nested structures
        complex_data = [
            pd.Timestamp("2023-01-01"),
            {"nested": pd.Timestamp("2023-01-02")},
            [pd.Timestamp("2023-01-03")],
            "regular_string",
            42,
        ]
        series = pd.Series(complex_data)

        result = convert_pandas_timestamps(series)

        # Verify Series was processed and nested structures handled
        assert isinstance(result, pd.Series)
        # First item should be converted from Timestamp to datetime
        assert isinstance(result.iloc[0], datetime)

    def test_pandas_dataframe_conversion_edge_cases(self):
        """Test lines 32-33 - pandas DataFrame handling edge cases."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Test DataFrame with mixed types containing nested structures
        df = pd.DataFrame(
            {
                "timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
                "nested_dicts": [{"date": pd.Timestamp("2023-01-03")}, {"date": pd.Timestamp("2023-01-04")}],
                "nested_lists": [[pd.Timestamp("2023-01-05")], [pd.Timestamp("2023-01-06")]],
                "regular": ["string1", "string2"],
            }
        )

        result = convert_pandas_timestamps(df)

        # Verify DataFrame was processed and nested structures handled
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result["timestamps"].iloc[0], datetime)

    def test_pandas_timestamp_to_pydatetime_conversion(self):
        """Test lines 36-37 - pd.Timestamp to pydatetime conversion."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Test direct pd.Timestamp conversion
        timestamp = pd.Timestamp("2023-01-01 12:30:45")
        result = convert_pandas_timestamps(timestamp)

        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45

    def test_datetime_passthrough(self):
        """Test lines 44-45 - datetime objects pass through unchanged."""
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        result = convert_pandas_timestamps(test_datetime)

        # Should return the same datetime object
        assert result is test_datetime
        assert isinstance(result, datetime)

    def test_other_types_passthrough(self):
        """Test lines 47-48 - other types pass through unchanged."""
        test_cases = ["string", 42, 3.14, None, {"non_timestamp": "data"}, [1, 2, 3], {1, 2, 3}]

        for test_input in test_cases:
            result = convert_pandas_timestamps(test_input)
            assert result == test_input

    def test_recursive_alias_function(self):
        """Test lines 51-65 - convert_pandas_timestamps_recursive alias."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        test_data = {
            "timestamp": pd.Timestamp("2023-01-01"),
            "nested": {"another_timestamp": pd.Timestamp("2023-01-02")},
        }

        result = convert_pandas_timestamps_recursive(test_data)

        # Verify it works the same as the main function
        assert isinstance(result["timestamp"], datetime)
        assert isinstance(result["nested"]["another_timestamp"], datetime)


class TestEnsureTimestampAdvancedEdgeCases:
    """Test advanced edge cases for ensure_timestamp function."""

    def test_ensure_timestamp_none_and_float_nan(self):
        """Test lines 82-83 - None and float NaN handling."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Test None
        result = ensure_timestamp(None)
        assert pd.isna(result)

        # Test float NaN
        result = ensure_timestamp(float("nan"))
        assert pd.isna(result)

    def test_ensure_timestamp_existing_timestamp(self):
        """Test lines 84-85 - existing pd.Timestamp passthrough."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        original_timestamp = pd.Timestamp("2023-01-01")
        result = ensure_timestamp(original_timestamp)

        assert result is original_timestamp

    def test_ensure_timestamp_invalid_types_with_logging(self):
        """Test lines 86-88 - invalid types raise TypeError with logging."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        invalid_inputs = [
            [1, 2, 3],  # list
            {"key": "value"},  # dict
            {1, 2, 3},  # set
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(TypeError, match="Cannot convert type"):
                ensure_timestamp(invalid_input)

    def test_ensure_timestamp_conversion_exception_with_logging(self):
        """Test exception handling during pd.to_datetime conversion."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Create an object that will cause pd.to_datetime to fail
        class ProblematicObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        problematic_obj = ProblematicObject()

        # Should return NaT when conversion fails
        result = ensure_timestamp(problematic_obj)
        assert pd.isna(result)


class TestEnsureDatesAdvancedEdgeCases:
    """Test advanced edge cases for ensure_dates function."""

    def test_ensure_dates_dict_with_date_fields(self):
        """Test lines 103, 136 - dictionary with specific date fields."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        test_dict = {
            "first_date": "2023-01-01",
            "last_date": "2023-12-31",
            "created_at": "2023-06-15T10:30:00",
            "updated_at": "2023-06-16T15:45:00",
            "date": "2023-07-01",
            "outer_date": "2023-08-01",
            "other_field": "not_a_date",
            "number_field": 42,
        }

        result = ensure_dates(test_dict)

        # Verify date fields were converted
        assert isinstance(result["first_date"], pd.Timestamp)
        assert isinstance(result["last_date"], pd.Timestamp)
        assert isinstance(result["created_at"], pd.Timestamp)
        assert isinstance(result["updated_at"], pd.Timestamp)
        assert isinstance(result["date"], pd.Timestamp)
        assert isinstance(result["outer_date"], pd.Timestamp)

        # Verify non-date fields unchanged
        assert result["other_field"] == "not_a_date"
        assert result["number_field"] == 42

    def test_ensure_dates_dict_timezone_stripping(self):
        """Test lines 155-157 - timezone stripping in dictionary processing."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Create timezone-aware datetime string
        tz_aware_date = "2023-01-01T12:00:00+05:00"
        test_dict = {"date": tz_aware_date, "created_at": tz_aware_date}

        result = ensure_dates(test_dict, strip_timezone=True)

        # Verify timezone was stripped (tzinfo should be None)
        assert result["date"].tzinfo is None
        assert result["created_at"].tzinfo is None

    def test_ensure_dates_dict_conversion_failure(self):
        """Test lines 162-164 - dictionary date conversion failure handling."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        test_dict = {"date": "not-a-valid-date-string", "created_at": "also-invalid", "valid_field": "normal_value"}

        # Should not raise exception, should leave invalid dates as-is
        result = ensure_dates(test_dict)

        # Invalid dates should remain unchanged
        assert result["date"] == "not-a-valid-date-string"
        assert result["created_at"] == "also-invalid"
        assert result["valid_field"] == "normal_value"

    def test_ensure_dates_dataframe_no_date_column(self):
        """Test KeyError when DataFrame missing date column."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        df_no_date = pd.DataFrame({"other_col": [1, 2, 3]})

        with pytest.raises(KeyError, match="DataFrame must contain a 'date' column"):
            ensure_dates(df_no_date)

    def test_ensure_dates_dataframe_weird_types_in_date_column(self):
        """Test ValueError when date column contains non-date objects."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        df_weird_dates = pd.DataFrame(
            {
                "date": [
                    "2023-01-01",
                    [1, 2, 3],  # list in date column
                    "2023-01-03",
                ]
            }
        )

        with pytest.raises(ValueError, match="Date column contains non-date-like objects"):
            ensure_dates(df_weird_dates)

    def test_ensure_dates_dataframe_simple_case(self):
        """Test basic DataFrame processing without timezone complications."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Create simple DataFrame with string dates
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"]})

        result = ensure_dates(df, strip_timezone=True)

        # All dates should be processed correctly
        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_ensure_dates_dataframe_conversion_error(self):
        """Test lines 194->200, 197-199 - DataFrame conversion error handling."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Create DataFrame with dates that will cause conversion errors
        df = pd.DataFrame({"date": ["2023-01-01", "completely-invalid-date", "2023-01-03"]})

        with pytest.raises(ValueError, match="Invalid date format"):
            ensure_dates(df)

    def test_ensure_dates_dataframe_already_datetime_timezone_localize_error(self):
        """Test timezone localize error handling for already datetime columns."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        # Create DataFrame with datetime column that's already timezone-aware
        dates = pd.to_datetime(["2023-01-01", "2023-01-02"]).tz_localize("UTC")
        df = pd.DataFrame({"date": dates})

        # Should handle the case where timezone localization fails
        result = ensure_dates(df, strip_timezone=True)

        # Should still process successfully
        assert isinstance(result, pd.DataFrame)


class TestPandasImportFallbackPaths:
    """Test pandas import fallback and conditional logic."""

    def test_convert_pandas_timestamps_no_pandas(self):
        """Test lines 13-14 - convert_pandas_timestamps when pandas not available."""
        test_data = {"date": "2023-01-01", "value": 42}

        with patch("datason.datetime_utils.pd", None):
            result = convert_pandas_timestamps(test_data)

            # Should return unchanged when pandas is None
            assert result == test_data

    def test_ensure_dates_dict_no_pandas(self):
        """Test ImportError when pandas not available for dict processing."""
        test_dict = {"date": "2023-01-01"}

        with patch("datason.datetime_utils.pd", None):
            with pytest.raises(ImportError, match="pandas is required"):
                ensure_dates(test_dict)

    def test_ensure_dates_non_dict_no_pandas(self):
        """Test TypeError when pandas not available for non-dict input."""
        with patch("datason.datetime_utils.pd", None):
            with pytest.raises(TypeError, match="pandas DataFrame or dict"):
                ensure_dates("not_a_dict")

    def test_ensure_dates_dataframe_pandas_none_check(self):
        """Test lines 103, 163, 167, 173-174 - pandas None check for DataFrame."""
        # This tests the pandas None check that happens after dict processing
        with patch("datason.datetime_utils.pd", None):
            with pytest.raises(TypeError, match="pandas DataFrame or dict"):
                # Pass a list to trigger the non-dict branch and pandas None check
                ensure_dates([1, 2, 3])


class TestSerializeDatetimesAdvanced:
    """Test advanced scenarios for serialize_datetimes."""

    def test_serialize_datetimes_deeply_nested_with_mixed_types(self):
        """Test serialize_datetimes with complex nested structures."""
        complex_data = {
            "level1": {
                "dates": [datetime(2023, 1, 1), datetime(2023, 1, 2, tzinfo=timezone.utc)],
                "nested": {
                    "more_dates": {"start": datetime(2023, 1, 3), "end": datetime(2023, 1, 4), "metadata": "not_a_date"}
                },
            },
            "simple_list": [datetime(2023, 1, 5), "string", 42, {"inner_date": datetime(2023, 1, 6)}],
        }

        result = serialize_datetimes(complex_data)

        # Verify all datetime objects were converted to strings
        assert isinstance(result["level1"]["dates"][0], str)
        assert isinstance(result["level1"]["dates"][1], str)
        assert isinstance(result["level1"]["nested"]["more_dates"]["start"], str)
        assert isinstance(result["level1"]["nested"]["more_dates"]["end"], str)
        assert isinstance(result["simple_list"][0], str)
        assert isinstance(result["simple_list"][3]["inner_date"], str)

        # Verify non-datetime objects unchanged
        assert result["level1"]["nested"]["more_dates"]["metadata"] == "not_a_date"
        assert result["simple_list"][1] == "string"
        assert result["simple_list"][2] == 42

    def test_serialize_datetimes_edge_case_types(self):
        """Test serialize_datetimes with edge case input types."""
        test_cases = [
            (None, None),
            (42, 42),
            ("string", "string"),
            (3.14, 3.14),
            (True, True),
            (datetime(2023, 1, 1), "2023-01-01T00:00:00"),
        ]

        for input_val, expected in test_cases:
            result = serialize_datetimes(input_val)
            if isinstance(expected, str) and "T" in expected:
                # For datetime, just check it's a string (exact format may vary)
                assert isinstance(result, str)
            else:
                assert result == expected


class TestLoggingAndErrorHandling:
    """Test logging and error handling paths."""

    def test_ensure_timestamp_logging_on_error(self):
        """Test that ensure_timestamp logs errors appropriately."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        with patch("datason.datetime_utils.logger") as mock_logger:
            # Test invalid type logging
            with pytest.raises(TypeError):
                ensure_timestamp([1, 2, 3])

            mock_logger.error.assert_called()

            # Test conversion failure logging
            mock_logger.reset_mock()
            result = ensure_timestamp("completely-invalid-date-format-12345")

            # Should log warning for conversion failure
            mock_logger.warning.assert_called()
            assert pd.isna(result)

    def test_ensure_dates_logging_on_conversion_failure(self):
        """Test that ensure_dates logs appropriately on conversion failures."""
        if not HAS_PANDAS:
            pytest.skip("pandas not available")

        with patch("datason.datetime_utils.logger") as mock_logger:
            test_dict = {"date": "invalid-date-format", "other": "value"}

            ensure_dates(test_dict)

            # Should log debug message for conversion failure
            mock_logger.debug.assert_called()


class TestPandasSpecificTypeHandling:
    """Test pandas-specific type handling and edge cases."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_ensure_dates_dataframe_with_datetime_dtype_already(self):
        """Test ensure_dates with DataFrame that already has datetime dtype."""
        # Create DataFrame with datetime dtype
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        df = pd.DataFrame({"date": dates})

        result = ensure_dates(df)

        # Should process without errors
        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_convert_pandas_timestamps_with_nat_values(self):
        """Test convert_pandas_timestamps with NaT (Not a Time) values."""
        series_with_nat = pd.Series([pd.Timestamp("2023-01-01"), pd.NaT, pd.Timestamp("2023-01-03")])

        result = convert_pandas_timestamps(series_with_nat)

        # Should handle NaT values appropriately
        assert isinstance(result, pd.Series)
        assert isinstance(result.iloc[0], datetime)
        assert pd.isna(result.iloc[1])  # NaT should remain as NaN
        assert isinstance(result.iloc[2], datetime)
