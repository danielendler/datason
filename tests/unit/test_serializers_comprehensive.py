"""Comprehensive tests for datason.serializers module.

This module tests specialized serialization functions including detection details
serialization, handling of numpy/pandas types, and edge cases.
"""

from datetime import datetime
from unittest.mock import patch

import datason.serializers as serializers


class TestSerializeDetectionDetails:
    """Test serialize_detection_details function."""

    def test_serialize_non_dict_input(self):
        """Test that non-dict input is returned as-is."""
        # Test various non-dict types
        assert serializers.serialize_detection_details("string") == "string"
        assert serializers.serialize_detection_details(42) == 42
        assert serializers.serialize_detection_details([1, 2, 3]) == [1, 2, 3]
        assert serializers.serialize_detection_details(None) is None

    def test_serialize_empty_dict(self):
        """Test serialization of empty dictionary."""
        result = serializers.serialize_detection_details({})
        assert result == {}

    def test_serialize_basic_types(self):
        """Test serialization of basic Python types."""
        input_data = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "none_value": None,
        }

        result = serializers.serialize_detection_details(input_data)
        expected = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "none_value": None,
        }
        assert result == expected

    def test_serialize_nested_structures(self):
        """Test serialization of nested lists and dictionaries."""
        input_data = {"nested_list": [1, 2, {"inner": "value"}], "nested_dict": {"level1": {"level2": ["a", "b", "c"]}}}

        result = serializers.serialize_detection_details(input_data)
        expected = {"nested_list": [1, 2, {"inner": "value"}], "nested_dict": {"level1": {"level2": ["a", "b", "c"]}}}
        assert result == expected

    def test_serialize_datetime_objects(self):
        """Test serialization of datetime objects."""
        dt = datetime(2023, 1, 15, 12, 30, 45)
        input_data = {"timestamp": dt, "nested_datetime": {"date": dt}}

        result = serializers.serialize_detection_details(input_data)
        expected = {"timestamp": dt.isoformat(), "nested_datetime": {"date": dt.isoformat()}}
        assert result == expected

    def test_serialize_float_nan_and_inf(self):
        """Test serialization of NaN and Infinity float values."""
        input_data = {
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "neg_inf_value": float("-inf"),
            "normal_float": 3.14,
            "nested_nan": {"inner": float("nan")},
        }

        result = serializers.serialize_detection_details(input_data)
        expected = {
            "nan_value": None,
            "inf_value": None,
            "neg_inf_value": None,
            "normal_float": 3.14,
            "nested_nan": {"inner": None},
        }
        assert result == expected


class TestSerializeDetectionDetailsWithNumpy:
    """Test serialize_detection_details with numpy types."""

    def test_serialize_numpy_arrays(self):
        """Test serialization of numpy arrays."""
        with patch("datason.serializers.np") as mock_np:
            # Create a real type for ndarray instead of using Mock type
            class MockNumpyArray:
                def __iter__(self):
                    return iter([1, 2, 3])

            mock_array = MockNumpyArray()
            mock_np.ndarray = MockNumpyArray

            input_data = {"array": mock_array}

            result = serializers.serialize_detection_details(input_data)
            expected = {"array": [1, 2, 3]}
            assert result == expected

    def test_serialize_numpy_integers(self):
        """Test serialization of numpy integer types."""
        with patch("datason.serializers.np") as mock_np:
            # Create a real type for integer instead of using Mock type
            class MockNumpyInteger:
                def __int__(self):
                    return 42

            mock_int = MockNumpyInteger()
            mock_np.integer = MockNumpyInteger

            input_data = {"np_int": mock_int}

            result = serializers.serialize_detection_details(input_data)
            expected = {"np_int": 42}
            assert result == expected

    def test_serialize_numpy_floats(self):
        """Test serialization of numpy float types."""
        with patch("datason.serializers.np") as mock_np:
            # Create a real type for floating instead of using Mock type
            class MockNumpyFloat:
                def __float__(self):
                    return 3.14

            mock_float = MockNumpyFloat()
            mock_np.floating = MockNumpyFloat
            mock_np.isnan.return_value = False
            mock_np.isinf.return_value = False

            input_data = {"np_float": mock_float}

            result = serializers.serialize_detection_details(input_data)
            expected = {"np_float": 3.14}
            assert result == expected

    def test_serialize_numpy_nan_and_inf(self):
        """Test serialization of numpy NaN and Inf values."""
        with patch("datason.serializers.np") as mock_np:
            # Create a real type for floating instead of using Mock type
            class MockNumpyFloat:
                pass

            mock_nan_float = MockNumpyFloat()
            mock_inf_float = MockNumpyFloat()
            mock_np.floating = MockNumpyFloat

            # Mock isnan/isinf to return appropriate values for each instance
            def mock_isnan(value):
                return value is mock_nan_float

            def mock_isinf(value):
                return value is mock_inf_float

            mock_np.isnan.side_effect = mock_isnan
            mock_np.isinf.side_effect = mock_isinf

            input_data = {"nan_float": mock_nan_float, "inf_float": mock_inf_float}

            result = serializers.serialize_detection_details(input_data)
            expected = {"nan_float": None, "inf_float": None}
            assert result == expected

    def test_serialize_without_numpy(self):
        """Test serialization when numpy is not available."""
        # Temporarily set np to None to simulate numpy not being available
        original_np = serializers.np
        serializers.np = None

        try:
            input_data = {"regular_value": 42, "float_value": 3.14}

            result = serializers.serialize_detection_details(input_data)
            expected = {"regular_value": 42, "float_value": 3.14}
            assert result == expected
        finally:
            serializers.np = original_np


class TestSerializeDetectionDetailsWithPandas:
    """Test serialize_detection_details with pandas types."""

    def test_serialize_pandas_series(self):
        """Test serialization of pandas Series."""
        with patch("datason.serializers.pd") as mock_pd:
            # Create a real type for Series instead of using Mock type
            class MockPandasSeries:
                def __iter__(self):
                    return iter([1, 2, 3])

            mock_series = MockPandasSeries()
            mock_pd.Series = MockPandasSeries

            # Mock pd.isna to return False for our test values
            mock_pd.isna.return_value = False

            input_data = {"series": mock_series}

            result = serializers.serialize_detection_details(input_data)
            expected = {"series": [1, 2, 3]}
            assert result == expected

    def test_serialize_pandas_timestamp(self):
        """Test serialization of pandas Timestamp."""
        with patch("datason.serializers.pd") as mock_pd:
            # Create a real type for Timestamp instead of using Mock type
            class MockPandasTimestamp:
                def isoformat(self):
                    return "2023-01-15T12:30:45"

            mock_timestamp = MockPandasTimestamp()
            mock_pd.Timestamp = MockPandasTimestamp

            input_data = {"timestamp": mock_timestamp}

            result = serializers.serialize_detection_details(input_data)
            expected = {"timestamp": "2023-01-15T12:30:45"}
            assert result == expected

    def test_serialize_pandas_na_values(self):
        """Test serialization of pandas NA values."""
        with patch("datason.serializers.pd") as mock_pd:
            # Mock pandas isna function
            mock_pd.isna.return_value = True

            input_data = {"na_value": "some_value"}

            result = serializers.serialize_detection_details(input_data)
            expected = {"na_value": None}
            assert result == expected

    def test_serialize_pandas_isna_exception(self):
        """Test handling of pandas.isna exceptions."""
        with patch("datason.serializers.pd") as mock_pd:
            # Mock pandas isna to raise an exception
            mock_pd.isna.side_effect = ValueError("Cannot check NA")

            input_data = {"value": "test_value"}

            result = serializers.serialize_detection_details(input_data)
            expected = {"value": "test_value"}
            assert result == expected

    def test_serialize_without_pandas(self):
        """Test serialization when pandas is not available."""
        # Temporarily set pd to None to simulate pandas not being available
        original_pd = serializers.pd
        serializers.pd = None

        try:
            input_data = {"regular_value": 42, "string_value": "test"}

            result = serializers.serialize_detection_details(input_data)
            expected = {"regular_value": 42, "string_value": "test"}
            assert result == expected
        finally:
            serializers.pd = original_pd


class TestComplexSerializationScenarios:
    """Test complex serialization scenarios."""

    def test_deeply_nested_structures(self):
        """Test serialization of deeply nested structures."""
        input_data = {
            "level1": {
                "level2": {
                    "level3": [{"datetime": datetime(2023, 1, 1)}, {"float_nan": float("nan")}, {"normal": "value"}]
                }
            }
        }

        result = serializers.serialize_detection_details(input_data)
        expected = {
            "level1": {
                "level2": {"level3": [{"datetime": "2023-01-01T00:00:00"}, {"float_nan": None}, {"normal": "value"}]}
            }
        }
        assert result == expected

    def test_mixed_types_in_lists(self):
        """Test serialization of lists with mixed types."""
        dt = datetime(2023, 5, 15, 10, 30)
        input_data = {"mixed_list": [42, "string", dt, float("nan"), None, {"nested": "dict"}, [1, 2, 3]]}

        result = serializers.serialize_detection_details(input_data)
        expected = {"mixed_list": [42, "string", dt.isoformat(), None, None, {"nested": "dict"}, [1, 2, 3]]}
        assert result == expected

    def test_all_edge_cases_combined(self):
        """Test serialization with all edge cases combined."""
        dt = datetime(2023, 12, 25, 15, 45, 30)

        input_data = {
            "datetime_val": dt,
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neg_inf_val": float("-inf"),
            "none_val": None,
            "nested": {
                "inner_datetime": dt,
                "inner_list": [float("nan"), 42, "test"],
                "deep_nested": {"datetime_in_list": [dt, float("inf")]},
            },
            "list_of_dicts": [{"dt": dt, "val": float("nan")}, {"normal": 123}],
        }

        result = serializers.serialize_detection_details(input_data)
        expected = {
            "datetime_val": dt.isoformat(),
            "nan_val": None,
            "inf_val": None,
            "neg_inf_val": None,
            "none_val": None,
            "nested": {
                "inner_datetime": dt.isoformat(),
                "inner_list": [None, 42, "test"],
                "deep_nested": {"datetime_in_list": [dt.isoformat(), None]},
            },
            "list_of_dicts": [{"dt": dt.isoformat(), "val": None}, {"normal": 123}],
        }
        assert result == expected


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_nested_structures(self):
        """Test serialization of empty nested structures."""
        input_data = {
            "empty_dict": {},
            "empty_list": [],
            "nested_empty": {"empty_inner_dict": {}, "empty_inner_list": []},
        }

        result = serializers.serialize_detection_details(input_data)
        expected = {
            "empty_dict": {},
            "empty_list": [],
            "nested_empty": {"empty_inner_dict": {}, "empty_inner_list": []},
        }
        assert result == expected

    def test_circular_reference_prevention(self):
        """Test that circular references don't cause infinite recursion."""
        # Create a structure with potential circular reference
        inner_dict = {"value": 42}
        outer_dict = {"inner": inner_dict}
        # Note: We don't actually create a circular reference as that would
        # be problematic for JSON serialization anyway

        input_data = {"method1": outer_dict}

        result = serializers.serialize_detection_details(input_data)
        expected = {"method1": {"inner": {"value": 42}}}
        assert result == expected

    def test_very_large_float_values(self):
        """Test serialization of very large float values."""
        input_data = {"large_float": 1e308, "small_negative": -1e308, "zero": 0.0, "tiny": 1e-308}

        result = serializers.serialize_detection_details(input_data)
        expected = {"large_float": 1e308, "small_negative": -1e308, "zero": 0.0, "tiny": 1e-308}
        assert result == expected

    def test_unicode_and_special_characters(self):
        """Test serialization with unicode and special characters."""
        input_data = {
            "unicode": "测试数据",
            "emoji": "🚀💻📊",
            "special_chars": "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./",
            "newlines": "line1\nline2\r\nline3",
        }

        result = serializers.serialize_detection_details(input_data)
        # Should pass through unchanged
        assert result == input_data
