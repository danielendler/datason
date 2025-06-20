"""Enhanced coverage tests for datason/utils.py.

Targets missing lines to improve coverage from 80% to 95%+.
Missing lines: 19-23, 129, 161-162, 188, 196-197, 221, 224-225, 250-251, 312-319, 322->333, 325-328, 354-357, 372, 381-388, 464->472, 490, 510-511, 515-517, 537->543, 608, 615, 618->624, 630, 641, 646, 650, 653->656, 662, 666, 672, 676, 710-713, 725, 731, 750-771, 791->794, 814, 820, 828, 833-834, 839, 850, 896-897, 910, 920->929, 939->947, 952-953, 984-985, 1004->1012, 1015->1028, 1018->1028, 1024-1025, 1031-1032, 1038-1039, 1043-1046
"""

import warnings
from datetime import datetime

import pytest

import datason.utils as utils
from datason.utils import UtilityConfig, UtilitySecurityError

# Try importing optional dependencies
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


class TestImportFallbacks:
    """Test import fallback scenarios - line 19-23."""

    def test_fallback_constants(self):
        """Test that fallback constants are properly set."""
        # Test the constants exist and have reasonable values
        assert hasattr(utils, "MAX_SERIALIZATION_DEPTH")
        assert hasattr(utils, "MAX_OBJECT_SIZE")
        assert hasattr(utils, "MAX_STRING_LENGTH")

        assert utils.MAX_SERIALIZATION_DEPTH == 50
        assert utils.MAX_OBJECT_SIZE == 100_000
        assert utils.MAX_STRING_LENGTH == 1_000_000


class TestSecurityLimits:
    """Test security limits and error paths."""

    def test_deep_compare_max_depth_exceeded(self):
        """Test depth limit security - line 129."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(1, 60):  # Exceed max depth
            current["next"] = {"level": i}
            current = current["next"]

        config = UtilityConfig(max_depth=5)

        with pytest.raises(UtilitySecurityError, match="Maximum comparison depth.*exceeded"):
            utils.deep_compare(nested, nested, config=config)

    def test_object_size_limits(self):
        """Test object size security limits - line 161-162."""
        # Create large objects
        large_obj1 = list(range(1000))
        large_obj2 = list(range(1000))
        config = UtilityConfig(max_object_size=100)

        with pytest.raises(UtilitySecurityError, match="Object size exceeds maximum"):
            utils.deep_compare(large_obj1, large_obj2, config=config)

    def test_circular_reference_detection_disabled(self):
        """Test circular reference handling when disabled - line 188."""
        config = UtilityConfig(enable_circular_reference_detection=False)

        # Create non-circular structure to test the disabled path
        obj1 = {"data": {"nested": "value"}}
        obj2 = {"data": {"nested": "value"}}

        result = utils.deep_compare(obj1, obj2, config=config)
        assert result["are_equal"] is True

    def test_circular_reference_detection_enabled(self):
        """Test circular reference detection - line 196-197."""
        config = UtilityConfig(enable_circular_reference_detection=True)

        # Create circular reference
        circular = {"data": "test"}
        circular["self"] = circular

        # Should handle gracefully without infinite recursion
        result = utils.deep_compare(circular, circular, config=config)
        assert result["are_equal"] is True


class TestDeepCompareEdgeCases:
    """Test deep comparison edge cases."""

    def test_dict_size_warning(self):
        """Test large dictionary warning - line 221."""
        config = UtilityConfig(max_collection_size=3)

        large_dict1 = {f"key_{i}": i for i in range(10)}
        large_dict2 = {f"key_{i}": i for i in range(10)}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils.deep_compare(large_dict1, large_dict2, config=config)

            # Should warn about large dictionary
            assert len(w) > 0
            assert "Large dictionary detected" in str(w[0].message)

    def test_missing_keys_in_dicts(self):
        """Test missing key detection - line 224-225."""
        dict1 = {"a": 1, "b": 2, "unique1": "value1"}
        dict2 = {"a": 1, "c": 3, "unique2": "value2"}

        result = utils.deep_compare(dict1, dict2)

        assert not result["are_equal"]
        # Should detect missing keys in both directions
        differences = result["differences"]
        assert any("Missing key in first object" in diff for diff in differences)
        assert any("Missing key in second object" in diff for diff in differences)
        assert result["summary"]["value_differences"] >= 4  # Missing keys count as value differences

    def test_list_length_differences(self):
        """Test list length differences - line 250-251."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 3]

        result = utils.deep_compare(list1, list2)

        assert not result["are_equal"]
        assert any("Length mismatch" in diff for diff in result["differences"])

    def test_string_length_warnings(self):
        """Test string length warnings - line 312-319."""
        config = UtilityConfig(max_string_length=5)

        long_str1 = "this_is_a_very_long_string"
        long_str2 = "this_is_another_very_long_string"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils.deep_compare(long_str1, long_str2, config=config)

            # Should warn about long strings
            assert len(w) >= 1
            assert any("Long string detected" in str(warning.message) for warning in w)


class TestDataAnomalyDetection:
    """Test anomaly detection functionality."""

    def test_anomaly_detection_with_custom_rules(self):
        """Test anomaly detection with rules - line 322->333, 325-328."""
        test_data = {
            "numbers": [1, 2, 3, 1000, 5],  # Contains outlier
            "strings": ["short", "medium", "a" * 100],  # Contains long string
            "nulls": [1, None, 3, None, None],  # High null ratio
            "duplicates": [1, 1, 1, 2, 2, 2],  # Many duplicates
        }

        custom_rules = {
            "detect_outliers": True,
            "outlier_threshold": 3.0,
            "detect_long_strings": True,
            "max_string_length": 50,
            "detect_high_null_ratio": True,
            "null_ratio_threshold": 0.3,
            "detect_duplicates": True,
            "duplicate_threshold": 0.5,
        }

        result = utils.find_data_anomalies(test_data, rules=custom_rules)

        assert "outliers" in result
        assert "long_strings" in result
        assert "high_null_ratios" in result
        assert "duplicates" in result
        assert result["summary"]["total_anomalies"] > 0

    def test_anomaly_detection_depth_limit(self):
        """Test anomaly detection depth limit - line 354-357."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(1, 60):
            current["next"] = {"level": i}
            current = current["next"]

        config = UtilityConfig(max_depth=10)

        with pytest.raises(UtilitySecurityError, match="Maximum anomaly detection depth.*exceeded"):
            utils.find_data_anomalies(nested, config=config)

    def test_anomaly_detection_circular_refs(self):
        """Test anomaly detection with circular references - line 372."""
        circular = {"data": [1, 2, 3]}
        circular["self"] = circular

        config = UtilityConfig(enable_circular_reference_detection=True)

        # Should handle circular references gracefully
        result = utils.find_data_anomalies(circular, config=config)
        assert result is not None
        assert "summary" in result

    def test_numeric_outlier_detection(self):
        """Test numeric outlier detection - line 381-388."""
        test_data = {
            "normal": [1, 2, 3, 4, 5],
            "with_outlier": [1, 2, 3, 100, 5],  # Clear outlier
            "too_small": [1, 2],  # Too small for outlier detection
        }

        result = utils.find_data_anomalies(test_data)

        # Should detect outlier in with_outlier list
        outliers = result.get("outliers", [])
        assert any("with_outlier" in str(outlier) for outlier in outliers)


class TestDataEnhancement:
    """Test data enhancement functionality."""

    def test_enhance_data_with_rules(self):
        """Test data enhancement with custom rules - line 464->472."""
        test_data = {
            "dates": ["2023-01-01", "2023-12-31", "invalid_date"],
            "numbers": ["123", "45.67", "not_a_number"],
            "booleans": ["true", "false", "yes", "no", "maybe"],
        }

        custom_rules = {
            "enhance_dates": True,
            "enhance_numbers": True,
            "enhance_booleans": True,
            "date_formats": ["%Y-%m-%d", "%Y/%m/%d"],
            "strict_parsing": False,
        }

        enhanced_data, report = utils.enhance_data_types(test_data, enhancement_rules=custom_rules)

        assert "dates" in enhanced_data
        assert "numbers" in enhanced_data
        assert "booleans" in enhanced_data
        assert "enhancements" in report
        assert report["summary"]["total_enhancements"] > 0

    def test_enhance_data_depth_limit(self):
        """Test enhancement depth limit - line 490."""
        # Create deeply nested structure
        nested = {"level": 0, "data": "2023-01-01"}
        current = nested
        for i in range(1, 60):
            current["next"] = {"level": i, "data": f"2023-01-{i:02d}"}
            current = current["next"]

        config = UtilityConfig(max_depth=10)

        with pytest.raises(UtilitySecurityError, match="Maximum enhancement depth.*exceeded"):
            utils.enhance_data_types(nested, config=config)

    def test_string_enhancement_parsing(self):
        """Test string enhancement - line 510-511, 515-517."""
        test_strings = [
            "2023-01-01T12:30:45",  # ISO datetime
            "123.456",  # Float
            "true",  # Boolean
            "not_parseable",  # Should remain string
        ]

        enhanced_data, report = utils.enhance_data_types(test_strings)

        # Should enhance parseable strings
        assert len(enhanced_data) == 4
        assert isinstance(enhanced_data[0], datetime)
        assert isinstance(enhanced_data[1], float)
        assert isinstance(enhanced_data[2], bool)
        assert isinstance(enhanced_data[3], str)

    def test_date_parsing_fallbacks(self):
        """Test date parsing with fallbacks - line 537->543."""
        date_strings = [
            "2023-01-01",
            "01/01/2023",
            "Jan 1, 2023",
            "invalid_date_format",
        ]

        enhanced_data, report = utils.enhance_data_types(date_strings)

        # Should parse some valid dates and leave invalid as strings
        valid_dates = sum(1 for item in enhanced_data if isinstance(item, datetime))
        assert valid_dates >= 1  # At least one should parse
        assert any(isinstance(item, str) for item in enhanced_data)  # Some remain strings


class TestDataNormalization:
    """Test data structure normalization."""

    def test_normalize_flatten_structure(self):
        """Test flattening structures - line 608."""
        nested_data = {
            "user": {"name": "John", "address": {"street": "123 Main St", "city": "Anytown"}},
            "scores": [85, 92, 78],
        }

        flattened = utils.normalize_data_structure(nested_data, target_structure="flat")

        assert "user.name" in flattened
        assert "user.address.street" in flattened
        assert "user.address.city" in flattened
        assert flattened["user.name"] == "John"

    def test_flatten_depth_limit(self):
        """Test flattening depth limit - line 615."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(1, 60):
            current["next"] = {"level": i}
            current = current["next"]

        config = UtilityConfig(max_depth=10)

        with pytest.raises(UtilitySecurityError, match="Maximum flattening depth.*exceeded"):
            utils.normalize_data_structure(nested, target_structure="flat", config=config)

    def test_flatten_circular_reference(self):
        """Test flattening with circular references - line 618->624."""
        circular = {"data": "value"}
        circular["self"] = circular

        config = UtilityConfig(enable_circular_reference_detection=True)

        result = utils.normalize_data_structure(circular, target_structure="flat", config=config)
        assert result is not None
        assert "data" in result

    def test_normalize_to_records(self):
        """Test normalization to records - line 630."""
        test_data = {"names": ["John", "Jane", "Bob"], "ages": [25, 30, 35], "cities": ["NYC", "LA", "Chicago"]}

        result = utils.normalize_data_structure(test_data, target_structure="records")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all("names" in record for record in result)

    def test_dict_to_records_validation(self):
        """Test dict to records conversion - line 641, 646, 650."""
        # Test with inconsistent list lengths
        inconsistent_data = {
            "names": ["John", "Jane"],
            "ages": [25, 30, 35],  # Different length
        }

        config = UtilityConfig()
        result = utils._dict_to_records(inconsistent_data, config)
        assert isinstance(result, list)
        assert len(result) == 3  # Should use max length

    def test_dict_to_records_size_warning(self):
        """Test dict to records size warning - line 653->656."""
        large_data = {"values": list(range(10000))}

        config = UtilityConfig(max_collection_size=1000)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils._dict_to_records(large_data, config)

            # Should warn about large collection
            assert len(w) > 0
            assert "Large collection detected" in str(w[0].message)


class TestDatetimeStandardization:
    """Test datetime standardization."""

    def test_standardize_datetime_formats(self):
        """Test datetime format standardization - line 662, 666, 672, 676."""
        test_data = {
            "dates": [
                datetime(2023, 1, 1, 12, 30, 45),
                datetime(2023, 12, 31, 23, 59, 59),
            ],
            "mixed": [datetime(2023, 6, 15), "not_a_date", {"nested": datetime(2023, 7, 4)}],
        }

        # Test ISO format
        result_iso, errors = utils.standardize_datetime_formats(test_data, target_format="iso")
        assert isinstance(result_iso, dict)

        # Test timestamp format
        result_ts, errors = utils.standardize_datetime_formats(test_data, target_format="timestamp")
        assert isinstance(result_ts, dict)

        # Test custom format
        result_custom, errors = utils.standardize_datetime_formats(test_data, target_format="%Y-%m-%d %H:%M:%S")
        assert isinstance(result_custom, dict)

    def test_datetime_conversion_depth_limit(self):
        """Test datetime conversion depth limit - line 710-713."""
        # Create deeply nested structure with datetimes
        nested = {"level": 0, "date": datetime(2023, 1, 1)}
        current = nested
        for i in range(1, 60):
            current["next"] = {"level": i, "date": datetime(2023, 1, i + 1)}
            current = current["next"]

        config = UtilityConfig(max_depth=10)

        with pytest.raises(UtilitySecurityError, match="Maximum datetime conversion depth.*exceeded"):
            utils.standardize_datetime_formats(nested, config=config)

    def test_datetime_circular_reference(self):
        """Test datetime conversion with circular refs - line 725."""
        circular = {"date": datetime(2023, 1, 1)}
        circular["self"] = circular

        config = UtilityConfig(enable_circular_reference_detection=True)

        result, errors = utils.standardize_datetime_formats(circular, config=config)
        assert result is not None

    def test_datetime_conversion_error_handling(self):
        """Test datetime conversion error handling - line 731."""
        # Test with problematic data
        problematic_data = {"dates": [datetime(2023, 1, 1), "invalid_datetime_object"]}

        # Should handle conversion errors gracefully
        result, errors = utils.standardize_datetime_formats(problematic_data)
        assert result is not None


class TestTemporalFeatures:
    """Test temporal feature extraction."""

    def test_extract_temporal_features(self):
        """Test temporal feature extraction - line 750-771."""
        test_data = {
            "events": [
                datetime(2023, 1, 15, 10, 30, 0),
                datetime(2023, 6, 20, 14, 45, 30),
                datetime(2023, 12, 25, 20, 0, 0),
            ],
            "mixed": {"start": datetime(2023, 1, 1), "end": datetime(2023, 12, 31), "metadata": "not_a_date"},
        }

        result = utils.extract_temporal_features(test_data)

        assert "datetime_counts" in result
        assert "temporal_ranges" in result
        assert "patterns" in result
        assert result["summary"]["total_datetimes"] >= 5

    def test_temporal_extraction_depth_limit(self):
        """Test temporal extraction depth limit - line 791->794."""
        # Create deeply nested structure with datetimes
        nested = {"level": 0, "date": datetime(2023, 1, 1)}
        current = nested
        for i in range(1, 60):
            current["next"] = {"level": i, "date": datetime(2023, 1, i + 1)}
            current = current["next"]

        config = UtilityConfig(max_depth=10)

        with pytest.raises(UtilitySecurityError, match="Maximum temporal extraction depth.*exceeded"):
            utils.extract_temporal_features(nested, config=config)

    def test_temporal_patterns_detection(self):
        """Test temporal pattern detection - line 814, 820, 828."""
        test_data = [
            datetime(2023, 1, 1, 9, 0, 0),  # Monday morning
            datetime(2023, 1, 2, 9, 0, 0),  # Tuesday morning
            datetime(2023, 1, 3, 9, 0, 0),  # Wednesday morning
            datetime(2023, 1, 1, 17, 0, 0),  # Monday evening
            datetime(2023, 1, 2, 17, 0, 0),  # Tuesday evening
        ]

        result = utils.extract_temporal_features(test_data)

        assert "patterns" in result
        assert "temporal_ranges" in result
        patterns = result["patterns"]
        assert "hour_distribution" in patterns
        assert "day_of_week_distribution" in patterns

    def test_temporal_circular_reference(self):
        """Test temporal extraction with circular refs - line 833-834."""
        circular = {"date": datetime(2023, 1, 1)}
        circular["self"] = circular

        config = UtilityConfig(enable_circular_reference_detection=True)

        result = utils.extract_temporal_features(circular, config=config)
        assert result is not None
        assert "summary" in result

    def test_temporal_statistics(self):
        """Test temporal statistics - line 839, 850."""
        test_dates = [
            datetime(2023, 1, 1),
            datetime(2023, 6, 1),
            datetime(2023, 12, 31),
        ]

        result = utils.extract_temporal_features(test_dates)

        ranges = result["temporal_ranges"]
        assert "earliest" in ranges
        assert "latest" in ranges
        assert "span_days" in ranges
        assert ranges["span_days"] > 300  # Should be close to a year


class TestPandasIntegration:
    """Test pandas-specific functionality."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_enhance_pandas_dataframe(self):
        """Test pandas DataFrame enhancement - line 896-897."""
        df = pd.DataFrame(
            {
                "dates": ["2023-01-01", "2023-12-31", "invalid"],
                "numbers": ["123", "45.67", "not_number"],
                "booleans": ["true", "false", "maybe"],
            }
        )

        enhanced_df, report = utils.enhance_pandas_dataframe(df)

        assert isinstance(enhanced_df, pd.DataFrame)
        assert "enhancements" in report

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_enhance_pandas_security(self):
        """Test pandas DataFrame security - line 910."""
        large_df = pd.DataFrame({"data": list(range(10000))})

        config = UtilityConfig(max_object_size=1000)

        with pytest.raises(UtilitySecurityError, match="DataFrame size exceeds maximum"):
            utils.enhance_pandas_dataframe(large_df, config=config)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_enhance_pandas_column_types(self):
        """Test pandas column type enhancement - line 920->929."""
        df = pd.DataFrame(
            {
                "mixed_dates": ["2023-01-01", "2023-12-31", pd.NaT],
                "mixed_numbers": [1, 2.5, "3.7", "invalid"],
                "mixed_booleans": [True, False, "yes", "no", "maybe"],
            }
        )

        custom_rules = {
            "enhance_dates": True,
            "enhance_numbers": True,
            "enhance_booleans": True,
            "handle_mixed_types": True,
        }

        enhanced_df, report = utils.enhance_pandas_dataframe(df, enhancement_rules=custom_rules)

        assert isinstance(enhanced_df, pd.DataFrame)
        assert len(enhanced_df) == len(df)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_enhance_pandas_error_handling(self):
        """Test pandas enhancement error handling - line 939->947, 952-953."""
        # Create DataFrame with problematic data
        df = pd.DataFrame({"problematic": [{"complex": "object"}, [1, 2, 3], {1, 2, 3}]})

        enhanced_df, report = utils.enhance_pandas_dataframe(df)

        assert isinstance(enhanced_df, pd.DataFrame)
        assert len(enhanced_df) == len(df)


class TestNumpyIntegration:
    """Test numpy-specific functionality."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_enhance_numpy_array(self):
        """Test numpy array enhancement - line 984-985."""
        arr = np.array([1, 2, 3, 4, 5])

        enhanced_arr, report = utils.enhance_numpy_array(arr)

        assert isinstance(enhanced_arr, np.ndarray)
        assert "enhancements" in report

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_enhance_numpy_security(self):
        """Test numpy array security - line 1004->1012."""
        large_arr = np.arange(100000)

        config = UtilityConfig(max_object_size=1000)

        with pytest.raises(UtilitySecurityError, match="Array size exceeds maximum"):
            utils.enhance_numpy_array(large_arr, config=config)

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_enhance_numpy_types(self):
        """Test numpy array types - line 1015->1028, 1018->1028."""
        int_arr = np.array([1, 2, 3])
        float_arr = np.array([1.1, 2.2, 3.3])
        str_arr = np.array(["2023-01-01", "2023-12-31"])

        enhanced_int, report_int = utils.enhance_numpy_array(int_arr)
        assert isinstance(enhanced_int, np.ndarray)

        enhanced_float, report_float = utils.enhance_numpy_array(float_arr)
        assert isinstance(enhanced_float, np.ndarray)

        enhanced_str, report_str = utils.enhance_numpy_array(str_arr)
        assert isinstance(enhanced_str, np.ndarray)

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_enhance_numpy_statistics(self):
        """Test numpy statistics - line 1024-1025, 1031-1032."""
        arr = np.array([1, 2, 3, 4, 5, 100])  # With outlier

        enhanced_arr, report = utils.enhance_numpy_array(arr)

        assert "statistics" in report
        stats = report["statistics"]
        assert "mean" in stats
        assert "std" in stats

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_enhance_numpy_error_handling(self):
        """Test numpy error handling - line 1038-1039, 1043-1046."""
        arr_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        enhanced_arr, report = utils.enhance_numpy_array(arr_with_nan)

        assert isinstance(enhanced_arr, np.ndarray)
        assert "enhancements" in report
