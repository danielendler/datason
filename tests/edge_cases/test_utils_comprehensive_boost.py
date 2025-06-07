"""Comprehensive test coverage for datason.utils module.

This file focuses on exercising all the major utility functions to boost code coverage,
particularly targeting the many helper functions and edge cases in utils.py.
"""

from datetime import datetime

import pytest

from datason.utils import (
    UtilityConfig,
    UtilitySecurityError,
    deep_compare,
    enhance_data_types,
    extract_temporal_features,
    find_data_anomalies,
    get_available_utilities,
    get_default_utility_config,
    normalize_data_structure,
    standardize_datetime_formats,
)


class TestDeepCompare:
    """Test deep_compare functionality."""

    def test_deep_compare_basic(self) -> None:
        """Test basic deep comparison."""
        result = deep_compare({"a": 1}, {"a": 1})
        assert result["are_equal"] is True
        assert result["summary"]["total_differences"] == 0

    def test_deep_compare_differences(self) -> None:
        """Test deep comparison with differences."""
        result = deep_compare({"a": 1}, {"a": 2})
        assert result["are_equal"] is False
        assert result["summary"]["total_differences"] > 0

    def test_deep_compare_type_mismatch(self) -> None:
        """Test type mismatch detection."""
        result = deep_compare("string", 123)
        assert result["are_equal"] is False
        assert result["summary"]["type_mismatches"] > 0

    def test_deep_compare_nested_structures(self) -> None:
        """Test comparison of nested structures."""
        obj1 = {"nested": {"deep": [1, 2, 3]}}
        obj2 = {"nested": {"deep": [1, 2, 4]}}
        result = deep_compare(obj1, obj2)
        assert result["are_equal"] is False

    def test_deep_compare_with_tolerance(self) -> None:
        """Test numeric comparison with tolerance."""
        result = deep_compare(1.0000001, 1.0000002, tolerance=1e-5)
        assert result["are_equal"] is True

    def test_deep_compare_lists(self) -> None:
        """Test list comparison."""
        result = deep_compare([1, 2, 3], [1, 2, 3])
        assert result["are_equal"] is True

    def test_deep_compare_tuples(self) -> None:
        """Test tuple comparison."""
        result = deep_compare((1, 2, 3), (1, 2, 3))
        assert result["are_equal"] is True

    def test_deep_compare_with_config(self) -> None:
        """Test comparison with custom config."""
        config = UtilityConfig(max_depth=5)
        result = deep_compare({"a": 1}, {"a": 1}, config=config)
        assert result["are_equal"] is True


class TestDataAnomalies:
    """Test find_data_anomalies functionality."""

    def test_find_anomalies_basic(self) -> None:
        """Test basic anomaly detection."""
        data = {"numbers": [1, 2, 1000], "text": "hello"}
        result = find_data_anomalies(data)
        assert "large_strings" in result
        assert "large_collections" in result
        assert "suspicious_patterns" in result
        assert "security_violations" in result

    def test_find_anomalies_with_rules(self) -> None:
        """Test anomaly detection with custom rules."""
        data = {"value": 999999}
        rules = {"max_numeric_value": 1000}
        result = find_data_anomalies(data, rules=rules)
        assert isinstance(result, dict)

    def test_find_anomalies_nested(self) -> None:
        """Test anomaly detection in nested structures."""
        data = {"level1": {"level2": {"numbers": [1, 2, 3, 1000]}}}
        result = find_data_anomalies(data)
        assert isinstance(result, dict)

    def test_find_anomalies_with_config(self) -> None:
        """Test anomaly detection with config."""
        config = UtilityConfig(max_collection_size=5)
        data = {"small_list": [1, 2, 3]}
        result = find_data_anomalies(data, config=config)
        assert isinstance(result, dict)


class TestDataEnhancement:
    """Test enhance_data_types functionality."""

    def test_enhance_basic(self) -> None:
        """Test basic data enhancement."""
        data = {"date_str": "2023-01-01", "number_str": "42"}
        enhanced, report = enhance_data_types(data)
        assert isinstance(enhanced, dict)
        assert isinstance(report, dict)

    def test_enhance_with_rules(self) -> None:
        """Test enhancement with custom rules."""
        data = {"value": "123.45"}
        rules = {"auto_convert_numbers": True}
        enhanced, report = enhance_data_types(data, enhancement_rules=rules)
        assert isinstance(enhanced, dict)
        assert isinstance(report, dict)

    def test_enhance_nested_data(self) -> None:
        """Test enhancement of nested data."""
        data = {"metadata": {"created": "2023-01-01T12:00:00", "count": "42"}}
        enhanced, report = enhance_data_types(data)
        assert isinstance(enhanced, dict)

    def test_enhance_with_config(self) -> None:
        """Test enhancement with config."""
        config = UtilityConfig(max_depth=10)
        data = {"simple": "value"}
        enhanced, report = enhance_data_types(data, config=config)
        assert isinstance(enhanced, dict)


class TestNormalizeStructure:
    """Test normalize_data_structure functionality."""

    def test_normalize_flatten(self) -> None:
        """Test structure normalization to flat format."""
        data = {"a": {"b": {"c": 1}}}
        result = normalize_data_structure(data, target_structure="flat")
        assert isinstance(result, dict)

    def test_normalize_records(self) -> None:
        """Test structure normalization to records format."""
        data = {"col1": [1, 2], "col2": [3, 4]}
        result = normalize_data_structure(data, target_structure="records")
        assert isinstance(result, list)

    def test_normalize_with_config(self) -> None:
        """Test normalization with config."""
        config = UtilityConfig(max_depth=5)
        data = {"simple": "data"}
        result = normalize_data_structure(data, config=config)
        assert isinstance(result, dict)


class TestDatetimeStandardization:
    """Test standardize_datetime_formats functionality."""

    def test_standardize_iso_format(self) -> None:
        """Test datetime standardization to ISO format."""
        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}
        result, errors = standardize_datetime_formats(data, target_format="iso")
        assert isinstance(result, dict)
        assert isinstance(errors, list)

    def test_standardize_timestamp_format(self) -> None:
        """Test datetime standardization to timestamp format."""
        data = {"date": datetime(2023, 1, 1)}
        result, errors = standardize_datetime_formats(data, target_format="timestamp")
        assert isinstance(result, dict)
        assert isinstance(errors, list)

    def test_standardize_nested_dates(self) -> None:
        """Test standardization of nested datetime objects."""
        data = {"events": [{"timestamp": datetime(2023, 1, 1)}, {"timestamp": datetime(2023, 1, 2)}]}
        result, errors = standardize_datetime_formats(data)
        assert isinstance(result, dict)

    def test_standardize_with_config(self) -> None:
        """Test standardization with config."""
        config = UtilityConfig(max_depth=3)
        data = {"date": datetime.now()}
        result, errors = standardize_datetime_formats(data, config=config)
        assert isinstance(result, dict)


class TestTemporalFeatures:
    """Test extract_temporal_features functionality."""

    def test_extract_features_basic(self) -> None:
        """Test basic temporal feature extraction."""
        data = {"timestamp": datetime(2023, 6, 15, 14, 30, 0)}
        result = extract_temporal_features(data)
        assert isinstance(result, dict)
        assert "datetime_fields" in result
        assert "date_ranges" in result

    def test_extract_features_multiple_dates(self) -> None:
        """Test extraction from multiple datetime objects."""
        data = {
            "start": datetime(2023, 1, 1),
            "end": datetime(2023, 12, 31),
            "events": [datetime(2023, 6, 1), datetime(2023, 7, 1)],
        }
        result = extract_temporal_features(data)
        assert isinstance(result, dict)
        assert len(result["datetime_fields"]) > 1

    def test_extract_features_with_config(self) -> None:
        """Test extraction with config."""
        config = UtilityConfig(max_depth=2)
        data = {"date": datetime.now()}
        result = extract_temporal_features(data, config=config)
        assert isinstance(result, dict)


class TestUtilityConfig:
    """Test UtilityConfig and related functionality."""

    def test_default_config(self) -> None:
        """Test default configuration creation."""
        config = get_default_utility_config()
        assert isinstance(config, UtilityConfig)
        assert config.max_depth > 0
        assert config.max_object_size > 0

    def test_custom_config(self) -> None:
        """Test custom configuration creation."""
        config = UtilityConfig(
            max_depth=10, max_object_size=1000, max_string_length=500, enable_circular_reference_detection=False
        )
        assert config.max_depth == 10
        assert config.max_object_size == 1000
        assert config.max_string_length == 500
        assert config.enable_circular_reference_detection is False

    def test_security_error_depth(self) -> None:
        """Test security error on depth limit."""
        config = UtilityConfig(max_depth=1)
        deeply_nested = {"a": {"b": {"c": 1}}}

        with pytest.raises(UtilitySecurityError):
            deep_compare(deeply_nested, deeply_nested, config=config)

    def test_security_error_size(self) -> None:
        """Test security error on size limit."""
        config = UtilityConfig(max_object_size=2)
        large_list = list(range(100))

        with pytest.raises(UtilitySecurityError):
            deep_compare(large_list, large_list, config=config)


class TestUtilityMeta:
    """Test utility meta functions."""

    def test_get_available_utilities(self) -> None:
        """Test getting list of available utilities."""
        utilities = get_available_utilities()
        assert isinstance(utilities, dict)
        assert "data_comparison" in utilities
        assert "data_enhancement" in utilities
        assert "datetime_utilities" in utilities


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_values(self) -> None:
        """Test handling of None values."""
        result = deep_compare(None, None)
        assert result["are_equal"] is True

    def test_empty_structures(self) -> None:
        """Test handling of empty structures."""
        result = deep_compare({}, {})
        assert result["are_equal"] is True

        result = deep_compare([], [])
        assert result["are_equal"] is True

    def test_mixed_types(self) -> None:
        """Test handling of mixed data types."""
        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        # Test various utility functions with mixed data
        deep_compare(data, data.copy())
        find_data_anomalies(data)
        enhance_data_types(data)
        normalize_data_structure(data)

    def test_circular_references(self) -> None:
        """Test handling of circular references."""
        data: dict = {}
        data["self"] = data

        config = UtilityConfig(enable_circular_reference_detection=True)

        # Should handle circular references gracefully
        try:
            deep_compare(data, data, config=config)
        except UtilitySecurityError:
            # This is acceptable behavior
            pass
