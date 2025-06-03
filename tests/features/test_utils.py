"""Tests for data transformation utilities (v0.5.5)."""

from datetime import datetime
from typing import Any, Dict

import pytest

from datason.utils import (
    UtilityConfig,
    UtilitySecurityError,
    deep_compare,
    enhance_data_types,
    enhance_numpy_array,
    enhance_pandas_dataframe,
    extract_temporal_features,
    find_data_anomalies,
    get_available_utilities,
    get_default_utility_config,
    normalize_data_structure,
    standardize_datetime_formats,
)


class TestUtilityConfig:
    """Test the utility configuration system."""

    def test_default_config(self) -> None:
        """Test default configuration creation."""
        config = get_default_utility_config()

        assert config.max_depth > 0
        assert config.max_object_size > 0
        assert config.max_string_length > 0
        assert config.enable_circular_reference_detection is True

    def test_custom_config(self) -> None:
        """Test custom configuration creation."""
        config = UtilityConfig(
            max_depth=10,
            max_object_size=1000,
            max_string_length=100,
            enable_circular_reference_detection=False,
        )

        assert config.max_depth == 10
        assert config.max_object_size == 1000
        assert config.max_string_length == 100
        assert config.enable_circular_reference_detection is False


class TestSecurityFeatures:
    """Test security features in utility functions."""

    def test_depth_limit_enforcement(self) -> None:
        """Test that depth limits are enforced."""
        config = UtilityConfig(max_depth=2)

        # Create deeply nested structure
        data = {"level1": {"level2": {"level3": {"level4": "deep"}}}}

        with pytest.raises(UtilitySecurityError):
            deep_compare(data, data, config=config)

    def test_object_size_limits(self) -> None:
        """Test that object size limits are enforced."""
        config = UtilityConfig(max_object_size=10)

        # Create large object
        large_data = {f"key{i}": f"value{i}" for i in range(20)}

        # This should trigger security violations in the results
        anomalies = find_data_anomalies(large_data, config=config)
        assert len(anomalies["security_violations"]) > 0
        assert "object_too_large" in anomalies["security_violations"][0]["violation"]

    def test_string_length_limits(self) -> None:
        """Test that string length limits are enforced."""
        config = UtilityConfig(max_string_length=10)

        # Create long string
        long_string = "x" * 100
        data = {"long": long_string}

        # Should trigger security violation in anomalies (not raise exception)
        anomalies = find_data_anomalies(data, config=config)
        assert len(anomalies["security_violations"]) > 0

    def test_circular_reference_detection(self) -> None:
        """Test circular reference detection."""
        config = UtilityConfig(enable_circular_reference_detection=True)

        # Create circular reference
        data: Dict[str, Any] = {"a": 1}
        data["self"] = data

        # Should handle circular references gracefully
        result = deep_compare(data, data, config=config)
        assert result["are_equal"] is True

    def test_circular_reference_disabled(self) -> None:
        """Test behavior when circular reference detection is disabled."""
        config = UtilityConfig(
            enable_circular_reference_detection=False,
            max_depth=10,  # Lower depth to trigger depth limit before recursion error
        )

        # Create circular reference
        data: Dict[str, Any] = {"a": 1}
        data["self"] = data

        # This should hit the depth limit instead of handling circular references
        with pytest.raises(UtilitySecurityError, match="Maximum comparison depth"):
            deep_compare(data, data, config=config)


class TestDeepCompare:
    """Test deep comparison functionality."""

    def test_identical_objects(self) -> None:
        """Test comparison of identical objects."""
        obj1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        obj2 = {"a": 1, "b": [2, 3], "c": {"d": 4}}

        result = deep_compare(obj1, obj2)

        assert result["are_equal"] is True
        assert len(result["differences"]) == 0

    def test_different_objects(self) -> None:
        """Test comparison of different objects."""
        obj1 = {"a": 1, "b": [2, 3]}
        obj2 = {"a": 2, "b": [2, 4]}

        result = deep_compare(obj1, obj2)

        assert result["are_equal"] is False
        assert len(result["differences"]) > 0
        assert result["summary"]["value_differences"] > 0

    def test_type_mismatches(self) -> None:
        """Test comparison with type mismatches."""
        obj1 = {"a": 1}
        obj2 = {"a": "1"}

        result = deep_compare(obj1, obj2)

        assert result["are_equal"] is False
        assert result["summary"]["type_mismatches"] > 0

    def test_numeric_tolerance(self) -> None:
        """Test numeric comparison with tolerance."""
        obj1 = {"value": 1.0000001}
        obj2 = {"value": 1.0000002}

        # Should be equal with default tolerance
        result = deep_compare(obj1, obj2, tolerance=1e-6)
        assert result["are_equal"] is True

        # Should be different with stricter tolerance
        result = deep_compare(obj1, obj2, tolerance=1e-8)
        assert result["are_equal"] is False

    def test_missing_keys(self) -> None:
        """Test comparison with missing keys."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1}

        result = deep_compare(obj1, obj2)

        assert result["are_equal"] is False
        assert any("Missing key" in diff for diff in result["differences"])

    def test_with_custom_config(self) -> None:
        """Test deep compare with custom configuration."""
        config = UtilityConfig(max_collection_size=5)

        obj1 = {"small": "data"}
        obj2 = {"small": "data"}

        result = deep_compare(obj1, obj2, config=config)
        assert result["are_equal"] is True


class TestFindDataAnomalies:
    """Test anomaly detection functionality."""

    def test_large_strings(self) -> None:
        """Test detection of large strings."""
        data = {"normal": "short", "large": "x" * 20000}

        anomalies = find_data_anomalies(data)

        assert len(anomalies["large_strings"]) > 0
        assert anomalies["large_strings"][0]["path"] == "large"

    def test_large_collections(self) -> None:
        """Test detection of large collections."""
        data = {"small_list": [1, 2, 3], "large_list": list(range(2000))}

        anomalies = find_data_anomalies(data)

        assert len(anomalies["large_collections"]) > 0

    def test_custom_rules(self) -> None:
        """Test anomaly detection with custom rules."""
        data = {"text": "x" * 500}
        rules = {"max_string_length": 100}

        anomalies = find_data_anomalies(data, rules)

        assert len(anomalies["large_strings"]) > 0

    def test_security_violations(self) -> None:
        """Test security violation detection."""
        config = UtilityConfig(max_string_length=10)
        data = {"long_string": "x" * 50}

        anomalies = find_data_anomalies(data, config=config)

        assert len(anomalies["security_violations"]) > 0
        assert "string_too_long" in anomalies["security_violations"][0]["violation"]

    def test_with_custom_config(self) -> None:
        """Test anomaly detection with custom configuration."""
        config = UtilityConfig(max_collection_size=5)
        data = {"items": list(range(10))}

        anomalies = find_data_anomalies(data, config=config)
        assert "large_collections" in anomalies


class TestEnhanceDataTypes:
    """Test data type enhancement functionality."""

    def test_numeric_string_conversion(self) -> None:
        """Test conversion of numeric strings."""
        data = {"price": "29.99", "quantity": "5"}
        rules = {"parse_numbers": True}

        enhanced, report = enhance_data_types(data, rules)

        assert enhanced["price"] == 29.99
        assert enhanced["quantity"] == 5
        assert len(report["type_conversions"]) > 0

    def test_date_parsing(self) -> None:
        """Test parsing of date strings."""
        data = {"date": "2023-12-25"}
        rules = {"parse_dates": True}

        enhanced, report = enhance_data_types(data, rules)

        assert isinstance(enhanced["date"], datetime)
        assert len(report["type_conversions"]) > 0

    def test_nested_enhancement(self) -> None:
        """Test enhancement of nested structures."""
        data = {"user": {"age": "25", "score": "98.5"}, "items": ["1", "2", "3"]}
        rules = {"parse_numbers": True}

        enhanced, report = enhance_data_types(data, rules)

        assert enhanced["user"]["age"] == 25
        assert enhanced["user"]["score"] == 98.5
        assert enhanced["items"] == [1, 2, 3]

    def test_with_security_limits(self) -> None:
        """Test enhancement with security limits."""
        config = UtilityConfig(max_object_size=5)
        large_data = {f"key{i}": f"value{i}" for i in range(10)}

        with pytest.raises(UtilitySecurityError):
            enhance_data_types(large_data, config=config)

    def test_circular_reference_error(self) -> None:
        """Test enhancement with circular references."""
        config = UtilityConfig(enable_circular_reference_detection=True)
        data: Dict[str, Any] = {"a": 1}
        data["self"] = data

        with pytest.raises(UtilitySecurityError):
            enhance_data_types(data, config=config)


class TestNormalizeDataStructure:
    """Test data structure normalization."""

    def test_flatten_structure(self) -> None:
        """Test flattening of nested structures."""
        data = {"a": {"b": {"c": 1}}, "d": [2, 3]}

        flattened = normalize_data_structure(data, "flat")

        assert "a.b.c" in flattened
        assert flattened["a.b.c"] == 1
        assert "d.0" in flattened
        assert flattened["d.0"] == 2

    def test_records_conversion(self) -> None:
        """Test conversion to records format."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}

        records = normalize_data_structure(data, "records")

        assert len(records) == 2
        assert records[0] == {"name": "Alice", "age": 25}
        assert records[1] == {"name": "Bob", "age": 30}

    def test_with_security_limits(self) -> None:
        """Test normalization with security limits."""
        config = UtilityConfig(max_object_size=5)
        large_data = {f"key{i}": i for i in range(10)}

        with pytest.raises(UtilitySecurityError):
            normalize_data_structure(large_data, "flat", config=config)

    def test_circular_reference_handling(self) -> None:
        """Test flattening with circular references."""
        config = UtilityConfig(enable_circular_reference_detection=True)
        data: Dict[str, Any] = {"a": {"b": 1}}
        data["a"]["self"] = data["a"]  # Create circular reference

        flattened = normalize_data_structure(data, "flat", config=config)

        # Should handle circular reference gracefully
        assert "a.b" in flattened
        assert flattened["a.b"] == 1


class TestDatetimeUtilities:
    """Test datetime utility functions."""

    def test_standardize_datetime_formats(self) -> None:
        """Test datetime format standardization."""
        dt = datetime(2023, 12, 25, 10, 30, 0)
        data = {"timestamp": dt, "nested": {"date": dt}}

        converted, log = standardize_datetime_formats(data, "iso")

        assert isinstance(converted["timestamp"], str)
        assert "2023-12-25T10:30:00" in converted["timestamp"]
        assert len(log) > 0

    def test_extract_temporal_features(self) -> None:
        """Test temporal feature extraction."""
        dt = datetime(2023, 12, 25, 10, 30, 0)
        data = {"event_time": dt, "other": "not a date"}

        features = extract_temporal_features(data)

        assert "event_time" in features["datetime_fields"]
        assert "event_time" in features["date_ranges"]
        assert features["date_ranges"]["event_time"]["min"] == dt

    def test_datetime_with_security_limits(self) -> None:
        """Test datetime utilities with security limits."""
        config = UtilityConfig(max_object_size=5)
        large_data = {f"date{i}": datetime.now() for i in range(10)}

        with pytest.raises(UtilitySecurityError):
            standardize_datetime_formats(large_data, config=config)


class TestUtilityDiscovery:
    """Test utility discovery functionality."""

    def test_get_available_utilities(self) -> None:
        """Test getting list of available utilities."""
        utilities = get_available_utilities()

        assert "data_comparison" in utilities
        assert "data_enhancement" in utilities
        assert "datetime_utilities" in utilities
        assert "configuration" in utilities
        assert "security" in utilities
        assert "deep_compare" in utilities["data_comparison"]
        assert "enhance_data_types" in utilities["data_enhancement"]
        assert "UtilityConfig" in utilities["configuration"]
        assert "UtilitySecurityError" in utilities["security"]


class TestPandasNumpyIntegration:
    """Test pandas and numpy specific utilities."""

    def test_pandas_dataframe_enhancement(self) -> None:
        """Test enhancement of pandas DataFrames."""
        try:
            import pandas as pd

            df = pd.DataFrame({"price": ["10.99", "20.50"], "quantity": ["1", "2"]})

            enhanced_df, report = enhance_pandas_dataframe(df)

            # Should enhance the DataFrame
            assert enhanced_df is not None
            assert "columns_processed" in report
            assert len(report["columns_processed"]) > 0

        except ImportError:
            pytest.skip("pandas not available")

    def test_pandas_with_security_limits(self) -> None:
        """Test pandas enhancement with security limits."""
        try:
            import pandas as pd

            config = UtilityConfig(max_object_size=5)
            df = pd.DataFrame({"col": range(10)})  # 10 rows > limit of 5

            with pytest.raises(UtilitySecurityError):
                enhance_pandas_dataframe(df, config=config)

        except ImportError:
            pytest.skip("pandas not available")

    def test_numpy_array_enhancement(self) -> None:
        """Test enhancement of numpy arrays."""
        try:
            import numpy as np

            arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)

            enhanced_arr, report = enhance_numpy_array(arr)

            # Should enhance the array
            assert enhanced_arr is not None
            assert "optimizations_applied" in report

        except ImportError:
            pytest.skip("numpy not available")

    def test_numpy_with_security_limits(self) -> None:
        """Test numpy enhancement with security limits."""
        try:
            import numpy as np

            config = UtilityConfig(max_object_size=5)
            arr = np.array(range(10))  # 10 elements > limit of 5

            with pytest.raises(UtilitySecurityError):
                enhance_numpy_array(arr, config=config)

        except ImportError:
            pytest.skip("numpy not available")

    def test_mixed_pandas_numpy_data(self) -> None:
        """Test enhancement of mixed pandas/numpy data structures."""
        try:
            import numpy as np
            import pandas as pd

            data = {
                "dataframe": pd.DataFrame({"values": ["1", "2", "3"]}),
                "array": np.array(["4.5", "5.5"]),
                "regular": {"number": "42"},
            }

            enhanced, report = enhance_data_types(data, {"parse_numbers": True})

            # Should handle mixed structure
            assert enhanced is not None
            assert enhanced["regular"]["number"] == 42

        except ImportError:
            pytest.skip("pandas/numpy not available")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_circular_references(self) -> None:
        """Test handling of circular references."""
        data: Dict[str, Any] = {"a": 1}
        data["self"] = data

        # Should not crash
        result = deep_compare(data, data)
        assert result["are_equal"] is True

    def test_empty_data(self) -> None:
        """Test handling of empty data."""
        assert deep_compare({}, {})["are_equal"] is True
        assert deep_compare([], [])["are_equal"] is True

        anomalies = find_data_anomalies({})
        assert len(anomalies["large_strings"]) == 0

    def test_none_values(self) -> None:
        """Test handling of None values."""
        data = {"value": None}

        enhanced, report = enhance_data_types(data)
        assert enhanced["value"] is None

        result = deep_compare(data, data)
        assert result["are_equal"] is True

    def test_invalid_inputs(self) -> None:
        """Test handling of invalid inputs."""
        # Test invalid DataFrame type
        try:
            enhance_pandas_dataframe("not a dataframe")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        except ImportError:
            pytest.skip("pandas not available")

        # Test invalid numpy array type
        try:
            enhance_numpy_array("not an array")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        except ImportError:
            pytest.skip("numpy not available")
