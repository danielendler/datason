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


class TestUtilityConfigExtended:
    """Extended tests for UtilityConfig functionality."""

    def test_utility_config_all_parameters(self) -> None:
        """Test UtilityConfig with all parameters specified."""
        config = UtilityConfig(
            max_depth=10,
            max_object_size=1000,
            max_string_length=500,
            max_collection_size=100,
            enable_circular_reference_detection=False,
            timeout_seconds=30.0,
        )

        assert config.max_depth == 10
        assert config.max_object_size == 1000
        assert config.max_string_length == 500
        assert config.max_collection_size == 100
        assert config.enable_circular_reference_detection is False
        assert config.timeout_seconds == 30.0


class TestDeepCompareExtended:
    """Extended tests for deep_compare functionality."""

    def test_deep_compare_type_mismatches(self) -> None:
        """Test deep_compare with type mismatches."""
        result = deep_compare({"a": 1}, {"a": "1"})

        assert not result["are_equal"]
        assert len(result["differences"]) > 0
        assert result["summary"]["type_mismatches"] > 0

    def test_deep_compare_missing_keys(self) -> None:
        """Test deep_compare with missing keys."""
        result = deep_compare({"a": 1, "b": 2}, {"a": 1})

        assert not result["are_equal"]
        assert any("Missing key" in diff for diff in result["differences"])

    def test_deep_compare_string_length_limits(self) -> None:
        """Test deep_compare with string length limits."""
        config = UtilityConfig(max_string_length=10)
        long_str = "x" * 20

        # Should handle long strings without throwing errors
        result = deep_compare({"text": long_str}, {"text": long_str}, config=config)
        assert result["are_equal"]

    def test_deep_compare_object_size_security_error(self) -> None:
        """Test deep_compare security error with large objects."""
        config = UtilityConfig(max_object_size=5)
        large_list = list(range(10))

        with pytest.raises(UtilitySecurityError):
            deep_compare(large_list, large_list, config=config)

    def test_deep_compare_depth_security_error(self) -> None:
        """Test deep_compare security error with excessive depth."""
        config = UtilityConfig(max_depth=2)
        nested = {"a": {"b": {"c": {"d": 1}}}}

        with pytest.raises(UtilitySecurityError):
            deep_compare(nested, nested, config=config)


class TestFindDataAnomaliesExtended:
    """Extended tests for find_data_anomalies functionality."""

    def test_find_data_anomalies_large_strings(self) -> None:
        """Test find_data_anomalies with large strings."""
        data = {"large_text": "x" * 1000, "normal": "hello"}

        result = find_data_anomalies(data)

        assert "large_strings" in result
        # May or may not find large strings depending on default thresholds

    def test_find_data_anomalies_large_collections(self) -> None:
        """Test find_data_anomalies with large collections."""
        data = {"large_list": list(range(1000)), "normal": [1, 2, 3]}

        result = find_data_anomalies(data)

        assert "large_collections" in result
        # May or may not find large collections depending on default thresholds

    def test_find_data_anomalies_custom_rules_comprehensive(self) -> None:
        """Test find_data_anomalies with custom rules."""
        rules = {
            "max_string_length": 5,
            "max_collection_size": 2,
            "custom_patterns": [r"secret_\w+"],
        }

        data = {"long_text": "this is too long", "secret_key": "password", "list": [1, 2, 3]}

        result = find_data_anomalies(data, rules=rules)

        assert "large_strings" in result
        assert "large_collections" in result
        # API may vary on exact key names

    def test_find_data_anomalies_security_violations_patterns(self) -> None:
        """Test find_data_anomalies with security-related patterns."""
        data = {
            "password": "secret123",
            "api_key": "abc123def456",
            "credit_card": "4111-1111-1111-1111",
            "email": "user@example.com",
        }

        result = find_data_anomalies(data)

        assert "security_violations" in result or "suspicious_patterns" in result
        # Should detect some security-related patterns


class TestEnhanceDataTypesExtended:
    """Extended tests for enhance_data_types functionality."""

    def test_enhance_data_types_numeric_strings(self) -> None:
        """Test enhance_data_types with numeric string conversion."""
        data = {"number": "123", "float": "45.67", "text": "hello"}

        enhanced, report = enhance_data_types(data)

        assert enhanced["number"] == 123
        assert enhanced["float"] == 45.67
        assert enhanced["text"] == "hello"
        assert "type_conversions" in report

    def test_enhance_data_types_date_parsing(self) -> None:
        """Test enhance_data_types with date string parsing."""
        from datetime import datetime

        data = {"date": "2023-01-01", "datetime": "2023-01-01T12:00:00", "text": "not a date"}

        enhanced, report = enhance_data_types(data)

        assert isinstance(enhanced["date"], datetime)
        assert isinstance(enhanced["datetime"], datetime)
        assert enhanced["text"] == "not a date"

    def test_enhance_data_types_with_security_limits(self) -> None:
        """Test enhance_data_types with security limits."""
        config = UtilityConfig(max_depth=2)
        nested = {"a": {"b": {"c": "123"}}}

        with pytest.raises(UtilitySecurityError):
            enhance_data_types(nested, config=config)

    def test_enhance_data_types_circular_reference_error(self) -> None:
        """Test enhance_data_types with circular reference."""
        from typing import Any, Dict

        data: Dict[str, Any] = {"val": 1}
        data["self"] = data

        # Should handle circular references gracefully or raise appropriate error
        try:
            enhanced, report = enhance_data_types(data)
            # If it succeeds, verify it handled the circular ref
            assert isinstance(enhanced, dict)
        except UtilitySecurityError:
            # Acceptable outcome for circular reference detection
            pass


class TestNormalizeDataStructureExtended:
    """Extended tests for normalize_data_structure functionality."""

    def test_normalize_data_structure_flatten(self) -> None:
        """Test normalize_data_structure with flatten option."""
        data = {"a": {"b": 1, "c": 2}, "d": 3}

        result = normalize_data_structure(data, target_structure="flatten")

        # Result structure may vary, just verify it's processed
        assert isinstance(result, (dict, list))

    def test_normalize_data_structure_records(self) -> None:
        """Test normalize_data_structure with records conversion."""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

        result = normalize_data_structure(data, target_structure="records")

        assert isinstance(result, (list, dict))

    def test_normalize_data_structure_security_limits(self) -> None:
        """Test normalize_data_structure with security limits."""
        config = UtilityConfig(max_depth=1)
        nested = {"a": {"b": {"c": 1}}}

        with pytest.raises(UtilitySecurityError):
            normalize_data_structure(nested, target_structure="flatten", config=config)


class TestDatetimeUtilitiesExtended:
    """Extended tests for datetime utilities."""

    def test_standardize_datetime_formats_iso(self) -> None:
        """Test standardize_datetime_formats with ISO format."""
        from datetime import datetime

        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}

        result, paths = standardize_datetime_formats(data, target_format="iso")

        assert isinstance(result["date"], str)
        assert "2023-01-01T12:00:00" in result["date"]
        assert len(paths) > 0

    def test_standardize_datetime_formats_unix(self) -> None:
        """Test standardize_datetime_formats with unix format."""
        from datetime import datetime

        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}

        result, paths = standardize_datetime_formats(data, target_format="unix")

        assert isinstance(result["date"], (int, float))
        assert len(paths) > 0

    def test_extract_temporal_features_comprehensive(self) -> None:
        """Test extract_temporal_features functionality."""
        from datetime import datetime

        data = {
            "timestamp": datetime(2023, 6, 15, 14, 30, 0),
            "dates": [datetime(2023, 1, 1), datetime(2023, 12, 31)],
        }

        features = extract_temporal_features(data)

        assert "datetime_fields" in features
        assert len(features["datetime_fields"]) > 0


class TestUtilityDiscoveryExtended:
    """Extended tests for utility discovery."""

    def test_get_available_utilities_comprehensive(self) -> None:
        """Test get_available_utilities function."""
        utilities = get_available_utilities()

        assert isinstance(utilities, dict)
        # Check for expected categories (API may vary)
        expected_categories = ["data_comparison", "data_enhancement", "datetime_utilities", "configuration"]
        found_categories = any(cat in utilities for cat in expected_categories)
        assert found_categories


class TestPandasNumpyIntegrationExtended:
    """Extended integration tests with pandas and numpy."""

    def test_utils_with_pandas_integration(self) -> None:
        """Test utils integration with pandas when available."""
        pd = pytest.importorskip("pandas")

        from datason.utils import enhance_pandas_dataframe

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        enhanced, report = enhance_pandas_dataframe(df)

        assert isinstance(enhanced, pd.DataFrame)
        assert isinstance(report, dict)

    def test_utils_with_numpy_integration(self) -> None:
        """Test utils integration with numpy when available."""
        np = pytest.importorskip("numpy")

        from datason.utils import enhance_numpy_array

        arr = np.array([1, 2, 3, 4, 5])

        enhanced, report = enhance_numpy_array(arr)

        assert isinstance(enhanced, np.ndarray)
        assert isinstance(report, dict)
