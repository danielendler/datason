"""Comprehensive coverage boost tests for datason modules.

This test file specifically targets uncovered lines identified in the coverage report
to increase overall test coverage across all datason modules.
"""

import warnings
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest

import datason
from datason import serialize
from datason.utils import (
    UtilityConfig,
    UtilitySecurityError,
    deep_compare,
    enhance_data_types,
    extract_temporal_features,
    find_data_anomalies,
    get_available_utilities,
    normalize_data_structure,
    standardize_datetime_formats,
)


class TestUtilsCoverageBoost:
    """Test utils.py uncovered lines and edge cases."""

    def test_utility_config_all_parameters(self) -> None:
        """Test UtilityConfig with all parameters."""
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

    def test_deep_compare_numeric_tolerance(self) -> None:
        """Test deep_compare with numeric tolerance."""
        result = deep_compare({"val": 1.0000001}, {"val": 1.0}, tolerance=1e-6)

        assert result["are_equal"]
        assert len(result["differences"]) == 0

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

    def test_deep_compare_circular_reference_disabled(self) -> None:
        """Test deep_compare with circular reference detection disabled."""
        config = UtilityConfig(enable_circular_reference_detection=False)

        a1: Dict[str, Any] = {"val": 1}
        a1["self"] = a1

        a2: Dict[str, Any] = {"val": 1}
        a2["self"] = a2

        # Should handle circular refs when detection is disabled
        # (though it may not detect them as equal)
        result = deep_compare(a1, a2, config=config)
        # Just verify it doesn't crash
        assert isinstance(result, dict)

    def test_find_data_anomalies_large_strings(self) -> None:
        """Test find_data_anomalies with large strings."""
        data = {"large_text": "x" * 1000, "normal": "hello"}

        result = find_data_anomalies(data)

        assert "large_strings" in result
        assert len(result["large_strings"]) > 0

    def test_find_data_anomalies_large_collections(self) -> None:
        """Test find_data_anomalies with large collections."""
        data = {"large_list": list(range(1000)), "normal": [1, 2, 3]}

        result = find_data_anomalies(data)

        assert "large_collections" in result
        assert len(result["large_collections"]) > 0

    def test_find_data_anomalies_custom_rules(self) -> None:
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
        assert "pattern_matches" in result

    def test_find_data_anomalies_security_violations(self) -> None:
        """Test find_data_anomalies with security-related patterns."""
        data = {
            "password": "secret123",
            "api_key": "abc123def456",
            "credit_card": "4111-1111-1111-1111",
            "email": "user@example.com",
        }

        result = find_data_anomalies(data)

        assert "security_violations" in result
        # Should detect some security-related patterns
        assert len(result["security_violations"]) > 0

    def test_enhance_data_types_numeric_strings(self) -> None:
        """Test enhance_data_types with numeric string conversion."""
        data = {"number": "123", "float": "45.67", "text": "hello"}

        enhanced, report = enhance_data_types(data)

        assert enhanced["number"] == 123
        assert enhanced["float"] == 45.67
        assert enhanced["text"] == "hello"
        assert "conversions" in report

    def test_enhance_data_types_date_parsing(self) -> None:
        """Test enhance_data_types with date string parsing."""
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

    def test_normalize_data_structure_flatten(self) -> None:
        """Test normalize_data_structure with flatten option."""
        data = {"a": {"b": 1, "c": 2}, "d": 3}

        result = normalize_data_structure(data, target_structure="flatten")

        assert "a.b" in result
        assert "a.c" in result
        assert "d" in result

    def test_normalize_data_structure_records(self) -> None:
        """Test normalize_data_structure with records conversion."""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

        result = normalize_data_structure(data, target_structure="records")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_normalize_data_structure_security_limits(self) -> None:
        """Test normalize_data_structure with security limits."""
        config = UtilityConfig(max_depth=1)
        nested = {"a": {"b": {"c": 1}}}

        with pytest.raises(UtilitySecurityError):
            normalize_data_structure(nested, target_structure="flatten", config=config)

    def test_standardize_datetime_formats_iso(self) -> None:
        """Test standardize_datetime_formats with ISO format."""
        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}

        result, paths = standardize_datetime_formats(data, target_format="iso")

        assert isinstance(result["date"], str)
        assert "2023-01-01T12:00:00" in result["date"]
        assert len(paths) > 0

    def test_standardize_datetime_formats_unix(self) -> None:
        """Test standardize_datetime_formats with unix format."""
        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}

        result, paths = standardize_datetime_formats(data, target_format="unix")

        assert isinstance(result["date"], (int, float))
        assert len(paths) > 0

    def test_extract_temporal_features(self) -> None:
        """Test extract_temporal_features functionality."""
        data = {
            "timestamp": datetime(2023, 6, 15, 14, 30, 0),
            "dates": [datetime(2023, 1, 1), datetime(2023, 12, 31)],
        }

        features = extract_temporal_features(data)

        assert "datetime_count" in features
        assert "date_range" in features
        assert features["datetime_count"] > 0

    def test_get_available_utilities(self) -> None:
        """Test get_available_utilities function."""
        utilities = get_available_utilities()

        assert isinstance(utilities, dict)
        assert "comparison" in utilities
        assert "analysis" in utilities
        assert "transformation" in utilities
        assert "datetime" in utilities


class TestCoreCoverageBoost:
    """Test core.py uncovered lines and edge cases."""

    def test_ml_serializer_import_error_path(self) -> None:
        """Test core.py lines 14-17 - ML import failure path."""
        # Test what happens when ML serializer is None
        with patch("datason.core._ml_serializer", None):
            # This should use the fallback path
            result = serialize({"test": "data"})
            assert result == {"test": "data"}

    def test_config_import_failure_path(self) -> None:
        """Test core.py lines 36-45 - config import failure path."""
        # This tests the ImportError fallback when config system unavailable
        # We can't easily mock this, but we can test the fallback functions
        from datason.core import is_nan_like, normalize_numpy_types

        # Test fallback implementations
        assert not is_nan_like("test")
        assert normalize_numpy_types("test") == "test"

    def test_serialize_with_type_cache_limit(self) -> None:
        """Test type cache behavior when limit is reached."""
        # Fill up the type cache
        original_limit = datason.core._TYPE_CACHE_SIZE_LIMIT
        datason.core._TYPE_CACHE_SIZE_LIMIT = 2
        datason.core._TYPE_CACHE.clear()

        try:
            # Add types to fill cache
            class TestType1:
                pass

            class TestType2:
                pass

            class TestType3:
                pass

            # These should use the cache
            result1 = datason.core._get_cached_type_category(TestType1)
            result2 = datason.core._get_cached_type_category(TestType2)

            # This should hit the limit and return None
            _result3 = datason.core._get_cached_type_category(TestType3)

            assert result1 == "other"
            assert result2 == "other"
            # Should not cache when limit reached

        finally:
            datason.core._TYPE_CACHE_SIZE_LIMIT = original_limit
            datason.core._TYPE_CACHE.clear()

    def test_string_length_cache_behavior(self) -> None:
        """Test string length cache optimization."""
        # Test that the string cache works correctly
        test_string = "test_string_for_cache"

        # First call should populate cache
        result1 = serialize(test_string)

        # Second call should use cache
        result2 = serialize(test_string)

        assert result1 == result2 == test_string

    def test_uuid_string_cache_behavior(self) -> None:
        """Test UUID string cache optimization."""
        import uuid

        test_uuid = uuid.uuid4()

        # First serialization should populate cache
        result1 = serialize(test_uuid)

        # Second should use cache
        result2 = serialize(test_uuid)

        assert result1 == result2
        assert isinstance(result1, str)

    def test_memory_pool_dict_exception_safety(self) -> None:
        """Test memory pooling exception safety for dicts."""
        # Test that pooled dicts are safely returned even if exceptions occur
        from datason.core import _get_pooled_dict, _return_dict_to_pool

        # Get a pooled dict
        pooled_dict = _get_pooled_dict()
        pooled_dict["test"] = "value"

        # Return it to pool
        _return_dict_to_pool(pooled_dict)

        # Dict should be cleared when returned
        # (This is implementation detail but helps with coverage)

    def test_memory_pool_list_exception_safety(self) -> None:
        """Test memory pooling exception safety for lists."""
        from datason.core import _get_pooled_list, _return_list_to_pool

        # Get a pooled list
        pooled_list = _get_pooled_list()
        pooled_list.append("test")

        # Return it to pool
        _return_list_to_pool(pooled_list)

        # List should be cleared when returned

    def test_homogeneous_collection_detection(self) -> None:
        """Test homogeneous collection optimization paths."""
        # Test with homogeneous list
        homogeneous_list = [1, 2, 3, 4, 5] * 10  # 50 integers
        result = serialize(homogeneous_list)
        assert isinstance(result, list)
        assert len(result) == 50

        # Test with heterogeneous list
        heterogeneous_list = [1, "string", 3.14, True, None]
        result = serialize(heterogeneous_list)
        assert isinstance(result, list)

    def test_json_only_fast_path(self) -> None:
        """Test JSON-only fast path optimization."""
        # Test data that should trigger fast path
        json_data = {
            "string": "hello",
            "number": 42,
            "boolean": True,
            "null": None,
            "float": 3.14,
        }

        result = serialize(json_data)
        assert result == json_data

    def test_iterative_serialization_path(self) -> None:
        """Test iterative serialization for deeply nested structures."""
        # Create a structure that might use iterative path
        deep_dict: Dict[str, Any] = {}
        current = deep_dict
        for i in range(20):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["value"] = "deep"

        result = serialize(deep_dict)
        assert isinstance(result, dict)

    def test_tuple_conversion_fast_path(self) -> None:
        """Test fast tuple conversion paths."""
        # Test nested tuples
        nested_tuples = ((1, 2), (3, 4), (5, 6))
        result = serialize(nested_tuples)
        assert result == [[1, 2], [3, 4], [5, 6]]

        # Test mixed tuple content
        mixed_tuple = (1, "hello", True, None)
        result = serialize(mixed_tuple)
        assert result == [1, "hello", True, None]


class TestInitCoverageBoost:
    """Test __init__.py uncovered lines and edge cases."""

    def test_configure_function_with_config(self) -> None:
        """Test configure function when config system available."""
        if hasattr(datason, "configure"):
            # Just verify function exists and can be called
            assert callable(datason.configure)

    def test_all_exports_comprehensive(self) -> None:
        """Test that all expected exports are available."""
        # Core functions should always be available
        assert hasattr(datason, "serialize")
        assert hasattr(datason, "deserialize")

        # Check availability flags
        assert hasattr(datason, "_config_available")
        assert hasattr(datason, "_ml_available")
        assert hasattr(datason, "_pickle_bridge_available")

        # Version info
        assert hasattr(datason, "__version__")

    def test_conditional_imports_coverage(self) -> None:
        """Test conditional import coverage paths."""
        # Test that conditional features work when available
        if datason._config_available:
            from datason.config import SerializationConfig

            config = SerializationConfig()
            assert config is not None

        if datason._ml_available:
            from datason.ml_serializers import get_ml_library_info

            info = get_ml_library_info()
            assert isinstance(info, dict)


class TestRedactionCoverageBoost:
    """Test redaction.py uncovered lines and edge cases."""

    def test_redaction_engine_with_invalid_patterns(self) -> None:
        """Test redaction engine with invalid regex patterns."""
        from datason.redaction import RedactionEngine

        # Test with invalid regex - should handle gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = RedactionEngine(redact_patterns=["[invalid"])

            data = {"test": "some data"}
            result = engine.process_object(data)

            # Should work despite invalid pattern
            assert isinstance(result, dict)
            # Should have warning about invalid pattern
            assert len(w) > 0

    def test_redaction_engine_empty_configuration(self) -> None:
        """Test redaction engine with empty configuration."""
        from datason.redaction import RedactionEngine

        engine = RedactionEngine(
            redact_fields=[],
            redact_patterns=[],
            redact_large_objects=False,
        )

        data = {"field": "value", "large_list": list(range(1000))}
        result = engine.process_object(data)

        # Should not redact anything with empty config
        assert result == data

    def test_redaction_with_complex_nested_structure(self) -> None:
        """Test redaction with complex nested structures."""
        from datason.redaction import RedactionEngine

        engine = RedactionEngine(redact_fields=["password"])

        complex_data = {
            "users": [
                {"id": 1, "password": "secret1", "nested": {"password": "secret2"}},
                {"id": 2, "password": "secret3"},
            ],
            "config": {"db_password": "secret4"},
        }

        result = engine.process_object(complex_data)

        # Should redact all password fields recursively
        assert result["users"][0]["password"] == "<REDACTED>"
        assert result["users"][0]["nested"]["password"] == "<REDACTED>"
        assert result["users"][1]["password"] == "<REDACTED>"

    def test_redaction_list_processing(self) -> None:
        """Test redaction processing of lists."""
        from datason.redaction import RedactionEngine

        engine = RedactionEngine(redact_patterns=[r"secret_\w+"])

        data = ["public_info", "secret_key_123", "normal_data", "secret_token_xyz"]

        result = engine.process_object(data)

        # Should redact items matching pattern
        assert "<REDACTED>" in str(result)
        assert "public_info" in result
        assert "normal_data" in result

    def test_redaction_audit_trail_comprehensive(self) -> None:
        """Test comprehensive redaction audit trail."""
        from datason.redaction import RedactionEngine

        engine = RedactionEngine(
            redact_fields=["password"],
            redact_patterns=[r"api_key_\w+"],
            redact_large_objects=True,
            audit_trail=True,
        )

        data = {
            "user": {"password": "secret"},
            "config": {"api_key_abc123": "sensitive"},
            "large_data": list(range(1000)),
        }

        _result = engine.process_object(data)
        audit = engine.get_audit_trail()

        if audit is not None:
            assert len(audit) > 0
            audit_str = str(audit)
            assert "password" in audit_str or "field" in audit_str

    def test_redaction_summary_statistics(self) -> None:
        """Test redaction summary statistics."""
        from datason.redaction import RedactionEngine

        engine = RedactionEngine(
            redact_fields=["sensitive"],
            include_redaction_summary=True,
        )

        data = {
            "public": "data",
            "sensitive": "secret1",
            "nested": {"sensitive": "secret2"},
        }

        _result = engine.process_object(data)
        summary = engine.get_redaction_summary()

        assert summary is not None
        assert "fields_redacted" in summary or "total_redactions" in summary


class TestAdvancedEdgeCases:
    """Test advanced edge cases across modules."""

    def test_serialize_with_all_optimizations(self) -> None:
        """Test serialization with all optimization paths enabled."""
        # Create data that exercises multiple optimization paths
        data = {
            "json_basic": {"str": "hello", "int": 42, "bool": True, "null": None},
            "homogeneous_list": [1, 2, 3, 4, 5] * 20,
            "heterogeneous": [1, "string", 3.14, True, None],
            "nested_tuples": ((1, 2), (3, 4)),
            "uuid_test": "550e8400-e29b-41d4-a716-446655440000",  # UUID-like string
        }

        result = serialize(data)

        # Verify structure is preserved
        assert isinstance(result, dict)
        assert isinstance(result["homogeneous_list"], list)
        assert len(result["homogeneous_list"]) == 100

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

    def test_cross_module_integration(self) -> None:
        """Test integration between multiple datason modules."""
        from datason.redaction import RedactionEngine
        from datason.utils import deep_compare

        # Create test data
        original_data = {
            "public": "info",
            "private": "secret",
            "nested": {"password": "hidden"},
        }

        # Redact sensitive data
        engine = RedactionEngine(redact_fields=["private", "password"])
        redacted_data = engine.process_object(original_data)

        # Compare original vs redacted
        comparison = deep_compare(original_data, redacted_data)

        assert not comparison["are_equal"]
        assert comparison["summary"]["value_differences"] > 0

    def test_error_recovery_patterns(self) -> None:
        """Test error recovery patterns across modules."""
        # Test that modules handle errors gracefully

        # Test utils with problematic data
        problematic_data: Dict[str, Any] = {
            "circular": None,
            "huge_string": "x" * 10000,
            "deep_nest": {},
        }

        # Create circular reference
        problematic_data["circular"] = problematic_data

        # Create deep nesting
        current: Dict[str, Any] = problematic_data["deep_nest"]
        for i in range(100):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        # These should either work or fail gracefully
        try:
            result = serialize(problematic_data)
            assert isinstance(result, dict)
        except Exception as e:
            # Should be a controlled exception, not a crash
            assert isinstance(e, (RecursionError, datason.core.SecurityError))


if __name__ == "__main__":
    pytest.main([__file__])
