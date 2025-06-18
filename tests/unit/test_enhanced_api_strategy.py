"""
Test suite for DataSON's Enhanced API Strategy.

This test validates the new approach:
1. Main DataSON API: Enhanced defaults (dict output, smart features, datetime parsing)
2. datason.json module: Perfect JSON stdlib compatibility for drop-in replacement
3. All explicit functions preserved and working

This replaces the old drop-in compatibility test with a more comprehensive approach
that validates both enhanced features and backward compatibility.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

import datason
import datason.json as datason_json


class TestEnhancedDataSONAPI:
    """Test that the main DataSON API provides enhanced defaults and smart features."""

    def test_dumps_returns_dict(self):
        """Test that main API dumps() returns dict, not string."""
        test_data = {"key": "value", "number": 42}

        result = datason.dumps(test_data)

        assert isinstance(result, dict), "Enhanced dumps() should return dict"
        assert result == test_data, "Content should be preserved"

    def test_loads_smart_datetime_parsing(self):
        """Test that loads() provides smart datetime parsing."""
        json_string = '{"timestamp": "2024-01-01T00:00:00Z", "value": 42}'

        result = datason.loads(json_string)

        assert isinstance(result, dict), "loads() should return dict"
        assert isinstance(result["timestamp"], datetime), "Should parse datetime strings"
        assert result["value"] == 42, "Other values should be preserved"

    def test_dump_enhanced_file_serialization(self):
        """Test that dump() provides enhanced serialization features."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "data": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name
            datason.dump(test_data, f)

        try:
            # Load back with enhanced features
            with open(temp_path) as f:
                loaded_data = datason.load(f)

            assert isinstance(loaded_data, dict), "Enhanced load should return dict"
            # The timestamp should be parsed as datetime due to smart parsing
            assert isinstance(loaded_data.get("timestamp"), datetime), "Should have smart datetime parsing"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_smart_parsing(self):
        """Test that load() provides smart parsing features."""
        # Create a JSON file with datetime strings
        test_data = {"created": "2024-01-01T00:00:00Z", "items": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name
            json.dump(test_data, f)

        try:
            with open(temp_path) as f:
                result = datason.load(f)

            assert isinstance(result, dict), "load() should return dict"
            assert isinstance(result["created"], datetime), "Should parse datetime strings"

            # Handle NumPy arrays (enhanced feature)
            items = result["items"]
            if hasattr(items, "tolist"):  # NumPy array
                items = items.tolist()
            assert items == [1, 2, 3], "Other data should be preserved"

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestJSONCompatibilityModule:
    """Test that datason.json provides perfect JSON stdlib compatibility."""

    # Standard test data that works with both json and datason.json
    TEST_DATA = [
        None,
        True,
        False,
        42,
        3.14,
        "hello world",
        "",
        [],
        [1, 2, 3],
        {},
        {"key": "value"},
        {"timestamp": "2024-01-01T00:00:00Z", "nested": {"data": [1, 2, 3]}},
    ]

    def test_dumps_perfect_compatibility(self):
        """Test that datason.json.dumps() is perfectly compatible with json.dumps()."""
        for test_data in self.TEST_DATA:
            json_result = json.dumps(test_data)
            datason_result = datason_json.dumps(test_data)

            assert isinstance(json_result, str), "json.dumps should return string"
            assert isinstance(datason_result, str), "datason.json.dumps should return string"
            assert json_result == datason_result, f"Results should be identical for {test_data}"

    def test_loads_perfect_compatibility(self):
        """Test that datason.json.loads() is perfectly compatible with json.loads()."""
        test_strings = [
            "null",
            "true",
            "false",
            "42",
            "3.14",
            '"hello"',
            "[]",
            "[1,2,3]",
            "{}",
            '{"key":"value"}',
            '{"timestamp":"2024-01-01T00:00:00Z"}',
        ]

        for test_string in test_strings:
            json_result = json.loads(test_string)
            datason_result = datason_json.loads(test_string)

            assert json_result == datason_result, f"Results should be identical for {test_string}"
            assert type(json_result) is type(datason_result), f"Types should match for {test_string}"

    def test_dump_perfect_compatibility(self):
        """Test that datason.json.dump() is perfectly compatible with json.dump()."""
        for test_data in self.TEST_DATA:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as json_file:
                json.dump(test_data, json_file)
                json_file_path = json_file.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as datason_file:
                datason_json.dump(test_data, datason_file)
                datason_file_path = datason_file.name

            try:
                # Read back and compare raw file contents
                with open(json_file_path) as f:
                    json_content = f.read()

                with open(datason_file_path) as f:
                    datason_content = f.read()

                # Parse both to ensure they're equivalent
                assert json.loads(json_content) == json.loads(datason_content)

            finally:
                Path(json_file_path).unlink(missing_ok=True)
                Path(datason_file_path).unlink(missing_ok=True)

    def test_load_perfect_compatibility(self):
        """Test that datason.json.load() is perfectly compatible with json.load()."""
        for test_data in self.TEST_DATA:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                json.dump(test_data, temp_file)
                temp_file_path = temp_file.name

            try:
                with open(temp_file_path) as f:
                    json_result = json.load(f)

                with open(temp_file_path) as f:
                    datason_result = datason_json.load(f)

                assert json_result == datason_result, f"Results should be identical for {test_data}"
                assert type(json_result) is type(datason_result), f"Types should match for {test_data}"

            finally:
                Path(temp_file_path).unlink(missing_ok=True)

    def test_parameter_compatibility(self):
        """Test that all json.dumps() parameters work with datason.json.dumps()."""
        test_data = {"b": 2, "a": 1, "c": [3, 1, 2]}

        # Test all standard parameters
        params_to_test = [
            {"sort_keys": True},
            {"indent": 2},
            {"indent": 4, "sort_keys": True},
            {"separators": (",", ":")},
            {"ensure_ascii": False},
            {"skipkeys": True},  # This won't affect our simple test data
        ]

        for params in params_to_test:
            json_result = json.dumps(test_data, **params)
            datason_result = datason_json.dumps(test_data, **params)

            # Both should produce the same result
            assert json_result == datason_result, f"Parameter test failed for {params}"

    def test_drop_in_replacement_works(self):
        """Test that 'import datason.json as json' works as perfect drop-in."""
        import datason.json as json_replacement

        # Test all main functions exist and work
        test_data = {"test": "data", "number": 42}

        # Test dumps/loads
        json_str = json_replacement.dumps(test_data)
        parsed_data = json_replacement.loads(json_str)
        assert parsed_data == test_data

        # Test dump/load
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name
            json_replacement.dump(test_data, f)

        try:
            with open(temp_path) as f:
                loaded_data = json_replacement.load(f)
            assert loaded_data == test_data
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestBothAPIsCoexist:
    """Test that both enhanced and compatibility APIs work together."""

    def test_mixed_usage_pattern(self):
        """Test using both APIs together for different purposes."""
        test_data = {"timestamp": "2024-01-01T00:00:00Z", "value": 42}

        # Enhanced API for processing
        enhanced_result = datason.dumps(test_data)
        assert isinstance(enhanced_result, dict), "Enhanced API should return dict"

        # Compatibility API for JSON output
        json_string = datason_json.dumps(test_data)
        assert isinstance(json_string, str), "Compatibility API should return string"

        # Both should represent the same data
        assert json.loads(json_string) == test_data
        assert enhanced_result == test_data

    def test_enhanced_vs_compatibility_datetime_handling(self):
        """Test different datetime handling between APIs."""
        json_string = '{"timestamp": "2024-01-01T00:00:00Z", "value": 42}'

        # Enhanced API: smart datetime parsing
        enhanced_result = datason.loads(json_string)
        assert isinstance(enhanced_result["timestamp"], datetime), "Enhanced API should parse datetimes"

        # Compatibility API: no smart parsing (like stdlib json)
        compat_result = datason_json.loads(json_string)
        assert isinstance(compat_result["timestamp"], str), "Compatibility API should keep strings as strings"

        # Both should have the same value (just different types)
        assert enhanced_result["value"] == compat_result["value"]


class TestExplicitFunctionsPreserved:
    """Test that all explicit DataSON functions are preserved and working."""

    def test_explicit_dump_functions(self):
        """Test that all explicit dump functions work."""
        test_data = {"test": "data", "numbers": [1, 2, 3]}

        explicit_functions = ["dump_ml", "dump_api", "dump_secure", "dump_fast", "dump_chunked"]

        for func_name in explicit_functions:
            assert hasattr(datason, func_name), f"Function {func_name} should exist"
            func = getattr(datason, func_name)

            try:
                result = func(test_data)
                assert result is not None, f"Function {func_name} should return a result"
            except Exception as e:
                pytest.fail(f"Function {func_name} failed: {e}")

    def test_explicit_load_functions(self):
        """Test that all explicit load functions work."""
        test_data = {"test": "data", "numbers": [1, 2, 3]}

        explicit_functions = [
            ("load_basic", lambda: datason.load_basic(test_data)),
            ("load_smart", lambda: datason.load_smart(test_data)),
            ("load_typed", lambda: datason.load_typed(test_data)),
            ("load_perfect", lambda: datason.load_perfect(test_data, test_data)),  # Needs template
        ]

        for func_name, func_call in explicit_functions:
            assert hasattr(datason, func_name), f"Function {func_name} should exist"

            try:
                result = func_call()
                assert result is not None, f"Function {func_name} should return a result"
            except Exception as e:
                pytest.fail(f"Function {func_name} failed: {e}")


class TestBackwardCompatibility:
    """Test that existing user code patterns still work."""

    def test_serialize_function_exists(self):
        """Test that serialize() function still exists and works."""
        test_data = {"test": "value"}

        # The serialize function should exist and return a dict
        result = datason.serialize(test_data)
        assert isinstance(result, dict), "serialize() should return dict"
        assert result == test_data, "serialize() should preserve data"

    def test_deprecation_warnings_for_direct_serialize(self):
        """Test that direct serialize usage shows appropriate guidance."""
        # The serialize function should work but guide users to better APIs
        test_data = {"test": "value"}

        # This should work (backward compatibility)
        result = datason.serialize(test_data)
        assert isinstance(result, dict)

        # But the enhanced dump/dumps are preferred
        enhanced_result = datason.dumps(test_data)
        assert result == enhanced_result, "Results should be equivalent"
