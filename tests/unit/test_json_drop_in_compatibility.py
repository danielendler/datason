"""
Test suite ensuring DataSON provides perfect JSON compatibility via the datason.json module.

This test file is CRITICAL for maintaining JSON compatibility. Any failure here means
we've broken the drop-in replacement capability that users depend on.

The tests compare datason.json's behavior exactly with Python's built-in json module,
while also verifying that the main datason API provides enhanced features.

NEW STRATEGY:
- datason.json provides exact json stdlib compatibility
- Main datason API provides enhanced defaults (dict output, smart parsing)
"""

import io
import json
import tempfile
from pathlib import Path

import pytest

import datason
import datason.json as datason_json  # The compatibility module


class TestJSONDropInCompatibility:
    """Test that DataSON can be used as a perfect drop-in replacement for json module."""

    # Standard test data that works with both json and datason
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
        ["a", "b", "c"],
        {},
        {"key": "value"},
        {"number": 42, "boolean": True, "null": None},
        {"nested": {"list": [1, 2, 3], "string": "test", "boolean": False}},
        # Complex but JSON-compatible structure
        {
            "users": [{"id": 1, "name": "Alice", "active": True}, {"id": 2, "name": "Bob", "active": False}],
            "metadata": {"total": 2, "page": 1, "has_more": False},
        },
    ]

    def test_dumps_compatibility(self):
        """Test that datason.json.dumps() behaves exactly like json.dumps()."""
        for test_data in self.TEST_DATA:
            # Test JSON compatibility module (should return strings)
            json_result = json.dumps(test_data)
            datason_json_result = datason_json.dumps(test_data)

            # Both should return strings
            assert isinstance(json_result, str), f"json.dumps should return string for {test_data}"
            assert isinstance(datason_json_result, str), f"datason.json.dumps should return string for {test_data}"

            # Results should be equivalent when parsed back
            assert json.loads(json_result) == json.loads(datason_json_result), (
                f"Results should be equivalent for {test_data}"
            )

            # Test that enhanced API behavior (different from JSON by design)
            datason_enhanced_result = datason.dumps(test_data)

            # Enhanced API preserves the data structure (dict for complex, unchanged for primitives)
            if isinstance(test_data, dict):
                assert isinstance(datason_enhanced_result, dict), (
                    f"datason.dumps should return dict for dict input {test_data}"
                )
            else:
                # For primitives (None, int, str, etc.), enhanced API may return unchanged
                pass

            # Enhanced result should be equivalent to original data
            assert datason_enhanced_result == test_data, f"Enhanced API should return equivalent data for {test_data}"

    def test_dumps_parameters(self):
        """Test that datason.json.dumps() supports standard json.dumps() parameters."""
        test_data = {"b": 2, "a": 1, "c": 3}

        # Test sort_keys parameter
        json_sorted = json.dumps(test_data, sort_keys=True)
        datason_sorted = datason_json.dumps(test_data, sort_keys=True)
        assert json.loads(json_sorted) == json.loads(datason_sorted)

        # Test indent parameter
        json_indented = json.dumps(test_data, indent=2)
        datason_indented = datason_json.dumps(test_data, indent=2)
        assert json.loads(json_indented) == json.loads(datason_indented)

        # Test separators parameter
        json_compact = json.dumps(test_data, separators=(",", ":"))
        datason_compact = datason_json.dumps(test_data, separators=(",", ":"))
        assert json.loads(json_compact) == json.loads(datason_compact)

    def test_loads_compatibility(self):
        """Test that datason.json.loads() behaves exactly like json.loads()."""
        test_strings = [
            "null",
            "true",
            "false",
            "42",
            "3.14",
            '"hello"',
            '""',
            "[]",
            "[1,2,3]",
            '["a","b","c"]',
            "{}",
            '{"key":"value"}',
            '{"number":42,"boolean":true,"null":null}',
            '{"nested":{"list":[1,2,3],"string":"test","boolean":false}}',
        ]

        for test_string in test_strings:
            json_result = json.loads(test_string)
            datason_json_result = datason_json.loads(test_string)

            assert json_result == datason_json_result, f"Results should be identical for {test_string}"
            assert type(json_result) is type(datason_json_result), f"Types should be identical for {test_string}"

    def test_dump_file_compatibility(self):
        """Test that datason.json.dump() behaves exactly like json.dump()."""
        for test_data in self.TEST_DATA:
            # Test with temporary files
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as json_file:
                json.dump(test_data, json_file)
                json_file_path = json_file.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as datason_file:
                datason_json.dump(test_data, datason_file)
                datason_file_path = datason_file.name

            try:
                # Read back and compare
                with open(json_file_path) as f:
                    json_content = json.load(f)

                with open(datason_file_path) as f:
                    datason_content = json.load(f)  # Use json.load to read datason output

                assert json_content == datason_content, f"File contents should be identical for {test_data}"

            finally:
                # Clean up
                Path(json_file_path).unlink(missing_ok=True)
                Path(datason_file_path).unlink(missing_ok=True)

    def test_dump_return_value(self):
        """Test that datason.json.dump() returns None like json.dump()."""
        test_data = {"test": "data"}

        with tempfile.NamedTemporaryFile(mode="w") as json_file:
            json_result = json.dump(test_data, json_file)

        with tempfile.NamedTemporaryFile(mode="w") as datason_file:
            datason_result = datason_json.dump(test_data, datason_file)

        assert json_result is None, "json.dump() should return None"
        assert datason_result is None, "datason.dump() should return None"
        assert type(json_result) is type(datason_result), "Return types should match"

    def test_load_file_compatibility(self):
        """Test that datason.load() behaves exactly like json.load()."""
        for test_data in self.TEST_DATA:
            # Create test file with json
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                json.dump(test_data, temp_file)
                temp_file_path = temp_file.name

            try:
                # Read with both json and datason
                with open(temp_file_path) as f:
                    json_result = json.load(f)

                with open(temp_file_path) as f:
                    datason_result = datason_json.load(f)

                assert json_result == datason_result, f"Results should be identical for {test_data}"
                assert type(json_result) is type(datason_result), f"Types should be identical for {test_data}"

            finally:
                Path(temp_file_path).unlink(missing_ok=True)

    def test_string_io_compatibility(self):
        """Test compatibility with StringIO objects."""
        test_data = {"string_io": "test", "number": 123}

        # Test dump with StringIO
        json_buffer = io.StringIO()
        datason_buffer = io.StringIO()

        json.dump(test_data, json_buffer)
        datason_json.dump(test_data, datason_buffer)

        json_content = json_buffer.getvalue()
        datason_content = datason_buffer.getvalue()

        # Both should produce valid JSON that parses to the same result
        assert json.loads(json_content) == json.loads(datason_content)

        # Test load with StringIO
        json_buffer.seek(0)
        datason_buffer.seek(0)

        json_loaded = json.load(json_buffer)
        datason_loaded = datason_json.load(datason_buffer)

        assert json_loaded == datason_loaded

    def test_error_compatibility(self):
        """Test that datason raises the same errors as json for invalid input."""

        # Test loads with invalid JSON
        invalid_json_strings = [
            '{"invalid": json}',  # Unquoted value
            '{invalid: "json"}',  # Unquoted key
            '{"incomplete": ',  # Incomplete JSON
            "{",  # Just opening brace
            '"unclosed string',  # Unclosed string
        ]

        for invalid_string in invalid_json_strings:
            json_error = None
            datason_error = None

            try:
                json.loads(invalid_string)
            except Exception as e:
                json_error = type(e)

            try:
                datason_json.loads(invalid_string)
            except Exception as e:
                datason_error = type(e)

            # Both should raise some kind of error (exact error type may vary)
            assert json_error is not None, f"json.loads should error on {invalid_string}"
            assert datason_error is not None, f"datason.loads should error on {invalid_string}"

    def test_complete_drop_in_replacement(self):
        """Comprehensive test that DataSON can completely replace json module."""

        # This test uses DataSON exactly like the json module
        # If this passes, DataSON is a perfect drop-in replacement

        original_data = {
            "api_response": {
                "status": "success",
                "data": [
                    {"id": 1, "name": "Item 1", "active": True, "value": 99.99},
                    {"id": 2, "name": "Item 2", "active": False, "value": None},
                ],
                "pagination": {"page": 1, "per_page": 10, "total": 2, "has_next": False},
            },
            "metadata": {"timestamp": "2024-01-01T00:00:00Z", "version": "1.0", "debug": False},
        }

        # Test the complete round-trip with file I/O
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

            # Write with datason (replacing json.dump)
            datason_json.dump(original_data, temp_file)

        try:
            # Read with datason (replacing json.load)
            with open(temp_path) as f:
                loaded_data = datason_json.load(f)

            # Convert to string with datason (replacing json.dumps)
            json_string = datason_json.dumps(loaded_data, indent=2, sort_keys=True)

            # Parse from string with datason (replacing json.loads)
            parsed_data = datason_json.loads(json_string)

            # All data should be identical
            assert original_data == loaded_data == parsed_data

            # Should also be readable by standard json module
            with open(temp_path) as f:
                json_loaded = json.load(f)
            assert json_loaded == original_data

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_parameter_forwarding(self):
        """Test that parameters are properly forwarded to underlying json functions."""
        test_data = {"z": 3, "a": 1, "m": 2}

        # Test dumps parameter forwarding with JSON compatibility module
        json_result = json.dumps(test_data, sort_keys=True, indent=4, separators=(",", ": "))
        datason_result = datason_json.dumps(test_data, sort_keys=True, indent=4, separators=(",", ": "))

        # Results should be identical
        assert json_result == datason_result

        # Test that both parse back to the same data
        assert json.loads(json_result) == json.loads(datason_result) == test_data

    def test_edge_cases(self):
        """Test edge cases that might break compatibility."""

        # Empty structures
        assert datason_json.dumps([]) == json.dumps([])
        assert datason_json.dumps({}) == json.dumps({})

        # Nested empty structures
        nested_empty = {"empty_list": [], "empty_dict": {}, "null": None}
        assert json.loads(datason_json.dumps(nested_empty)) == nested_empty

        # Large numbers (within JSON spec)
        large_number = 1234567890123456789
        assert datason_json.dumps(large_number) == json.dumps(large_number)

        # Unicode strings
        unicode_data = {"unicode": "Hello 🌍 世界", "emoji": "🚀🎉"}
        json_unicode = json.dumps(unicode_data, ensure_ascii=False)
        datason_unicode = datason_json.dumps(unicode_data, ensure_ascii=False)
        assert json.loads(json_unicode) == json.loads(datason_unicode)


class TestJSONCompatibilityRegression:
    """Additional regression tests to prevent breaking changes."""

    def test_import_compatibility(self):
        """Test that datason can be imported and used like json."""
        # This should work exactly like: import json
        # User should be able to do: import datason as json

        # Test that all expected functions exist
        assert hasattr(datason, "dump")
        assert hasattr(datason, "dumps")
        assert hasattr(datason, "load")
        assert hasattr(datason, "loads")

        # Test that functions are callable
        assert callable(datason.dump)
        assert callable(datason.dumps)
        assert callable(datason.load)
        assert callable(datason.loads)

    def test_signature_compatibility(self):
        """Test that JSON compatibility module signatures are compatible with json module."""
        import inspect

        # Get signatures for JSON compatibility module (not enhanced API)
        json_dumps_sig = inspect.signature(json.dumps)
        datason_json_dumps_sig = inspect.signature(datason_json.dumps)

        # DataSON JSON module should accept the same parameters as json
        # (it can accept more, but must accept all json parameters)
        json_params = set(json_dumps_sig.parameters.keys())
        datason_json_params = set(datason_json_dumps_sig.parameters.keys())

        # Special handling for **kwargs vs **kw (functionally equivalent)
        json_params_normalized = json_params.copy()
        if "kw" in json_params_normalized:
            json_params_normalized.remove("kw")  # json uses **kw
        if "kwargs" in datason_json_params:
            # datason.json uses **kwargs, which is equivalent to **kw
            pass

        # DataSON JSON module should support all standard json parameters (excluding **kw)
        missing_params = json_params_normalized - datason_json_params
        assert not missing_params, f"datason.json.dumps missing json parameters: {missing_params}"

    def test_readme_examples_work(self):
        """Test that common README examples work with DataSON as json replacement."""

        # Common pattern: serialize to string using JSON compatibility module
        data = {"name": "DataSON", "version": "1.0", "features": ["fast", "compatible"]}
        json_str = datason_json.dumps(data)
        assert isinstance(json_str, str)
        assert datason_json.loads(json_str) == data

        # Common pattern: save to file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            datason_json.dump(data, f)
            temp_path = f.name

        try:
            with open(temp_path) as f:
                loaded = datason_json.load(f)
            assert loaded == data
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Common pattern: pretty printing
        pretty = datason_json.dumps(data, indent=2)
        assert "\n" in pretty  # Should be formatted
        assert datason_json.loads(pretty) == data

    def test_no_breaking_changes(self):
        """Test that core JSON functionality never breaks."""

        # This test should NEVER fail in any future version
        # If it fails, we've broken JSON compatibility

        test_cases = [
            # Basic types
            (None, "null"),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (3.14, "3.14"),
            ("hello", '"hello"'),
            # Collections
            ([], "[]"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({}, "{}"),
            ({"key": "value"}, '{"key": "value"}'),
        ]

        for data, expected_pattern in test_cases:
            # dumps should work with JSON compatibility module
            result = datason_json.dumps(data)
            assert isinstance(result, str)

            # loads should work
            parsed = datason_json.loads(result)
            assert parsed == data

            # Should be valid JSON
            json_parsed = json.loads(result)
            assert json_parsed == data


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
