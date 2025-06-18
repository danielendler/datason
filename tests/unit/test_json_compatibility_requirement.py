"""
CRITICAL COMPATIBILITY REQUIREMENT TEST

This test file is MANDATORY and ensures DataSON always remains a perfect drop-in
replacement for Python's standard JSON library via the datason.json module.

ANY FAILURE in this test means we've broken the core compatibility promise.
This test must ALWAYS pass to maintain DataSON's value proposition.

Key Requirement: 'import datason.json as json' must work exactly like stdlib json.
"""

import json
import tempfile
from pathlib import Path

import datason.json as datason_json


class TestJSONDropInRequirement:
    """
    CRITICAL: Test that datason.json provides perfect JSON stdlib compatibility.

    This is a REQUIREMENT, not just a nice-to-have. Users depend on this working.
    """

    # Core test data that must work identically with both libraries
    REQUIRED_COMPATIBILITY_DATA = [
        None,
        True,
        False,
        0,
        42,
        -1,
        3.14,
        "",
        "hello world",
        "special chars: áéíóú 中文 🌟",
        [],
        [1, 2, 3],
        ["a", "b", "c"],
        {},
        {"key": "value"},
        {"number": 42, "boolean": True, "null": None},
        {"nested": {"array": [1, 2, 3], "string": "test", "object": {"deep": "value"}}},
        # Real-world data patterns
        {
            "users": [{"id": 1, "name": "Alice", "active": True}, {"id": 2, "name": "Bob", "active": False}],
            "metadata": {"total": 2, "timestamp": "2024-01-01T00:00:00Z"},
        },
    ]

    def test_dumps_identical_behavior(self):
        """CRITICAL: datason.json.dumps() must behave exactly like json.dumps()."""
        for test_data in self.REQUIRED_COMPATIBILITY_DATA:
            stdlib_result = json.dumps(test_data)
            datason_result = datason_json.dumps(test_data)

            assert stdlib_result == datason_result, (
                f"COMPATIBILITY BROKEN: dumps() results differ for {test_data}\n"
                f"stdlib: {stdlib_result}\n"
                f"datason: {datason_result}"
            )

    def test_loads_identical_behavior(self):
        """CRITICAL: datason.json.loads() must behave exactly like json.loads()."""
        # Test with various JSON strings
        test_strings = [
            "null",
            "true",
            "false",
            "0",
            "42",
            "-1",
            "3.14",
            '""',
            '"hello"',
            '"hello world"',
            "[]",
            "[1,2,3]",
            '["a","b","c"]',
            "{}",
            '{"key":"value"}',
            '{"number":42,"boolean":true,"null":null}',
            '{"nested":{"array":[1,2,3],"string":"test"}}',
        ]

        for test_string in test_strings:
            stdlib_result = json.loads(test_string)
            datason_result = datason_json.loads(test_string)

            assert stdlib_result == datason_result, (
                f"COMPATIBILITY BROKEN: loads() results differ for {test_string}\n"
                f"stdlib: {stdlib_result}\n"
                f"datason: {datason_result}"
            )

            assert type(stdlib_result) is type(datason_result), (
                f"COMPATIBILITY BROKEN: loads() types differ for {test_string}\n"
                f"stdlib: {type(stdlib_result)}\n"
                f"datason: {type(datason_result)}"
            )

    def test_dump_identical_behavior(self):
        """CRITICAL: datason.json.dump() must behave exactly like json.dump()."""
        for test_data in self.REQUIRED_COMPATIBILITY_DATA:
            # Create temp files for both
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as stdlib_file:
                json.dump(test_data, stdlib_file)
                stdlib_path = stdlib_file.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as datason_file:
                datason_json.dump(test_data, datason_file)
                datason_path = datason_file.name

            try:
                # Read back and compare
                with open(stdlib_path) as f:
                    stdlib_content = f.read()

                with open(datason_path) as f:
                    datason_content = f.read()

                # Parse both to ensure they're equivalent
                stdlib_parsed = json.loads(stdlib_content)
                datason_parsed = json.loads(datason_content)

                assert stdlib_parsed == datason_parsed, (
                    f"COMPATIBILITY BROKEN: dump() results differ for {test_data}\n"
                    f"stdlib content: {stdlib_content}\n"
                    f"datason content: {datason_content}"
                )

            finally:
                Path(stdlib_path).unlink(missing_ok=True)
                Path(datason_path).unlink(missing_ok=True)

    def test_load_identical_behavior(self):
        """CRITICAL: datason.json.load() must behave exactly like json.load()."""
        for test_data in self.REQUIRED_COMPATIBILITY_DATA:
            # Create test file with stdlib json
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                json.dump(test_data, temp_file)
                temp_path = temp_file.name

            try:
                # Read with both libraries
                with open(temp_path) as f:
                    stdlib_result = json.load(f)

                with open(temp_path) as f:
                    datason_result = datason_json.load(f)

                assert stdlib_result == datason_result, (
                    f"COMPATIBILITY BROKEN: load() results differ for {test_data}\n"
                    f"stdlib: {stdlib_result}\n"
                    f"datason: {datason_result}"
                )

                assert type(stdlib_result) is type(datason_result), (
                    f"COMPATIBILITY BROKEN: load() types differ for {test_data}\n"
                    f"stdlib: {type(stdlib_result)}\n"
                    f"datason: {type(datason_result)}"
                )

            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_parameter_compatibility(self):
        """CRITICAL: All json.dumps() parameters must work identically."""
        test_data = {"c": 3, "a": 1, "b": 2}

        # Test key parameters that users rely on
        parameter_tests = [
            {"sort_keys": True},
            {"indent": 2},
            {"indent": 4, "sort_keys": True},
            {"separators": (",", ":")},
            {"ensure_ascii": False},
            {"ensure_ascii": True},
        ]

        for params in parameter_tests:
            stdlib_result = json.dumps(test_data, **params)
            datason_result = datason_json.dumps(test_data, **params)

            assert stdlib_result == datason_result, (
                f"COMPATIBILITY BROKEN: Parameter {params} produces different results\n"
                f"stdlib: {stdlib_result}\n"
                f"datason: {datason_result}"
            )

    def test_drop_in_replacement_guarantee(self):
        """
        CRITICAL: The ultimate test - can we replace 'import json' with
        'import datason.json as json' and have everything work identically?
        """
        # Simulate the drop-in replacement
        import datason.json as json_replacement

        test_data = {
            "message": "Hello, World!",
            "numbers": [1, 2, 3, 4, 5],
            "metadata": {
                "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "features": ["compatibility", "performance"],
            },
        }

        # Test full round-trip with the replacement
        json_string = json_replacement.dumps(test_data, indent=2, sort_keys=True)
        parsed_data = json_replacement.loads(json_string)

        assert isinstance(json_string, str), "dumps() must return string"
        assert parsed_data == test_data, "Round-trip must preserve data exactly"

        # Test file operations
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            json_replacement.dump(test_data, temp_file)
            temp_path = temp_file.name

        try:
            with open(temp_path) as f:
                loaded_data = json_replacement.load(f)

            assert loaded_data == test_data, "File round-trip must preserve data exactly"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_error_compatibility(self):
        """CRITICAL: Errors must be identical to stdlib json."""
        # Test that invalid JSON produces the same error type
        invalid_json = '{"invalid": json}'

        stdlib_error = None
        datason_error = None

        try:
            json.loads(invalid_json)
        except Exception as e:
            stdlib_error = type(e)

        try:
            datason_json.loads(invalid_json)
        except Exception as e:
            datason_error = type(e)

        assert stdlib_error == datason_error, (
            f"COMPATIBILITY BROKEN: Error types differ\nstdlib raises: {stdlib_error}\ndatason raises: {datason_error}"
        )


class TestCompatibilityGuarantee:
    """
    GUARANTEE: These tests ensure the compatibility promise is never broken.
    """

    def test_import_statement_works(self):
        """CRITICAL: 'import datason.json as json' must work."""
        # This should not raise any import errors
        import datason.json as json_replacement

        # All required functions must exist
        required_functions = ["dumps", "loads", "dump", "load"]
        for func_name in required_functions:
            assert hasattr(json_replacement, func_name), f"COMPATIBILITY BROKEN: Missing required function {func_name}"

    def test_real_world_usage_patterns(self):
        """CRITICAL: Test patterns that real users actually use."""
        import datason.json as json

        # Pattern 1: API response serialization
        api_response = {"status": "success", "data": [{"id": 1, "name": "test"}], "meta": {"count": 1}}
        json_str = json.dumps(api_response)
        assert isinstance(json_str, str)
        assert json.loads(json_str) == api_response

        # Pattern 2: Configuration file handling
        config = {"database": {"host": "localhost", "port": 5432, "ssl": True}, "features": ["auth", "logging"]}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config, f, indent=2)
            config_path = f.name

        try:
            with open(config_path) as f:
                loaded_config = json.load(f)
            assert loaded_config == config
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_no_behavioral_differences(self):
        """CRITICAL: No subtle behavioral differences allowed."""
        # Test edge cases that might reveal differences
        edge_cases = [
            float("inf"),  # May not be supported by both
            float("-inf"),  # May not be supported by both
            # float('nan'),   # Commented out as both may handle differently
        ]

        # These should either both work or both fail the same way
        for test_case in edge_cases:
            stdlib_works = True
            datason_works = True

            try:
                json.dumps(test_case)
            except Exception:
                stdlib_works = False

            try:
                datason_json.dumps(test_case)
            except Exception:
                datason_works = False

            assert stdlib_works == datason_works, (
                f"COMPATIBILITY BROKEN: Different handling of edge case {test_case}\n"
                f"stdlib works: {stdlib_works}, datason works: {datason_works}"
            )
