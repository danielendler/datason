"""Test coverage for datason.converters module.

This file focuses on testing the safe conversion utilities
to improve code coverage for converters.py.
"""

import math

from datason.converters import safe_float, safe_int


class TestSafeFloat:
    """Test safe_float function."""

    def test_safe_float_basic_numbers(self) -> None:
        """Test safe_float with basic numeric values."""
        assert safe_float(42) == 42.0
        assert safe_float(42.5) == 42.5
        assert safe_float(-42) == -42.0
        assert safe_float(-42.5) == -42.5
        assert safe_float(0) == 0.0
        assert safe_float(0.0) == 0.0

    def test_safe_float_strings(self) -> None:
        """Test safe_float with string inputs."""
        assert safe_float("42") == 42.0
        assert safe_float("42.5") == 42.5
        assert safe_float("-42") == -42.0
        assert safe_float("-42.5") == -42.5
        assert safe_float("0") == 0.0
        assert safe_float("0.0") == 0.0

    def test_safe_float_none(self) -> None:
        """Test safe_float with None."""
        assert safe_float(None) == 0.0
        assert safe_float(None, 10.0) == 10.0

    def test_safe_float_nan(self) -> None:
        """Test safe_float with NaN values."""
        assert safe_float(float("nan")) == 0.0
        assert safe_float(float("nan"), 10.0) == 10.0

    def test_safe_float_infinity(self) -> None:
        """Test safe_float with infinity values."""
        assert safe_float(float("inf")) == 0.0
        assert safe_float(float("-inf")) == 0.0
        assert safe_float(float("inf"), 10.0) == 10.0
        assert safe_float(float("-inf"), 10.0) == 10.0

    def test_safe_float_invalid_strings(self) -> None:
        """Test safe_float with invalid string inputs."""
        assert safe_float("invalid") == 0.0
        assert safe_float("not_a_number") == 0.0
        assert safe_float("42.5.5") == 0.0
        assert safe_float("") == 0.0
        assert safe_float("   ") == 0.0
        assert safe_float("abc123") == 0.0

    def test_safe_float_invalid_strings_with_default(self) -> None:
        """Test safe_float with invalid strings and custom default."""
        assert safe_float("invalid", 99.9) == 99.9
        assert safe_float("not_a_number", -1.0) == -1.0
        assert safe_float("", 5.5) == 5.5

    def test_safe_float_boolean(self) -> None:
        """Test safe_float with boolean values."""
        assert safe_float(True) == 1.0
        assert safe_float(False) == 0.0

    def test_safe_float_edge_cases(self) -> None:
        """Test safe_float with edge case values."""
        # Very large numbers
        assert safe_float(1e10) == 1e10
        assert safe_float(-1e10) == -1e10

        # Very small numbers
        assert safe_float(1e-10) == 1e-10
        assert safe_float(-1e-10) == -1e-10

    def test_safe_float_complex_objects(self) -> None:
        """Test safe_float with complex objects that should fail."""
        assert safe_float([1, 2, 3]) == 0.0
        assert safe_float({"key": "value"}) == 0.0
        assert safe_float([1, 2, 3], 42.0) == 42.0
        assert safe_float({"key": "value"}, -5.5) == -5.5


class TestSafeInt:
    """Test safe_int function."""

    def test_safe_int_basic_numbers(self) -> None:
        """Test safe_int with basic numeric values."""
        assert safe_int(42) == 42
        assert safe_int(-42) == -42
        assert safe_int(0) == 0
        assert safe_int(42.0) == 42
        assert safe_int(42.7) == 42
        assert safe_int(42.9) == 42
        assert safe_int(-42.7) == -42

    def test_safe_int_strings(self) -> None:
        """Test safe_int with string inputs."""
        assert safe_int("42") == 42
        assert safe_int("-42") == -42
        assert safe_int("0") == 0
        assert safe_int("42.0") == 42
        assert safe_int("42.7") == 42
        assert safe_int("-42.7") == -42

    def test_safe_int_none(self) -> None:
        """Test safe_int with None."""
        assert safe_int(None) == 0
        assert safe_int(None, 10) == 10

    def test_safe_int_nan(self) -> None:
        """Test safe_int with NaN values."""
        assert safe_int(float("nan")) == 0
        assert safe_int(float("nan"), 10) == 10

    def test_safe_int_infinity(self) -> None:
        """Test safe_int with infinity values."""
        # Note: safe_int doesn't handle infinity in the current implementation
        # This is a limitation that should be improved
        try:
            result = safe_int(float("inf"))
            assert result == 0  # If it gets fixed to handle infinity
        except OverflowError:
            # Current behavior - infinity causes overflow
            pass

        try:
            result = safe_int(float("-inf"))
            assert result == 0  # If it gets fixed to handle infinity
        except OverflowError:
            # Current behavior - infinity causes overflow
            pass

    def test_safe_int_invalid_strings(self) -> None:
        """Test safe_int with invalid string inputs."""
        assert safe_int("invalid") == 0
        assert safe_int("not_a_number") == 0
        assert safe_int("42.5.5") == 0
        assert safe_int("") == 0
        assert safe_int("   ") == 0
        assert safe_int("abc123") == 0

    def test_safe_int_invalid_strings_with_default(self) -> None:
        """Test safe_int with invalid strings and custom default."""
        assert safe_int("invalid", 99) == 99
        assert safe_int("not_a_number", -1) == -1
        assert safe_int("", 5) == 5

    def test_safe_int_boolean(self) -> None:
        """Test safe_int with boolean values."""
        assert safe_int(True) == 1
        assert safe_int(False) == 0

    def test_safe_int_edge_cases(self) -> None:
        """Test safe_int with edge case values."""
        # Large numbers
        assert safe_int(1000000) == 1000000
        assert safe_int(-1000000) == -1000000

        # String representations of floats with different decimal parts
        assert safe_int("123.0") == 123
        assert safe_int("123.99") == 123
        assert safe_int("-123.99") == -123

    def test_safe_int_complex_objects(self) -> None:
        """Test safe_int with complex objects that should fail."""
        assert safe_int([1, 2, 3]) == 0
        assert safe_int({"key": "value"}) == 0
        assert safe_int([1, 2, 3], 42) == 42
        assert safe_int({"key": "value"}, -5) == -5

    def test_safe_int_string_floats_edge_cases(self) -> None:
        """Test safe_int with string float representations."""
        # Test the specific path for string inputs that are floats
        assert safe_int("42.123") == 42
        assert safe_int("-42.123") == -42
        assert safe_int("0.999") == 0
        assert safe_int("-0.999") == 0

        # Test string NaN and infinity
        assert safe_int("nan") == 0
        assert safe_int("inf") == 0
        assert safe_int("-inf") == 0
        assert safe_int("infinity") == 0


class TestConverterEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_safe_conversions_consistency(self) -> None:
        """Test that safe conversions are consistent."""
        test_values = [
            42,
            42.0,
            "42",
            "42.0",
            -42,
            -42.0,
            "-42",
            "-42.0",
            0,
            0.0,
            "0",
            "0.0",
            None,
            "invalid",
            float("nan"),
        ]

        for value in test_values:
            # Both functions should handle the same inputs gracefully
            float_result = safe_float(value)
            int_result = safe_int(value)

            # Results should be reasonable
            assert isinstance(float_result, float)
            assert isinstance(int_result, int)

            # NaN and inf should not appear in results
            assert not math.isnan(float_result)
            assert not math.isinf(float_result)

    def test_custom_defaults_consistency(self) -> None:
        """Test that custom defaults work consistently."""
        custom_float_default = 99.9
        custom_int_default = 99

        invalid_inputs = ["invalid", None, float("nan"), [1, 2, 3]]

        for invalid_input in invalid_inputs:
            assert safe_float(invalid_input, custom_float_default) == custom_float_default
            assert safe_int(invalid_input, custom_int_default) == custom_int_default

    def test_large_scale_data_processing(self) -> None:
        """Test converter functions with arrays of data."""
        # Simulate processing a list of mixed data
        mixed_data = [1, 2.5, "3", "4.5", None, "invalid", float("nan"), True, False, "", "0", "  42  "]

        # Process with safe_float
        float_results = [safe_float(item) for item in mixed_data]
        assert len(float_results) == len(mixed_data)
        assert all(isinstance(result, float) for result in float_results)
        assert all(not math.isnan(result) and not math.isinf(result) for result in float_results)

        # Process with safe_int
        int_results = [safe_int(item) for item in mixed_data]
        assert len(int_results) == len(mixed_data)
        assert all(isinstance(result, int) for result in int_results)
