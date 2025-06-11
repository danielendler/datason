"""Comprehensive tests for datason.converters module.

This module tests safe conversion utilities including safe_float and safe_int
functions with various edge cases and error conditions.
"""

import datason.converters as converters


class TestSafeFloat:
    """Test safe_float function."""

    def test_safe_float_basic_conversions(self):
        """Test basic float conversions."""
        assert converters.safe_float(42) == 42.0
        assert converters.safe_float(42.5) == 42.5
        assert converters.safe_float("42.5") == 42.5
        assert converters.safe_float("42") == 42.0
        assert converters.safe_float(-3.14) == -3.14
        assert converters.safe_float("-3.14") == -3.14

    def test_safe_float_none_handling(self):
        """Test None input handling."""
        assert converters.safe_float(None) == 0.0
        assert converters.safe_float(None, 10.0) == 10.0
        assert converters.safe_float(None, -5.5) == -5.5

    def test_safe_float_nan_handling(self):
        """Test NaN value handling."""
        assert converters.safe_float(float("nan")) == 0.0
        assert converters.safe_float(float("nan"), 42.0) == 42.0
        assert converters.safe_float(float("nan"), -10.5) == -10.5

    def test_safe_float_infinity_handling(self):
        """Test infinity value handling."""
        assert converters.safe_float(float("inf")) == 0.0
        assert converters.safe_float(float("-inf")) == 0.0
        assert converters.safe_float(float("inf"), 99.9) == 99.9
        assert converters.safe_float(float("-inf"), -99.9) == -99.9

    def test_safe_float_invalid_string_handling(self):
        """Test invalid string input handling."""
        assert converters.safe_float("invalid") == 0.0
        assert converters.safe_float("not_a_number") == 0.0
        assert converters.safe_float("") == 0.0
        assert converters.safe_float("abc123") == 0.0
        assert converters.safe_float("12.34.56") == 0.0

        # With custom defaults
        assert converters.safe_float("invalid", 25.0) == 25.0
        assert converters.safe_float("", -7.5) == -7.5

    def test_safe_float_type_error_handling(self):
        """Test TypeError handling for non-convertible types."""
        assert converters.safe_float([1, 2, 3]) == 0.0
        assert converters.safe_float({"key": "value"}) == 0.0
        assert converters.safe_float(lambda x: x) == 0.0

        # With custom defaults
        assert converters.safe_float([1, 2, 3], 15.0) == 15.0
        assert converters.safe_float({"key": "value"}, -20.0) == -20.0

    def test_safe_float_boolean_handling(self):
        """Test boolean input handling."""
        assert converters.safe_float(True) == 1.0
        assert converters.safe_float(False) == 0.0

    def test_safe_float_edge_cases(self):
        """Test edge cases and boundary values."""
        # Very large numbers
        assert converters.safe_float(1e308) == 1e308
        assert converters.safe_float(-1e308) == -1e308

        # Very small numbers
        assert converters.safe_float(1e-308) == 1e-308
        assert converters.safe_float(-1e-308) == -1e-308

        # Zero variations
        assert converters.safe_float(0) == 0.0
        assert converters.safe_float(0.0) == 0.0
        assert converters.safe_float(-0.0) == 0.0
        assert converters.safe_float("0") == 0.0
        assert converters.safe_float("0.0") == 0.0

    def test_safe_float_string_edge_cases(self):
        """Test string edge cases."""
        # Whitespace handling
        assert converters.safe_float("  42.5  ") == 42.5
        assert converters.safe_float("\t-3.14\n") == -3.14

        # Scientific notation
        assert converters.safe_float("1e3") == 1000.0
        assert converters.safe_float("1.5e-2") == 0.015
        assert converters.safe_float("-2.5e4") == -25000.0

        # Special string values that should return default
        assert converters.safe_float("inf") == 0.0  # Infinity string becomes infinity float, then default
        assert converters.safe_float("-inf") == 0.0  # Negative infinity string becomes infinity float, then default
        assert converters.safe_float("nan") == 0.0  # NaN string becomes NaN float, then default

    def test_safe_float_custom_defaults(self):
        """Test various custom default values."""
        assert converters.safe_float(None, 100.0) == 100.0
        assert converters.safe_float("invalid", -50.25) == -50.25
        assert converters.safe_float(float("nan"), 0.001) == 0.001
        assert converters.safe_float(float("inf"), 999.999) == 999.999


class TestSafeInt:
    """Test safe_int function."""

    def test_safe_int_basic_conversions(self):
        """Test basic integer conversions."""
        assert converters.safe_int(42) == 42
        assert converters.safe_int(42.0) == 42
        assert converters.safe_int(42.7) == 42  # Truncates
        assert converters.safe_int(-42.9) == -42  # Truncates towards zero
        assert converters.safe_int("42") == 42
        assert converters.safe_int("-42") == -42

    def test_safe_int_none_handling(self):
        """Test None input handling."""
        assert converters.safe_int(None) == 0
        assert converters.safe_int(None, 10) == 10
        assert converters.safe_int(None, -5) == -5

    def test_safe_int_nan_handling(self):
        """Test NaN value handling."""
        assert converters.safe_int(float("nan")) == 0
        assert converters.safe_int(float("nan"), 42) == 42
        assert converters.safe_int(float("nan"), -10) == -10

    def test_safe_int_infinity_handling(self):
        """Test infinity handling in strings."""
        # Direct float infinity should be handled by the float check
        assert converters.safe_int(float("inf"), 99) == 99
        assert converters.safe_int(float("-inf"), -99) == -99

    def test_safe_int_string_float_conversions(self):
        """Test string representations of floats."""
        assert converters.safe_int("42.0") == 42
        assert converters.safe_int("42.7") == 42
        assert converters.safe_int("-42.9") == -42
        assert converters.safe_int("3.14159") == 3

    def test_safe_int_string_infinity_handling(self):
        """Test string representations of infinity."""
        assert converters.safe_int("inf") == 0  # Default case
        assert converters.safe_int("-inf") == 0  # Default case
        assert converters.safe_int("inf", 50) == 50
        assert converters.safe_int("-inf", -50) == -50

    def test_safe_int_string_nan_handling(self):
        """Test string representations of NaN."""
        assert converters.safe_int("nan") == 0  # Default case
        assert converters.safe_int("nan", 25) == 25

    def test_safe_int_invalid_string_handling(self):
        """Test invalid string input handling."""
        assert converters.safe_int("invalid") == 0
        assert converters.safe_int("not_a_number") == 0
        assert converters.safe_int("") == 0
        assert converters.safe_int("abc123") == 0
        assert converters.safe_int("12.34.56") == 0

        # With custom defaults
        assert converters.safe_int("invalid", 25) == 25
        assert converters.safe_int("", -7) == -7

    def test_safe_int_type_error_handling(self):
        """Test TypeError handling for non-convertible types."""
        assert converters.safe_int([1, 2, 3]) == 0
        assert converters.safe_int({"key": "value"}) == 0
        assert converters.safe_int(lambda x: x) == 0

        # With custom defaults
        assert converters.safe_int([1, 2, 3], 15) == 15
        assert converters.safe_int({"key": "value"}, -20) == -20

    def test_safe_int_boolean_handling(self):
        """Test boolean input handling."""
        assert converters.safe_int(True) == 1
        assert converters.safe_int(False) == 0

    def test_safe_int_large_numbers(self):
        """Test large number handling."""
        # Large but valid integers
        assert converters.safe_int(2**31 - 1) == 2**31 - 1  # Max 32-bit signed int
        assert converters.safe_int(2**63 - 1) == 2**63 - 1  # Max 64-bit signed int
        assert converters.safe_int(-(2**31)) == -(2**31)

        # Large floats that can be converted to int
        assert converters.safe_int(1e6) == 1000000
        assert converters.safe_int(-1e6) == -1000000

    def test_safe_int_string_edge_cases(self):
        """Test string edge cases."""
        # Whitespace handling
        assert converters.safe_int("  42  ") == 42
        assert converters.safe_int("\t-42\n") == -42

        # Scientific notation in strings
        assert converters.safe_int("1e3") == 1000
        assert converters.safe_int("1.5e2") == 150
        assert converters.safe_int("-2e2") == -200

        # Edge case: scientific notation that results in float
        assert converters.safe_int("1.5e-1") == 0  # 0.15 -> 0

    def test_safe_int_zero_variations(self):
        """Test various zero representations."""
        assert converters.safe_int(0) == 0
        assert converters.safe_int(0.0) == 0
        assert converters.safe_int(-0.0) == 0
        assert converters.safe_int("0") == 0
        assert converters.safe_int("0.0") == 0
        assert converters.safe_int("-0") == 0
        assert converters.safe_int("-0.0") == 0

    def test_safe_int_custom_defaults(self):
        """Test various custom default values."""
        assert converters.safe_int(None, 100) == 100
        assert converters.safe_int("invalid", -50) == -50
        assert converters.safe_int(float("nan"), 1) == 1
        assert converters.safe_int(float("inf"), 999) == 999
        assert converters.safe_int([1, 2, 3], 0) == 0

    def test_safe_int_floating_point_precision(self):
        """Test floating point precision edge cases."""
        # Very close to integer but not quite
        assert converters.safe_int(41.99999999999999) == 41
        assert converters.safe_int(42.00000000000001) == 42

        # Large floats that might have precision issues
        assert converters.safe_int(1.23456789e10) == int(1.23456789e10)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for both functions."""

    def test_extreme_values(self):
        """Test extreme input values."""
        # Test with very large numbers
        large_float = 1e100
        assert converters.safe_float(large_float) == large_float

        # Test with very small numbers
        small_float = 1e-100
        assert converters.safe_float(small_float) == small_float

        # Test safe_int with floats that are too large for int
        try:
            huge_float = float(10**1000)  # This might be inf
            result = converters.safe_int(huge_float, 999)
            assert result == 999  # Should use default due to overflow or inf
        except OverflowError:
            # Some systems might raise OverflowError for very large floats
            pass

    def test_multiple_consecutive_calls(self):
        """Test that functions work correctly with multiple calls."""
        # Test that function state is not affected by previous calls
        assert converters.safe_float("invalid") == 0.0
        assert converters.safe_float(42.5) == 42.5
        assert converters.safe_float("invalid") == 0.0

        assert converters.safe_int("invalid") == 0
        assert converters.safe_int(42) == 42
        assert converters.safe_int("invalid") == 0

    def test_function_with_same_default_values(self):
        """Test functions with the same default values they already use."""
        # These should behave the same as without specifying default
        assert converters.safe_float(None, 0.0) == 0.0
        assert converters.safe_float("invalid", 0.0) == 0.0

        assert converters.safe_int(None, 0) == 0
        assert converters.safe_int("invalid", 0) == 0

    def test_docstring_examples(self):
        """Test examples from function docstrings to ensure they work."""
        # safe_float examples
        assert converters.safe_float(42.5) == 42.5
        assert converters.safe_float(None) == 0.0
        assert converters.safe_float(float("nan")) == 0.0
        assert converters.safe_float(float("inf")) == 0.0
        assert converters.safe_float("invalid", 10.0) == 10.0

        # safe_int examples
        assert converters.safe_int(42) == 42
        assert converters.safe_int(42.7) == 42
        assert converters.safe_int(None) == 0
        assert converters.safe_int(float("nan")) == 0
        assert converters.safe_int("invalid", 10) == 10

    def test_type_consistency(self):
        """Test that return types are consistent."""
        # safe_float should always return float
        assert isinstance(converters.safe_float(42), float)
        assert isinstance(converters.safe_float("42"), float)
        assert isinstance(converters.safe_float(None), float)
        assert isinstance(converters.safe_float("invalid"), float)

        # safe_int should always return int
        assert isinstance(converters.safe_int(42.5), int)
        assert isinstance(converters.safe_int("42"), int)
        assert isinstance(converters.safe_int(None), int)
        assert isinstance(converters.safe_int("invalid"), int)
