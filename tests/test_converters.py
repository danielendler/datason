"""Tests for the safe converter functions."""

from datetime import datetime

from serialpy.converters import safe_float, safe_int
from serialpy.datetime_utils import convert_pandas_timestamps_recursive


class TestSafeFloat:
    """Test the safe_float function."""

    def test_safe_float_valid_numbers(self) -> None:
        """Test safe_float with valid numeric inputs."""
        assert safe_float(42) == 42.0
        assert safe_float(3.14) == 3.14
        assert safe_float("42.5") == 42.5
        assert safe_float("-10.7") == -10.7

    def test_safe_float_none(self) -> None:
        """Test safe_float with None input."""
        assert safe_float(None) == 0.0
        assert safe_float(None, default=5.0) == 5.0

    def test_safe_float_nan(self) -> None:
        """Test safe_float with NaN input."""
        assert safe_float(float("nan")) == 0.0
        assert safe_float(float("nan"), default=10.0) == 10.0

    def test_safe_float_infinity(self) -> None:
        """Test safe_float with infinity values."""
        assert safe_float(float("inf")) == 0.0
        assert safe_float(float("-inf")) == 0.0
        assert safe_float(float("inf"), default=42.0) == 42.0

    def test_safe_float_invalid_strings(self) -> None:
        """Test safe_float with invalid string inputs."""
        assert safe_float("invalid") == 0.0
        assert safe_float("") == 0.0
        assert safe_float("not_a_number", default=7.5) == 7.5

    def test_safe_float_invalid_types(self) -> None:
        """Test safe_float with invalid type inputs."""
        assert safe_float([1, 2, 3]) == 0.0
        assert safe_float({"key": "value"}) == 0.0
        assert safe_float(object(), default=15.0) == 15.0

    def test_safe_float_boolean(self) -> None:
        """Test safe_float with boolean inputs."""
        assert safe_float(True) == 1.0
        assert safe_float(False) == 0.0


class TestSafeInt:
    """Test the safe_int function."""

    def test_safe_int_valid_numbers(self) -> None:
        """Test safe_int with valid numeric inputs."""
        assert safe_int(42) == 42
        assert safe_int(3.14) == 3
        assert safe_int("42") == 42
        assert safe_int("-10") == -10

    def test_safe_int_none(self) -> None:
        """Test safe_int with None input."""
        assert safe_int(None) == 0
        assert safe_int(None, default=5) == 5

    def test_safe_int_nan(self) -> None:
        """Test safe_int with NaN input."""
        assert safe_int(float("nan")) == 0
        assert safe_int(float("nan"), default=10) == 10

    def test_safe_int_float_truncation(self) -> None:
        """Test safe_int truncates float values."""
        assert safe_int(42.7) == 42
        assert safe_int(-10.9) == -10
        assert safe_int(0.1) == 0

    def test_safe_int_invalid_strings(self) -> None:
        """Test safe_int with invalid string inputs."""
        assert safe_int("invalid") == 0
        assert safe_int("") == 0
        assert safe_int("not_a_number", default=7) == 7

    def test_safe_int_invalid_types(self) -> None:
        """Test safe_int with invalid type inputs."""
        assert safe_int([1, 2, 3]) == 0
        assert safe_int({"key": "value"}) == 0
        assert safe_int(object(), default=15) == 15

    def test_safe_int_boolean(self) -> None:
        """Test safe_int with boolean inputs."""
        assert safe_int(True) == 1
        assert safe_int(False) == 0

    def test_safe_int_edge_cases(self) -> None:
        """Test safe_int with edge case inputs."""
        # Large numbers should work
        assert safe_int(1000000) == 1000000
        assert safe_int(-1000000) == -1000000

        # String representations of floats
        assert safe_int("42.0") == 42


class TestDateTimeUtilities:
    """Test the datetime utility functions."""

    def test_convert_pandas_timestamps_recursive(self) -> None:
        """Test the recursive pandas timestamp converter."""
        # Test with basic data that doesn't require pandas
        data = {"text": "hello", "number": 42, "date": datetime(2023, 1, 1)}
        result = convert_pandas_timestamps_recursive(data)
        assert result["text"] == "hello"
        assert result["number"] == 42
        assert result["date"] == datetime(2023, 1, 1)
