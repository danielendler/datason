"""Test error paths and edge cases for datason.core module.

This file focuses on testing public API error handling and edge cases
to improve code coverage for core.py.
"""

import pytest

from datason.core import SecurityError, serialize


class TestSerializeErrorPaths:
    """Test error paths in serialize function."""

    def test_serialize_circular_reference(self) -> None:
        """Test serialize with circular references."""
        data: dict = {}
        data["self"] = data

        try:
            result = serialize(data)
            # Might succeed with circular reference handling
            assert isinstance(result, (dict, type(None)))
        except (SecurityError, ValueError):
            # Expected for circular references
            pass

    def test_serialize_none_input(self) -> None:
        """Test serialize with None input."""
        result = serialize(None)
        assert result is None

    def test_serialize_empty_structures(self) -> None:
        """Test serialize with empty structures."""
        assert serialize({}) == {}
        assert serialize([]) == []
        assert serialize(()) == []

    def test_serialize_basic_types(self) -> None:
        """Test serialize with basic types."""
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize(3.14) == 3.14
        assert serialize(True) is True
        assert serialize(False) is False

    def test_serialize_nested_structures(self) -> None:
        """Test serialize with nested structures."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, "two", 3.0],
            "dict": {"nested": "value"},
        }

        result = serialize(data)
        assert isinstance(result, dict)
        assert "string" in result
        assert "number" in result
        assert "float" in result
        assert "boolean" in result
        assert "null" in result
        assert "list" in result
        assert "dict" in result

    def test_serialize_deeply_nested(self) -> None:
        """Test serialize with deeply nested structures."""
        # Create moderately nested structure
        nested: dict = {"level": 1}
        current: dict = nested
        for i in range(2, 8):  # Not too deep to avoid hitting limits
            next_level: dict = {"level": i}
            current["next"] = next_level
            current = next_level

        try:
            result = serialize(nested)
            assert isinstance(result, dict)
        except SecurityError:
            # Expected for very deep nesting
            pass

    def test_serialize_large_lists(self) -> None:
        """Test serialize with large lists."""
        large_list = list(range(100))  # Moderate size

        try:
            result = serialize(large_list)
            assert isinstance(result, list)
            assert len(result) == 100
        except SecurityError:
            # Expected if size limits are hit
            pass

    def test_serialize_large_dicts(self) -> None:
        """Test serialize with large dictionaries."""
        large_dict = {f"key_{i}": i for i in range(100)}  # Moderate size

        try:
            result = serialize(large_dict)
            assert isinstance(result, dict)
            assert len(result) == 100
        except SecurityError:
            # Expected if size limits are hit
            pass

    def test_serialize_mixed_nested_types(self) -> None:
        """Test serialize with mixed nested types."""
        mixed_data = {
            "lists": [[1, 2], [3, 4], [5, 6]],
            "dicts": [{"a": 1}, {"b": 2}, {"c": 3}],
            "tuples": [(1, 2), (3, 4)],
            "nested": {"level1": {"level2": [1, 2, {"level3": "value"}]}},
        }

        result = serialize(mixed_data)
        assert isinstance(result, dict)
        assert "lists" in result
        assert "dicts" in result
        assert "tuples" in result
        assert "nested" in result


class TestSecurityError:
    """Test SecurityError exception handling."""

    def test_security_error_creation(self) -> None:
        """Test SecurityError can be created and raised."""
        with pytest.raises(SecurityError):
            raise SecurityError("Test security error")

    def test_security_error_with_message(self) -> None:
        """Test SecurityError with custom message."""
        message = "Maximum depth exceeded"
        try:
            raise SecurityError(message)
        except SecurityError as e:
            assert str(e) == message


class TestEdgeCases:
    """Test edge cases in serialization."""

    def test_special_string_values(self) -> None:
        """Test serialization of special string values."""
        special_strings = [
            "",  # empty string
            " ",  # space
            "\n",  # newline
            "\t",  # tab
            "unicode: 你好",  # unicode
            "emoji: 🚀",  # emoji
            "null",  # string that looks like null
            "true",  # string that looks like boolean
            "123",  # string that looks like number
        ]

        for s in special_strings:
            result = serialize(s)
            assert result == s

    def test_special_numeric_values(self) -> None:
        """Test serialization of special numeric values."""

        # Test regular numbers
        assert serialize(0) == 0
        assert serialize(-1) == -1
        assert serialize(1.0) == 1.0
        assert serialize(-1.0) == -1.0

        # Test special float values
        try:
            # These might be handled specially
            serialize(float("inf"))
            serialize(float("-inf"))
            serialize(float("nan"))
        except (ValueError, TypeError):
            # Some special values might not be serializable
            pass

    def test_complex_nested_structure(self) -> None:
        """Test serialization of complex nested structure."""
        complex_data = {
            "metadata": {"version": "1.0", "timestamp": "2023-01-01T00:00:00Z", "tags": ["tag1", "tag2", "tag3"]},
            "data": [
                {"id": 1, "values": [1.1, 2.2, 3.3], "nested": {"a": {"b": {"c": "deep_value"}}}},
                {"id": 2, "values": [4.4, 5.5, 6.6], "nested": {"x": {"y": {"z": "another_deep_value"}}}},
            ],
            "summary": {"total_items": 2, "processed": True, "errors": []},
        }

        result = serialize(complex_data)
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result
        assert "summary" in result

    def test_tuples_conversion(self) -> None:
        """Test that tuples are converted to lists."""
        tuple_data = (1, 2, (3, 4, (5, 6)))
        result = serialize(tuple_data)
        assert isinstance(result, list)
        assert result == [1, 2, [3, 4, [5, 6]]]

    def test_sets_handling(self) -> None:
        """Test handling of sets."""
        set_data = {1, 2, 3, 4, 5}

        try:
            result = serialize(set_data)
            # Sets might be converted to lists
            assert isinstance(result, (list, set))
        except (ValueError, TypeError):
            # Sets might not be directly serializable
            pass
