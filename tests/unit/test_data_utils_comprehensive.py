"""Comprehensive tests for datason.data_utils module.

This module tests data conversion utilities including convert_string_method_votes
function with various edge cases and data structures.
"""

import datason.data_utils as data_utils


class TestConvertStringMethodVotes:
    """Test convert_string_method_votes function."""

    def test_none_input(self):
        """Test None input handling."""
        result = data_utils.convert_string_method_votes(None)
        assert result is None

    def test_empty_dict_input(self):
        """Test empty dictionary input."""
        result = data_utils.convert_string_method_votes({})
        assert result == {}

    def test_dict_without_method_votes(self):
        """Test dictionary without method_votes key."""
        input_data = {"other_key": "value", "number": 42}
        result = data_utils.convert_string_method_votes(input_data)
        assert result == {"other_key": "value", "number": 42}

    def test_dict_with_list_method_votes(self):
        """Test dictionary with already-list method_votes."""
        input_data = {"method_votes": ["method1", "method2"], "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": ["method1", "method2"], "other": "data"}
        assert result == expected

    def test_dict_with_string_list_method_votes(self):
        """Test dictionary with string representation of list method_votes."""
        input_data = {"method_votes": "['method1', 'method2', 'method3']", "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": ["method1", "method2", "method3"], "other": "data"}
        assert result == expected

    def test_dict_with_plain_string_method_votes(self):
        """Test dictionary with plain string method_votes."""
        input_data = {"method_votes": "single_method", "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": ["single_method"], "other": "data"}
        assert result == expected

    def test_dict_with_none_method_votes(self):
        """Test dictionary with None method_votes."""
        input_data = {"method_votes": None, "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": [], "other": "data"}
        assert result == expected

    def test_dict_with_empty_list_method_votes(self):
        """Test dictionary with empty list method_votes."""
        input_data = {"method_votes": [], "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": [], "other": "data"}
        assert result == expected

    def test_dict_with_empty_string_list_method_votes(self):
        """Test dictionary with empty string list method_votes."""
        input_data = {"method_votes": "[]", "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": [], "other": "data"}
        assert result == expected

    def test_empty_list_input(self):
        """Test empty list input."""
        result = data_utils.convert_string_method_votes([])
        assert result == []

    def test_list_with_valid_transactions(self):
        """Test list with valid transaction dictionaries."""
        input_data = [
            {"method_votes": "['method1', 'method2']", "id": 1},
            {"method_votes": "method_single", "id": 2},
            {"method_votes": ["already", "list"], "id": 3},
        ]
        result = data_utils.convert_string_method_votes(input_data)
        expected = [
            {"method_votes": ["method1", "method2"], "id": 1},
            {"method_votes": ["method_single"], "id": 2},
            {"method_votes": ["already", "list"], "id": 3},
        ]
        assert result == expected

    def test_list_with_none_entries(self):
        """Test list containing None entries."""
        input_data = [{"method_votes": "['method1']", "id": 1}, None, {"method_votes": "method2", "id": 2}]
        result = data_utils.convert_string_method_votes(input_data)
        expected = [{"method_votes": ["method1"], "id": 1}, {"method_votes": ["method2"], "id": 2}]
        assert result == expected

    def test_invalid_string_list_syntax(self):
        """Test handling of invalid string list syntax."""
        input_data = {"method_votes": "['method1', 'method2'", "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": [], "other": "data"}
        assert result == expected

    def test_malformed_string_list(self):
        """Test handling of malformed string list."""
        input_data = {"method_votes": "[method1, method2]", "other": "data"}
        result = data_utils.convert_string_method_votes(input_data)
        expected = {"method_votes": [], "other": "data"}
        assert result == expected
