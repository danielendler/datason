"""Tests for datason deserialization engine."""

from __future__ import annotations

import pytest

import datason
from datason._errors import SecurityError


class TestLoadsBasicTypes:
    """Test loads() with JSON-native types."""

    def test_string(self):
        assert datason.loads('"hello"') == "hello"

    def test_integer(self):
        assert datason.loads("42") == 42

    def test_float(self):
        assert datason.loads("3.14") == 3.14

    def test_boolean_true(self):
        assert datason.loads("true") is True

    def test_boolean_false(self):
        assert datason.loads("false") is False

    def test_null(self):
        assert datason.loads("null") is None


class TestLoadsContainers:
    """Test loads() with dicts and lists."""

    def test_empty_dict(self):
        assert datason.loads("{}") == {}

    def test_empty_list(self):
        assert datason.loads("[]") == []

    def test_nested_dict(self):
        data = '{"a": {"b": {"c": 1}}}'
        assert datason.loads(data) == {"a": {"b": {"c": 1}}}

    def test_list_of_mixed(self):
        data = '[1, "two", null, true]'
        assert datason.loads(data) == [1, "two", None, True]


class TestDeserializeDepthLimit:
    """Test that depth limits are enforced during deserialization."""

    def test_deep_nesting_raises(self):
        # Build a deeply nested JSON string
        deep = '{"a": ' * 60 + "1" + "}" * 60
        with pytest.raises(SecurityError, match="depth"):
            datason.loads(deep)
