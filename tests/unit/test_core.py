"""Tests for datason core serialization engine."""

from __future__ import annotations

import json

import pytest

import datason
from datason._config import NanHandling
from datason._errors import SecurityError, SerializationError


class TestDumpsBasicTypes:
    """Test dumps() with JSON-native types."""

    def test_string(self):
        assert datason.dumps("hello") == '"hello"'

    def test_integer(self):
        assert datason.dumps(42) == "42"

    def test_float(self):
        assert datason.dumps(3.14) == "3.14"

    def test_boolean_true(self):
        assert datason.dumps(True) == "true"

    def test_boolean_false(self):
        assert datason.dumps(False) == "false"

    def test_none(self):
        assert datason.dumps(None) == "null"

    def test_empty_dict(self):
        assert datason.dumps({}) == "{}"

    def test_empty_list(self):
        assert datason.dumps([]) == "[]"


class TestDumpsContainers:
    """Test dumps() with dicts, lists, tuples, sets."""

    def test_flat_dict(self):
        result = json.loads(datason.dumps({"a": 1, "b": 2}))
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        data = {"outer": {"inner": {"deep": 42}}}
        result = json.loads(datason.dumps(data))
        assert result == data

    def test_list_of_ints(self):
        assert datason.dumps([1, 2, 3]) == "[1, 2, 3]"

    def test_tuple_becomes_list(self):
        result = json.loads(datason.dumps((1, 2, 3)))
        assert result == [1, 2, 3]

    def test_set_becomes_sorted_list(self):
        result = json.loads(datason.dumps({3, 1, 2}))
        assert sorted(result) == [1, 2, 3]

    def test_mixed_nested(self, sample_data):
        result = json.loads(datason.dumps(sample_data))
        assert result == sample_data


class TestRoundTrip:
    """Test dumps() -> loads() round-trip for basic types."""

    @pytest.mark.parametrize(
        "value",
        [
            "hello",
            42,
            3.14,
            True,
            False,
            None,
            [1, 2, 3],
            {"key": "value"},
            {"nested": {"list": [1, "two", None]}},
        ],
    )
    def test_round_trip(self, value):
        assert datason.loads(datason.dumps(value)) == value


class TestNanHandling:
    """Test NaN and Infinity handling."""

    def test_nan_to_null_default(self):
        result = json.loads(datason.dumps(float("nan")))
        assert result is None

    def test_nan_to_string(self):
        result = json.loads(datason.dumps(float("nan"), nan_handling=NanHandling.STRING))
        assert result == "nan"

    def test_inf_to_null_default(self):
        result = json.loads(datason.dumps(float("inf")))
        assert result is None

    def test_neg_inf_to_null(self):
        result = json.loads(datason.dumps(float("-inf")))
        assert result is None


class TestSecurityLimits:
    """Test that security limits are enforced."""

    def test_depth_limit_raises(self):
        # Build deeply nested dict
        data: dict = {}
        current = data
        for _i in range(60):
            current["next"] = {}
            current = current["next"]

        with pytest.raises(SecurityError, match="depth"):
            datason.dumps(data)

    def test_dict_size_limit_raises(self):
        big = {f"key_{i}": i for i in range(200_000)}
        with pytest.raises(SecurityError, match="size"):
            datason.dumps(big)

    def test_list_size_limit_raises(self):
        big = list(range(200_000))
        with pytest.raises(SecurityError, match="size"):
            datason.dumps(big)

    def test_circular_reference_raises(self):
        data: dict = {}
        data["self"] = data
        with pytest.raises(SecurityError, match="Circular"):
            datason.dumps(data)


class TestUnknownTypes:
    """Test behavior with unregistered types."""

    def test_unknown_type_raises_by_default(self):
        with pytest.raises(SerializationError, match="Cannot serialize"):
            datason.dumps(object())

    def test_unknown_type_fallback_to_string(self):
        result = datason.dumps(object(), fallback_to_string=True)
        assert "object" in result


class TestConfigContext:
    """Test the config() context manager."""

    def test_sort_keys(self):
        data = {"b": 2, "a": 1}
        with datason.config(sort_keys=True):
            result = datason.dumps(data)
        assert result == '{"a": 1, "b": 2}'
