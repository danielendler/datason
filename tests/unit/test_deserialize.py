"""Tests for datason deserialization engine."""

from __future__ import annotations

import pytest

import datason
from datason._errors import DeserializationError, SecurityError


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


class TestPluginDeserializationSafety:
    """Test plugin deserialization safety controls for untrusted inputs."""

    def test_plugins_disabled_blocks_plugin_execution(self, monkeypatch):
        def _raise_if_called(data, ctx):
            raise RuntimeError("exploit plugin executed")

        monkeypatch.setattr("datason._deserialize.default_registry.find_deserializer", _raise_if_called)
        payload = '{"__datason_type__": "exploit", "__datason_value__": "boom"}'

        with pytest.raises(DeserializationError, match="allow_plugin_deserialization"):
            datason.loads(payload, allow_plugin_deserialization=False)

    def test_plugins_enabled_executes_plugin_path(self, monkeypatch):
        def _raise_if_called(data, ctx):
            raise RuntimeError("exploit plugin executed")

        monkeypatch.setattr("datason._deserialize.default_registry.find_deserializer", _raise_if_called)
        payload = '{"__datason_type__": "exploit", "__datason_value__": "boom"}'

        with pytest.raises(RuntimeError, match="exploit plugin executed"):
            datason.loads(payload, allow_plugin_deserialization=True)

    def test_builtin_collection_hints_still_work_when_plugins_disabled(self):
        payload = '{"__datason_type__": "tuple", "__datason_value__": [1, 2, 3]}'
        assert datason.loads(payload, allow_plugin_deserialization=False) == (1, 2, 3)


class TestLoadKwargValidation:
    def test_load_reports_correct_function_name(self):
        with pytest.raises(TypeError, match=r"load\(\) got an unexpected keyword argument"):
            datason.load(__import__("io").StringIO("{}"), nope=True)
