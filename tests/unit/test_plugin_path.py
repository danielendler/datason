"""Tests for the path plugin."""

from __future__ import annotations

import json
import pathlib

import pytest

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.path import PathPlugin


@pytest.fixture()
def plugin() -> PathPlugin:
    return PathPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestPathPluginProperties:
    def test_name(self, plugin: PathPlugin) -> None:
        assert plugin.name == "path"

    def test_priority(self, plugin: PathPlugin) -> None:
        assert plugin.priority == 103


class TestCanHandle:
    def test_posix_path(self, plugin: PathPlugin) -> None:
        assert plugin.can_handle(pathlib.PurePosixPath("/usr/bin/python"))

    def test_windows_path(self, plugin: PathPlugin) -> None:
        assert plugin.can_handle(pathlib.PureWindowsPath("C:/Users/test"))

    def test_path(self, plugin: PathPlugin) -> None:
        assert plugin.can_handle(pathlib.Path("/home/user/file.txt"))

    def test_rejects_string(self, plugin: PathPlugin) -> None:
        assert not plugin.can_handle("/home/user/file.txt")

    def test_rejects_int(self, plugin: PathPlugin) -> None:
        assert not plugin.can_handle(42)


class TestSerialize:
    def test_with_type_hints(self, plugin: PathPlugin, ser_ctx: SerializeContext) -> None:
        obj = pathlib.PurePosixPath("/home/user/data.csv")
        result = plugin.serialize(obj, ser_ctx)
        assert result == {TYPE_METADATA_KEY: "pathlib.Path", VALUE_METADATA_KEY: "/home/user/data.csv"}

    def test_without_type_hints(self, plugin: PathPlugin) -> None:
        obj = pathlib.PurePosixPath("/home/user/data.csv")
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(obj, ctx)
        assert result == "/home/user/data.csv"

    def test_relative_path(self, plugin: PathPlugin, ser_ctx: SerializeContext) -> None:
        obj = pathlib.PurePosixPath("relative/path/file.txt")
        result = plugin.serialize(obj, ser_ctx)
        assert result[VALUE_METADATA_KEY] == "relative/path/file.txt"


class TestCanDeserialize:
    def test_path_type(self, plugin: PathPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "pathlib.Path"})

    def test_rejects_other(self, plugin: PathPlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})

    def test_rejects_missing_key(self, plugin: PathPlugin) -> None:
        assert not plugin.can_deserialize({"value": "something"})


class TestDeserialize:
    def test_from_string(self, plugin: PathPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pathlib.Path", VALUE_METADATA_KEY: "/home/user/data.csv"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == pathlib.Path("/home/user/data.csv")
        assert isinstance(result, pathlib.Path)

    def test_bad_type_raises(self, plugin: PathPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "pathlib.Path", VALUE_METADATA_KEY: 12345}
        with pytest.raises(PluginError, match="Expected string for Path"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_path_roundtrip(self) -> None:
        obj = pathlib.Path("/home/user/documents/report.pdf")
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj
        assert isinstance(result, pathlib.Path)

    def test_path_in_dict(self) -> None:
        data = {"config": pathlib.Path("/etc/app/config.yaml"), "name": "app"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_multiple_paths(self) -> None:
        data = {
            "input": pathlib.Path("/data/input.csv"),
            "output": pathlib.Path("/data/output.json"),
            "log": pathlib.Path("/var/log/app.log"),
        }
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_json_valid(self) -> None:
        obj = pathlib.Path("/home/user/file.txt")
        serialized = datason.dumps(obj)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
