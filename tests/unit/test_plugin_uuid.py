"""Tests for the UUID plugin."""

from __future__ import annotations

import json
import uuid

import pytest

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.uuid import UUIDPlugin


@pytest.fixture()
def plugin() -> UUIDPlugin:
    return UUIDPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


@pytest.fixture()
def sample_uuid() -> uuid.UUID:
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


class TestUUIDPluginProperties:
    def test_name(self, plugin: UUIDPlugin) -> None:
        assert plugin.name == "uuid"

    def test_priority(self, plugin: UUIDPlugin) -> None:
        assert plugin.priority == 101


class TestCanHandle:
    def test_uuid4(self, plugin: UUIDPlugin) -> None:
        assert plugin.can_handle(uuid.uuid4())

    def test_uuid_from_string(self, plugin: UUIDPlugin) -> None:
        assert plugin.can_handle(uuid.UUID("12345678-1234-5678-1234-567812345678"))

    def test_rejects_string(self, plugin: UUIDPlugin) -> None:
        assert not plugin.can_handle("12345678-1234-5678-1234-567812345678")

    def test_rejects_int(self, plugin: UUIDPlugin) -> None:
        assert not plugin.can_handle(12345)


class TestSerialize:
    def test_with_type_hints(self, plugin: UUIDPlugin, sample_uuid: uuid.UUID) -> None:
        ctx = SerializeContext(config=SerializationConfig())
        result = plugin.serialize(sample_uuid, ctx)
        assert result == {
            TYPE_METADATA_KEY: "uuid.UUID",
            VALUE_METADATA_KEY: "12345678-1234-5678-1234-567812345678",
        }

    def test_without_type_hints(self, plugin: UUIDPlugin, sample_uuid: uuid.UUID) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(sample_uuid, ctx)
        assert result == "12345678-1234-5678-1234-567812345678"

    def test_uuid4_format(self, plugin: UUIDPlugin, ser_ctx: SerializeContext) -> None:
        obj = uuid.uuid4()
        result = plugin.serialize(obj, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "uuid.UUID"
        # Validate it's a proper UUID string
        uuid.UUID(result[VALUE_METADATA_KEY])


class TestCanDeserialize:
    def test_uuid_type(self, plugin: UUIDPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "uuid.UUID"})

    def test_rejects_other(self, plugin: UUIDPlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})

    def test_rejects_missing_key(self, plugin: UUIDPlugin) -> None:
        assert not plugin.can_deserialize({"value": "something"})


class TestDeserialize:
    def test_from_string(self, plugin: UUIDPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "uuid.UUID", VALUE_METADATA_KEY: "12345678-1234-5678-1234-567812345678"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert isinstance(result, uuid.UUID)

    def test_bad_type_raises(self, plugin: UUIDPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "uuid.UUID", VALUE_METADATA_KEY: 12345}
        with pytest.raises(PluginError, match="Expected string for UUID"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_uuid_roundtrip(self, sample_uuid: uuid.UUID) -> None:
        serialized = datason.dumps(sample_uuid)
        result = datason.loads(serialized)
        assert result == sample_uuid
        assert isinstance(result, uuid.UUID)

    def test_uuid4_roundtrip(self) -> None:
        obj = uuid.uuid4()
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_uuid_in_dict(self, sample_uuid: uuid.UUID) -> None:
        data = {"id": sample_uuid, "name": "test"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_uuid_in_list(self) -> None:
        uuids = [uuid.uuid4() for _ in range(3)]
        data = {"ids": uuids}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_json_valid(self, sample_uuid: uuid.UUID) -> None:
        serialized = datason.dumps(sample_uuid)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
