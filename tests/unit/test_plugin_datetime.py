"""Tests for the datetime plugin."""

from __future__ import annotations

import datetime as dt
import json

import pytest

import datason
from datason._config import DateFormat, SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.datetime import DatetimePlugin


@pytest.fixture()
def plugin() -> DatetimePlugin:
    return DatetimePlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestDatetimePluginProperties:
    def test_name(self, plugin: DatetimePlugin) -> None:
        assert plugin.name == "datetime"

    def test_priority(self, plugin: DatetimePlugin) -> None:
        assert plugin.priority == 100


class TestCanHandle:
    def test_datetime(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_handle(dt.datetime(2024, 1, 15, 10, 30))

    def test_date(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_handle(dt.date(2024, 1, 15))

    def test_time(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_handle(dt.time(10, 30, 45))

    def test_timedelta(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_handle(dt.timedelta(days=5, hours=3))

    def test_rejects_string(self, plugin: DatetimePlugin) -> None:
        assert not plugin.can_handle("2024-01-15")

    def test_rejects_int(self, plugin: DatetimePlugin) -> None:
        assert not plugin.can_handle(1705312200)


class TestSerializeDatetime:
    def test_iso_format(self, plugin: DatetimePlugin) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.ISO))
        result = plugin.serialize(obj, ctx)
        assert result == {TYPE_METADATA_KEY: "datetime", VALUE_METADATA_KEY: "2024-01-15T10:30:00"}

    def test_unix_format(self, plugin: DatetimePlugin) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0, tzinfo=dt.timezone.utc)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.UNIX))
        result = plugin.serialize(obj, ctx)
        assert result[TYPE_METADATA_KEY] == "datetime"
        assert isinstance(result[VALUE_METADATA_KEY], float)

    def test_unix_ms_format(self, plugin: DatetimePlugin) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0, tzinfo=dt.timezone.utc)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.UNIX_MS))
        result = plugin.serialize(obj, ctx)
        assert result[VALUE_METADATA_KEY] == obj.timestamp() * 1000

    def test_string_format(self, plugin: DatetimePlugin) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.STRING))
        result = plugin.serialize(obj, ctx)
        assert result[VALUE_METADATA_KEY] == str(obj)

    def test_no_type_hints(self, plugin: DatetimePlugin) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(obj, ctx)
        assert isinstance(result, str)
        assert TYPE_METADATA_KEY not in str(result)


class TestSerializeDate:
    def test_iso_format(self, plugin: DatetimePlugin) -> None:
        obj = dt.date(2024, 1, 15)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.ISO))
        result = plugin.serialize(obj, ctx)
        assert result == {TYPE_METADATA_KEY: "date", VALUE_METADATA_KEY: "2024-01-15"}

    def test_unix_falls_back_to_iso(self, plugin: DatetimePlugin) -> None:
        """date has no timestamp(), so UNIX format falls back to isoformat."""
        obj = dt.date(2024, 1, 15)
        ctx = SerializeContext(config=SerializationConfig(date_format=DateFormat.UNIX))
        result = plugin.serialize(obj, ctx)
        assert result[VALUE_METADATA_KEY] == "2024-01-15"


class TestSerializeTime:
    def test_time_iso(self, plugin: DatetimePlugin) -> None:
        obj = dt.time(10, 30, 45)
        ctx = SerializeContext(config=SerializationConfig())
        result = plugin.serialize(obj, ctx)
        assert result == {TYPE_METADATA_KEY: "time", VALUE_METADATA_KEY: "10:30:45"}


class TestSerializeTimedelta:
    def test_timedelta_seconds(self, plugin: DatetimePlugin) -> None:
        obj = dt.timedelta(hours=2, minutes=30)
        ctx = SerializeContext(config=SerializationConfig())
        result = plugin.serialize(obj, ctx)
        assert result == {TYPE_METADATA_KEY: "timedelta", VALUE_METADATA_KEY: 9000.0}


class TestCanDeserialize:
    def test_datetime(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})

    def test_date(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "date"})

    def test_time(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "time"})

    def test_timedelta(self, plugin: DatetimePlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "timedelta"})

    def test_rejects_unknown(self, plugin: DatetimePlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "uuid.UUID"})

    def test_rejects_missing_key(self, plugin: DatetimePlugin) -> None:
        assert not plugin.can_deserialize({"some_key": "value"})


class TestDeserialize:
    def test_datetime_from_iso(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "datetime", VALUE_METADATA_KEY: "2024-01-15T10:30:00"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == dt.datetime(2024, 1, 15, 10, 30, 0)

    def test_datetime_from_timestamp(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        ts = dt.datetime(2024, 1, 15, 10, 30, 0, tzinfo=dt.timezone.utc).timestamp()
        data = {TYPE_METADATA_KEY: "datetime", VALUE_METADATA_KEY: ts}
        result = plugin.deserialize(data, deser_ctx)
        assert result == dt.datetime(2024, 1, 15, 10, 30, 0, tzinfo=dt.timezone.utc)

    def test_date_from_iso(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "date", VALUE_METADATA_KEY: "2024-01-15"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == dt.date(2024, 1, 15)

    def test_time_from_iso(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "time", VALUE_METADATA_KEY: "10:30:45"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == dt.time(10, 30, 45)

    def test_timedelta_from_seconds(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "timedelta", VALUE_METADATA_KEY: 9000.0}
        result = plugin.deserialize(data, deser_ctx)
        assert result == dt.timedelta(hours=2, minutes=30)

    def test_datetime_bad_type_raises(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "datetime", VALUE_METADATA_KEY: [2024, 1, 15]}
        with pytest.raises(PluginError, match="Cannot deserialize datetime"):
            plugin.deserialize(data, deser_ctx)

    def test_date_bad_type_raises(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "date", VALUE_METADATA_KEY: 12345}
        with pytest.raises(PluginError, match="Cannot deserialize date"):
            plugin.deserialize(data, deser_ctx)

    def test_time_bad_type_raises(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "time", VALUE_METADATA_KEY: 12345}
        with pytest.raises(PluginError, match="Cannot deserialize time"):
            plugin.deserialize(data, deser_ctx)

    def test_timedelta_bad_type_raises(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "timedelta", VALUE_METADATA_KEY: "2 hours"}
        with pytest.raises(PluginError, match="Cannot deserialize timedelta"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: DatetimePlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "unknown_dt", VALUE_METADATA_KEY: "value"}
        with pytest.raises(PluginError, match="Unknown datetime type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    """Test full serialize → JSON → deserialize round-trips via datason API."""

    def test_datetime_roundtrip(self) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_date_roundtrip(self) -> None:
        obj = dt.date(2024, 1, 15)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_time_roundtrip(self) -> None:
        obj = dt.time(10, 30, 45)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_timedelta_roundtrip(self) -> None:
        obj = dt.timedelta(days=5, hours=3, seconds=42)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_naive_datetime_unix_ms_roundtrip_preserves_naive(self) -> None:
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        serialized = datason.dumps(obj, date_format=DateFormat.UNIX_MS)
        result = datason.loads(serialized)
        assert result == obj
        assert result.tzinfo is None

    def test_datetime_in_dict(self) -> None:
        data = {"created": dt.datetime(2024, 1, 15, 10, 30), "name": "test"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_mixed_datetime_types(self) -> None:
        data = {
            "timestamp": dt.datetime(2024, 1, 15, 10, 30),
            "date": dt.date(2024, 1, 15),
            "time": dt.time(10, 30),
            "duration": dt.timedelta(hours=2),
        }
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_json_valid(self) -> None:
        """Ensure serialized output is valid JSON."""
        obj = dt.datetime(2024, 1, 15, 10, 30, 0)
        serialized = datason.dumps(obj)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
