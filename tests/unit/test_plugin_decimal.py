"""Tests for the decimal/complex plugin."""

from __future__ import annotations

import decimal
import json

import pytest

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.decimal import DecimalPlugin


@pytest.fixture()
def plugin() -> DecimalPlugin:
    return DecimalPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestDecimalPluginProperties:
    def test_name(self, plugin: DecimalPlugin) -> None:
        assert plugin.name == "decimal"

    def test_priority(self, plugin: DecimalPlugin) -> None:
        assert plugin.priority == 102


class TestCanHandle:
    def test_decimal(self, plugin: DecimalPlugin) -> None:
        assert plugin.can_handle(decimal.Decimal("3.14"))

    def test_complex(self, plugin: DecimalPlugin) -> None:
        assert plugin.can_handle(complex(1, 2))

    def test_rejects_float(self, plugin: DecimalPlugin) -> None:
        assert not plugin.can_handle(3.14)

    def test_rejects_int(self, plugin: DecimalPlugin) -> None:
        assert not plugin.can_handle(42)


class TestSerializeDecimal:
    def test_with_type_hints(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = decimal.Decimal("3.14159")
        result = plugin.serialize(obj, ser_ctx)
        assert result == {TYPE_METADATA_KEY: "decimal.Decimal", VALUE_METADATA_KEY: "3.14159"}

    def test_without_type_hints(self, plugin: DecimalPlugin) -> None:
        obj = decimal.Decimal("3.14")
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(obj, ctx)
        assert result == "3.14"

    def test_negative(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = decimal.Decimal("-99.5")
        result = plugin.serialize(obj, ser_ctx)
        assert result[VALUE_METADATA_KEY] == "-99.5"

    def test_infinity(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = decimal.Decimal("Infinity")
        result = plugin.serialize(obj, ser_ctx)
        assert result[VALUE_METADATA_KEY] == "Infinity"

    def test_nan(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = decimal.Decimal("NaN")
        result = plugin.serialize(obj, ser_ctx)
        assert result[VALUE_METADATA_KEY] == "NaN"


class TestSerializeComplex:
    def test_with_type_hints(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = complex(3, 4)
        result = plugin.serialize(obj, ser_ctx)
        assert result == {TYPE_METADATA_KEY: "complex", VALUE_METADATA_KEY: [3.0, 4.0]}

    def test_without_type_hints(self, plugin: DecimalPlugin) -> None:
        obj = complex(1, -2)
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(obj, ctx)
        assert result == [1.0, -2.0]

    def test_real_only(self, plugin: DecimalPlugin, ser_ctx: SerializeContext) -> None:
        obj = complex(5, 0)
        result = plugin.serialize(obj, ser_ctx)
        assert result[VALUE_METADATA_KEY] == [5.0, 0.0]


class TestCanDeserialize:
    def test_decimal_type(self, plugin: DecimalPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "decimal.Decimal"})

    def test_complex_type(self, plugin: DecimalPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "complex"})

    def test_rejects_other(self, plugin: DecimalPlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})


class TestDeserialize:
    def test_decimal_from_string(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "decimal.Decimal", VALUE_METADATA_KEY: "3.14159"}
        result = plugin.deserialize(data, deser_ctx)
        assert result == decimal.Decimal("3.14159")
        assert isinstance(result, decimal.Decimal)

    def test_decimal_bad_type_raises(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "decimal.Decimal", VALUE_METADATA_KEY: 3.14}
        with pytest.raises(PluginError, match="Expected string for Decimal"):
            plugin.deserialize(data, deser_ctx)

    def test_complex_from_list(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "complex", VALUE_METADATA_KEY: [3.0, 4.0]}
        result = plugin.deserialize(data, deser_ctx)
        assert result == complex(3, 4)
        assert isinstance(result, complex)

    def test_complex_bad_format_raises(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "complex", VALUE_METADATA_KEY: "3+4j"}
        with pytest.raises(PluginError, match="Expected \\[real, imag\\]"):
            plugin.deserialize(data, deser_ctx)

    def test_complex_wrong_length_raises(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "complex", VALUE_METADATA_KEY: [1.0, 2.0, 3.0]}
        with pytest.raises(PluginError, match="Expected \\[real, imag\\]"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: DecimalPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "unknown_decimal", VALUE_METADATA_KEY: "value"}
        with pytest.raises(PluginError, match="Unknown decimal type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_decimal_roundtrip(self) -> None:
        obj = decimal.Decimal("3.14159265358979323846")
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_complex_roundtrip(self) -> None:
        obj = complex(3, 4)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result == obj

    def test_decimal_in_dict(self) -> None:
        data = {"price": decimal.Decimal("19.99"), "name": "widget"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_complex_in_dict(self) -> None:
        data = {"impedance": complex(100, -50), "frequency": 60}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_mixed_numeric_types(self) -> None:
        data = {
            "decimal": decimal.Decimal("99.99"),
            "complex": complex(1, 2),
            "int": 42,
            "float": 3.14,
        }
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result == data

    def test_json_valid(self) -> None:
        obj = decimal.Decimal("3.14")
        serialized = datason.dumps(obj)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
