"""Tests for the NumPy plugin."""

from __future__ import annotations

import json

import numpy as np
import pytest

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.numpy import NumpyPlugin


@pytest.fixture()
def plugin() -> NumpyPlugin:
    return NumpyPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestNumpyPluginProperties:
    def test_name(self, plugin: NumpyPlugin) -> None:
        assert plugin.name == "numpy"

    def test_priority(self, plugin: NumpyPlugin) -> None:
        assert plugin.priority == 200


class TestCanHandle:
    def test_ndarray(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_handle(np.array([1, 2, 3]))

    def test_int64(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_handle(np.int64(42))

    def test_float64(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_handle(np.float64(3.14))

    def test_bool_(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_handle(np.bool_(True))

    def test_complex128(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_handle(np.complex128(1 + 2j))

    def test_rejects_python_list(self, plugin: NumpyPlugin) -> None:
        assert not plugin.can_handle([1, 2, 3])

    def test_rejects_python_int(self, plugin: NumpyPlugin) -> None:
        assert not plugin.can_handle(42)

    def test_rejects_python_float(self, plugin: NumpyPlugin) -> None:
        assert not plugin.can_handle(3.14)


class TestSerializeNdarray:
    def test_1d_with_hints(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        arr = np.array([1, 2, 3])
        result = plugin.serialize(arr, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "numpy.ndarray"
        assert result[VALUE_METADATA_KEY]["data"] == [1, 2, 3]
        assert result[VALUE_METADATA_KEY]["shape"] == [3]

    def test_2d_array(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        arr = np.array([[1, 2], [3, 4]])
        result = plugin.serialize(arr, ser_ctx)
        assert result[VALUE_METADATA_KEY]["data"] == [[1, 2], [3, 4]]
        assert result[VALUE_METADATA_KEY]["shape"] == [2, 2]

    def test_dtype_preserved(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = plugin.serialize(arr, ser_ctx)
        assert result[VALUE_METADATA_KEY]["dtype"] == "float32"

    def test_without_hints(self, plugin: NumpyPlugin) -> None:
        arr = np.array([1, 2, 3])
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        result = plugin.serialize(arr, ctx)
        assert result == [1, 2, 3]


class TestSerializeScalars:
    def test_int64(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        result = plugin.serialize(np.int64(42), ser_ctx)
        assert result == {TYPE_METADATA_KEY: "numpy.integer", VALUE_METADATA_KEY: 42}

    def test_float64(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        result = plugin.serialize(np.float64(3.14), ser_ctx)
        assert result[TYPE_METADATA_KEY] == "numpy.floating"
        assert abs(result[VALUE_METADATA_KEY] - 3.14) < 1e-10

    def test_bool_(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        result = plugin.serialize(np.bool_(True), ser_ctx)
        assert result == {TYPE_METADATA_KEY: "numpy.bool_", VALUE_METADATA_KEY: True}

    def test_complex128(self, plugin: NumpyPlugin, ser_ctx: SerializeContext) -> None:
        result = plugin.serialize(np.complex128(3 + 4j), ser_ctx)
        assert result == {TYPE_METADATA_KEY: "numpy.complex", VALUE_METADATA_KEY: [3.0, 4.0]}

    def test_scalar_without_hints(self, plugin: NumpyPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        assert plugin.serialize(np.int64(42), ctx) == 42
        assert plugin.serialize(np.float64(3.14), ctx) == pytest.approx(3.14)
        assert plugin.serialize(np.bool_(False), ctx) is False


class TestCanDeserialize:
    def test_ndarray(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "numpy.ndarray"})

    def test_integer(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "numpy.integer"})

    def test_floating(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "numpy.floating"})

    def test_bool(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "numpy.bool_"})

    def test_complex(self, plugin: NumpyPlugin) -> None:
        assert plugin.can_deserialize({TYPE_METADATA_KEY: "numpy.complex"})

    def test_rejects_non_numpy(self, plugin: NumpyPlugin) -> None:
        assert not plugin.can_deserialize({TYPE_METADATA_KEY: "datetime"})

    def test_rejects_missing_key(self, plugin: NumpyPlugin) -> None:
        assert not plugin.can_deserialize({"value": "something"})


class TestDeserialize:
    def test_ndarray_1d(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "numpy.ndarray",
            VALUE_METADATA_KEY: {"data": [1, 2, 3], "dtype": "int64", "shape": [3]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_ndarray_2d(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "numpy.ndarray",
            VALUE_METADATA_KEY: {"data": [[1, 2], [3, 4]], "dtype": "int64", "shape": [2, 2]},
        }
        result = plugin.deserialize(data, deser_ctx)
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_ndarray_float32(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "numpy.ndarray",
            VALUE_METADATA_KEY: {"data": [1.0, 2.5], "dtype": "float32", "shape": [2]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert result.dtype == np.float32

    def test_integer(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.integer", VALUE_METADATA_KEY: 42}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, np.int64)
        assert result == 42

    def test_floating(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.floating", VALUE_METADATA_KEY: 3.14}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, np.float64)

    def test_bool_(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.bool_", VALUE_METADATA_KEY: True}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, np.bool_)
        assert result is np.bool_(True)

    def test_complex(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.complex", VALUE_METADATA_KEY: [3.0, 4.0]}
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, np.complex128)
        assert result == 3 + 4j

    def test_ndarray_bad_value_raises(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict for ndarray"):
            plugin.deserialize(data, deser_ctx)

    def test_complex_bad_format_raises(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.complex", VALUE_METADATA_KEY: "3+4j"}
        with pytest.raises(PluginError, match="Expected \\[real, imag\\]"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: NumpyPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "numpy.unknown", VALUE_METADATA_KEY: 42}
        with pytest.raises(PluginError, match="Unknown numpy type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_ndarray_1d(self) -> None:
        arr = np.array([1, 2, 3, 4, 5])
        serialized = datason.dumps(arr)
        result = datason.loads(serialized)
        np.testing.assert_array_equal(result, arr)

    def test_ndarray_2d_float(self) -> None:
        arr = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
        serialized = datason.dumps(arr)
        result = datason.loads(serialized)
        np.testing.assert_array_almost_equal(result, arr)

    def test_int64_scalar(self) -> None:
        obj = np.int64(42)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert isinstance(result, np.int64)
        assert result == 42

    def test_float64_scalar(self) -> None:
        """np.float64 inherits from float — JSON fast-path preserves value, not type."""
        obj = np.float64(3.14159)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        # Value is preserved; type degrades to Python float (expected)
        assert abs(result - 3.14159) < 1e-10

    def test_bool_scalar(self) -> None:
        """np.bool_ inherits from bool — JSON fast-path preserves value, not type."""
        obj = np.bool_(True)
        serialized = datason.dumps(obj)
        result = datason.loads(serialized)
        assert result is True or result == True  # noqa: E712

    def test_ndarray_in_dict(self) -> None:
        data = {"weights": np.array([0.1, 0.9]), "name": "model"}
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        assert result["name"] == "model"
        np.testing.assert_array_almost_equal(result["weights"], data["weights"])

    def test_json_valid(self) -> None:
        arr = np.array([1, 2, 3])
        serialized = datason.dumps(arr)
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_mixed_numpy_types(self) -> None:
        data = {
            "array": np.array([1.0, 2.0, 3.0]),
            "count": np.int64(100),
            "score": np.float64(0.95),
            "flag": np.bool_(True),
        }
        serialized = datason.dumps(data)
        result = datason.loads(serialized)
        np.testing.assert_array_almost_equal(result["array"], data["array"])
        assert result["count"] == 100
        assert isinstance(result["flag"], np.bool_)
