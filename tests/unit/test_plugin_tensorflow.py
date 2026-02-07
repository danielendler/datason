"""Tests for the TensorFlow plugin."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.tensorflow import TensorFlowPlugin


@pytest.fixture()
def plugin() -> TensorFlowPlugin:
    return TensorFlowPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig())


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


class TestTensorFlowPluginProperties:
    def test_name(self, plugin: TensorFlowPlugin) -> None:
        assert plugin.name == "tensorflow"

    def test_priority(self, plugin: TensorFlowPlugin) -> None:
        assert plugin.priority == 301


class TestCanHandle:
    def test_eager_tensor(self, plugin: TensorFlowPlugin) -> None:
        assert plugin.can_handle(tf.constant([1.0, 2.0]))

    def test_variable(self, plugin: TensorFlowPlugin) -> None:
        assert plugin.can_handle(tf.Variable([1.0, 2.0]))

    def test_sparse_tensor(self, plugin: TensorFlowPlugin) -> None:
        sparse = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[3, 3])
        assert plugin.can_handle(sparse)

    def test_rejects_numpy_array(self, plugin: TensorFlowPlugin) -> None:
        assert not plugin.can_handle(np.array([1.0]))

    def test_rejects_python_list(self, plugin: TensorFlowPlugin) -> None:
        assert not plugin.can_handle([1.0, 2.0])


class TestSerializeTensor:
    def test_1d_with_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = tf.constant([1.0, 2.0, 3.0])
        result = plugin.serialize(tensor, ctx)
        assert result[TYPE_METADATA_KEY] == "tf.Tensor"
        assert result[VALUE_METADATA_KEY]["data"] == [1.0, 2.0, 3.0]
        assert result[VALUE_METADATA_KEY]["dtype"] == "float32"
        assert result[VALUE_METADATA_KEY]["shape"] == [3]

    def test_2d_with_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = tf.constant([[1, 2], [3, 4]])
        result = plugin.serialize(tensor, ctx)
        assert result[VALUE_METADATA_KEY]["shape"] == [2, 2]

    def test_dtype_preserved(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        tensor = tf.constant([1.0], dtype=tf.float64)
        result = plugin.serialize(tensor, ctx)
        assert result[VALUE_METADATA_KEY]["dtype"] == "float64"

    def test_without_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        tensor = tf.constant([1.0, 2.0])
        result = plugin.serialize(tensor, ctx)
        assert result == [1.0, 2.0]


class TestSerializeVariable:
    def test_with_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        var = tf.Variable([1.0, 2.0, 3.0])
        result = plugin.serialize(var, ctx)
        assert result[TYPE_METADATA_KEY] == "tf.Variable"
        assert result[VALUE_METADATA_KEY]["data"] == [1.0, 2.0, 3.0]

    def test_without_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        var = tf.Variable([1.0, 2.0])
        result = plugin.serialize(var, ctx)
        assert result == [1.0, 2.0]


class TestSerializeSparseTensor:
    def test_with_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=True))
        sparse = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[3, 3])
        result = plugin.serialize(sparse, ctx)
        assert result[TYPE_METADATA_KEY] == "tf.SparseTensor"
        value = result[VALUE_METADATA_KEY]
        assert value["indices"] == [[0, 0], [1, 2]]
        assert value["values"] == [1.0, 2.0]
        assert value["dense_shape"] == [3, 3]

    def test_without_hints(self, plugin: TensorFlowPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        sparse = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[2, 2])
        result = plugin.serialize(sparse, ctx)
        assert isinstance(result, dict)
        assert "indices" in result


class TestCanDeserialize:
    def test_tensor(self, plugin: TensorFlowPlugin) -> None:
        data = {TYPE_METADATA_KEY: "tf.Tensor", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_variable(self, plugin: TensorFlowPlugin) -> None:
        data = {TYPE_METADATA_KEY: "tf.Variable", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_sparse_tensor(self, plugin: TensorFlowPlugin) -> None:
        data = {TYPE_METADATA_KEY: "tf.SparseTensor", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_rejects_non_tf(self, plugin: TensorFlowPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.Tensor", VALUE_METADATA_KEY: {}}
        assert not plugin.can_deserialize(data)

    def test_rejects_missing_key(self, plugin: TensorFlowPlugin) -> None:
        assert not plugin.can_deserialize({"foo": "bar"})


class TestDeserialize:
    def test_tensor_1d(self, plugin: TensorFlowPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "tf.Tensor",
            VALUE_METADATA_KEY: {"data": [1.0, 2.0, 3.0], "dtype": "float32", "shape": [3]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, tf.Tensor)
        np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])

    def test_variable(self, plugin: TensorFlowPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "tf.Variable",
            VALUE_METADATA_KEY: {"data": [1.0, 2.0], "dtype": "float32", "shape": [2]},
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, tf.Variable)

    def test_sparse_tensor(self, plugin: TensorFlowPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "tf.SparseTensor",
            VALUE_METADATA_KEY: {
                "indices": [[0, 0], [1, 2]],
                "values": [1.0, 2.0],
                "dense_shape": [3, 3],
                "dtype": "float32",
            },
        }
        result = plugin.deserialize(data, deser_ctx)
        assert isinstance(result, tf.SparseTensor)

    def test_tensor_bad_value_raises(self, plugin: TensorFlowPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "tf.Tensor", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: TensorFlowPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "tf.unknown", VALUE_METADATA_KEY: {}}
        with pytest.raises(PluginError, match="Unknown TensorFlow type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_tensor_1d(self) -> None:
        tensor = tf.constant([1.0, 2.0, 3.0])
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, tf.Tensor)
        np.testing.assert_array_almost_equal(result.numpy(), tensor.numpy())

    def test_variable(self) -> None:
        var = tf.Variable([1.0, 2.0])
        s = datason.dumps(var, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, tf.Variable)

    def test_sparse_tensor(self) -> None:
        sparse = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[3, 3])
        s = datason.dumps(sparse, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, tf.SparseTensor)

    def test_tensor_in_dict(self) -> None:
        data = {"weights": tf.constant([0.1, 0.9]), "label": "model"}
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result["weights"], tf.Tensor)
        assert result["label"] == "model"

    def test_json_valid(self) -> None:
        tensor = tf.constant([1.0, 2.0])
        s = datason.dumps(tensor, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_float64_dtype(self) -> None:
        tensor = tf.constant([1.0, 2.0], dtype=tf.float64)
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert result.dtype == tf.float64

    def test_int32_tensor(self) -> None:
        tensor = tf.constant([1, 2, 3], dtype=tf.int32)
        s = datason.dumps(tensor, include_type_hints=True)
        result = datason.loads(s)
        assert result.dtype == tf.int32
