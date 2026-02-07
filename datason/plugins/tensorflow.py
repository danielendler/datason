"""Plugin for TensorFlow type serialization.

Handles tf.Tensor (EagerTensor), tf.Variable, and tf.SparseTensor.
Requires TensorFlow eager execution (default in TF2).

This module imports tensorflow directly — if tensorflow is not installed,
the ImportError is caught by plugins/__init__.py and this plugin is
simply not registered.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class TensorFlowPlugin:
    """Handles serialization/deserialization of TensorFlow types."""

    @property
    def name(self) -> str:
        return "tensorflow"

    @property
    def priority(self) -> int:
        return 301

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, tf.Tensor | tf.Variable | tf.SparseTensor)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_tensorflow(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("tf.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_tensorflow(data)


def _serialize_tensorflow(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a TensorFlow object to JSON-safe representation."""
    # Check SparseTensor first (most specific)
    if isinstance(obj, tf.SparseTensor):
        return _serialize_sparse_tensor(obj, ctx)
    # Variable before Tensor (Variable is not a subclass of Tensor)
    if isinstance(obj, tf.Variable):
        return _serialize_dense(obj, ctx, "tf.Variable")
    if isinstance(obj, tf.Tensor):
        return _serialize_dense(obj, ctx, "tf.Tensor")
    raise PluginError(f"Unsupported TensorFlow type: {type(obj).__name__}")


def _serialize_dense(tensor: Any, ctx: SerializeContext, type_name: str) -> Any:
    """Serialize a dense tensor or variable."""
    value = {
        "data": tensor.numpy().tolist(),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: value}
    return tensor.numpy().tolist()


def _serialize_sparse_tensor(sparse: tf.SparseTensor, ctx: SerializeContext) -> Any:
    """Serialize a SparseTensor with indices, values, and shape."""
    value = {
        "indices": sparse.indices.numpy().tolist(),
        "values": sparse.values.numpy().tolist(),
        "dense_shape": sparse.dense_shape.numpy().tolist(),
        "dtype": sparse.values.dtype.name,
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "tf.SparseTensor", VALUE_METADATA_KEY: value}
    return value


def _deserialize_tensorflow(data: dict[str, Any]) -> Any:
    """Reconstruct a TensorFlow object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "tf.Tensor":
            return _reconstruct_dense(value, as_variable=False)
        case "tf.Variable":
            return _reconstruct_dense(value, as_variable=True)
        case "tf.SparseTensor":
            return _reconstruct_sparse(value)
        case _:
            raise PluginError(f"Unknown TensorFlow type: {type_name}")


def _reconstruct_dense(value: Any, *, as_variable: bool) -> Any:
    """Reconstruct a dense tensor or variable from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for TF tensor, got {type(value).__name__}")
    dtype = tf.dtypes.as_dtype(value.get("dtype", "float32"))
    tensor = tf.constant(value["data"], dtype=dtype)
    if as_variable:
        return tf.Variable(tensor)
    return tensor


def _reconstruct_sparse(value: Any) -> tf.SparseTensor:
    """Reconstruct a SparseTensor from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for SparseTensor, got {type(value).__name__}")
    return tf.SparseTensor(
        indices=value["indices"],
        values=tf.constant(value["values"], dtype=tf.dtypes.as_dtype(value.get("dtype", "float32"))),
        dense_shape=value["dense_shape"],
    )
