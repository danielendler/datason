"""Plugin for TensorFlow type serialization.

Handles tf.Tensor (EagerTensor), tf.Variable, and tf.SparseTensor.
Requires TensorFlow eager execution (default in TF2).

This module imports tensorflow directly — if tensorflow is not installed,
the ImportError is caught by plugins/__init__.py and this plugin is
simply not registered.
"""

from __future__ import annotations

from typing import Any, cast

import tensorflow as tf

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._reconstruction import check_dense_allocation
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
        return _deserialize_tensorflow(data, ctx)


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
    indices, values, dense_shape = sparse.indices, sparse.values, sparse.dense_shape
    if indices is None or values is None or dense_shape is None or values.dtype is None:
        raise PluginError("Incomplete TensorFlow sparse components")
    value = {
        "indices": _eager_list(indices),
        "values": _eager_list(values),
        "dense_shape": _eager_list(dense_shape),
        "dtype": values.dtype.name,
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "tf.SparseTensor", VALUE_METADATA_KEY: value}
    return value


def _deserialize_tensorflow(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct a TensorFlow object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "tf.Tensor":
            return _reconstruct_dense(value, ctx, as_variable=False)
        case "tf.Variable":
            return _reconstruct_dense(value, ctx, as_variable=True)
        case "tf.SparseTensor":
            return _reconstruct_sparse(value, ctx)
        case _:
            raise PluginError(f"Unknown TensorFlow type: {type_name}")


def _reconstruct_dense(value: Any, ctx: DeserializeContext, *, as_variable: bool) -> Any:
    """Reconstruct a dense tensor or variable from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for TF tensor, got {type(value).__name__}")
    dtype = tf.dtypes.as_dtype(value.get("dtype", "float32"))
    shape = check_dense_allocation(value["data"], value.get("shape"), dtype.size or 8, ctx)
    with tf.device("/CPU:0"):
        tensor = tf.constant(value["data"], dtype=dtype)
        if shape is not None:
            tensor = tf.reshape(tensor, shape)
        return tf.Variable(tensor) if as_variable else tensor


def _reconstruct_sparse(value: Any, ctx: DeserializeContext) -> tf.SparseTensor:
    """Reconstruct a SparseTensor from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for SparseTensor, got {type(value).__name__}")
    from .._errors import DeserializationError, SecurityError

    shape = value["dense_shape"]
    indices, values = value["indices"], value["values"]
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 or dim > 2**63 - 1 for dim in shape):
        raise DeserializationError("Sparse shape must contain non-negative integer dimensions")
    if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
        raise DeserializationError("Sparse indices and values must have matching lengths")
    if any(not isinstance(row, list) or len(row) != len(shape) for row in indices):
        raise DeserializationError("Sparse indices must match shape rank")
    if any(type(n) is not int or n < 0 or n >= dim for row in indices for n, dim in zip(row, shape, strict=True)):
        raise DeserializationError("Sparse index lies outside shape")
    dtype = tf.dtypes.as_dtype(value.get("dtype", "float32"))
    size = len(values) * ((dtype.size or 8) + 8 * len(shape)) + 8 * len(shape)
    if size > ctx.config.max_input_bytes:
        raise SecurityError("Sparse tensor exceeds reconstruction byte budget")
    with tf.device("/CPU:0"):
        index_tensor = tf.reshape(tf.constant(indices, dtype=tf.int64), [len(values), len(shape)])
        return tf.SparseTensor(indices=index_tensor, values=tf.constant(values, dtype=dtype), dense_shape=shape)


def _eager_list(tensor: Any) -> Any:
    """Reject deferred sparse components before calling their eager-only export."""
    method = getattr(tensor, "numpy", None)
    if not callable(method):
        raise PluginError("TensorFlow sparse serialization requires eager tensors")
    return cast(Any, method()).tolist()
