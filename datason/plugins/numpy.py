"""Plugin for NumPy type serialization.

Handles ndarray, scalar types (integer, floating, bool_, str_),
and complex types. This module imports numpy directly — if numpy
is not installed, the ImportError is caught by plugins/__init__.py
and this plugin is simply not registered.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._errors import DeserializationError, PluginError, SecurityError, SerializationError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class NumpyPlugin:
    """Handles serialization/deserialization of NumPy types."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def priority(self) -> int:
        return 200

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray | np.generic)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_numpy(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("numpy.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        from .._deserialize import _deserialize_recursive

        try:
            if data[TYPE_METADATA_KEY] == "numpy.ndarray":
                value = data[VALUE_METADATA_KEY]
                if isinstance(value, dict) and np.dtype(value.get("dtype")).kind == "O":
                    data = {
                        **data,
                        VALUE_METADATA_KEY: {**value, "data": _deserialize_recursive(value["data"], ctx.child())},
                    }
            return _deserialize_numpy(data, ctx)
        except (ValueError, TypeError, OverflowError, IndexError, KeyError) as exc:
            raise DeserializationError("Invalid NumPy payload") from exc


def _serialize_numpy(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a NumPy object to JSON-safe representation."""
    if isinstance(obj, np.ndarray):
        return _serialize_ndarray(obj, ctx)
    if isinstance(obj, np.datetime64 | np.timedelta64):
        return _serialize_scalar(obj, ctx, int(obj.view("i8")), "numpy.temporal")
    if isinstance(obj, np.void):
        raise SerializationError("Structured and void NumPy dtypes require an explicit custom plugin")
    if isinstance(obj, np.integer):
        return _serialize_scalar(obj, ctx, int(obj), "numpy.integer")
    if isinstance(obj, np.floating):
        return _serialize_scalar(obj, ctx, float(obj), "numpy.floating")
    if isinstance(obj, np.bool_):
        return _serialize_scalar(obj, ctx, bool(obj), "numpy.bool_")
    if isinstance(obj, np.complexfloating):
        value = [float(obj.real), float(obj.imag)]
        return _serialize_scalar(obj, ctx, value, "numpy.complex")
    if isinstance(obj, np.str_):
        return str(obj)
    # Generic fallback for other numpy scalars
    return _serialize_scalar(obj, ctx, obj.item(), "numpy.generic")


def _serialize_ndarray(arr: Any, ctx: SerializeContext) -> Any:
    """Serialize an ndarray with shape and dtype metadata."""
    if arr.dtype.fields is not None or arr.dtype.kind == "V":
        raise SerializationError("Structured and void NumPy dtypes require an explicit custom plugin")
    raw = arr.tolist()
    encoding = None
    if ctx.config.include_type_hints and arr.dtype.kind == "c":
        raw = [[float(x.real), float(x.imag)] for x in arr.flat]
        encoding = "complex_pairs"
    elif ctx.config.include_type_hints and arr.dtype.kind in "mM":
        raw = arr.view("i8").tolist()
        encoding = "temporal_int64"
    value = {
        "data": raw,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    if encoding is not None:
        value["encoding"] = encoding
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: value}
    return arr.tolist()


def _serialize_scalar(obj: Any, ctx: SerializeContext, native_value: Any, type_name: str) -> Any:
    """Serialize a numpy scalar, preserving type info if configured."""
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: native_value, "dtype": str(obj.dtype)}
    return native_value


def _deserialize_numpy(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct a NumPy object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "numpy.ndarray":
            if not isinstance(value, dict):
                raise PluginError(f"Expected dict for ndarray, got {type(value).__name__}")
            return _restore_array(value, ctx)
        case "numpy.integer":
            return _restore_scalar(value, data.get("dtype", "int64"), "iu")
        case "numpy.floating":
            return _restore_scalar(value, data.get("dtype", "float64"), "f")
        case "numpy.bool_":
            return _restore_scalar(value, data.get("dtype", "bool"), "b")
        case "numpy.complex":
            if not isinstance(value, list) or len(value) != 2:  # noqa: PLR2004
                raise PluginError(f"Expected [real, imag] for complex, got {type(value).__name__}")
            return _restore_scalar(complex(value[0], value[1]), data.get("dtype", "complex128"), "c")
        case "numpy.temporal":
            dtype = np.dtype(data["dtype"])
            if dtype.kind not in "mM":
                raise DeserializationError("Temporal scalar requires a datetime or timedelta dtype")
            return np.array(value, dtype="int64").view(dtype)[()]
        case "numpy.generic":
            return value
        case _:
            raise PluginError(f"Unknown numpy type: {type_name}")


def _restore_scalar(value: Any, dtype_name: str, allowed_kinds: str) -> Any:
    """Preserve scalar width without accepting a mismatched dtype family."""
    dtype = np.dtype(dtype_name)
    if dtype.kind not in allowed_kinds:
        raise DeserializationError("NumPy scalar dtype does not match its type tag")
    return dtype.type(value)


def _restore_array(value: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Restore shape only when it agrees with the supplied element count."""
    dtype = np.dtype(value.get("dtype"))
    if dtype.fields is not None or dtype.kind == "V":
        raise DeserializationError("Structured and void NumPy dtypes are unsupported")
    shape = value.get("shape")
    if shape is not None and (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)):
        raise DeserializationError("NumPy shape must contain non-negative integer dimensions")
    raw = value["data"]
    _check_array_allocation(dtype, shape, raw, ctx)
    encoding = value.get("encoding")
    if encoding == "complex_pairs" and dtype.kind == "c":
        raw = [complex(real, imag) for real, imag in raw]
        result = np.array(raw, dtype=dtype)
    elif encoding == "temporal_int64" and dtype.kind in "mM":
        result = np.array(raw, dtype="int64").view(dtype)
    elif encoding is None:
        result = np.array(raw, dtype=dtype)
    else:
        raise DeserializationError("Invalid NumPy array encoding for dtype")
    if shape is not None:
        try:
            result = result.reshape(shape)
        except ValueError as exc:
            raise DeserializationError("NumPy shape does not match payload") from exc
    return result


def _check_array_allocation(dtype: Any, shape: Any, raw: Any, ctx: DeserializeContext) -> None:
    """Bound dtype-driven allocations before calling NumPy constructors."""
    budget = ctx.config.max_input_bytes
    if dtype.itemsize > budget:
        raise SecurityError("NumPy dtype item size exceeds reconstruction budget")
    count = 1
    if shape is not None:
        for dim in shape:
            if dim > ctx.config.max_size:
                raise SecurityError("NumPy dimension exceeds container limit")
            count *= dim
            if count * max(dtype.itemsize, 1) > budget:
                raise SecurityError("NumPy array exceeds reconstruction byte budget")
    pending = [raw]
    count = 0
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        else:
            count += 1
            if count * max(dtype.itemsize, 1) > budget:
                raise SecurityError("NumPy array exceeds reconstruction byte budget")
