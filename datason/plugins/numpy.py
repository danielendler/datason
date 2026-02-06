"""Plugin for NumPy type serialization.

Handles ndarray, scalar types (integer, floating, bool_, str_),
and complex types. This module imports numpy directly — if numpy
is not installed, the ImportError is caught by plugins/__init__.py
and this plugin is simply not registered.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._errors import PluginError
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
        return _deserialize_numpy(data)


def _serialize_numpy(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a NumPy object to JSON-safe representation."""
    if isinstance(obj, np.ndarray):
        return _serialize_ndarray(obj, ctx)
    if isinstance(obj, np.integer):
        return _serialize_scalar(obj, ctx, int(obj), "numpy.integer")
    if isinstance(obj, np.floating):
        return _serialize_scalar(obj, ctx, float(obj), "numpy.floating")
    if isinstance(obj, np.bool_):
        return _serialize_scalar(obj, ctx, bool(obj), "numpy.bool_")
    if isinstance(obj, np.complexfloating):
        value = [float(obj.real), float(obj.imag)]
        if ctx.config.include_type_hints:
            return {TYPE_METADATA_KEY: "numpy.complex", VALUE_METADATA_KEY: value}
        return value
    if isinstance(obj, np.str_):
        return str(obj)
    # Generic fallback for other numpy scalars
    return _serialize_scalar(obj, ctx, obj.item(), "numpy.generic")


def _serialize_ndarray(arr: Any, ctx: SerializeContext) -> Any:
    """Serialize an ndarray with shape and dtype metadata."""
    value = {
        "data": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "numpy.ndarray", VALUE_METADATA_KEY: value}
    return arr.tolist()


def _serialize_scalar(obj: Any, ctx: SerializeContext, native_value: Any, type_name: str) -> Any:
    """Serialize a numpy scalar, preserving type info if configured."""
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: native_value}
    return native_value


def _deserialize_numpy(data: dict[str, Any]) -> Any:
    """Reconstruct a NumPy object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "numpy.ndarray":
            if not isinstance(value, dict):
                raise PluginError(f"Expected dict for ndarray, got {type(value).__name__}")
            return np.array(value["data"], dtype=value.get("dtype"))
        case "numpy.integer":
            return np.int64(value)
        case "numpy.floating":
            return np.float64(value)
        case "numpy.bool_":
            return np.bool_(value)
        case "numpy.complex":
            if not isinstance(value, list) or len(value) != 2:  # noqa: PLR2004
                raise PluginError(f"Expected [real, imag] for complex, got {type(value).__name__}")
            return np.complex128(complex(value[0], value[1]))
        case "numpy.generic":
            return value
        case _:
            raise PluginError(f"Unknown numpy type: {type_name}")
