"""Plugin for PyTorch type serialization.

Handles torch.Tensor, torch.device, torch.dtype, and torch.Size.
Tensors are always moved to CPU for serialization; the original device
is recorded as metadata. Deserialization always produces CPU tensors.

This module imports torch directly — if torch is not installed,
the ImportError is caught by plugins/__init__.py and this plugin is
simply not registered.
"""

from __future__ import annotations

from typing import Any

import torch

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY


class TorchPlugin:
    """Handles serialization/deserialization of PyTorch types."""

    @property
    def name(self) -> str:
        return "torch"

    @property
    def priority(self) -> int:
        return 300

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, torch.Tensor | torch.device | torch.dtype | torch.Size)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_torch(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("torch.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_torch(data)


def _serialize_torch(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a PyTorch object to JSON-safe representation."""
    if isinstance(obj, torch.Tensor):
        return _serialize_tensor(obj, ctx)
    if isinstance(obj, torch.device):
        return _serialize_simple(str(obj), "torch.device", ctx)
    if isinstance(obj, torch.dtype):
        return _serialize_simple(_dtype_to_str(obj), "torch.dtype", ctx)
    if isinstance(obj, torch.Size):
        return _serialize_simple(list(obj), "torch.Size", ctx)
    raise PluginError(f"Unsupported PyTorch type: {type(obj).__name__}")


def _serialize_tensor(tensor: torch.Tensor, ctx: SerializeContext) -> Any:
    """Serialize a tensor with dtype, shape, and device metadata."""
    value = {
        "data": tensor.detach().cpu().tolist(),
        "dtype": _dtype_to_str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "torch.Tensor", VALUE_METADATA_KEY: value}
    return tensor.detach().cpu().tolist()


def _serialize_simple(native: Any, type_name: str, ctx: SerializeContext) -> Any:
    """Serialize a simple PyTorch type (device, dtype, Size)."""
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: type_name, VALUE_METADATA_KEY: native}
    return native


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to a short string (e.g. 'float32')."""
    return str(dtype).removeprefix("torch.")


def _deserialize_torch(data: dict[str, Any]) -> Any:
    """Reconstruct a PyTorch object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "torch.Tensor":
            return _reconstruct_tensor(value)
        case "torch.device":
            if not isinstance(value, str):
                raise PluginError(f"Expected str for device, got {type(value).__name__}")
            return torch.device(value)
        case "torch.dtype":
            if not isinstance(value, str):
                raise PluginError(f"Expected str for dtype, got {type(value).__name__}")
            return _str_to_dtype(value)
        case "torch.Size":
            if not isinstance(value, list):
                raise PluginError(f"Expected list for Size, got {type(value).__name__}")
            return torch.Size(value)
        case _:
            raise PluginError(f"Unknown torch type: {type_name}")


def _reconstruct_tensor(value: Any) -> torch.Tensor:
    """Reconstruct a tensor from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for Tensor, got {type(value).__name__}")
    dtype_str = value.get("dtype", "float32")
    dtype = _str_to_dtype(dtype_str)
    return torch.tensor(value["data"], dtype=dtype)


def _str_to_dtype(name: str) -> torch.dtype:
    """Convert a dtype string back to torch.dtype."""
    result = getattr(torch, name, None)
    if not isinstance(result, torch.dtype):
        raise PluginError(f"Unknown torch dtype: {name}")
    return result
