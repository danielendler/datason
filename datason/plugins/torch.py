"""Plugin for PyTorch type serialization.

Handles torch.Tensor, torch.device, torch.dtype, and torch.Size.
Tensors are always moved to CPU for serialization; the original device
is recorded as metadata. Deserialization always produces CPU tensors.

This module imports its library when activated by the lazy
loader (or explicitly imported). Unavailable optional dependencies are skipped
on first use.
"""

from __future__ import annotations

from typing import Any

import torch

from .._errors import PluginError
from .._protocols import DeserializeContext, SerializeContext
from .._reconstruction import check_dense_allocation
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
        return _deserialize_torch(data, ctx)


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


def _deserialize_torch(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct a PyTorch object from serialized data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "torch.Tensor":
            return _reconstruct_tensor(value, ctx)
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


def _reconstruct_tensor(value: Any, ctx: DeserializeContext) -> torch.Tensor:
    """Reconstruct a tensor from serialized dict."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for Tensor, got {type(value).__name__}")
    dtype_str = value.get("dtype", "float32")
    dtype = _str_to_dtype(dtype_str)
    shape = check_dense_allocation(value["data"], value.get("shape"), _dtype_itemsize(dtype), ctx)
    result = torch.tensor(value["data"], dtype=dtype, device="cpu")
    return result.reshape(shape) if shape is not None else result


def _str_to_dtype(name: str) -> torch.dtype:
    """Convert a dtype string back to torch.dtype."""
    result = getattr(torch, name, None)
    if not isinstance(result, torch.dtype):
        raise PluginError(f"Unknown torch dtype: {name}")
    return result


def _dtype_itemsize(dtype: torch.dtype) -> int:
    """Use dtype information available on the supported older Torch releases."""
    width = getattr(dtype, "itemsize", None)
    if isinstance(width, int):
        return width
    if dtype is torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return max(torch.finfo(dtype).bits // 8, 1) * (2 if dtype.is_complex else 1)
    return max(torch.iinfo(dtype).bits // 8, 1)
