"""Pure Python allocation and shape checks for optional dense-array loaders."""

from __future__ import annotations

from typing import Any

from ._errors import DeserializationError, SecurityError
from ._protocols import DeserializeContext


def checked_shape(shape: Any, ctx: DeserializeContext) -> tuple[int, ...] | None:
    """Validate declared dimensions without constructing an array."""
    if shape is None:
        return None
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise DeserializationError("Shape must contain non-negative integer dimensions")
    if len(shape) > ctx.config.max_size or any(dim > ctx.config.max_size for dim in shape):
        raise SecurityError("Array dimension exceeds container limit")
    return tuple(shape)


def check_dense_allocation(raw: Any, shape: Any, itemsize: int, ctx: DeserializeContext) -> tuple[int, ...] | None:
    """Bound represented buffers and check element counts before constructors.

    The estimate is per representation, not a process-wide or peak-memory cap.
    Missing legacy shape metadata permits library shape inference.
    """
    dims = checked_shape(shape, ctx)
    width = max(itemsize, 1)
    limit = ctx.config.max_input_bytes // width
    if width > ctx.config.max_input_bytes:
        raise SecurityError("Array dtype item size exceeds reconstruction budget")
    declared = None
    if dims is not None:
        declared = 0 if 0 in dims else 1
        if declared:
            for dim in dims:
                declared *= dim
                if declared > limit:
                    raise SecurityError("Array exceeds reconstruction byte budget")
    pending, count = [raw], 0
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        else:
            count += 1
            if count > limit:
                raise SecurityError("Array exceeds reconstruction byte budget")
    if declared is not None and count != declared:
        raise DeserializationError("Shape does not match payload element count")
    return dims
