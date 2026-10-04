"""Plugin for SciPy sparse matrix serialization.

Handles csr_matrix/csr_array, csc_matrix/csc_array, coo_matrix/coo_array,
and other sparse formats. All formats are normalized to COO for storage,
with the original format recorded for reconstruction.

This module imports its library when activated by the lazy
loader (or explicitly imported). Unavailable optional dependencies are skipped
on first use.
"""

from __future__ import annotations

from typing import Any

import scipy.sparse as sp

from .._errors import DeserializationError, PluginError, SecurityError
from .._protocols import DeserializeContext, SerializeContext
from .._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY

# Formats we can reconstruct; others fall back to COO
_SUPPORTED_FORMATS = frozenset({"csr", "csc", "coo"})


class ScipySparsePlugin:
    """Handles serialization/deserialization of SciPy sparse matrices."""

    @property
    def name(self) -> str:
        return "scipy_sparse"

    @property
    def priority(self) -> int:
        return 250

    def can_handle(self, obj: Any) -> bool:
        return sp.issparse(obj)

    def serialize(self, obj: Any, ctx: SerializeContext) -> Any:
        return _serialize_sparse(obj, ctx)

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        type_name = data.get(TYPE_METADATA_KEY, "")
        return isinstance(type_name, str) and type_name.startswith("scipy.sparse.")

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return _deserialize_sparse(data, ctx)


def _serialize_sparse(obj: Any, ctx: SerializeContext) -> Any:
    """Serialize a sparse matrix/array to COO representation."""
    coo = obj.tocoo()
    value = {
        "format": obj.format,
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": coo.data.tolist(),
        "shape": list(coo.shape),
        "dtype": str(coo.dtype),
    }
    if ctx.config.include_type_hints:
        return {TYPE_METADATA_KEY: "scipy.sparse.matrix", VALUE_METADATA_KEY: value}
    return value


def _deserialize_sparse(data: dict[str, Any], ctx: DeserializeContext) -> Any:
    """Reconstruct a sparse matrix from serialized COO data."""
    type_name = data[TYPE_METADATA_KEY]
    value = data[VALUE_METADATA_KEY]

    match type_name:
        case "scipy.sparse.matrix":
            return _reconstruct_sparse(value, ctx)
        case _:
            raise PluginError(f"Unknown scipy.sparse type: {type_name}")


def _reconstruct_sparse(value: Any, ctx: DeserializeContext) -> Any:
    """Reconstruct the specific sparse format from COO data."""
    if not isinstance(value, dict):
        raise PluginError(f"Expected dict for sparse matrix, got {type(value).__name__}")
    row = value["row"]
    col = value["col"]
    data = value["data"]
    shape = value["shape"]
    dtype = value.get("dtype")
    fmt = value.get("format", "coo")
    _check_sparse_allocation(row, col, data, shape, dtype, fmt, ctx)

    coo = sp.coo_matrix((data, (row, col)), shape=tuple(shape), dtype=dtype)

    match fmt:
        case "csr":
            return coo.tocsr()
        case "csc":
            return coo.tocsc()
        case "coo":
            return coo
        case _:
            return coo  # Unsupported formats fall back to COO


def _check_sparse_allocation(
    row: Any, col: Any, data: Any, shape: Any, dtype: Any, fmt: Any, ctx: DeserializeContext
) -> None:
    """Bound COO buffers and CSR/CSC pointer arrays, not logical dense size."""
    import numpy as np

    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dim) is not int or dim < 0 or dim > 2**63 - 1 for dim in shape)
    ):
        raise DeserializationError("Sparse shape requires two non-negative integer dimensions")
    if not all(isinstance(items, list) for items in (row, col, data)) or len(row) != len(col) or len(row) != len(data):
        raise DeserializationError("Sparse coordinates and data must have matching lengths")
    if any(type(n) is not int or n < 0 or n >= shape[0] for n in row):
        raise DeserializationError("Sparse row index lies outside shape")
    if any(type(n) is not int or n < 0 or n >= shape[1] for n in col):
        raise DeserializationError("Sparse column index lies outside shape")
    resolved = np.dtype(dtype or "float64")
    if resolved.fields is not None or resolved.kind not in "biufc":
        raise DeserializationError("Sparse reconstruction requires a numeric or boolean dtype")
    estimate = len(data) * (resolved.itemsize + 16)
    if fmt in ("csr", "csc"):
        estimate += (shape[0 if fmt == "csr" else 1] + 1) * 8
    if estimate > ctx.config.max_input_bytes:
        raise SecurityError("Sparse matrix exceeds reconstruction byte budget")
