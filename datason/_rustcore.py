"""Optional Rust accelerator integration for datason.

This module loads the optional Rust extension and exposes helper
functions that verify objects are eligible for the fast path.
"""

from __future__ import annotations

import math
from typing import Any

from .config import get_accel_mode
from .core_new import (
    MAX_OBJECT_SIZE,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    SecurityError,
)

try:  # pragma: no cover - prefer in-package extension
    from ._datason_rust import SecurityError as _RustSecurityError
    from ._datason_rust import dumps_core as _dumps_core
    from ._datason_rust import loads_core as _loads_core

    AVAILABLE = True
except Exception:  # pragma: no cover - fall back to top-level module name
    try:
        from _datason_rust import SecurityError as _RustSecurityError  # type: ignore
        from _datason_rust import dumps_core as _dumps_core  # type: ignore
        from _datason_rust import loads_core as _loads_core  # type: ignore

        AVAILABLE = True
    except Exception:
        _dumps_core = _loads_core = _RustSecurityError = None  # type: ignore
        AVAILABLE = False


class UnsupportedType(Exception):
    """Raised when object tree contains unsupported types."""


def _accel_enabled() -> bool:
    return get_accel_mode() != "off"


def _eligible_basic_tree(obj: Any) -> bool:
    """Return True if the object tree only contains supported basic types."""
    if isinstance(obj, (str, bool, type(None), int)):
        return True
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, (list, tuple)):
        return all(_eligible_basic_tree(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _eligible_basic_tree(v) for k, v in obj.items())
    return False


def dumps(obj: Any, *, ensure_ascii: bool = False, allow_nan: bool = False) -> bytes:
    """Serialize obj using the Rust core, enforcing security limits."""
    if not (AVAILABLE and _accel_enabled() and _eligible_basic_tree(obj)):
        raise UnsupportedType("Object not eligible for Rust path")
    try:
        return _dumps_core(  # type: ignore[misc]
            obj,
            ensure_ascii,
            allow_nan,
            MAX_SERIALIZATION_DEPTH,
            MAX_OBJECT_SIZE,
            MAX_STRING_LENGTH,
        )
    except Exception as e:  # pragma: no cover - any rust error triggers fallback
        # Convert Rust SecurityError to Python SecurityError for consistency
        if _RustSecurityError and isinstance(e, _RustSecurityError):
            raise SecurityError(str(e)) from e
        if isinstance(e, SecurityError):
            raise
        raise UnsupportedType(str(e)) from e


def loads(data: Any) -> Any:
    """Deserialize data using the Rust core, enforcing security limits."""
    if not (AVAILABLE and _accel_enabled()):
        raise UnsupportedType("Rust core unavailable")
    try:
        return _loads_core(  # type: ignore[misc]
            data,
            MAX_SERIALIZATION_DEPTH,
            MAX_OBJECT_SIZE,
            MAX_STRING_LENGTH,
        )
    except Exception as e:  # pragma: no cover
        # Convert Rust SecurityError to Python SecurityError for consistency
        if _RustSecurityError and isinstance(e, _RustSecurityError):
            raise SecurityError(str(e)) from e
        if isinstance(e, SecurityError):
            raise
        raise UnsupportedType(str(e)) from e


__all__ = ["AVAILABLE", "UnsupportedType", "dumps", "loads", "_eligible_basic_tree"]
