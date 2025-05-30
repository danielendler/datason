"""Core serialization functionality for SerialPy.

This module contains the main serialize function that handles recursive
serialization of complex Python data structures to JSON-compatible formats.
"""

from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set, Union
import uuid
import warnings

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

# Import ML serializers
try:
    from .ml_serializers import detect_and_serialize_ml_object

    _ml_serializer: Optional[
        Callable[[Any], Optional[Dict[str, Any]]]
    ] = detect_and_serialize_ml_object
except ImportError:
    _ml_serializer = None

# Security constants
MAX_SERIALIZATION_DEPTH = 1000  # Prevent stack overflow
MAX_OBJECT_SIZE = 10_000_000  # Prevent memory exhaustion (10MB worth of items)
MAX_STRING_LENGTH = 1_000_000  # Prevent excessive string processing


class SecurityError(Exception):
    """Raised when security limits are exceeded during serialization."""


def serialize(obj: Any, _depth: int = 0, _seen: Optional[Set[int]] = None) -> Any:
    """Recursively serialize an object for JSON compatibility.

    Handles pandas, datetime, UUID, NaT, numpy data types, Pydantic models,
    and nested dict/list/tuple. Includes security protections against
    circular references, excessive depth, and resource exhaustion.

    Args:
        obj: The object to serialize. Can be any Python data type.
        _depth: Internal parameter for tracking recursion depth
        _seen: Internal parameter for tracking circular references

    Returns:
        A JSON-compatible representation of the input object.

    Raises:
        SecurityError: If security limits are exceeded

    Examples:
        >>> import datetime
        >>> serialize({'date': datetime.datetime.now()})
        {'date': '2023-...'}

        >>> serialize([1, 2, float('nan'), 4])
        [1, 2, None, 4]
    """
    # Security check: prevent excessive recursion depth
    if _depth > MAX_SERIALIZATION_DEPTH:
        raise SecurityError(
            f"Maximum serialization depth ({MAX_SERIALIZATION_DEPTH}) exceeded. "
            "This may indicate circular references or extremely nested data."
        )

    # Initialize circular reference tracking on first call
    if _seen is None:
        _seen = set()

    # Security check: detect circular references for mutable objects
    if isinstance(obj, (dict, list, set)) and id(obj) in _seen:
        warnings.warn(
            "Circular reference detected. Replacing with null to prevent infinite recursion.",
            stacklevel=2,
        )
        return None

    # Add current object to seen set for mutable types
    if isinstance(obj, (dict, list, set)):
        _seen.add(id(obj))

    try:
        return _serialize_object(obj, _depth, _seen)
    finally:
        # Clean up: remove from seen set when done processing
        if isinstance(obj, (dict, list, set)):
            _seen.discard(id(obj))


def _serialize_object(obj: Any, _depth: int, _seen: Set[int]) -> Any:
    """Internal serialization logic without circular reference management."""
    # Handle None
    if obj is None:
        return None

    # OPTIMIZATION: Early return for already JSON-serializable basic types
    # This prevents unnecessary processing of values that are already serialized
    if isinstance(obj, (str, int, bool)):
        # Security check: prevent excessive string processing
        if isinstance(obj, str) and len(obj) > MAX_STRING_LENGTH:
            warnings.warn(
                f"String length ({len(obj)}) exceeds maximum ({MAX_STRING_LENGTH}). Truncating.",
                stacklevel=3,
            )
            return obj[:MAX_STRING_LENGTH] + "...[TRUNCATED]"
        return obj

    # OPTIMIZATION: Handle float early, including NaN/Inf cases
    if isinstance(obj, float):
        if obj != obj or obj in (
            float("inf"),
            float("-inf"),
        ):  # obj != obj checks for NaN
            return None
        return obj

    # OPTIMIZATION: Check if dict/list are already fully serialized
    # This prevents recursive processing when not needed
    if isinstance(obj, dict):
        # Security check: prevent excessive object sizes
        if len(obj) > MAX_OBJECT_SIZE:
            raise SecurityError(
                f"Dictionary size ({len(obj)}) exceeds maximum ({MAX_OBJECT_SIZE}). "
                "This may indicate a resource exhaustion attempt."
            )

        # Quick check: if all keys are strings and all values are basic types, return as-is
        # BUT only if we haven't seen this object before (to prevent infinite recursion in optimization check)
        if id(obj) not in _seen and _is_already_serialized_dict(obj):
            return obj
        # Otherwise, recursively serialize
        return {k: serialize(v, _depth + 1, _seen) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        # Security check: prevent excessive object sizes
        if len(obj) > MAX_OBJECT_SIZE:
            raise SecurityError(
                f"List/tuple size ({len(obj)}) exceeds maximum ({MAX_OBJECT_SIZE}). "
                "This may indicate a resource exhaustion attempt."
            )

        # Quick check: if all items are basic types, return as-is (convert tuple to list)
        # BUT only if we haven't seen this object before (to prevent infinite recursion in optimization check)
        if id(obj) not in _seen and _is_already_serialized_list(obj):
            return list(obj) if isinstance(obj, tuple) else obj
        # Otherwise, recursively serialize
        return [serialize(x, _depth + 1, _seen) for x in obj]

    # Handle numpy data types
    if np is not None:
        # Handle numpy boolean types
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle numpy integer types
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle numpy floating point types
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            # Security check: prevent excessive array sizes
            if obj.size > MAX_OBJECT_SIZE:
                raise SecurityError(
                    f"NumPy array size ({obj.size}) exceeds maximum ({MAX_OBJECT_SIZE}). "
                    "This may indicate a resource exhaustion attempt."
                )
            return [serialize(x, _depth + 1, _seen) for x in obj.tolist()]
        # Handle numpy string types
        if isinstance(obj, np.str_):
            str_val = str(obj)
            if len(str_val) > MAX_STRING_LENGTH:
                warnings.warn(
                    f"NumPy string length ({len(str_val)}) exceeds maximum ({MAX_STRING_LENGTH}). Truncating.",
                    stacklevel=3,
                )
                return str_val[:MAX_STRING_LENGTH] + "...[TRUNCATED]"
            return str_val

    # Handle datetime, pd.Timestamp, NaT
    if isinstance(obj, datetime):
        return obj.isoformat()
    if pd is not None and isinstance(obj, (pd.Timestamp,)):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    # Handle UUID
    if isinstance(obj, uuid.UUID):
        return str(obj)
    # Handle pandas DataFrame
    if pd is not None and isinstance(obj, pd.DataFrame):
        # Security check: prevent excessive DataFrame sizes
        total_size = obj.shape[0] * obj.shape[1]
        if total_size > MAX_OBJECT_SIZE:
            raise SecurityError(
                f"DataFrame size ({total_size} cells) exceeds maximum ({MAX_OBJECT_SIZE}). "
                "This may indicate a resource exhaustion attempt."
            )
        records = obj.to_dict(orient="records")
        return [serialize(row, _depth + 1, _seen) for row in records]
    # Handle pandas Series/Index
    if pd is not None and isinstance(obj, (pd.Series, pd.Index)):
        if len(obj) > MAX_OBJECT_SIZE:
            raise SecurityError(
                f"Series/Index size ({len(obj)}) exceeds maximum ({MAX_OBJECT_SIZE}). "
                "This may indicate a resource exhaustion attempt."
            )
        return [serialize(x, _depth + 1, _seen) for x in obj.tolist()]

    # Handle Pydantic models and other objects with .dict() method
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return serialize(obj.dict(), _depth + 1, _seen)
        except Exception:
            # Log the specific error for debugging while preventing information leakage
            warnings.warn(
                "Failed to serialize object using .dict() method. Falling back to alternative methods.",
                stacklevel=3,
            )
            # Continue to next handler rather than using bare pass

    # Try ML/AI object serialization before fallback
    try:
        from .ml_serializers import detect_and_serialize_ml_object

        ml_result = detect_and_serialize_ml_object(obj)
        if ml_result is not None:
            return ml_result
    except ImportError:
        pass

    # Handle objects with __dict__ attribute
    try:
        if hasattr(obj, "__dict__"):
            try:
                obj_dict = vars(obj)
                # Only use dict serialization if the object has meaningful attributes
                if obj_dict:
                    return serialize(obj_dict, _depth + 1, _seen)
                # If __dict__ is empty, fall through to string conversion
            except Exception:
                # Log the specific error for debugging while preventing information leakage
                warnings.warn(
                    f"Failed to serialize object using __dict__. Object type: {type(obj).__name__}. "
                    "Falling back to string representation.",
                    stacklevel=3,
                )
                # Continue to fallback rather than using bare pass
    except Exception:
        # Handle case where even hasattr() fails (e.g., in our test)
        # This is intentional - we want to catch all exceptions here as a final fallback
        pass  # nosec: B110

    # Fallback: try to convert to string with length limit
    try:
        str_repr = str(obj)
        if len(str_repr) > MAX_STRING_LENGTH:
            warnings.warn(
                f"String representation length ({len(str_repr)}) exceeds maximum ({MAX_STRING_LENGTH}). Truncating.",
                stacklevel=3,
            )
            return str_repr[:MAX_STRING_LENGTH] + "...[TRUNCATED]"
        return str_repr
    except Exception:
        # Last resort: return a safe representation
        return f"<{type(obj).__name__} object at {hex(id(obj))}>"


def _is_already_serialized_dict(d: dict) -> bool:
    """Check if a dictionary is already fully serialized (contains only JSON-compatible values)."""
    try:
        for key, value in d.items():
            # Keys must be strings for JSON compatibility
            if not isinstance(key, str):
                return False
            # Values must be JSON-serializable basic types
            if not _is_json_serializable_basic_type(value):
                return False
        return True
    except Exception:
        return False


def _is_already_serialized_list(lst: Union[list, tuple]) -> bool:
    """Check if a list/tuple is already fully serialized (contains only JSON-compatible values)."""
    try:
        for item in lst:
            if not _is_json_serializable_basic_type(item):
                return False
        # Always return False for tuples so they get converted to lists
        return not isinstance(lst, tuple)
    except Exception:
        return False


def _is_json_serializable_basic_type(value: Any) -> bool:
    """Check if a value is a JSON-serializable basic type."""
    if value is None:
        return True
    if isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        # NaN and Inf are not JSON serializable, but we handle them specially
        return not (
            value != value or value in (float("inf"), float("-inf"))
        )  # value != value checks for NaN
    if isinstance(value, dict):
        # Recursively check if nested dict is serialized
        return _is_already_serialized_dict(value)
    if isinstance(value, (list, tuple)):
        # Recursively check if nested list is serialized
        return _is_already_serialized_list(value)
    return False
