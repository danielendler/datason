"""Deserialization functionality for datason.

This module provides functions to convert JSON-compatible data back to appropriate
Python objects, including datetime parsing, UUID reconstruction, and pandas types.
"""

import uuid
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import numpy as np
    except ImportError:
        np = None


# NEW: Type metadata constants for round-trip serialization
TYPE_METADATA_KEY = "__datason_type__"
VALUE_METADATA_KEY = "__datason_value__"


def deserialize(obj: Any, parse_dates: bool = True, parse_uuids: bool = True) -> Any:
    """Recursively deserialize JSON-compatible data back to Python objects.

    Attempts to intelligently restore datetime objects, UUIDs, and other types
    that were serialized to strings by the serialize function.

    Args:
        obj: The JSON-compatible object to deserialize
        parse_dates: Whether to attempt parsing ISO datetime strings back to datetime objects
        parse_uuids: Whether to attempt parsing UUID strings back to UUID objects

    Returns:
        Python object with restored types where possible

    Examples:
        >>> data = {"date": "2023-01-01T12:00:00", "id": "12345678-1234-5678-9012-123456789abc"}
        >>> deserialize(data)
        {"date": datetime(2023, 1, 1, 12, 0), "id": UUID('12345678-1234-5678-9012-123456789abc')}
    """
    if obj is None:
        return None

    # NEW: Handle type metadata for round-trip serialization
    if isinstance(obj, dict) and TYPE_METADATA_KEY in obj:
        return _deserialize_with_type_metadata(obj)

    # Handle basic types (already in correct format)
    if isinstance(obj, (int, float, bool)):
        return obj

    # Handle strings - attempt intelligent parsing
    if isinstance(obj, str):
        # Try to parse as datetime if enabled
        if parse_dates and _looks_like_datetime(obj):
            try:
                return datetime.fromisoformat(obj.replace("Z", "+00:00"))
            except ValueError:
                # Log parsing failure but continue with string
                warnings.warn(
                    f"Failed to parse datetime string: {obj[:50]}{'...' if len(obj) > 50 else ''}",
                    stacklevel=2,
                )

        # Try to parse as UUID if enabled
        if parse_uuids and _looks_like_uuid(obj):
            try:
                return uuid.UUID(obj)
            except ValueError:
                # Log parsing failure but continue with string
                warnings.warn(f"Failed to parse UUID string: {obj}", stacklevel=2)

        # Return as string if no parsing succeeded
        return obj

    # Handle lists
    if isinstance(obj, list):
        return [deserialize(item, parse_dates, parse_uuids) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: deserialize(v, parse_dates, parse_uuids) for k, v in obj.items()}

    # For any other type, return as-is
    return obj


def auto_deserialize(obj: Any, aggressive: bool = False) -> Any:
    """NEW: Intelligent auto-detection deserialization with heuristics.

    Uses pattern recognition and heuristics to automatically detect and restore
    complex data types without explicit configuration.

    Args:
        obj: JSON-compatible object to deserialize
        aggressive: Whether to use aggressive type detection (may have false positives)

    Returns:
        Python object with auto-detected types restored

    Examples:
        >>> data = {"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        >>> auto_deserialize(data, aggressive=True)
        {"records": DataFrame(...)}  # May detect as DataFrame
    """
    if obj is None:
        return None

    # Handle type metadata first
    if isinstance(obj, dict) and TYPE_METADATA_KEY in obj:
        return _deserialize_with_type_metadata(obj)

    # Handle basic types
    if isinstance(obj, (int, float, bool)):
        return obj

    # Handle strings with auto-detection
    if isinstance(obj, str):
        return _auto_detect_string_type(obj, aggressive)

    # Handle lists with auto-detection
    if isinstance(obj, list):
        deserialized_list = [auto_deserialize(item, aggressive) for item in obj]

        if aggressive and pd is not None and _looks_like_series_data(deserialized_list):
            # Try to detect if this should be a pandas Series or DataFrame
            try:
                return pd.Series(deserialized_list)
            except Exception:  # nosec B110
                pass

        return deserialized_list

    # Handle dictionaries with auto-detection
    if isinstance(obj, dict):
        # Check for pandas DataFrame patterns first
        if aggressive and pd is not None and _looks_like_dataframe_dict(obj):
            try:
                return _reconstruct_dataframe(obj)
            except Exception:  # nosec B110
                pass

        # Check for pandas split format
        if pd is not None and _looks_like_split_format(obj):
            try:
                return _reconstruct_from_split(obj)
            except Exception:  # nosec B110
                pass

        # Standard dictionary deserialization
        return {k: auto_deserialize(v, aggressive) for k, v in obj.items()}

    return obj


def deserialize_to_pandas(obj: Any, **kwargs: Any) -> Any:
    """Deserialize with pandas-specific optimizations.

    When pandas is available, attempts to reconstruct pandas objects
    from their serialized representations.

    Args:
        obj: JSON-compatible object to deserialize
        **kwargs: Additional arguments passed to deserialize()

    Returns:
        Deserialized object with pandas types restored where possible
    """
    if pd is None:
        return deserialize(obj, **kwargs)

    # First do standard deserialization
    result = deserialize(obj, **kwargs)

    # Then apply pandas-specific post-processing
    return _restore_pandas_types(result)


def _deserialize_with_type_metadata(obj: Dict[str, Any]) -> Any:
    """NEW: Deserialize objects with embedded type metadata for perfect round-trips."""
    if TYPE_METADATA_KEY not in obj or VALUE_METADATA_KEY not in obj:
        return obj

    type_name = obj[TYPE_METADATA_KEY]
    value = obj[VALUE_METADATA_KEY]

    try:
        if type_name == "datetime":
            return datetime.fromisoformat(value)
        if type_name == "uuid.UUID":
            return uuid.UUID(value)
        if type_name == "pandas.DataFrame":
            if pd is not None:
                return pd.DataFrame.from_dict(value)
        elif type_name == "pandas.Series":
            if pd is not None:
                # Handle Series with name preservation
                if isinstance(value, dict) and "_series_name" in value:
                    series_name = value["_series_name"]
                    series_data = {k: v for k, v in value.items() if k != "_series_name"}
                    return pd.Series(series_data, name=series_name)
                return pd.Series(value)
        elif type_name == "numpy.ndarray":
            if np is not None:
                return np.array(value)
        elif type_name == "set":
            return set(value)
        elif type_name == "tuple":
            return tuple(value)
        elif type_name == "complex":
            return complex(value["real"], value["imag"])
        # Add more type reconstructors as needed

    except Exception as e:
        warnings.warn(f"Failed to reconstruct type {type_name}: {e}", stacklevel=2)

    # Fallback to the original value
    return value


def _auto_detect_string_type(s: str, aggressive: bool = False) -> Any:
    """NEW: Auto-detect the most likely type for a string value."""
    # Always try datetime and UUID detection
    if _looks_like_datetime(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            pass

    if _looks_like_uuid(s):
        try:
            return uuid.UUID(s)
        except ValueError:
            pass

    if not aggressive:
        return s

    # Aggressive detection - more prone to false positives
    # Try to detect numbers
    if _looks_like_number(s):
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            pass

    # Try to detect boolean
    if s.lower() in ("true", "false"):
        return s.lower() == "true"

    return s


def _looks_like_series_data(data: List[Any]) -> bool:
    """NEW: Check if a list looks like it should be a pandas Series."""
    if len(data) < 2:
        return False

    # Check if all items are the same type and numeric/datetime
    first_type = type(data[0])
    if not all(isinstance(item, first_type) for item in data):
        return False

    return first_type in (int, float, datetime)


def _looks_like_dataframe_dict(obj: Dict[str, Any]) -> bool:
    """NEW: Check if a dict looks like it represents a DataFrame."""
    if not isinstance(obj, dict) or len(obj) < 1:
        return False

    # Check if all values are lists of the same length
    values = list(obj.values())
    if not all(isinstance(v, list) for v in values):
        return False

    if len({len(v) for v in values}) != 1:  # All lists same length
        return False

    # Must have at least a few rows to be worth converting
    return len(values[0]) >= 2


def _looks_like_split_format(obj: Dict[str, Any]) -> bool:
    """NEW: Check if a dict looks like pandas split format."""
    if not isinstance(obj, dict):
        return False

    required_keys = {"index", "columns", "data"}
    return required_keys.issubset(obj.keys())


def _reconstruct_dataframe(obj: Dict[str, Any]) -> "pd.DataFrame":
    """NEW: Reconstruct a DataFrame from a column-oriented dict."""
    return pd.DataFrame(obj)


def _reconstruct_from_split(obj: Dict[str, Any]) -> "pd.DataFrame":
    """NEW: Reconstruct a DataFrame from split format."""
    return pd.DataFrame(data=obj["data"], index=obj["index"], columns=obj["columns"])


def _looks_like_number(s: str) -> bool:
    """NEW: Check if a string looks like a number."""
    if not s:
        return False

    # Handle negative/positive signs
    s = s.strip()
    if s.startswith(("+", "-")):
        s = s[1:]

    if not s:
        return False

    # Scientific notation
    if "e" in s.lower():
        parts = s.lower().split("e")
        if len(parts) == 2:
            mantissa, exponent = parts
            # Check mantissa
            if not _is_numeric_part(mantissa):
                return False
            # Check exponent (can have +/- sign)
            exp = exponent.strip()
            if exp.startswith(("+", "-")):
                exp = exp[1:]
            return exp.isdigit() if exp else False

    # Regular number (integer or float)
    return _is_numeric_part(s)


def _is_numeric_part(s: str) -> bool:
    """Helper to check if a string part is numeric."""
    if not s:
        return False
    # Allow decimal points but only one
    if s.count(".") > 1:
        return False
    # Remove decimal point for digit check
    s_no_decimal = s.replace(".", "")
    return s_no_decimal.isdigit() if s_no_decimal else False


def _looks_like_datetime(s: str) -> bool:
    """Check if a string looks like an ISO datetime string."""
    if not isinstance(s, str) or len(s) < 10:
        return False

    # Check for ISO format patterns
    patterns = [
        # Basic ISO patterns
        s.count("-") >= 2 and ("T" in s or " " in s),
        # Common datetime patterns
        s.count(":") >= 1 and s.count("-") >= 2,
        # Z or timezone offset
        s.endswith("Z") or s.count("+") == 1 or s.count("-") >= 3,
    ]

    return any(patterns)


def _looks_like_uuid(s: str) -> bool:
    """Check if a string looks like a UUID."""
    if not isinstance(s, str) or len(s) != 36:
        return False

    # Check UUID pattern: 8-4-4-4-12 hex digits
    parts = s.split("-")
    if len(parts) != 5:
        return False

    expected_lengths = [8, 4, 4, 4, 12]
    for part, expected_len in zip(parts, expected_lengths):
        if len(part) != expected_len:
            return False
        try:
            int(part, 16)  # Check if hex
        except ValueError:
            return False

    return True


def _restore_pandas_types(obj: Any) -> Any:
    """Attempt to restore pandas-specific types from deserialized data."""
    if pd is None:
        return obj

    # This is a placeholder for pandas-specific restoration logic
    # In a full implementation, this could:
    # - Detect lists that should be Series
    # - Detect list-of-dicts that should be DataFrames
    # - Restore pandas Timestamps from datetime objects
    # etc.

    if isinstance(obj, dict):
        return {k: _restore_pandas_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_pandas_types(item) for item in obj]

    return obj


# Convenience functions for common use cases
def safe_deserialize(json_str: str, **kwargs: Any) -> Any:
    """Safely deserialize a JSON string, handling parse errors gracefully.

    Args:
        json_str: JSON string to parse and deserialize
        **kwargs: Arguments passed to deserialize()

    Returns:
        Deserialized Python object, or the original string if parsing fails
    """
    import json

    try:
        parsed = json.loads(json_str)
        return deserialize(parsed, **kwargs)
    except (json.JSONDecodeError, TypeError, ValueError):
        return json_str


def parse_datetime_string(s: Any) -> Optional[datetime]:
    """Parse a string as a datetime object if possible.

    Args:
        s: String that might represent a datetime (or other type for graceful handling)

    Returns:
        datetime object if parsing succeeds, None otherwise
    """
    if not _looks_like_datetime(s):
        return None

    try:
        # Handle various common formats
        # ISO format with Z
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        # Standard ISO format
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            # Try pandas parsing if available
            if pd is not None:
                ts = pd.to_datetime(s)
                if hasattr(ts, "to_pydatetime"):
                    return ts.to_pydatetime()
                return None
        except Exception as e:
            # Log specific error instead of silently failing
            warnings.warn(
                f"Failed to parse datetime string '{s}' using pandas: {e!s}",
                stacklevel=2,
            )
            return None

    return None


def parse_uuid_string(s: Any) -> Optional[uuid.UUID]:
    """Parse a string as a UUID object if possible.

    Args:
        s: String that might represent a UUID (or other type for graceful handling)

    Returns:
        UUID object if parsing succeeds, None otherwise
    """
    if not _looks_like_uuid(s):
        return None

    try:
        return uuid.UUID(s)
    except ValueError:
        return None
