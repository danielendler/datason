"""Deserialization functionality for datason.

This module provides functions to convert JSON-compatible data back to appropriate
Python objects, including datetime parsing, UUID reconstruction, and pandas types.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
import uuid
import warnings

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
