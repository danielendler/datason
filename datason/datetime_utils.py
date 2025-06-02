"""DateTime utilities for datason.

This module provides functions for handling datetime objects, pandas timestamps,
and related date/time conversion tasks.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


def convert_pandas_timestamps(obj: Any) -> Any:
    """Convert pandas Timestamp objects to datetime objects recursively.

    Args:
        obj: Any object that might contain pandas Timestamp objects

    Returns:
        Object with all pandas Timestamp objects converted to Python datetime
    """
    if pd is None:
        return obj

    # Handle pandas Series or DataFrame
    if isinstance(obj, pd.Series):
        return obj.map(lambda x: convert_pandas_timestamps(x) if isinstance(x, (pd.Timestamp, list, dict)) else x)

    if isinstance(obj, pd.DataFrame):
        # Convert each column
        return obj.map(lambda x: convert_pandas_timestamps(x) if isinstance(x, (pd.Timestamp, list, dict)) else x)

    # Handle pd.Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()

    # Handle lists
    if isinstance(obj, list):
        return [convert_pandas_timestamps(item) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        return {key: convert_pandas_timestamps(value) for key, value in obj.items()}

    # Handle datetimes - no conversion needed
    if isinstance(obj, datetime):
        return obj

    # Return unchanged for other types
    return obj


def convert_pandas_timestamps_recursive(obj: Any) -> Any:
    """Convert pandas Timestamp objects to datetime objects recursively.

    This is an alias for convert_pandas_timestamps to match the original function name.

    Args:
        obj: Any object that might contain pandas Timestamp objects

    Returns:
        Object with all pandas Timestamp objects converted to Python datetime
    """
    return convert_pandas_timestamps(obj)


def serialize_datetimes(obj: Any) -> Union[Any, str, List[Any], Dict[str, Any]]:
    """Recursively serialize datetime objects within a structure to ISO format strings.

    Args:
        obj: Object to serialize (could be a datetime, dict, list, or other type)

    Returns:
        Serialized object with datetimes converted to ISO format strings
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_datetimes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_datetimes(item) for item in obj]
    return obj


def ensure_timestamp(val: Any) -> Any:
    """Ensure a scalar date value is a pandas Timestamp. Use this for group-level date fields.

    Args:
        val: A date value (can be pd.Timestamp, datetime, or string)

    Returns:
        pd.Timestamp or pd.NaT
    Raises:
        TypeError: If input is a list, dict, or other non-date-like object
    """
    if pd is None:
        raise ImportError("pandas is required for ensure_timestamp function")

    if val is None or (isinstance(val, float) and pd.isna(val)):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, (list, dict, set)):
        logger.error(f"ensure_timestamp: Invalid type {type(val)} for value {val}")
        raise TypeError(f"Cannot convert type {type(val)} to Timestamp")
    try:
        return pd.to_datetime(val)
    except Exception as e:
        logger.warning(f"ensure_timestamp: Could not convert {val!r} to Timestamp: {e}")
        return pd.NaT


def ensure_dates(df: Any, strip_timezone: bool = True) -> Any:
    """Ensure dates are in datetime format and handle common date formats.

    Args:
        df: Input DataFrame or dict with date-like fields
        strip_timezone: If True, convert all dates to timezone-naive (default: True)

    Returns:
        DataFrame or dict with standardized dates

    Raises:
        KeyError: If date column is missing in DataFrame
        ValueError: If date format is invalid
    """
    if pd is None:
        raise ImportError("pandas is required for ensure_dates function")

    # Handle dictionaries
    if isinstance(df, dict):
        result = df.copy()
        # Convert date fields in the dictionary
        date_fields = [
            "first_date",
            "last_date",
            "created_at",
            "updated_at",
            "date",
            "outer_date",
        ]
        for field in date_fields:
            if field in result and result[field] is not None:
                try:
                    # Try to convert to timestamp
                    converted = pd.to_datetime(result[field])
                    if strip_timezone and hasattr(converted, "tzinfo") and converted.tzinfo is not None:
                        converted = converted.replace(tzinfo=None)
                    result[field] = converted
                except (ValueError, TypeError) as e:
                    # If conversion fails, leave as is
                    logger.debug(f"Could not convert {field} value '{result[field]}' to timestamp: {e}")
        return result

    # Handle DataFrames - original implementation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame or dict, got {type(df)}")
    if "date" not in df.columns:
        raise KeyError("DataFrame must contain a 'date' column")
    df = df.copy()
    # Handle empty DataFrame
    if df.empty:
        df["date"] = pd.to_datetime(df["date"])
        return df
    # Check for weird types in date column
    if df["date"].apply(lambda x: isinstance(x, (list, dict, set))).any():
        raise ValueError("Date column contains non-date-like objects (list, dict, set)")
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            # If mixed tz-aware and naive, normalize all to UTC then localize to naive
            def normalize_tz(val: Any) -> Any:
                from datetime import datetime as dt

                if isinstance(val, dt) and val.tzinfo is not None:
                    return val.astimezone(timezone.utc).replace(tzinfo=None)
                return val

            df["date"] = df["date"].map(normalize_tz)
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            logger.error(f"ensure_dates: Could not convert date column to datetime: {e}")
            raise ValueError(f"Invalid date format: {e!s}") from e
    if strip_timezone and hasattr(df["date"], "dt"):
        try:
            df["date"] = df["date"].dt.tz_localize(None)
        except TypeError:
            # Already tz-naive
            pass
    return df
