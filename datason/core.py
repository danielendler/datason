"""Core serialization functionality for datason.

This module contains the main serialize function that handles recursive
serialization of complex Python data structures to JSON-compatible formats.
"""

import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set, Union

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

# Import configuration and type handling
try:
    from .config import (
        DateFormat,
        NanHandling,
        OutputType,
        SerializationConfig,
        get_default_config,
    )
    from .type_handlers import TypeHandler, is_nan_like, normalize_numpy_types

    _config_available = True
except ImportError:
    _config_available = False

# Import ML serializers
try:
    from .ml_serializers import detect_and_serialize_ml_object

    _ml_serializer: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = detect_and_serialize_ml_object
except ImportError:
    _ml_serializer = None

# Security constants
MAX_SERIALIZATION_DEPTH = 1000  # Prevent stack overflow
MAX_OBJECT_SIZE = 10_000_000  # Prevent memory exhaustion (10MB worth of items)
MAX_STRING_LENGTH = 1_000_000  # Prevent excessive string processing


class SecurityError(Exception):
    """Raised when security limits are exceeded during serialization."""


def serialize(
    obj: Any,
    config: Optional["SerializationConfig"] = None,
    _depth: int = 0,
    _seen: Optional[Set[int]] = None,
    _type_handler: Optional[TypeHandler] = None,
) -> Any:
    """Recursively serialize an object for JSON compatibility.

    Handles pandas, datetime, UUID, NaT, numpy data types, Pydantic models,
    and nested dict/list/tuple. Includes security protections against
    circular references, excessive depth, and resource exhaustion.

    Args:
        obj: The object to serialize. Can be any Python data type.
        config: Optional serialization configuration. If None, uses global default.
        _depth: Internal parameter for tracking recursion depth
        _seen: Internal parameter for tracking circular references
        _type_handler: Internal parameter for type handling

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

        >>> from datason.config import get_ml_config
        >>> serialize(data, config=get_ml_config())
        # Uses ML-optimized settings
    """
    # Initialize configuration and type handler on first call
    if config is None and _config_available:
        config = get_default_config()

    # NEW: Performance optimization - skip processing if already serialized
    if (
        config
        and hasattr(config, "check_if_serialized")
        and config.check_if_serialized
        and _depth == 0
        and _is_json_serializable_basic_type(obj)
    ):
        return obj

    if _type_handler is None and config is not None:
        _type_handler = TypeHandler(config)

    # Use config limits if available, otherwise use defaults
    max_depth = config.max_depth if config else MAX_SERIALIZATION_DEPTH
    max_size = config.max_size if config else MAX_OBJECT_SIZE
    max_string_length = config.max_string_length if config else MAX_STRING_LENGTH

    # Security check: prevent excessive recursion depth
    if _depth > max_depth:
        raise SecurityError(
            f"Maximum serialization depth ({max_depth}) exceeded. "
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

    # For mutable objects, check optimization BEFORE adding to _seen
    optimization_result = None
    if isinstance(obj, dict):
        # Security check: prevent excessive object sizes
        if len(obj) > max_size:
            raise SecurityError(
                f"Dictionary size ({len(obj)}) exceeds maximum ({max_size}). "
                "This may indicate a resource exhaustion attempt."
            )
        # Only use optimization when it's safe (no config that could alter output)
        if (
            _depth == 0
            and (
                config is None
                or (
                    not config.sort_keys
                    and config.nan_handling == NanHandling.NULL
                    and not config.custom_serializers
                    and not config.include_type_hints  # NEW: Don't optimize if type hints are enabled
                    and config.max_depth >= 1000
                )
            )  # Only optimize with reasonable depth limits
            and _is_already_serialized_dict(obj)
        ):
            optimization_result = obj
    elif isinstance(obj, (list, tuple)):
        # Security check: prevent excessive object sizes
        if len(obj) > max_size:
            raise SecurityError(
                f"List/tuple size ({len(obj)}) exceeds maximum ({max_size}). "
                "This may indicate a resource exhaustion attempt."
            )
        # Only use optimization when it's safe (no config that could alter output)
        if (
            _depth == 0
            and (
                config is None
                or (
                    config.nan_handling == NanHandling.NULL
                    and not config.custom_serializers
                    and not config.include_type_hints  # NEW: Don't optimize if type hints are enabled
                    and config.max_depth >= 1000
                )
            )  # Only optimize with reasonable depth limits
            and _is_already_serialized_list(obj)
        ):
            optimization_result = list(obj) if isinstance(obj, tuple) else obj

    # Add current object to seen set for mutable types
    if isinstance(obj, (dict, list, set)):
        _seen.add(id(obj))

    try:
        # Use optimization result if available
        if optimization_result is not None:
            return optimization_result
        return _serialize_object(obj, config, _depth, _seen, _type_handler, max_string_length)
    finally:
        # Clean up: remove from seen set when done processing
        if isinstance(obj, (dict, list, set)):
            _seen.discard(id(obj))


def _serialize_object(
    obj: Any,
    config: Optional["SerializationConfig"],
    _depth: int,
    _seen: Set[int],
    _type_handler: Optional[TypeHandler],
    max_string_length: int,
) -> Any:
    """Internal serialization logic without circular reference management."""
    # Handle None
    if obj is None:
        return None

    # Check for NaN-like values first if type handler is available
    if _type_handler and is_nan_like(obj):
        return _type_handler.handle_nan_value(obj)

    # OPTIMIZATION: Early return for already JSON-serializable basic types
    # This prevents unnecessary processing of values that are already serialized
    if isinstance(obj, (str, int, bool)):
        # Security check: prevent excessive string processing
        if isinstance(obj, str) and len(obj) > max_string_length:
            warnings.warn(
                f"String length ({len(obj)}) exceeds maximum ({max_string_length}). Truncating.",
                stacklevel=3,
            )
            return obj[:max_string_length] + "...[TRUNCATED]"
        return obj

    # OPTIMIZATION: Handle float early, including NaN/Inf cases
    if isinstance(obj, float):
        if obj != obj or obj in (
            float("inf"),
            float("-inf"),
        ):  # obj != obj checks for NaN
            return _type_handler.handle_nan_value(obj) if _type_handler else None
        return obj

    # Try advanced type handler first if available
    if _type_handler:
        handler = _type_handler.get_type_handler(obj)
        if handler:
            try:
                return handler(obj)
            except Exception as e:
                # If custom handler fails, fall back to default handling
                warnings.warn(f"Custom type handler failed for {type(obj)}: {e}", stacklevel=3)

    # OPTIMIZATION: Check if dict/list are already fully serialized
    # This prevents recursive processing when not needed
    if isinstance(obj, dict):
        # Handle dict with optional key sorting
        result = {}
        for k, v in obj.items():
            serialized_value = serialize(v, config, _depth + 1, _seen, _type_handler)
            # Handle NaN dropping at collection level
            if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(v):
                continue
            result[k] = serialized_value

        # Sort keys if configured
        if config and config.sort_keys:
            return dict(sorted(result.items()))
        return result

    if isinstance(obj, (list, tuple)):
        result = []
        for x in obj:
            serialized_value = serialize(x, config, _depth + 1, _seen, _type_handler)
            # Handle NaN dropping at collection level
            if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(x):
                continue
            result.append(serialized_value)

        # NEW: Handle type metadata for tuples
        if isinstance(obj, tuple) and config and config.include_type_hints:
            return _create_type_metadata("tuple", result)

        return result

    # Handle numpy data types with normalization
    if np is not None:
        normalized = normalize_numpy_types(obj)
        # Use 'is' comparison for object identity to avoid DataFrame truth value issues
        if normalized is not obj:  # Something was converted
            return serialize(normalized, config, _depth + 1, _seen, _type_handler)

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            # Security check: prevent excessive array sizes
            if obj.size > (config.max_size if config else MAX_OBJECT_SIZE):
                raise SecurityError(
                    f"NumPy array size ({obj.size}) exceeds maximum. This may indicate a resource exhaustion attempt."
                )

            serialized_array = [serialize(x, config, _depth + 1, _seen, _type_handler) for x in obj.tolist()]

            # NEW: Handle type metadata for numpy arrays
            if config and config.include_type_hints:
                return _create_type_metadata("numpy.ndarray", serialized_array)

            return serialized_array

    # Handle datetime with configurable format and output type
    if isinstance(obj, datetime):
        # NEW: Check output type preference first
        if config and hasattr(config, "datetime_output") and config.datetime_output == OutputType.OBJECT:
            return obj  # Return datetime object as-is

        # Handle format configuration for JSON-safe output
        iso_string = None
        if config and hasattr(config, "date_format"):
            if config.date_format == DateFormat.ISO:
                iso_string = obj.isoformat()
            elif config.date_format == DateFormat.UNIX:
                return obj.timestamp()
            elif config.date_format == DateFormat.UNIX_MS:
                return int(obj.timestamp() * 1000)
            elif config.date_format == DateFormat.STRING:
                return str(obj)
            elif config.date_format == DateFormat.CUSTOM and config.custom_date_format:
                return obj.strftime(config.custom_date_format)

        # Default to ISO format
        if iso_string is None:
            iso_string = obj.isoformat()

        # NEW: Handle type metadata for datetimes
        if config and config.include_type_hints:
            return _create_type_metadata("datetime", iso_string)

        return iso_string

    # Handle pandas DataFrame with configurable orientation and output type
    if pd is not None and isinstance(obj, pd.DataFrame):
        # NEW: Check output type preference first
        if config and hasattr(config, "dataframe_output") and config.dataframe_output == OutputType.OBJECT:
            return obj  # Return DataFrame object as-is

        # Handle orientation configuration for JSON-safe output
        serialized_df = None
        if config and hasattr(config, "dataframe_orient"):
            orient = config.dataframe_orient.value
            try:
                # Special handling for VALUES orientation
                serialized_df = obj.values.tolist() if orient == "values" else obj.to_dict(orient=orient)
            except Exception:
                # Fall back to records if the specified orientation fails
                serialized_df = obj.to_dict(orient="records")
        else:
            serialized_df = obj.to_dict(orient="records")  # Default orientation

        # NEW: Handle type metadata for DataFrames
        if config and config.include_type_hints:
            return _create_type_metadata("pandas.DataFrame", serialized_df)

        return serialized_df

    # Handle pandas Series with configurable output type
    if pd is not None and isinstance(obj, pd.Series):
        # NEW: Check output type preference first
        if config and hasattr(config, "series_output") and config.series_output == OutputType.OBJECT:
            return obj  # Return Series object as-is

        # Default: convert to dict for JSON-safe output
        serialized_series = obj.to_dict()

        # NEW: Handle type metadata for Series with name preservation
        if config and config.include_type_hints:
            # Include series name if it exists
            if obj.name is not None:
                serialized_series = {"_series_name": obj.name, **serialized_series}
            return _create_type_metadata("pandas.Series", serialized_series)

        return serialized_series

    if pd is not None and isinstance(obj, (pd.Timestamp,)):
        if pd.isna(obj):
            return _type_handler.handle_nan_value(obj) if _type_handler else None
        # Convert to datetime and then serialize with date format
        dt = obj.to_pydatetime()
        return serialize(dt, config, _depth + 1, _seen, _type_handler)

    # Handle UUID
    if isinstance(obj, uuid.UUID):
        uuid_string = str(obj)

        # NEW: Handle type metadata for UUIDs
        if config and config.include_type_hints:
            return _create_type_metadata("uuid.UUID", uuid_string)

        return uuid_string

    # Handle set (convert to list for JSON compatibility)
    if isinstance(obj, set):
        serialized_set = [serialize(x, config, _depth + 1, _seen, _type_handler) for x in obj]

        # NEW: Handle type metadata for sets
        if config and config.include_type_hints:
            return _create_type_metadata("set", serialized_set)

        return serialized_set

    # Try ML serializer if available
    if _ml_serializer:
        try:
            ml_result = _ml_serializer(obj)
            if ml_result is not None:
                return ml_result
        except Exception:
            # If ML serializer fails, continue with fallback
            pass  # nosec B110

    # Handle objects with __dict__ (custom classes)
    if hasattr(obj, "__dict__"):
        try:
            return serialize(obj.__dict__, config, _depth + 1, _seen, _type_handler)
        except Exception:
            pass  # nosec B110

    # Fallback: convert to string representation
    try:
        str_repr = str(obj)
        if len(str_repr) > max_string_length:
            warnings.warn(
                f"Object string representation length ({len(str_repr)}) exceeds maximum. Truncating.",
                stacklevel=3,
            )
            return str_repr[:max_string_length] + "...[TRUNCATED]"
        return str_repr
    except Exception:
        return f"<{type(obj).__name__} object>"


def _create_type_metadata(type_name: str, value: Any) -> Dict[str, Any]:
    """NEW: Create a type metadata wrapper for round-trip serialization."""
    # Import here to avoid circular imports
    type_metadata_key = "__datason_type__"
    value_metadata_key = "__datason_value__"

    return {type_metadata_key: type_name, value_metadata_key: value}


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
        return not (value != value or value in (float("inf"), float("-inf")))  # value != value checks for NaN
    if isinstance(value, dict):
        # Recursively check if nested dict is serialized
        return _is_already_serialized_dict(value)
    if isinstance(value, (list, tuple)):
        # Recursively check if nested list is serialized
        return _is_already_serialized_list(value)
    return False
