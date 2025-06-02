"""Core serialization functionality for datason.

This module contains the main serialize function that handles recursive
serialization of complex Python data structures to JSON-compatible formats.
"""

import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Union

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


# OPTIMIZATION: Module-level type cache for repeated type checks
# This significantly reduces isinstance() overhead for repeated serialization
_TYPE_CACHE: Dict[type, str] = {}
_TYPE_CACHE_SIZE_LIMIT = 1000  # Prevent memory growth

# OPTIMIZATION: String length cache for repeated string processing
_STRING_LENGTH_CACHE: Dict[int, bool] = {}  # Maps string id to "is_long" boolean
_STRING_CACHE_SIZE_LIMIT = 500  # Smaller cache for strings

# OPTIMIZATION: Common UUID string cache for frequently used UUIDs
_UUID_STRING_CACHE: Dict[int, str] = {}  # Maps UUID object id to string
_UUID_CACHE_SIZE_LIMIT = 100  # Small cache for common UUIDs

# OPTIMIZATION: Collection processing cache for bulk operations
_COLLECTION_COMPATIBILITY_CACHE: Dict[int, str] = {}  # Maps collection id to compatibility status
_COLLECTION_CACHE_SIZE_LIMIT = 200  # Smaller cache for collections

# OPTIMIZATION: Memory allocation optimization - Phase 1 Step 1.4
# String interning for frequently used values
_COMMON_STRING_POOL: Dict[str, str] = {
    "true": "true",
    "false": "false",
    "null": "null",
    "True": "True",
    "False": "False",
    "None": "None",
    "": "",
    "0": "0",
    "1": "1",
    "-1": "-1",
}

# Pre-allocated result containers for reuse
_RESULT_DICT_POOL: List[Dict] = []
_RESULT_LIST_POOL: List[List] = []
_POOL_SIZE_LIMIT = 20  # Limit pool size to prevent memory bloat

# OPTIMIZATION: Function call overhead reduction - Phase 1 Step 1.5
# Pre-computed type sets for ultra-fast membership testing
_JSON_BASIC_TYPES = (str, int, bool, type(None))
_NUMERIC_TYPES = (int, float)
_CONTAINER_TYPES = (dict, list, tuple)

# Inline type checking constants for hot path optimization
_TYPE_STR = str
_TYPE_INT = int
_TYPE_BOOL = bool
_TYPE_NONE = type(None)
_TYPE_FLOAT = float
_TYPE_DICT = dict
_TYPE_LIST = list
_TYPE_TUPLE = tuple


class SecurityError(Exception):
    """Raised when security limits are exceeded during serialization."""


def _get_cached_type_category(obj_type: type) -> Optional[str]:
    """Get cached type category to optimize isinstance checks.

    Categories:
    - 'json_basic': str, int, bool, NoneType
    - 'float': float
    - 'dict': dict
    - 'list': list, tuple
    - 'datetime': datetime
    - 'numpy': numpy types
    - 'pandas': pandas types
    - 'uuid': UUID
    - 'set': set
    - 'other': everything else
    """
    if obj_type in _TYPE_CACHE:
        return _TYPE_CACHE[obj_type]

    # Only cache if we haven't hit the limit
    if len(_TYPE_CACHE) >= _TYPE_CACHE_SIZE_LIMIT:
        return None

    # Determine category - ordered by frequency in typical usage
    category = None
    if obj_type in (str, int, bool, type(None)):
        category = "json_basic"
    elif obj_type is float:
        category = "float"
    elif obj_type is dict:
        category = "dict"
    elif obj_type in (list, tuple):
        category = "list"
    elif obj_type is datetime:
        category = "datetime"
    elif obj_type is uuid.UUID:
        category = "uuid"
    elif obj_type is set:
        category = "set"
    elif np is not None and (
        obj_type is np.ndarray
        or (hasattr(np, "generic") and issubclass(obj_type, np.generic))
        or (hasattr(np, "number") and issubclass(obj_type, np.number))
        or (hasattr(np, "ndarray") and issubclass(obj_type, np.ndarray))
    ):
        category = "numpy"
    elif pd is not None and (
        obj_type is pd.DataFrame
        or obj_type is pd.Series
        or obj_type is pd.Timestamp
        or issubclass(obj_type, (pd.DataFrame, pd.Series, pd.Timestamp))
    ):
        category = "pandas"
    else:
        category = "other"

    _TYPE_CACHE[obj_type] = category
    return category


def _is_json_compatible_dict(obj: dict) -> bool:
    """Fast check if a dict is already JSON-compatible.

    This is much faster than the existing _is_already_serialized_dict
    for simple cases.
    """
    # Quick check: if empty, it's compatible
    if not obj:
        return True

    # Sample a few keys/values for quick assessment
    # For large dicts, avoid checking every single item
    items_to_check = list(obj.items())[:10]  # Check first 10 items

    for key, value in items_to_check:
        # Keys must be strings
        if not isinstance(key, str):
            return False
        # Check if value is basic JSON type
        if not _is_json_basic_type(value):
            return False

    return True


def _is_json_basic_type(value: Any) -> bool:
    """Ultra-fast check for basic JSON types without recursion."""
    # OPTIMIZATION: Use type() comparison for most common cases first
    value_type = type(value)

    if value_type in (str, int, bool, type(None)):
        return True
    elif value_type is float:
        # Efficient NaN/Inf check without function calls
        return value == value and value not in (float("inf"), float("-inf"))
    else:
        return False


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
        and config.max_string_length >= MAX_STRING_LENGTH  # NEW: Don't optimize if string truncation might be needed
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
                    and config.max_string_length >= MAX_STRING_LENGTH  # NEW: Don't optimize if string truncation might be needed
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
                    and config.max_string_length >= MAX_STRING_LENGTH  # NEW: Don't optimize if string truncation might be needed
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

        # OPTIMIZATION: Try hot path first for maximum performance
        hot_result = _serialize_hot_path(obj, config, max_string_length)
        if hot_result is not None or obj is None:
            return hot_result

        # OPTIMIZATION: Try fast path for common containers
        obj_type = type(obj)
        if obj_type in _CONTAINER_TYPES:
            # Handle dict with minimal function calls
            if obj_type is _TYPE_DICT:
                # Quick check for empty dict
                if not obj:
                    return obj

                # Quick JSON compatibility check for small dicts
                if (len(obj) <= 5 and 
                    all(isinstance(k, str) and type(v) in _JSON_BASIC_TYPES and 
                        (type(v) is not str or len(v) <= max_string_length) 
                        for k, v in obj.items())):
                    return obj

                # Needs full dict processing
                result = _process_homogeneous_dict(obj, config, _depth, _seen, _type_handler)
                # Sort keys if configured
                if config and config.sort_keys:
                    return dict(sorted(result.items()))
                return result

            # Handle list/tuple with minimal function calls
            elif obj_type in (_TYPE_LIST, _TYPE_TUPLE):
                # Quick check for empty list/tuple
                if not obj:
                    return [] if obj_type is _TYPE_TUPLE else obj

                # Quick JSON compatibility check for small lists (but only if type hints are not needed)
                if (len(obj) <= 5 and all(type(item) in _JSON_BASIC_TYPES for item in obj) 
                    and not (config and config.include_type_hints and obj_type is _TYPE_TUPLE)):
                    return list(obj) if obj_type is _TYPE_TUPLE else obj

                # Needs full list processing
                result = _process_homogeneous_list(obj, config, _depth, _seen, _type_handler)
                # Handle type metadata for tuples
                if isinstance(obj, tuple) and config and config.include_type_hints:
                    return _create_type_metadata("tuple", result)
                return result

        # Fall back to full processing for complex types
        return _serialize_full_path(obj, config, _depth, _seen, _type_handler, max_string_length)
    finally:
        # Clean up: remove from seen set when done processing
        if isinstance(obj, (dict, list, set)):
            _seen.discard(id(obj))


def _serialize_hot_path(obj: Any, config: Optional["SerializationConfig"], max_string_length: int) -> Any:
    """Ultra-optimized hot path for common serialization cases.

    This function inlines the most common operations to minimize function call overhead.
    Returns None if the object needs full processing.
    """
    # OPTIMIZATION: Check if type hints are enabled - if so, skip hot path for containers
    # that might need type metadata
    if config and config.include_type_hints:
        obj_type = type(obj)
        # Skip hot path for tuples, numpy arrays, sets, and complex containers when type hints are enabled
        if obj_type in (_TYPE_TUPLE, set) or (np is not None and isinstance(obj, np.ndarray)):
            return None  # Let full processing handle type metadata

    # OPTIMIZATION: Inline type checking without function calls
    obj_type = type(obj)

    # Handle None first (most common in sparse data)
    if obj_type is _TYPE_NONE:
        return None

    # Handle basic JSON types with minimal overhead
    if obj_type is _TYPE_STR:
        # Inline string processing for short strings
        if len(obj) <= 10:
            # Try to intern common strings
            interned = _COMMON_STRING_POOL.get(obj, obj)
            return interned
        elif len(obj) <= max_string_length:
            return obj
        else:
            # Needs full string processing
            return None

    elif obj_type is _TYPE_INT or obj_type is _TYPE_BOOL:
        return obj

    elif obj_type is _TYPE_FLOAT:
        # Inline NaN/Inf check
        if obj == obj and obj not in (float("inf"), float("-inf")):
            return obj
        else:
            # Needs NaN handling
            return None

    # PHASE 1.6: CONTAINER HOT PATH EXPANSION
    # Handle containers with aggressive inlining for common cases
    elif obj_type is _TYPE_DICT:
        # Handle empty dict (very common)
        if not obj:
            return obj
        
        # Handle small dicts with string keys and basic values (common in APIs)
        if len(obj) <= 3:
            # Quick check: all keys strings, all values basic JSON types
            try:
                for k, v in obj.items():
                    if type(k) is not _TYPE_STR:
                        return None  # Non-string key, needs full processing
                    v_type = type(v)
                    if v_type not in (_TYPE_STR, _TYPE_INT, _TYPE_BOOL, _TYPE_NONE):
                        if v_type is _TYPE_FLOAT:
                            # Inline float check
                            if v != v or v in (float("inf"), float("-inf")):
                                return None  # NaN/Inf, needs full processing
                        elif np is not None and isinstance(v, (np.bool_, np.integer, np.floating)):
                            # Inline numpy scalar normalization for hot path
                            if isinstance(v, np.floating) and (np.isnan(v) or np.isinf(v)):
                                return None  # NaN/Inf numpy float, needs full processing
                            # Simple numpy scalars are fine, will be normalized in full processing
                            return None  # Let full processing handle the conversion
                        else:
                            return None  # Complex type, needs full processing
                    elif v_type is _TYPE_STR and len(v) > max_string_length:
                        return None  # String too long, needs truncation in full processing
                # All checks passed - dict is JSON-compatible
                return obj
            except (AttributeError, TypeError):
                return None  # Some issue with iteration, needs full processing
        else:
            return None  # Large dict, needs full processing

    elif obj_type is _TYPE_LIST:
        # Handle empty list (very common)
        if not obj:
            return obj
        
        # Handle small lists with basic JSON types (common in APIs)
        if len(obj) <= 5:
            # Quick check: all items are basic JSON types
            try:
                for item in obj:
                    item_type = type(item)
                    if item_type not in (_TYPE_STR, _TYPE_INT, _TYPE_BOOL, _TYPE_NONE):
                        if item_type is _TYPE_FLOAT:
                            # Inline float check
                            if item != item or item in (float("inf"), float("-inf")):
                                return None  # NaN/Inf, needs full processing
                        elif np is not None and isinstance(item, (np.bool_, np.integer, np.floating)):
                            # Inline numpy scalar normalization for hot path
                            if isinstance(item, np.floating) and (np.isnan(item) or np.isinf(item)):
                                return None  # NaN/Inf numpy float, needs full processing
                            # Simple numpy scalars are fine, will be normalized in full processing
                            return None  # Let full processing handle the conversion
                        else:
                            return None  # Complex type, needs full processing
                    elif item_type is _TYPE_STR and len(item) > max_string_length:
                        return None  # String too long, needs truncation in full processing
                # All checks passed - list is JSON-compatible
                return obj
            except (AttributeError, TypeError):
                return None  # Some issue with iteration, needs full processing
        else:
            return None  # Large list, needs full processing

    elif obj_type is _TYPE_TUPLE:
        # Handle empty tuple (less common but still worth optimizing)
        if not obj:
            return []  # Convert empty tuple to list
        
        # Handle small tuples with basic JSON types
        if len(obj) <= 5:
            # Quick check: all items are basic JSON types
            try:
                for item in obj:
                    item_type = type(item)
                    if item_type not in (_TYPE_STR, _TYPE_INT, _TYPE_BOOL, _TYPE_NONE):
                        if item_type is _TYPE_FLOAT:
                            # Inline float check
                            if item != item or item in (float("inf"), float("-inf")):
                                return None  # NaN/Inf, needs full processing
                        elif np is not None and isinstance(item, (np.bool_, np.integer, np.floating)):
                            # Inline numpy scalar normalization for hot path
                            if isinstance(item, np.floating) and (np.isnan(item) or np.isinf(item)):
                                return None  # NaN/Inf numpy float, needs full processing
                            # Simple numpy scalars are fine, will be normalized in full processing
                            return None  # Let full processing handle the conversion
                        else:
                            return None  # Complex type, needs full processing
                    elif item_type is _TYPE_STR and len(item) > max_string_length:
                        return None  # String too long, needs truncation in full processing
                # All checks passed - convert tuple to list
                return list(obj)
            except (AttributeError, TypeError):
                return None  # Some issue with iteration, needs full processing
        else:
            return None  # Large tuple, needs full processing

    # For complex types, return None to indicate full processing needed
    return None


def _serialize_full_path(
    obj: Any,
    config: Optional["SerializationConfig"],
    _depth: int,
    _seen: Set[int],
    _type_handler: Optional[TypeHandler],
    max_string_length: int,
) -> Any:
    """Full serialization path for complex objects."""
    # OPTIMIZATION: Use faster type cache for type detection
    obj_type = type(obj)
    type_category = _get_cached_type_category_fast(obj_type)

    # Handle float with streamlined NaN/Inf checking
    if type_category == "float":
        if obj != obj or obj in (float("inf"), float("-inf")):  # obj != obj checks for NaN
            return _type_handler.handle_nan_value(obj) if _type_handler else None
        return obj

    # Handle string processing
    if type_category == "json_basic" and obj_type is _TYPE_STR:
        return _process_string_optimized(obj, max_string_length)

    # Check for NaN-like values if type handler is available (for non-float types)
    if _type_handler and type_category != "float" and is_nan_like(obj):
        return _type_handler.handle_nan_value(obj)

    # Try advanced type handler first if available
    if _type_handler:
        handler = _type_handler.get_type_handler(obj)
        if handler:
            try:
                return handler(obj)
            except Exception as e:
                # If custom handler fails, fall back to default handling
                warnings.warn(f"Custom type handler failed for {type(obj)}: {e}", stacklevel=3)

    # Handle dicts with full processing
    if type_category == "dict":
        result = _process_homogeneous_dict(obj, config, _depth, _seen, _type_handler)
        # Sort keys if configured
        if config and config.sort_keys:
            return dict(sorted(result.items()))
        return result

    # Handle lists/tuples with full processing
    if type_category == "list":
        result = _process_homogeneous_list(obj, config, _depth, _seen, _type_handler)
        # Handle type metadata for tuples
        if isinstance(obj, tuple) and config and config.include_type_hints:
            return _create_type_metadata("tuple", result)
        return result

    # OPTIMIZATION: Streamlined datetime handling (frequent type)
    if type_category == "datetime":
        # Check output type preference first
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

        # Handle type metadata for datetimes
        if config and config.include_type_hints:
            return _create_type_metadata("datetime", iso_string)

        return iso_string

    # Handle UUID efficiently with caching (frequent in APIs)
    if type_category == "uuid":
        uuid_string = _uuid_to_string_optimized(obj)

        # Handle type metadata for UUIDs
        if config and config.include_type_hints:
            return _create_type_metadata("uuid.UUID", uuid_string)

        return uuid_string

    # Handle sets efficiently
    if type_category == "set":
        serialized_set = [serialize(x, config, _depth + 1, _seen, _type_handler) for x in obj]

        # Handle type metadata for sets
        if config and config.include_type_hints:
            return _create_type_metadata("set", serialized_set)

        return serialized_set

    # Handle numpy data types with normalization (less frequent, but important for ML)
    if type_category == "numpy" and np is not None:
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

            # Handle type metadata for numpy arrays
            if config and config.include_type_hints:
                return _create_type_metadata("numpy.ndarray", serialized_array)

            return serialized_array

    # Handle pandas types (less frequent but important for data science)
    if type_category == "pandas" and pd is not None:
        # Handle pandas DataFrame with configurable orientation and output type
        if isinstance(obj, pd.DataFrame):
            # Check output type preference first
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

            # Handle type metadata for DataFrames
            if config and config.include_type_hints:
                return _create_type_metadata("pandas.DataFrame", serialized_df)

            return serialized_df

        # Handle pandas Series with configurable output type
        if isinstance(obj, pd.Series):
            # Check output type preference first
            if config and hasattr(config, "series_output") and config.series_output == OutputType.OBJECT:
                return obj  # Return Series object as-is

            # Default: convert to dict for JSON-safe output
            serialized_series = obj.to_dict()

            # Handle type metadata for Series with name preservation
            if config and config.include_type_hints:
                # Include series name if it exists
                if obj.name is not None:
                    serialized_series = {"_series_name": obj.name, **serialized_series}
                return _create_type_metadata("pandas.Series", serialized_series)

            return serialized_series

        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return _type_handler.handle_nan_value(obj) if _type_handler else None
            # Convert to datetime and then serialize with date format
            dt = obj.to_pydatetime()
            return serialize(dt, config, _depth + 1, _seen, _type_handler)

    # For all other types (fallback path)
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
        # OPTIMIZATION: Intern common string representations
        if len(str_repr) <= 20:  # Only intern short string representations
            str_repr = _intern_common_string(str_repr)

        if len(str_repr) > max_string_length:
            warnings.warn(
                f"Object string representation length ({len(str_repr)}) exceeds maximum. Truncating.",
                stacklevel=3,
            )
            # OPTIMIZATION: Memory-efficient truncation
            return str_repr[:max_string_length] + "...[TRUNCATED]"
        return str_repr
    except Exception:
        # OPTIMIZATION: Return interned fallback string
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


# NEW: v0.4.0 Chunked Processing & Streaming Capabilities


class ChunkedSerializationResult:
    """Result container for chunked serialization operations."""

    def __init__(self, chunks: Iterator[Any], metadata: Dict[str, Any]):
        """Initialize chunked result.

        Args:
            chunks: Iterator of serialized chunks
            metadata: Metadata about the chunking operation
        """
        self.chunks = chunks
        self.metadata = metadata

    def to_list(self) -> list:
        """Convert all chunks to a list (loads everything into memory)."""
        return list(self.chunks)

    def save_to_file(self, file_path: Union[str, Path], format: str = "jsonl") -> None:
        """Save chunks to a file.

        Args:
            file_path: Path to save the chunks
            format: Format to save ('jsonl' for JSON lines, 'json' for array)
        """
        file_path = Path(file_path)

        with file_path.open("w") as f:
            if format == "jsonl":
                # JSON Lines format - one JSON object per line
                for chunk in self.chunks:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write("\n")
            elif format == "json":
                # JSON array format
                chunk_list = list(self.chunks)
                json.dump({"chunks": chunk_list, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'jsonl' or 'json'")


def serialize_chunked(
    obj: Any,
    chunk_size: int = 1000,
    config: Optional["SerializationConfig"] = None,
    memory_limit_mb: Optional[int] = None,
) -> ChunkedSerializationResult:
    """Serialize large objects in memory-bounded chunks.

    This function breaks large objects (lists, DataFrames, arrays) into smaller chunks
    to enable processing of datasets larger than available memory.

    Args:
        obj: Object to serialize (typically list, DataFrame, or array)
        chunk_size: Number of items per chunk
        config: Serialization configuration
        memory_limit_mb: Optional memory limit in MB (not enforced yet, for future use)

    Returns:
        ChunkedSerializationResult with iterator of serialized chunks

    Examples:
        >>> large_list = list(range(10000))
        >>> result = serialize_chunked(large_list, chunk_size=100)
        >>> chunks = result.to_list()  # Get all chunks
        >>> len(chunks)  # 100 chunks of 100 items each
        100

        >>> # Save directly to file without loading all chunks
        >>> result = serialize_chunked(large_data, chunk_size=1000)
        >>> result.save_to_file("large_data.jsonl", format="jsonl")
    """
    if config is None and _config_available:
        config = get_default_config()

    # Determine chunking strategy based on object type
    if isinstance(obj, (list, tuple)):
        return _chunk_sequence(obj, chunk_size, config)
    elif pd is not None and isinstance(obj, pd.DataFrame):
        return _chunk_dataframe(obj, chunk_size, config)
    elif np is not None and isinstance(obj, np.ndarray):
        return _chunk_numpy_array(obj, chunk_size, config)
    elif isinstance(obj, dict):
        return _chunk_dict(obj, chunk_size, config)
    else:
        # For non-chunnable objects, return single chunk
        single_chunk = serialize(obj, config)
        metadata = {
            "total_chunks": 1,
            "chunk_size": chunk_size,
            "object_type": type(obj).__name__,
            "chunking_strategy": "single_object",
        }
        return ChunkedSerializationResult(iter([single_chunk]), metadata)


def _chunk_sequence(
    seq: Union[list, tuple], chunk_size: int, config: Optional["SerializationConfig"]
) -> ChunkedSerializationResult:
    """Chunk a sequence (list or tuple) into smaller pieces."""
    total_items = len(seq)
    total_chunks = (total_items + chunk_size - 1) // chunk_size  # Ceiling division

    def chunk_generator():
        for i in range(0, total_items, chunk_size):
            chunk = seq[i : i + chunk_size]
            yield serialize(chunk, config)

    metadata = {
        "total_chunks": total_chunks,
        "total_items": total_items,
        "chunk_size": chunk_size,
        "object_type": type(seq).__name__,
        "chunking_strategy": "sequence",
    }

    return ChunkedSerializationResult(chunk_generator(), metadata)


def _chunk_dataframe(
    df: "pd.DataFrame", chunk_size: int, config: Optional["SerializationConfig"]
) -> ChunkedSerializationResult:
    """Chunk a pandas DataFrame by rows."""
    total_rows = len(df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    def chunk_generator():
        for i in range(0, total_rows, chunk_size):
            chunk_df = df.iloc[i : i + chunk_size]
            yield serialize(chunk_df, config)

    metadata = {
        "total_chunks": total_chunks,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "chunk_size": chunk_size,
        "object_type": "pandas.DataFrame",
        "chunking_strategy": "dataframe_rows",
        "columns": list(df.columns),
    }

    return ChunkedSerializationResult(chunk_generator(), metadata)


def _chunk_numpy_array(
    arr: "np.ndarray", chunk_size: int, config: Optional["SerializationConfig"]
) -> ChunkedSerializationResult:
    """Chunk a numpy array along the first axis."""
    total_items = arr.shape[0] if arr.ndim > 0 else 1
    total_chunks = (total_items + chunk_size - 1) // chunk_size

    def chunk_generator():
        if arr.ndim == 0:
            # Scalar array
            yield serialize(arr, config)
        else:
            for i in range(0, total_items, chunk_size):
                chunk_arr = arr[i : i + chunk_size]
                yield serialize(chunk_arr, config)

    metadata = {
        "total_chunks": total_chunks,
        "total_items": total_items,
        "chunk_size": chunk_size,
        "object_type": "numpy.ndarray",
        "chunking_strategy": "array_rows",
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }

    return ChunkedSerializationResult(chunk_generator(), metadata)


def _chunk_dict(d: dict, chunk_size: int, config: Optional["SerializationConfig"]) -> ChunkedSerializationResult:
    """Chunk a dictionary by grouping key-value pairs."""
    items = list(d.items())
    total_items = len(items)
    total_chunks = (total_items + chunk_size - 1) // chunk_size

    def chunk_generator():
        for i in range(0, total_items, chunk_size):
            chunk_items = items[i : i + chunk_size]
            chunk_dict = dict(chunk_items)
            yield serialize(chunk_dict, config)

    metadata = {
        "total_chunks": total_chunks,
        "total_items": total_items,
        "chunk_size": chunk_size,
        "object_type": "dict",
        "chunking_strategy": "dict_items",
    }

    return ChunkedSerializationResult(chunk_generator(), metadata)


class StreamingSerializer:
    """Context manager for streaming serialization to files.

    Enables processing of datasets larger than available memory by writing
    serialized data directly to files without keeping everything in memory.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional["SerializationConfig"] = None,
        format: str = "jsonl",
        buffer_size: int = 8192,
    ):
        """Initialize streaming serializer.

        Args:
            file_path: Path to output file
            config: Serialization configuration
            format: Output format ('jsonl' or 'json')
            buffer_size: Write buffer size in bytes
        """
        self.file_path = Path(file_path)
        self.config = config or (get_default_config() if _config_available else None)
        self.format = format
        self.buffer_size = buffer_size
        self._file = None
        self._items_written = 0
        self._json_array_started = False

    def __enter__(self) -> "StreamingSerializer":
        """Enter context manager."""
        self._file = self.file_path.open("w", buffering=self.buffer_size)

        if self.format == "json":
            # Start JSON array
            self._file.write('{"data": [')
            self._json_array_started = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._file:
            if self.format == "json" and self._json_array_started:
                # Close JSON array and add metadata
                self._file.write(f'], "metadata": {{"items_written": {self._items_written}}}}}')

            self._file.close()
            self._file = None

    def write(self, obj: Any) -> None:
        """Write a single object to the stream.

        Args:
            obj: Object to serialize and write
        """
        if not self._file:
            raise RuntimeError("StreamingSerializer not in context manager")

        serialized = serialize(obj, self.config)

        if self.format == "jsonl":
            # JSON Lines: one object per line
            json.dump(serialized, self._file, ensure_ascii=False)
            self._file.write("\n")
        elif self.format == "json":
            # JSON array format
            if self._items_written > 0:
                self._file.write(", ")
            json.dump(serialized, self._file, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        self._items_written += 1

    def write_chunked(self, obj: Any, chunk_size: int = 1000) -> None:
        """Write a large object using chunked serialization.

        Args:
            obj: Large object to chunk and write
            chunk_size: Size of each chunk
        """
        chunked_result = serialize_chunked(obj, chunk_size, self.config)

        for chunk in chunked_result.chunks:
            self.write(chunk)


def stream_serialize(
    file_path: Union[str, Path],
    config: Optional["SerializationConfig"] = None,
    format: str = "jsonl",
    buffer_size: int = 8192,
) -> StreamingSerializer:
    """Create a streaming serializer context manager.

    Args:
        file_path: Path to output file
        config: Serialization configuration
        format: Output format ('jsonl' or 'json')
        buffer_size: Write buffer size in bytes

    Returns:
        StreamingSerializer context manager

    Examples:
        >>> with stream_serialize("large_data.jsonl") as stream:
        ...     for item in large_dataset:
        ...         stream.write(item)

        >>> # Or write chunked data
        >>> with stream_serialize("massive_data.jsonl") as stream:
        ...     stream.write_chunked(massive_dataframe, chunk_size=1000)
    """
    return StreamingSerializer(file_path, config, format, buffer_size)


def deserialize_chunked_file(
    file_path: Union[str, Path], format: str = "jsonl", chunk_processor: Optional[Callable[[Any], Any]] = None
) -> Generator[Any, None, None]:
    """Deserialize a chunked file created with streaming serialization.

    Args:
        file_path: Path to the chunked file
        format: File format ('jsonl' or 'json')
        chunk_processor: Optional function to process each chunk

    Yields:
        Deserialized chunks from the file

    Examples:
        >>> # Process chunks one at a time (memory efficient)
        >>> for chunk in deserialize_chunked_file("large_data.jsonl"):
        ...     process_chunk(chunk)

        >>> # Apply custom processing to each chunk
        >>> def process_chunk(chunk):
        ...     return [item * 2 for item in chunk]
        >>>
        >>> processed_chunks = list(deserialize_chunked_file(
        ...     "data.jsonl",
        ...     chunk_processor=process_chunk
        ... ))
    """
    file_path = Path(file_path)

    if format == "jsonl":
        # JSON Lines format - one object per line
        with file_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    if chunk_processor:
                        chunk = chunk_processor(chunk)
                    yield chunk

    elif format == "json":
        # JSON format with array
        with file_path.open("r") as f:
            data = json.load(f)
            # Support both 'chunks' (from ChunkedSerializationResult) and 'data' (from StreamingSerializer)
            chunks = data.get("chunks", data.get("data", []))
            for chunk in chunks:
                if chunk_processor:
                    chunk = chunk_processor(chunk)
                yield chunk

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'jsonl' or 'json'")


def estimate_memory_usage(obj: Any, config: Optional["SerializationConfig"] = None) -> Dict[str, Any]:
    """Estimate memory usage for serializing an object.

    This is a rough estimation to help users decide on chunking strategies.

    Args:
        obj: Object to analyze
        config: Serialization configuration

    Returns:
        Dictionary with memory usage estimates

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': range(10000), 'b': range(10000)})
        >>> stats = estimate_memory_usage(df)
        >>> print(f"Estimated serialized size: {stats['estimated_serialized_mb']:.1f} MB")
        >>> print(f"Recommended chunk size: {stats['recommended_chunk_size']}")
    """
    import sys

    # Get basic object size
    object_size_bytes = sys.getsizeof(obj)

    # Estimate based on object type
    if isinstance(obj, (list, tuple)) or pd is not None and isinstance(obj, pd.DataFrame):
        item_count = len(obj)
        estimated_item_size = object_size_bytes / max(item_count, 1)
    elif np is not None and isinstance(obj, np.ndarray):
        item_count = obj.shape[0] if obj.ndim > 0 else 1
        estimated_item_size = object_size_bytes / max(item_count, 1)
    elif isinstance(obj, dict):
        item_count = len(obj)
        estimated_item_size = object_size_bytes / max(item_count, 1)
    else:
        item_count = 1
        estimated_item_size = object_size_bytes

    # Serialization typically increases size by 1.5-3x for complex objects
    serialization_overhead = 2.0
    estimated_serialized_bytes = object_size_bytes * serialization_overhead

    # Recommend chunk size to keep chunks under 50MB
    target_chunk_size_mb = 50
    target_chunk_size_bytes = target_chunk_size_mb * 1024 * 1024

    if estimated_item_size > 0:
        recommended_chunk_size = max(1, int(target_chunk_size_bytes / (estimated_item_size * serialization_overhead)))
    else:
        recommended_chunk_size = 1000  # Default fallback

    return {
        "object_type": type(obj).__name__,
        "object_size_mb": object_size_bytes / (1024 * 1024),
        "estimated_serialized_mb": estimated_serialized_bytes / (1024 * 1024),
        "item_count": item_count,
        "estimated_item_size_bytes": estimated_item_size,
        "recommended_chunk_size": recommended_chunk_size,
        "recommended_chunks": max(1, item_count // recommended_chunk_size),
    }


def _process_string_optimized(obj: str, max_string_length: int) -> str:
    """Optimized string processing with length caching and interning."""
    # OPTIMIZATION: Try to intern common strings first
    if len(obj) <= 10:  # Only check short strings for interning
        interned = _intern_common_string(obj)
        if interned is not obj:  # String was interned
            return interned

    obj_id = id(obj)

    # Check cache first for long strings
    if obj_id in _STRING_LENGTH_CACHE:
        is_long = _STRING_LENGTH_CACHE[obj_id]
        if not is_long:
            return obj  # Short string, return as-is
    else:
        # Calculate and cache length check
        obj_len = len(obj)
        is_long = obj_len > max_string_length

        # Only cache if we haven't hit the limit
        if len(_STRING_LENGTH_CACHE) < _STRING_CACHE_SIZE_LIMIT:
            _STRING_LENGTH_CACHE[obj_id] = is_long

        if not is_long:
            return obj  # Short string, return as-is

    # Handle long string truncation - use memory-efficient slicing
    warnings.warn(
        f"String length ({len(obj)}) exceeds maximum ({max_string_length}). Truncating.",
        stacklevel=4,
    )
    # OPTIMIZATION: Build truncated string efficiently
    return obj[:max_string_length] + "...[TRUNCATED]"


def _uuid_to_string_optimized(obj: uuid.UUID) -> str:
    """Optimized UUID to string conversion with caching."""
    obj_id = id(obj)

    # Check cache first
    if obj_id in _UUID_STRING_CACHE:
        return _UUID_STRING_CACHE[obj_id]

    # Convert and cache if space available
    uuid_string = str(obj)
    if len(_UUID_STRING_CACHE) < _UUID_CACHE_SIZE_LIMIT:
        _UUID_STRING_CACHE[obj_id] = uuid_string

    return uuid_string


def _get_cached_type_category_fast(obj_type: type) -> Optional[str]:
    """Faster version of type category lookup with optimized cache access."""
    # Direct cache lookup (most common case)
    cached = _TYPE_CACHE.get(obj_type)
    if cached is not None:
        return cached

    # Only compute and cache if we have space
    if len(_TYPE_CACHE) >= _TYPE_CACHE_SIZE_LIMIT:
        # Cache full, do direct type checking without caching
        if obj_type in (str, int, bool, type(None)):
            return "json_basic"
        elif obj_type is float:
            return "float"
        elif obj_type is dict:
            return "dict"
        elif obj_type in (list, tuple):
            return "list"
        elif obj_type is datetime:
            return "datetime"
        elif obj_type is uuid.UUID:
            return "uuid"
        else:
            return "other"  # Skip expensive checks when cache is full

    # Compute and cache (same logic as before, but optimized)
    if obj_type in (str, int, bool, type(None)):
        category = "json_basic"
    elif obj_type is float:
        category = "float"
    elif obj_type is dict:
        category = "dict"
    elif obj_type in (list, tuple):
        category = "list"
    elif obj_type is datetime:
        category = "datetime"
    elif obj_type is uuid.UUID:
        category = "uuid"
    elif obj_type is set:
        category = "set"
    elif np is not None and (
        obj_type is np.ndarray
        or (hasattr(np, "generic") and issubclass(obj_type, np.generic))
        or (hasattr(np, "number") and issubclass(obj_type, np.number))
        or (hasattr(np, "ndarray") and issubclass(obj_type, np.ndarray))
    ):
        category = "numpy"
    elif pd is not None and (
        obj_type is pd.DataFrame
        or obj_type is pd.Series
        or obj_type is pd.Timestamp
        or issubclass(obj_type, (pd.DataFrame, pd.Series, pd.Timestamp))
    ):
        category = "pandas"
    else:
        category = "other"

    _TYPE_CACHE[obj_type] = category
    return category


def _is_homogeneous_collection(obj: Union[list, tuple, dict], sample_size: int = 20) -> Optional[str]:
    """Check if a collection contains homogeneous types for bulk processing.

    Returns:
    - 'json_basic': All items are JSON-basic types
    - 'single_type': All items are the same non-basic type
    - 'mixed': Mixed types (requires individual processing)
    - None: Unable to determine or too small
    """
    # OPTIMIZATION: Check cache first for collections we've seen before
    obj_id = id(obj)
    if obj_id in _COLLECTION_COMPATIBILITY_CACHE:
        return _COLLECTION_COMPATIBILITY_CACHE[obj_id]

    homogeneity_result = None

    if isinstance(obj, dict):
        if not obj:
            homogeneity_result = "json_basic"
        else:
            # Sample values for type analysis
            values = list(obj.values())
            sample = values[:sample_size] if len(values) > sample_size else values

            if not sample:
                homogeneity_result = "json_basic"
            elif all(_is_json_basic_type(v) for v in sample):
                # Check if all values are JSON-basic types
                homogeneity_result = "json_basic"
            else:
                # Check if all values are the same type
                first_type = type(sample[0])
                homogeneity_result = "single_type" if all(isinstance(v, first_type) for v in sample) else "mixed"

    elif isinstance(obj, (list, tuple)):
        if not obj:
            homogeneity_result = "json_basic"
        else:
            # Sample items for type analysis
            sample = obj[:sample_size] if len(obj) > sample_size else obj

            if all(_is_json_basic_type(item) for item in sample):
                # Check if all items are JSON-basic types
                homogeneity_result = "json_basic"
            else:
                # Check if all items are the same type
                first_type = type(sample[0])
                homogeneity_result = "single_type" if all(isinstance(item, first_type) for item in sample) else "mixed"

    # Cache the result if we have space
    if homogeneity_result is not None and len(_COLLECTION_COMPATIBILITY_CACHE) < _COLLECTION_CACHE_SIZE_LIMIT:
        _COLLECTION_COMPATIBILITY_CACHE[obj_id] = homogeneity_result

    return homogeneity_result


def _process_homogeneous_dict(
    obj: dict,
    config: Optional["SerializationConfig"],
    _depth: int,
    _seen: Set[int],
    _type_handler: Optional[TypeHandler],
) -> dict:
    """Optimized processing for dictionaries with homogeneous values."""
    # For JSON-basic values, we can skip individual processing
    homogeneity = _is_homogeneous_collection(obj)

    if homogeneity == "json_basic":
        # All values are JSON-compatible, just return as-is
        return obj

    # OPTIMIZATION: Use pooled dictionary for memory efficiency
    result = _get_pooled_dict()
    try:
        if homogeneity == "single_type" and len(obj) > 10:
            # Batch process items of the same type - memory efficient iteration
            for k, v in obj.items():
                # Use the optimized serialization path
                serialized_value = serialize(v, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(v):
                    continue
                result[k] = serialized_value
        else:
            # Fall back to individual processing for mixed types
            for k, v in obj.items():
                serialized_value = serialize(v, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(v):
                    continue
                result[k] = serialized_value

        # Create final result and return dict to pool
        final_result = dict(result)  # Copy the result
        return final_result
    finally:
        # Always return dict to pool, even if exception occurs
        _return_dict_to_pool(result)


def _process_homogeneous_list(
    obj: Union[list, tuple],
    config: Optional["SerializationConfig"],
    _depth: int,
    _seen: Set[int],
    _type_handler: Optional[TypeHandler],
) -> list:
    """Optimized processing for lists/tuples with homogeneous items."""
    # Check homogeneity
    homogeneity = _is_homogeneous_collection(obj)

    if homogeneity == "json_basic":
        # All items are JSON-compatible, just convert tuple to list if needed
        return list(obj) if isinstance(obj, tuple) else obj

    # OPTIMIZATION: Use pooled list for memory efficiency
    result = _get_pooled_list()
    try:
        if homogeneity == "single_type" and len(obj) > 10:
            # Batch process items of the same type - memory efficient iteration
            for x in obj:
                serialized_value = serialize(x, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(x):
                    continue
                result.append(serialized_value)
        else:
            # Fall back to individual processing for mixed types
            for x in obj:
                serialized_value = serialize(x, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(x):
                    continue
                result.append(serialized_value)

        # Create final result and return list to pool
        final_result = list(result)  # Copy the result
        return final_result
    finally:
        # Always return list to pool, even if exception occurs
        _return_list_to_pool(result)


def _get_pooled_dict() -> Dict:
    """Get a dictionary from the pool or create new one."""
    if _RESULT_DICT_POOL:
        result = _RESULT_DICT_POOL.pop()
        result.clear()  # Ensure it's clean
        return result
    return {}


def _return_dict_to_pool(d: Dict) -> None:
    """Return a dictionary to the pool for reuse."""
    if len(_RESULT_DICT_POOL) < _POOL_SIZE_LIMIT:
        d.clear()
        _RESULT_DICT_POOL.append(d)


def _get_pooled_list() -> List:
    """Get a list from the pool or create new one."""
    if _RESULT_LIST_POOL:
        result = _RESULT_LIST_POOL.pop()
        result.clear()  # Ensure it's clean
        return result
    return []


def _return_list_to_pool(lst: List) -> None:
    """Return a list to the pool for reuse."""
    if len(_RESULT_LIST_POOL) < _POOL_SIZE_LIMIT:
        lst.clear()
        _RESULT_LIST_POOL.append(lst)


def _intern_common_string(s: str) -> str:
    """Intern common strings to reduce memory allocation."""
    return _COMMON_STRING_POOL.get(s, s)
