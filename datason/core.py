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
    np = None  # type: ignore

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
    # Define dummy functions/classes for when imports fail with matching signatures
    TypeHandler = None  # type: ignore

    def is_nan_like(obj: Any) -> bool:  # Match exact signature from type_handlers.py
        return False

    def normalize_numpy_types(obj: Any) -> Any:  # Match exact signature from type_handlers.py
        return obj


# Import ML serializers
try:
    from .ml_serializers import detect_and_serialize_ml_object

    _ml_serializer: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = detect_and_serialize_ml_object
except ImportError:
    _ml_serializer = None

# Security constants
MAX_SERIALIZATION_DEPTH = 50  # Prevent stack overflow (reasonable depth for legitimate data)
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
        # Note: For strings, this only checks if it's a string type
        # Length validation should be done separately when config is available
        return True
    elif value_type is float:
        # Efficient NaN/Inf check without function calls
        return value == value and value not in (float("inf"), float("-inf"))
    else:
        return False


def _is_json_basic_type_with_config(value: Any, max_string_length: int) -> bool:
    """Optimized JSON basic type check with configurable string length limit."""
    value_type = type(value)

    if value_type in (int, bool, type(None)):
        return True
    elif value_type is float:
        # Efficient NaN/Inf check without function calls
        return value == value and value not in (float("inf"), float("-inf"))
    elif value_type is str:
        return len(value) <= max_string_length
    else:
        return False


def serialize(
    obj: Any,
    config: Optional["SerializationConfig"] = None,
    _depth: int = 0,
    _seen: Optional[Set[int]] = None,
    _type_handler: Optional["TypeHandler"] = None,
) -> Any:
    """Serialize any Python object to JSON-compatible types with security-first design.

    SECURITY-FIRST ARCHITECTURE:
    1. SECURITY CHECKS (immediate, cheap, always enforced)
    2. PERFORMANCE OPTIMIZATIONS (expensive, only for safe objects)
    3. HOT PATHS (fastest, only for verified safe objects)

    Args:
        obj: The object to serialize.
        config: Optional serialization configuration.
        _depth: Current recursion depth (for internal use).
        _seen: Set of object IDs already seen (for internal use).
        _type_handler: Optional custom type handler (for internal use).

    Returns:
        JSON-compatible representation of the object.

    Raises:
        SecurityError: If security limits are exceeded.
        ValueError: If the object cannot be serialized.
        TypeError: If an unsupported type is encountered.
    """
    # ==================================================================================
    # EMERGENCY CIRCUIT BREAKER: Prevent infinite recursion at all costs
    # ==================================================================================
    if _depth > 100:  # Emergency fallback - should never reach this with proper depth=50 limit
        return f"<EMERGENCY_CIRCUIT_BREAKER: depth={_depth}, type={type(obj).__name__}>"

    # ==================================================================================
    # PHASE 1: SECURITY CHECKS (ALWAYS FIRST, NEVER BYPASSED)
    # ==================================================================================

    # INITIALIZATION: Set up config and type handler if needed
    if config is None and _config_available:
        config = get_default_config()

    # Create type handler if config is available but no type handler provided
    if config and _type_handler is None and _config_available:
        _type_handler = TypeHandler(config)

    # SECURITY CHECK 1: Depth limit enforcement (IMMEDIATE, CHEAP)
    # This prevents ALL depth bomb attacks, including homogeneity bypass
    # USER CONFIGURABLE: Users can set config.max_depth for their specific needs
    max_depth = config.max_depth if config else MAX_SERIALIZATION_DEPTH
    if _depth > max_depth:
        raise SecurityError(
            f"Maximum serialization depth ({max_depth}) exceeded. "
            f"Current depth: {_depth}. This may indicate circular references, "
            "extremely nested data, or a potential depth bomb attack. "
            f"You can increase max_depth in your SerializationConfig if needed."
        )

    # SECURITY CHECK 2: Initialize circular reference tracking
    if _seen is None:
        _seen = set()

    # SECURITY CHECK 3: Circular reference detection for container types
    if isinstance(obj, (dict, list, set, tuple)):
        obj_id = id(obj)
        if obj_id in _seen:
            # CIRCULAR REFERENCE DETECTED: Handle gracefully with warning
            warnings.warn(
                f"Circular reference detected at depth {_depth}. Replacing with None to prevent infinite loops.",
                stacklevel=4,
            )
            return None  # Handle circular reference by returning None
        _seen.add(obj_id)

    try:
        # SECURITY CHECK 4: Early size limits for containers (CHEAP)
        obj_type = type(obj)
        if isinstance(obj, dict) and len(obj) > (config.max_size if config else MAX_OBJECT_SIZE):
            raise SecurityError(f"Dictionary size ({len(obj)}) exceeds maximum allowed size.")
        elif isinstance(obj, (list, tuple)) and len(obj) > (config.max_size if config else MAX_OBJECT_SIZE):
            raise SecurityError(f"List/tuple size ({len(obj)}) exceeds maximum allowed size.")

        # Handle None early (most common case in sparse data)
        if obj is None:
            return None

        # ==================================================================================
        # PHASE 2: PERFORMANCE OPTIMIZATIONS (ONLY FOR SECURITY-VERIFIED OBJECTS)
        # ==================================================================================

        # PERFORMANCE CHECK 1: Try hot path for basic types and simple containers
        # This is safe because we already passed security checks
        max_string_length = config.max_string_length if config else MAX_STRING_LENGTH
        hot_result = _serialize_hot_path(obj, config, max_string_length)
        if hot_result is not None:
            return hot_result

        # PERFORMANCE CHECK 2: Advanced optimizations for verified safe containers
        # Handle dictionaries with SECURITY-VERIFIED performance optimizations
        if isinstance(obj, dict):
            # Quick path for empty dicts (already security-checked)
            if not obj:
                return obj

            # PERFORMANCE OPTIMIZATION: Homogeneity check (ONLY after security verification)
            # This is now safe because we already enforced depth limits
            homogeneity = None
            if _depth < 5:  # Only use optimization at reasonable depths
                # Safe homogeneity check with strict limits
                homogeneity = _is_homogeneous_collection(obj, sample_size=10, _max_check_depth=2)

            # Quick JSON compatibility check for small dicts
            if (
                homogeneity == "json_basic"
                and len(obj) <= 5
                and all(
                    isinstance(k, str)
                    and type(v) in _JSON_BASIC_TYPES
                    and (type(v) is not str or len(v) <= max_string_length)
                    for k, v in obj.items()
                )
            ):
                return obj

            # GUARANTEED SAFE PROCESSING: Each recursive call has verified depth increment
            result = {}
            for k, v in obj.items():
                # SECURITY: EVERY recursive call MUST increment depth (verified safe)
                serialized_value = serialize(v, config, _depth + 1, _seen, _type_handler)
                result[k] = serialized_value

            # Sort keys if configured
            if config and config.sort_keys:
                return dict(sorted(result.items()))
            return result

        # Handle list/tuple with SECURITY-VERIFIED performance optimizations
        elif obj_type in (_TYPE_LIST, _TYPE_TUPLE):
            # Quick path for empty list/tuple (already security-checked)
            if not obj:
                return [] if obj_type is _TYPE_TUPLE else obj

            # PERFORMANCE OPTIMIZATION: Homogeneity check (ONLY after security verification)
            homogeneity = None
            if _depth < 5:  # Only use optimization at reasonable depths
                # Safe homogeneity check with strict limits
                homogeneity = _is_homogeneous_collection(obj, sample_size=10, _max_check_depth=2)

                # Quick JSON compatibility check for small lists
            if (
                homogeneity == "json_basic"
                and len(obj) <= 5
                and all(type(item) in _JSON_BASIC_TYPES for item in obj)
                and not (config and config.include_type_hints and obj_type is _TYPE_TUPLE)
            ):
                return list(obj) if obj_type is _TYPE_TUPLE else obj

            # GUARANTEED SAFE PROCESSING: Each recursive call has verified depth increment
            result_list = []
            for item in obj:
                # SECURITY: EVERY recursive call MUST increment depth (verified safe)
                serialized_value = serialize(item, config, _depth + 1, _seen, _type_handler)
                result_list.append(serialized_value)

            # Handle type metadata for tuples
            if isinstance(obj, tuple) and config and config.include_type_hints:
                return _create_type_metadata("tuple", result_list)
            return result_list

        # ==================================================================================
        # PHASE 3: COMPLEX TYPE PROCESSING (FULL SECURITY + CUSTOM HANDLING)
        # ==================================================================================

        # For complex types, use full processing path with security verification
        return _serialize_full_path(obj, config, _depth, _seen, _type_handler, max_string_length)

    finally:
        # Clean up: remove from seen set when done processing
        if isinstance(obj, (dict, list, set, tuple)):
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
    _type_handler: Optional["TypeHandler"],
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
                result = handler(obj)
                return result
            except Exception as e:
                # If custom handler fails, log warning and continue with default handling
                warnings.warn(f"Custom type handler failed for {type(obj)}: {e}", stacklevel=3)
        # If no handler or handler failed, continue with default processing below

    # Handle dicts with full processing - SECURITY: Apply homogeneity bypass protection here too!
    if type_category == "dict":
        # CRITICAL SECURITY CHECK: Prevent homogeneity bypass attack in full path too
        # The attack was bypassing the main serialize() check by going through this path
        if _contains_potentially_exploitable_nested_structure(obj, _depth):
            # SECURITY: Raise SecurityError for potentially exploitable structures
            # This prevents the homogeneity bypass attack from succeeding
            raise SecurityError(
                f"Maximum serialization depth exceeded due to potentially exploitable nested structure at depth {_depth}. "
                "This structure appears designed to bypass security measures through "
                "homogeneity optimization paths, which is characteristic of depth bomb attacks."
            )

        # Safe to use homogeneous processing for non-exploitable structures
        result = _process_homogeneous_dict(obj, config, _depth, _seen, _type_handler)
        # Sort keys if configured
        if config and config.sort_keys:
            return dict(sorted(result.items()))
        return result

    # Handle lists/tuples with full processing
    if type_category == "list":
        # CRITICAL SECURITY CHECK: Prevent homogeneity bypass attack for lists in full path too
        if isinstance(obj, list) and _contains_potentially_exploitable_nested_list_structure(obj, _depth):
            # SECURITY: Raise SecurityError for potentially exploitable list structures
            raise SecurityError(
                f"Maximum serialization depth exceeded due to potentially exploitable nested list structure at depth {_depth}. "
                "This structure appears designed to bypass security measures through "
                "homogeneity optimization paths, which is characteristic of depth bomb attacks."
            )

        result_list = _process_homogeneous_list(obj, config, _depth, _seen, _type_handler)
        # Handle type metadata for tuples
        if isinstance(obj, tuple) and config and config.include_type_hints:
            return _create_type_metadata("tuple", result_list)
        return result_list

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
        # SECURITY: Early detection of problematic objects that can cause deep recursion
        obj_class_name = type(obj).__name__
        obj_module = getattr(type(obj), "__module__", "")

        # Detect MagicMock and other unittest.mock objects early
        if obj_module == "unittest.mock":
            warnings.warn(
                f"Detected potentially problematic mock object: {obj_class_name}. Using safe string representation.",
                stacklevel=3,
            )
            return f"<{obj_class_name} object>"

        # Detect other problematic types early
        if obj_module in ("io", "_io") and hasattr(obj, "__dict__"):
            warnings.warn(
                f"Detected potentially problematic IO object: {obj_class_name}. Using safe string representation.",
                stacklevel=3,
            )
            return f"<{obj_class_name} object>"

        try:
            # BUGFIX: Check for circular references in __dict__ before recursing
            # The __dict__ is a different object with different ID, so we need explicit checks
            obj_dict = obj.__dict__

            # Safety check: prevent processing extremely complex objects
            if len(obj_dict) > 100:  # Limit the number of attributes
                warnings.warn(
                    f"Object has too many attributes ({len(obj_dict)}). Using string representation.",
                    stacklevel=3,
                )
                return f"<{type(obj).__name__} object with {len(obj_dict)} attributes>"

            # Check if the __dict__ itself creates a circular reference
            if id(obj_dict) in _seen:
                warnings.warn(
                    "Circular reference detected in object.__dict__. Replacing with placeholder.",
                    stacklevel=3,
                )
                return {"__circular_reference__": f"<{type(obj).__name__} object>"}

            # Check if any values in __dict__ refer back to the original object
            # This catches cases where obj.__dict__['attr'] == obj
            for key, value in obj_dict.items():
                if id(value) in _seen:
                    warnings.warn(
                        f"Circular reference detected in object.__dict__['{key}']. Using safe serialization.",
                        stacklevel=3,
                    )
                    # Create a safe copy of __dict__ without circular references
                    safe_dict = {}
                    for k, v in obj_dict.items():
                        if id(v) in _seen:
                            safe_dict[k] = f"<circular reference to {type(v).__name__}>"
                        else:
                            safe_dict[k] = v
                    return serialize(safe_dict, config, _depth + 1, _seen, _type_handler)

            # Additional safety: check for problematic object types that often cause issues
            for key, value in obj_dict.items():
                # Skip known problematic types that can cause infinite recursion
                if (
                    hasattr(value, "__class__")
                    and value.__class__.__module__ in ("unittest.mock", "io", "_io")
                    and hasattr(value, "__dict__")
                    and len(getattr(value, "__dict__", {})) > 10
                ):
                    warnings.warn(
                        f"Skipping potentially problematic object in __dict__['{key}'] of type {type(value).__name__}",
                        stacklevel=3,
                    )
                    obj_dict = dict(obj_dict)  # Make a copy
                    obj_dict[key] = f"<{type(value).__name__} object - serialization skipped>"

            # If no immediate circular references found, proceed normally
            return serialize(obj_dict, config, _depth + 1, _seen, _type_handler)
        except RecursionError:
            # Catch recursion errors specifically
            warnings.warn(
                f"RecursionError while serializing {type(obj).__name__}. Using string representation.",
                stacklevel=3,
            )
            return f"<{type(obj).__name__} object - recursion limit reached>"
        except Exception as e:
            # Catch any other exceptions during __dict__ processing
            warnings.warn(
                f"Error serializing {type(obj).__name__}: {e}. Using string representation.",
                stacklevel=3,
            )
            pass  # Continue to fallback

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
    if isinstance(value, (int, bool)):
        return True
    if isinstance(value, str):
        # Check string length against the default limit
        return len(value) <= MAX_STRING_LENGTH
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

    def chunk_generator() -> Generator[Any, None, None]:
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

    def chunk_generator() -> Generator[Any, None, None]:
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

    def chunk_generator() -> Generator[Any, None, None]:
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

    def chunk_generator() -> Generator[Any, None, None]:
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
        self._file: Optional[Any] = None
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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


def _is_homogeneous_collection(
    obj: Union[list, tuple, dict],
    sample_size: int = 20,
    _seen_ids: Optional[Set[int]] = None,
    _max_check_depth: int = 5,
) -> Optional[str]:
    """Check if a collection contains homogeneous types for bulk processing.

    SECURITY: Added circular reference protection and depth limits to prevent infinite recursion.

    Args:
        obj: Collection to check for homogeneity
        sample_size: Number of items to sample for checking
        _seen_ids: Set of object IDs already seen (for circular reference detection)
        _max_check_depth: Maximum depth to check for homogeneity (prevents deep recursion)

    Returns:
    - 'json_basic': All items are JSON-basic types
    - 'single_type': All items are the same non-basic type
    - 'mixed': Mixed types (requires individual processing)
    - None: Unable to determine, too small, or depth limit exceeded
    """
    # SECURITY: Depth limit check - prevent deep recursion in homogeneity analysis
    if _max_check_depth <= 0:
        return "mixed"  # Force individual processing when depth limit exceeded

    # Initialize circular reference tracking on first call
    if _seen_ids is None:
        _seen_ids = set()

    # SECURITY: Check for circular references
    obj_id = id(obj)
    if obj_id in _seen_ids:
        # Circular reference detected, assume mixed to force full processing
        return "mixed"

    # Add to seen set for this traversal
    _seen_ids.add(obj_id)

    try:
        # OPTIMIZATION: Check cache first for collections we've seen before
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
                elif all(_is_json_basic_type_safe(v, _seen_ids, _max_check_depth - 1) for v in sample):
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
                # Sample items for type analysis - handle Union type properly
                sample_items = list(obj[:sample_size]) if len(obj) > sample_size else list(obj)

                if all(_is_json_basic_type_safe(item, _seen_ids, _max_check_depth - 1) for item in sample_items):
                    # Check if all items are JSON-basic types
                    homogeneity_result = "json_basic"
                else:
                    # Check if all items are the same type
                    first_type = type(sample_items[0])
                    homogeneity_result = (
                        "single_type" if all(isinstance(item, first_type) for item in sample_items) else "mixed"
                    )

        # Cache the result if we have space
        if homogeneity_result is not None and len(_COLLECTION_COMPATIBILITY_CACHE) < _COLLECTION_CACHE_SIZE_LIMIT:
            _COLLECTION_COMPATIBILITY_CACHE[obj_id] = homogeneity_result

        return homogeneity_result

    finally:
        # Clean up: remove from seen set when done
        _seen_ids.discard(obj_id)


def _is_json_basic_type_safe(value: Any, _seen_ids: Set[int], _max_check_depth: int) -> bool:
    """Safe version of _is_json_basic_type that prevents infinite recursion.

    Args:
        value: The value to check
        _seen_ids: Set of object IDs already seen to prevent circular references
        _max_check_depth: Maximum depth to check for homogeneity (prevents deep recursion)

    Returns:
        True if value is a basic JSON type, False otherwise
    """
    # SECURITY: Depth limit check - if we're too deep, assume not basic
    if _max_check_depth <= 0:
        return False

    # For non-container types, use regular check
    value_type = type(value)

    if value_type in (str, int, bool, type(None)):
        return True
    elif value_type is float:
        # Efficient NaN/Inf check without function calls
        return value == value and value not in (float("inf"), float("-inf"))
    elif value_type in (dict, list, tuple):
        # SECURITY: For containers, check for circular references first
        value_id = id(value)
        if value_id in _seen_ids:
            # Circular reference - not safe for basic type assumption
            return False

        # For containers, they're not "basic" types in the JSON sense
        # Only primitives are truly basic
        return False
    else:
        return False


def _process_homogeneous_dict(
    obj: dict,
    config: Optional["SerializationConfig"],
    _depth: int,
    _seen: Set[int],
    _type_handler: Optional["TypeHandler"],
) -> dict:
    """Optimized processing for dictionaries with homogeneous values.

    SECURITY: Enforces depth limits before processing to prevent depth bomb attacks.
    """
    # SECURITY: Enforce depth limit FIRST before any processing
    max_depth = config.max_depth if config else MAX_SERIALIZATION_DEPTH
    if _depth > max_depth:
        raise SecurityError(
            f"Maximum serialization depth ({max_depth}) exceeded in dict processing. "
            "This may indicate circular references or extremely nested data."
        )

    # CRITICAL SECURITY FIX: COMPLETELY DISABLE all homogeneity optimization if dict contains nested containers
    # This prevents the homogeneity bypass attack where nested structures exploit optimization paths
    homogeneity: Optional[str] = "mixed"  # Default to safe processing

    # Check if this dict contains any nested container types (dicts/lists/tuples)
    contains_nested_containers = False
    for value in obj.values():
        if isinstance(value, (dict, list, tuple)):
            contains_nested_containers = True
            break

    # Only allow homogeneity optimization for very shallow depths AND no nested containers
    if _depth < 2 and not contains_nested_containers:
        # SECURITY: Use minimal depth check to prevent any deep recursion
        homogeneity = _is_homogeneous_collection(obj, _max_check_depth=1)

    if homogeneity == "json_basic" and _depth < 2 and not contains_nested_containers:
        # All values are JSON-compatible, but need to check string lengths with config
        if config and config.max_string_length < MAX_STRING_LENGTH:
            # Need to validate string lengths
            for v in obj.values():
                if isinstance(v, str) and len(v) > config.max_string_length:
                    # Has long strings, needs full processing
                    break
            else:
                # All strings are within limits, can return as-is
                return obj
        else:
            # No length limits or using defaults, just return as-is
            return obj

    # OPTIMIZATION: Use pooled dictionary for memory efficiency
    result = _get_pooled_dict()
    try:
        if homogeneity == "single_type" and len(obj) > 10 and _depth < 2 and not contains_nested_containers:
            # Batch process items of the same type - memory efficient iteration
            # SECURITY: Only allow this optimization at very shallow depths with no nested containers
            for k, v in obj.items():
                # Use the optimized serialization path with SECURE depth increment
                serialized_value = serialize(v, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(v):
                    continue
                result[k] = serialized_value
        else:
            # Fall back to individual processing for mixed types OR deep structures OR nested containers
            # This path has proper depth enforcement in each serialize() call
            for k, v in obj.items():
                # SECURITY: Ensure depth is properly incremented for ALL recursive calls
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
    _type_handler: Optional["TypeHandler"],
) -> list:
    """Optimized processing for lists/tuples with homogeneous items.

    SECURITY: Enforces depth limits before processing to prevent depth bomb attacks.
    """
    # SECURITY: Enforce depth limit FIRST before any processing
    max_depth = config.max_depth if config else MAX_SERIALIZATION_DEPTH
    if _depth > max_depth:
        raise SecurityError(
            f"Maximum serialization depth ({max_depth}) exceeded in list processing. "
            "This may indicate circular references or extremely nested data."
        )

    # CRITICAL SECURITY FIX: COMPLETELY DISABLE homogeneity optimization for depth >= 2
    # The homogeneity bypass attack exploits the optimization path by creating structures
    # that appear homogeneous at each level, allowing infinite recursion before depth
    # limits can be enforced. For maximum security, we disable ALL homogeneity checking
    # for any depth >= 2, forcing individual processing which has proper depth enforcement.
    homogeneity: Optional[str] = "mixed"  # Force non-optimized path for all depths >= 2
    if _depth < 2:
        # Only allow homogeneity optimization at very shallow depths (0-1)
        # SECURITY: Use minimal depth check to prevent any deep recursion
        homogeneity = _is_homogeneous_collection(obj, _max_check_depth=1)

    if homogeneity == "json_basic" and _depth < 2:
        # All items are JSON-compatible, but need to check string lengths with config
        if config and config.max_string_length < MAX_STRING_LENGTH:
            # Need to validate string lengths
            for item in obj:
                if isinstance(item, str) and len(item) > config.max_string_length:
                    # Has long strings, needs full processing
                    break
            else:
                # All strings are within limits, can convert and return
                return list(obj) if isinstance(obj, tuple) else obj
        else:
            # No length limits or using defaults, just convert tuple to list if needed
            return list(obj) if isinstance(obj, tuple) else obj

    # OPTIMIZATION: Use pooled list for memory efficiency
    result = _get_pooled_list()
    try:
        if homogeneity == "single_type" and len(obj) > 10 and _depth < 2:
            # Batch process items of the same type - memory efficient iteration
            # SECURITY: Only allow this optimization at very shallow depths
            for x in obj:
                # SECURITY: Ensure depth is properly incremented for ALL recursive calls
                serialized_value = serialize(x, config, _depth + 1, _seen, _type_handler)
                # Handle NaN dropping at collection level
                if config and config.nan_handling == NanHandling.DROP and serialized_value is None and is_nan_like(x):
                    continue
                result.append(serialized_value)
        else:
            # Fall back to individual processing for mixed types OR deep structures
            # This path has proper depth enforcement in each serialize() call
            for x in obj:
                # SECURITY: Ensure depth is properly incremented for ALL recursive calls
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


# PHASE 2.1: JSON-FIRST SERIALIZATION STRATEGY
# Ultra-fast JSON compatibility detection and processing


def _is_fully_json_compatible(obj: Any, max_depth: int = 3, _current_depth: int = 0) -> bool:
    """Ultra-fast JSON compatibility check with depth limit and aggressive inlining.

    This function uses aggressive optimizations to determine if an object is
    fully JSON-compatible without requiring any custom serialization logic.

    Returns True only if the object can be passed directly to json.dumps()
    without any processing.
    """
    # Security: prevent excessive depth traversal
    if _current_depth > max_depth:
        return False

    # OPTIMIZATION: Inline type checking for hot path
    obj_type = type(obj)

    # Handle primitives with zero function calls
    if obj_type in (_TYPE_INT, _TYPE_BOOL, _TYPE_NONE):
        return True

    # Handle strings with length check
    if obj_type is _TYPE_STR:
        # Strings longer than MAX_STRING_LENGTH need processing
        return len(obj) <= MAX_STRING_LENGTH

    # Handle float with inline NaN/Inf check
    if obj_type is _TYPE_FLOAT:
        # Inline check: NaN and Inf are not JSON compatible
        return obj == obj and obj not in (float("inf"), float("-inf"))

    # Handle dict with aggressive optimization
    if obj_type is _TYPE_DICT:
        # Empty dict is compatible
        if not obj:
            return True

        # ENHANCEMENT: Increase size limits for better coverage
        if len(obj) > 200:  # Increased from 50 to handle larger API responses
            return False

        # Check all keys/values with early bailout
        try:
            for k, v in obj.items():
                # Keys must be strings for JSON compatibility
                if type(k) is not _TYPE_STR:
                    return False
                # Recursively check value (with depth limit)
                if not _is_fully_json_compatible(v, max_depth, _current_depth + 1):
                    return False
            return True
        except (AttributeError, TypeError):
            return False

    # Handle list with aggressive optimization
    if obj_type is _TYPE_LIST:
        # Empty list is compatible
        if not obj:
            return True

        # ENHANCEMENT: Increase size limits for better coverage
        if len(obj) > 500:  # Increased from 100 to handle larger arrays
            return False

        # Check all items with early bailout
        try:
            return all(_is_fully_json_compatible(item, max_depth, _current_depth + 1) for item in obj)
        except (AttributeError, TypeError):
            return False

    # ENHANCEMENT: Handle tuple as potential JSON list
    if obj_type is _TYPE_TUPLE:
        # Empty tuple is compatible (converts to empty list)
        if not obj:
            return True

        # Reasonable size limit for tuples
        if len(obj) > 500:
            return False

        # Check all items with early bailout
        try:
            return all(_is_fully_json_compatible(item, max_depth, _current_depth + 1) for item in obj)
        except (AttributeError, TypeError):
            return False

    # All other types are not JSON-compatible
    # This includes: set, datetime, UUID, numpy, pandas, custom objects
    return False


def _serialize_json_only_fast_path(obj: Any) -> Any:
    """Ultra-fast serialization for JSON-compatible objects.

    This function assumes the object is fully JSON-compatible and performs
    minimal processing. Should only be called after _is_fully_json_compatible
    returns True.
    """
    # ENHANCEMENT: Handle tuple-to-list conversion recursively
    obj_type = type(obj)

    if obj_type is _TYPE_TUPLE:
        # Convert tuple to list for JSON compatibility
        return _convert_tuple_to_list_fast(obj)
    elif obj_type is _TYPE_DICT:
        # Check if dict contains any tuples that need conversion
        if any(type(v) is _TYPE_TUPLE for v in obj.values()):
            return _convert_dict_tuples_fast(obj)
        return obj
    elif obj_type is _TYPE_LIST:
        # Check if list contains any tuples that need conversion
        if any(type(item) is _TYPE_TUPLE for item in obj):
            return _convert_list_tuples_fast(obj)
        return obj

    # For all other JSON-compatible objects, return as-is
    return obj


def _convert_dict_tuples_fast(obj: dict) -> dict:
    """Fast dict processing to convert nested tuples to lists."""
    result: Dict[str, Any] = {}
    for k, v in obj.items():
        if type(v) is _TYPE_TUPLE:
            result[k] = _convert_tuple_to_list_fast(v)
        elif type(v) is _TYPE_DICT:
            # Check for nested tuples in dicts
            if any(type(nested_v) is _TYPE_TUPLE for nested_v in v.values()):
                result[k] = _convert_dict_tuples_fast(v)
            else:
                result[k] = v
        elif type(v) is _TYPE_LIST:
            # Check for nested tuples in lists
            if any(type(item) is _TYPE_TUPLE for item in v):
                result[k] = _convert_list_tuples_fast(v)
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def _convert_list_tuples_fast(obj: list) -> list:
    """Fast list processing to convert nested tuples to lists."""
    result: List[Any] = []
    for item in obj:
        if type(item) is _TYPE_TUPLE:
            result.append(_convert_tuple_to_list_fast(item))
        elif type(item) is _TYPE_DICT:
            # Check for nested tuples in dicts
            if any(type(v) is _TYPE_TUPLE for v in item.values()):
                result.append(_convert_dict_tuples_fast(item))
            else:
                result.append(item)
        elif type(item) is _TYPE_LIST:
            # Check for nested tuples in lists
            if any(type(nested_item) is _TYPE_TUPLE for nested_item in item):
                result.append(_convert_list_tuples_fast(item))
            else:
                result.append(item)
        else:
            result.append(item)
    return result


def _convert_tuple_to_list_fast(obj: tuple) -> list:
    """Fast tuple-to-list conversion for JSON-compatible tuples.

    Assumes all items in the tuple are already JSON-compatible.
    """
    # For small tuples, direct list conversion is fastest
    if len(obj) <= 5:
        return [_convert_tuple_to_list_fast(item) if type(item) is _TYPE_TUPLE else item for item in obj]

    # For larger tuples, process iteratively
    result: List[Any] = []
    for item in obj:
        if type(item) is _TYPE_TUPLE:
            result.append(_convert_tuple_to_list_fast(item))
        elif type(item) is _TYPE_DICT:
            if any(type(v) is _TYPE_TUPLE for v in item.values()):
                result.append(_convert_dict_tuples_fast(item))
            else:
                result.append(item)
        elif type(item) is _TYPE_LIST:
            if any(type(nested_item) is _TYPE_TUPLE for nested_item in item):
                result.append(_convert_list_tuples_fast(item))
            else:
                result.append(item)
        else:
            result.append(item)
    return result


# PHASE 2.2: RECURSIVE CALL ELIMINATION
# Iterative processing to eliminate serialize() → serialize() overhead


def _serialize_iterative_path(obj: Any, config: Optional["SerializationConfig"]) -> Any:
    """Iterative serialization to eliminate recursive function call overhead.

    This processes nested structures using a stack-based approach instead of
    recursive serialize() calls, significantly reducing function call overhead.
    """
    if config is None and _config_available:
        config = get_default_config()

    # Use iterative processing for homogeneous collections
    obj_type = type(obj)

    if obj_type is _TYPE_DICT:
        return _process_dict_iterative(obj, config)
    elif obj_type in (_TYPE_LIST, _TYPE_TUPLE):
        result = _process_list_iterative(obj, config)
        # Handle type metadata for tuples
        if isinstance(obj, tuple) and config and config.include_type_hints:
            return _create_type_metadata("tuple", result)
        return result

    # For non-collection types, fall back to normal processing
    return None  # Indicates full processing needed


def _process_dict_iterative(obj: dict, config: Optional["SerializationConfig"]) -> dict:
    """Iterative dict processing to eliminate recursive calls."""
    if not obj:
        return obj

    # Quick homogeneity check
    if len(obj) <= 20:
        # For small dicts, check if all values are basic JSON types
        all_basic = True
        for value in obj.values():
            if not _is_json_basic_type(value):
                all_basic = False
                break

        if all_basic:
            return obj  # Already JSON-compatible

    # Process with minimal function calls
    result = {}
    for k, v in obj.items():
        v_type = type(v)

        # Inline processing for common types
        if v_type in (_TYPE_STR, _TYPE_INT, _TYPE_BOOL, _TYPE_NONE):
            result[k] = v
        elif v_type is _TYPE_FLOAT:
            # Inline NaN/Inf handling
            if v == v and v not in (float("inf"), float("-inf")):
                result[k] = v
            else:
                # Handle NaN/Inf according to config
                if config and hasattr(config, "nan_handling") and config.nan_handling == NanHandling.DROP:
                    continue  # Skip this key-value pair
                result[k] = None  # Default NaN handling
        elif v_type is _TYPE_DICT:
            # Recursive dict processing (but only one level of function calls)
            result[k] = _process_dict_iterative(v, config)
        elif v_type in (_TYPE_LIST, _TYPE_TUPLE):
            # Recursive list processing (but only one level of function calls)
            processed = _process_list_iterative(v, config)
            if v_type is _TYPE_TUPLE and config and config.include_type_hints:
                result[k] = _create_type_metadata("tuple", processed)
            else:
                result[k] = processed
        else:
            # For complex types, use safe string representation to avoid infinite recursion
            # This eliminates the circular call: serialize() -> _serialize_full_path() -> _process_dict_iterative() -> serialize()
            try:
                str_repr = str(v)
                if len(str_repr) > 100:  # Limit string length
                    str_repr = str_repr[:100] + "...[TRUNCATED]"
                result[k] = str_repr
            except Exception:
                result[k] = f"<{type(v).__name__} object>"

    return result


def _process_list_iterative(obj: Union[list, tuple], config: Optional["SerializationConfig"]) -> list:
    """Process a list or tuple using an iterative approach to avoid deep recursion."""
    result: List[Any] = []
    stack = [(obj, result, 0)]  # (current_obj, current_result, index)

    while stack:
        current_obj, current_result, index = stack.pop()

        if isinstance(current_obj, (list, tuple)) and index < len(current_obj):
            item = current_obj[index]

            # Push back the same object with incremented index
            if index + 1 < len(current_obj):
                stack.append((current_obj, current_result, index + 1))

            # If item is a container, we'll need to process it
            if isinstance(item, (dict, list, tuple)):
                # For now, just convert to avoid deep recursion
                if isinstance(item, dict):
                    current_result.append(dict(item))
                elif isinstance(item, tuple):
                    current_result.append(list(item))
                else:
                    current_result.append(list(item))
            else:
                # Simple type, return as-is to avoid recursive calls
                current_result.append(item)  # SECURITY FIX: No recursive serialize call

    return result


# SECURITY FUNCTION: Check if a dictionary contains nested structures that could exploit homogeneity optimization
def _contains_potentially_exploitable_nested_structure(obj: dict, _depth: int) -> bool:
    """
    Check if a dictionary contains nested structures that could exploit the homogeneity bypass attack.

    This function prevents the attack by identifying structures where:
    1. The dictionary contains other dictionaries that could create recursive potential
    2. The structure appears designed to bypass depth limits through homogeneity optimization
    3. The pattern suggests an intentional exploit rather than legitimate nested data

    IMPORTANT: This does NOT trigger on circular references (same object) - those are handled
    by the circular reference detection mechanism. This only triggers on new nested dictionaries.

    KEY INSIGHT: The attack exploits homogeneity optimization by creating structures that LOOK
    homogeneous (identical patterns at each level) to bypass depth checks. Legitimate data
    rarely has this perfect homogeneity.

    Args:
        obj: Dictionary to check
        _depth: Current depth in the serialization tree

    Returns:
        True if the structure is potentially exploitable, False otherwise
    """
    # SECURITY: Focus on the specific homogeneity bypass attack pattern
    # The attack creates structures with identical keys at multiple levels to appear homogeneous

    # At any depth, check for the specific attack signature: identical keys at nested levels
    if len(obj) == 1:
        key, value = next(iter(obj.items()))
        if isinstance(value, dict) and len(value) == 1:
            # CRITICAL: Don't flag circular references - those are legitimate and handled elsewhere
            if id(value) == id(obj):
                return False  # This is a circular reference, not an attack
            inner_key = next(iter(value.keys()))
            if key == inner_key:
                # This is the specific homogeneous attack pattern - identical keys at multiple levels
                return True

    # ADDITIONAL CHECK: Only at very deep levels (depth >= 5) where attacks become critical
    # AND only if it looks like a homogeneous pattern (multiple identical structures)
    if _depth >= 5:
        # Look for signs of homogeneous attack: multiple values that are identical in structure
        dict_values = [v for v in obj.values() if isinstance(v, dict)]
        if len(dict_values) >= 2:
            # Check if the nested dicts have identical key patterns (sign of attack)
            first_keys = set(dict_values[0].keys()) if dict_values else set()
            if first_keys and all(set(d.keys()) == first_keys for d in dict_values[1:]):
                # All nested dicts have identical key structure - potential attack
                return True

    return False


# SECURITY FUNCTION: Check if a list contains nested structures that could exploit homogeneity optimization
def _contains_potentially_exploitable_nested_list_structure(obj: list, _depth: int) -> bool:
    """
    Check if a list contains nested structures that could exploit the homogeneity bypass attack.

    This function prevents the attack by identifying structures where:
    1. The list contains other lists that could create recursive potential
    2. The structure appears designed to bypass depth limits through homogeneity optimization
    3. The pattern suggests an intentional exploit rather than legitimate nested data

    IMPORTANT: This does NOT trigger on circular references (same object) - those are handled
    by the circular reference detection mechanism. This only triggers on new nested lists.

    Args:
        obj: List to check
        _depth: Current depth in the serialization tree

    Returns:
        True if the structure is potentially exploitable, False otherwise
    """
    # SECURITY: Only check at deeper levels where the bypass attack becomes dangerous
    # Allow normal nesting at shallow depths (depth 0-2 should be fine for legitimate use)
    if _depth < 3:
        # At shallow depths, only block the specific attack pattern
        if len(obj) == 1 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 1:
            # CRITICAL: Don't flag circular references - those are legitimate and handled elsewhere
            # Single-item list containing single-item list - classic attack pattern (unless circular ref)
            return id(obj[0]) != id(obj)  # False if circular reference, True if attack pattern
        return False

    # At deeper levels (depth >= 3), check for nested lists (but not circular refs)
    obj_id = id(obj)
    for item in obj:
        if isinstance(item, (list, tuple)):
            # Don't flag circular references
            if id(item) == obj_id:
                continue  # This is a circular reference, skip it
            return True  # This is a new nested list at deep level - potentially suspicious

    return False
