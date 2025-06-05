"""Deserialization functionality for datason.

This module provides functions to convert JSON-compatible data back to appropriate
Python objects, including datetime parsing, UUID reconstruction, and pandas types.
"""

import decimal
import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

# Import configuration and security constants
try:
    from .config import SerializationConfig, get_default_config

    _config_available = True
except ImportError:
    _config_available = False
    SerializationConfig = None

    def get_default_config():
        return None


# SECURITY: Import same constants as core.py for consistency
try:
    from .core import MAX_OBJECT_SIZE, MAX_SERIALIZATION_DEPTH, MAX_STRING_LENGTH
except ImportError:
    # Fallback constants if core import fails - SECURITY FIX: Use secure values
    MAX_SERIALIZATION_DEPTH = 50
    MAX_OBJECT_SIZE = 100_000  # Prevent size bomb attacks
    MAX_STRING_LENGTH = 1_000_000


# OPTIMIZATION: Module-level caches for ultra-fast deserialization (mirroring core.py patterns)
_DESERIALIZATION_TYPE_CACHE: Dict[str, str] = {}  # Maps string patterns to detected types
_TYPE_CACHE_SIZE_LIMIT = 1000  # Prevent memory growth

# OPTIMIZATION: String pattern caches for repeated type detection
_STRING_PATTERN_CACHE: Dict[int, str] = {}  # Maps string id to detected pattern type
_STRING_CACHE_SIZE_LIMIT = 500  # Smaller cache for strings

# OPTIMIZATION: Common parsed objects cache for frequently used values
_PARSED_OBJECT_CACHE: Dict[str, Any] = {}  # Maps string to parsed object
_PARSED_CACHE_SIZE_LIMIT = 200  # Cache for common UUIDs/datetimes

# OPTIMIZATION: Memory allocation optimization - Phase 1 Step 1.4 (mirroring core.py)
# Pre-allocated result containers for reuse
_RESULT_DICT_POOL: List[Dict] = []
_RESULT_LIST_POOL: List[List] = []
_POOL_SIZE_LIMIT = 20  # Limit pool size to prevent memory bloat

# OPTIMIZATION: Function call overhead reduction - Phase 1 Step 1.5 (mirroring core.py)
# Pre-computed type sets for ultra-fast membership testing
_JSON_BASIC_TYPES = (str, int, bool, type(None))
_NUMERIC_TYPES = (int, float)
_CONTAINER_TYPES = (dict, list)

# Inline type checking constants for hot path optimization
_TYPE_STR = str
_TYPE_INT = int
_TYPE_BOOL = bool
_TYPE_NONE = type(None)
_TYPE_FLOAT = float
_TYPE_DICT = dict
_TYPE_LIST = list

# OPTIMIZATION: Pre-compiled pattern matchers for ultra-fast string detection
_UUID_CHAR_SET = set("0123456789abcdefABCDEF-")
_DATETIME_CHAR_SET = set("0123456789-T:Z.+")
_PATH_CHAR_SET = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/\\._-~:")


class DeserializationSecurityError(Exception):
    """Raised when security limits are exceeded during deserialization."""

    pass


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
        # Try to parse as UUID first (more specific pattern)
        if parse_uuids and _looks_like_uuid(obj):
            try:
                import uuid as uuid_module  # Fresh import to avoid state issues

                return uuid_module.UUID(obj)
            except (ValueError, ImportError):
                # Log parsing failure but continue with string
                warnings.warn(f"Failed to parse UUID string: {obj}", stacklevel=2)

        # Try to parse as datetime if enabled
        if parse_dates and _looks_like_datetime(obj):
            try:
                from datetime import datetime as datetime_class  # Fresh import

                return datetime_class.fromisoformat(obj.replace("Z", "+00:00"))
            except (ValueError, ImportError):
                # Log parsing failure but continue with string
                warnings.warn(
                    f"Failed to parse datetime string: {obj[:50]}{'...' if len(obj) > 50 else ''}",
                    stacklevel=2,
                )

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
    """Enhanced: Deserialize objects with embedded type metadata for perfect round-trips.

    Supports both new format (__datason_type__) and legacy format (_type) with
    comprehensive ML framework support and robust error handling.
    """

    # NEW TYPE METADATA FORMAT (priority 1)
    if TYPE_METADATA_KEY in obj and VALUE_METADATA_KEY in obj:
        type_name = obj[TYPE_METADATA_KEY]
        value = obj[VALUE_METADATA_KEY]

        try:
            # Basic types
            if type_name == "datetime":
                return datetime.fromisoformat(value)
            elif type_name == "uuid.UUID":
                return uuid.UUID(value)

            # Complex number reconstruction
            elif type_name == "complex":
                if isinstance(value, dict) and "real" in value and "imag" in value:
                    return complex(value["real"], value["imag"])
                return complex(value)

            # Decimal reconstruction
            elif type_name == "decimal.Decimal":
                return Decimal(str(value))

            # Path reconstruction
            elif type_name == "pathlib.Path":
                from pathlib import Path

                return Path(value)

            # Set and tuple reconstruction
            elif type_name == "set":
                return set(value)
            elif type_name == "tuple":
                return tuple(value)

            # Pandas types
            elif type_name == "pandas.DataFrame":
                if pd is not None:
                    # Enhanced DataFrame reconstruction - handle different orientations
                    if isinstance(value, list):
                        # Records format (most common) or VALUES format (list of lists)
                        if value and isinstance(value[0], list):
                            # VALUES format: list of lists without column/index info
                            # This loses column names, but that's expected for VALUES orientation
                            return pd.DataFrame(value)
                        else:
                            # Records format: list of dicts
                            return pd.DataFrame(value)
                    elif isinstance(value, dict):
                        # Dict format, split format, or index format
                        if "index" in value and "columns" in value and "data" in value:
                            # Split format
                            return pd.DataFrame(data=value["data"], index=value["index"], columns=value["columns"])
                        else:
                            # Index format: {0: {'a': 1, 'b': 3}, 1: {'a': 2, 'b': 4}}
                            # Need to use orient='index' to properly reconstruct
                            return pd.DataFrame.from_dict(value, orient="index")
                    return pd.DataFrame(value)  # Fallback

            elif type_name == "pandas.Series":
                if pd is not None:
                    # Enhanced Series reconstruction with name preservation and categorical support
                    if isinstance(value, dict) and "_series_name" in value:
                        series_name = value["_series_name"]
                        series_data = {
                            k: v
                            for k, v in value.items()
                            if k not in ("_series_name", "_dtype", "_categories", "_ordered")
                        }
                        # CRITICAL FIX: Handle JSON string keys that should be integers
                        series_data = _convert_string_keys_to_int_if_possible(series_data)

                        # Handle categorical dtype reconstruction
                        if value.get("_dtype") == "category":
                            categories = value.get("_categories", [])
                            ordered = value.get("_ordered", False)
                            # Create Series with categorical dtype
                            series = pd.Series(
                                list(series_data.values()), index=list(series_data.keys()), name=series_name
                            )
                            series = series.astype(pd.CategoricalDtype(categories=categories, ordered=ordered))
                            return series
                        else:
                            return pd.Series(series_data, name=series_name)
                    elif isinstance(value, dict):
                        # Handle categorical dtype reconstruction for unnamed series
                        if value.get("_dtype") == "category":
                            categories = value.get("_categories", [])
                            ordered = value.get("_ordered", False)
                            series_data = {
                                k: v for k, v in value.items() if k not in ("_dtype", "_categories", "_ordered")
                            }
                            # CRITICAL FIX: Handle JSON string keys that should be integers
                            series_data = _convert_string_keys_to_int_if_possible(series_data)
                            # Create Series with categorical dtype
                            series = pd.Series(list(series_data.values()), index=list(series_data.keys()))
                            series = series.astype(pd.CategoricalDtype(categories=categories, ordered=ordered))
                            return series
                        else:
                            # CRITICAL FIX: Handle JSON string keys that should be integers
                            series_data = _convert_string_keys_to_int_if_possible(value)
                            return pd.Series(series_data)
                    elif isinstance(value, list):
                        return pd.Series(value)
                    return pd.Series(value)  # Fallback

            # NumPy types
            elif type_name == "numpy.ndarray":
                if np is not None:
                    array_data = value.get("data", value) if isinstance(value, dict) else value
                    dtype = value.get("dtype") if isinstance(value, dict) else None
                    shape = value.get("shape") if isinstance(value, dict) else None

                    # CRITICAL FIX: Recursively deserialize array elements to handle complex numbers and other types
                    if isinstance(array_data, list):
                        deserialized_data = []
                        for item in array_data:
                            # Recursively deserialize each element to handle complex numbers, etc.
                            if isinstance(item, dict) and "_type" in item:
                                deserialized_item = _deserialize_with_type_metadata(item)
                            else:
                                deserialized_item = item
                            deserialized_data.append(deserialized_item)
                        array_data = deserialized_data

                    result = np.array(array_data)
                    if dtype:
                        result = result.astype(dtype)
                    if shape:
                        result = result.reshape(shape)
                    return result

            # Enhanced NumPy scalar types
            elif type_name.startswith("numpy."):
                if np is not None:
                    # Handle numpy scalar types (int32, float64, bool_, etc.)
                    if type_name == "numpy.int32":
                        return np.int32(value)
                    elif type_name == "numpy.int64":
                        return np.int64(value)
                    elif type_name == "numpy.float32":
                        return np.float32(value)
                    elif type_name == "numpy.float64":
                        return np.float64(value)
                    elif type_name == "numpy.bool_":
                        # Handle the deprecation warning for np.bool
                        return np.bool_(value)
                    elif type_name == "numpy.complex64":
                        return np.complex64(value)
                    elif type_name == "numpy.complex128":
                        return np.complex128(value)
                    # Generic fallback for other numpy types
                    try:
                        numpy_type_name = type_name.split(".", 1)[1]
                        # Handle special case for bool_ deprecation
                        if numpy_type_name == "bool":
                            numpy_type_name = "bool_"
                        numpy_type = getattr(np, numpy_type_name)
                        return numpy_type(value)
                    except (AttributeError, ValueError, TypeError):
                        pass

            # ML Types - PyTorch
            elif type_name.startswith("torch."):
                try:
                    import torch

                    if type_name == "torch.Tensor":
                        # CRITICAL FIX: Handle both new format (data, dtype) and legacy format (_data, _dtype)
                        tensor_data = value.get("_data", value.get("data", value)) if isinstance(value, dict) else value
                        dtype = value.get("_dtype", value.get("dtype")) if isinstance(value, dict) else None
                        device = value.get("_device", value.get("device", "cpu")) if isinstance(value, dict) else "cpu"
                        requires_grad = (
                            value.get("_requires_grad", value.get("requires_grad", False))
                            if isinstance(value, dict)
                            else False
                        )
                        shape = value.get("_shape", value.get("shape")) if isinstance(value, dict) else None

                        # Create tensor with proper attributes
                        result = torch.tensor(tensor_data, device=device, requires_grad=requires_grad)
                        if dtype and hasattr(torch, dtype.replace("torch.", "")):
                            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
                            result = result.to(torch_dtype)

                        # Reshape if needed (for tensors that were reshaped during serialization)
                        if shape and result.numel() > 0:
                            try:
                                result = result.reshape(shape)
                            except RuntimeError:
                                # If reshape fails, keep original shape
                                pass

                        return result
                except ImportError:
                    warnings.warn("PyTorch not available for tensor reconstruction", stacklevel=2)

            # ML Types - Scikit-learn
            elif type_name.startswith("sklearn.") or type_name.startswith("scikit_learn."):
                try:
                    # Handle new type metadata format for sklearn models
                    if isinstance(value, dict) and "_class" in value and "_params" in value:
                        # This is the new format: reconstruct the sklearn model from class and params
                        class_name = value["_class"]
                        params = value["_params"]

                        # Import the sklearn class dynamically
                        module_path, class_name_only = class_name.rsplit(".", 1)
                        try:
                            import importlib

                            module = importlib.import_module(module_path)
                            model_class = getattr(module, class_name_only)

                            # Create the model with the saved parameters
                            model = model_class(**params)

                            # Note: We can't restore fitted state without the actual fitted data
                            # This is a limitation of the current serialization format
                            return model
                        except (ImportError, AttributeError) as e:
                            warnings.warn(f"Could not import sklearn class {class_name}: {e}", stacklevel=2)
                            # Fall back to returning the dict
                            return value

                    # Handle legacy pickle format
                    elif isinstance(value, str):
                        # Assume base64 encoded pickle
                        import base64
                        import pickle  # nosec B403

                        pickle_data = base64.b64decode(value)
                        return pickle.loads(pickle_data)  # nosec B301
                    elif isinstance(value, dict) and "_pickle_data" in value:
                        # Alternative pickle storage format
                        import base64
                        import pickle  # nosec B403

                        pickle_data = base64.b64decode(value["_pickle_data"])
                        return pickle.loads(pickle_data)  # nosec B301
                except (ImportError, Exception) as e:
                    warnings.warn(f"Could not reconstruct sklearn model: {e}", stacklevel=2)

        except Exception as e:
            warnings.warn(f"Failed to reconstruct type {type_name}: {e}", stacklevel=2)

        # Fallback to the original value
        return value

    # ENHANCED LEGACY TYPE FORMATS (priority 2) - Handle older serialization formats
    if isinstance(obj, dict) and "_type" in obj:
        type_name = obj["_type"]

        try:
            # Enhanced Decimal legacy format with precision preservation
            if type_name == "decimal":
                if "value" in obj:
                    return Decimal(obj["value"])
                elif "precision" in obj and "scale" in obj:
                    # Enhanced precision format
                    return Decimal(obj["value"])
                return Decimal(str(obj.get("value", "0")))

            # Enhanced Complex number legacy format
            elif type_name == "complex":
                if "real" in obj and "imag" in obj:
                    return complex(obj["real"], obj["imag"])
                return complex(obj.get("value", 0))

            # Path legacy format
            elif type_name == "path":
                from pathlib import Path

                return Path(obj.get("value", ""))

            # Enhanced PyTorch tensor legacy format
            elif type_name == "torch.Tensor":
                try:
                    import torch

                    # Extract tensor data and metadata
                    data = obj.get("_data", obj.get("data", []))
                    shape = obj.get("_shape", obj.get("shape"))
                    dtype_str = obj.get("_dtype", obj.get("dtype", "torch.float32"))
                    device = obj.get("_device", obj.get("device", "cpu"))
                    requires_grad = obj.get("_requires_grad", obj.get("requires_grad", False))

                    # Create tensor
                    tensor = torch.tensor(data, device=device, requires_grad=requires_grad)

                    # Set dtype if specified
                    if dtype_str and hasattr(torch, dtype_str.replace("torch.", "")):
                        torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
                        tensor = tensor.to(torch_dtype)

                    # Reshape if needed
                    if shape and tensor.numel() == sum(shape):
                        tensor = tensor.reshape(shape)

                    return tensor
                except ImportError:
                    warnings.warn("PyTorch not available for tensor reconstruction", stacklevel=2)
                except Exception as e:
                    warnings.warn(f"Failed to reconstruct PyTorch tensor: {e}", stacklevel=2)

            # Enhanced NumPy array legacy format
            elif type_name == "numpy.ndarray" or type_name.startswith("numpy."):
                if np is not None:
                    try:
                        data = obj.get("data", obj.get("_data", []))
                        dtype = obj.get("dtype", obj.get("_dtype"))
                        shape = obj.get("shape", obj.get("_shape"))

                        result = np.array(data)
                        if dtype:
                            result = result.astype(dtype)
                        if shape:
                            result = result.reshape(shape)
                        return result
                    except Exception as e:
                        warnings.warn(f"Failed to reconstruct NumPy array: {e}", stacklevel=2)

            # Enhanced pandas DataFrame legacy format
            elif type_name == "pandas.DataFrame":
                if pd is not None:
                    try:
                        data = obj.get("data", obj.get("_data"))
                        if isinstance(data, list):
                            return pd.DataFrame(data)
                        elif isinstance(data, dict):
                            return pd.DataFrame.from_dict(data)
                    except Exception as e:
                        warnings.warn(f"Failed to reconstruct DataFrame: {e}", stacklevel=2)

            # Enhanced pandas Series legacy format
            elif type_name == "pandas.Series":
                if pd is not None:
                    try:
                        data = obj.get("data", obj.get("_data"))
                        name = obj.get("name", obj.get("_name"))
                        return pd.Series(data, name=name)
                    except Exception as e:
                        warnings.warn(f"Failed to reconstruct Series: {e}", stacklevel=2)

            # ML Models - Scikit-learn legacy format
            elif type_name.startswith("sklearn.") or type_name.startswith("scikit_learn."):
                try:
                    import base64
                    import pickle  # nosec B403

                    # Try different pickle storage formats
                    pickle_data = None
                    if "_pickle_data" in obj:
                        pickle_data = base64.b64decode(obj["_pickle_data"])
                    elif "pickle_data" in obj:
                        pickle_data = base64.b64decode(obj["pickle_data"])
                    elif "data" in obj and isinstance(obj["data"], str):
                        pickle_data = base64.b64decode(obj["data"])

                    if pickle_data:
                        return pickle.loads(pickle_data)  # nosec B301
                except (ImportError, Exception) as e:
                    warnings.warn(f"Could not reconstruct sklearn model: {e}", stacklevel=2)

        except Exception as e:
            warnings.warn(f"Failed to reconstruct legacy type {type_name}: {e}", stacklevel=2)

    # Not a type metadata object
    return obj


def _auto_detect_string_type(s: str, aggressive: bool = False) -> Any:
    """NEW: Auto-detect the most likely type for a string value."""
    # Always try UUID detection first (more specific pattern)
    if _looks_like_uuid(s):
        try:
            import uuid as uuid_module  # Fresh import to avoid state issues

            return uuid_module.UUID(s)
        except (ValueError, ImportError):
            pass

    # Then try datetime detection
    if _looks_like_datetime(s):
        try:
            from datetime import datetime as datetime_class  # Fresh import

            return datetime_class.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, ImportError):
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
    if not isinstance(obj, dict) or len(obj) < 2:  # FIXED: Require at least 2 columns
        return False

    # Check if all values are lists of the same length
    values = list(obj.values())
    if not all(isinstance(v, list) for v in values):
        return False

    if len({len(v) for v in values}) != 1:  # All lists same length
        return False

    # ENHANCED: Additional checks to avoid false positives on basic nested data

    # Must have at least a few rows to be worth converting
    if len(values[0]) < 2:
        return False

    # ENHANCED: Check if keys look like column names (not nested dict keys)
    # Avoid converting basic nested structures like {'nested': {'key': [1,2,3]}}
    keys = list(obj.keys())

    # FIXED: Be more lenient with single-character column names
    # Single character column names are common in DataFrames (a, b, c, x, y, z, etc.)
    # Only reject if ALL keys are single character AND we have very few columns
    # AND the data is very simple (all integers in a single column)
    if len(keys) == 1:
        # Special case: single column with simple integer data - probably not a DataFrame
        single_key = keys[0]
        single_value = values[0]
        if len(single_key) == 1 and all(isinstance(item, int) for item in single_value) and len(single_value) <= 5:
            # This looks like {'e': [1, 2, 3]} - probably basic nested data
            return False

    # If we have multiple columns, it's probably a legitimate DataFrame
    # Even with single-character names like {'a': [1,2,3], 'b': [4,5,6]}
    return True


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

    # First check if it looks like a UUID - if so, it's not a datetime
    if _looks_like_uuid(s):
        return False

    # Check for ISO format patterns
    patterns = [
        # Basic ISO patterns - must have time component
        s.count("-") >= 2 and ("T" in s or " " in s),
        # Common datetime patterns - must have time (colons)
        s.count(":") >= 1 and s.count("-") >= 2,
        # Z or timezone offset - but only if it also has time indicators
        (s.endswith("Z") or s.count("+") == 1) and (":" in s or "T" in s),
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
    """Parse a UUID string into a UUID object.

    Args:
        s: String that might be a UUID

    Returns:
        UUID object if parsing succeeds, None otherwise

    Examples:
        >>> parse_uuid_string("12345678-1234-5678-9012-123456789abc")
        UUID('12345678-1234-5678-9012-123456789abc')
        >>> parse_uuid_string("not a uuid")
        None
    """
    if not isinstance(s, str):
        return None

    try:
        return uuid.UUID(s)
    except ValueError:
        return None


# NEW: v0.4.5 Template-Based Deserialization & Enhanced Type Fidelity


class TemplateDeserializer:
    """Template-based deserializer for enhanced type fidelity and round-trip scenarios.

    This class allows users to provide a template object that guides the deserialization
    process, ensuring that the output matches the expected structure and types.
    """

    def __init__(self, template: Any, strict: bool = True, fallback_auto_detect: bool = True):
        """Initialize template deserializer.

        Args:
            template: Template object to guide deserialization
            strict: If True, raise errors when structure doesn't match
            fallback_auto_detect: If True, use auto-detection when template doesn't match
        """
        self.template = template
        self.strict = strict
        self.fallback_auto_detect = fallback_auto_detect
        self._template_info = self._analyze_template()

    def _analyze_template(self) -> Dict[str, Any]:
        """Analyze the template to understand expected structure and types."""
        info = {"type": type(self.template).__name__, "structure": {}, "expected_types": {}}

        if isinstance(self.template, dict):
            info["structure"] = "dict"
            for key, value in self.template.items():
                info["expected_types"][key] = type(value).__name__

        elif isinstance(self.template, (list, tuple)):
            info["structure"] = "sequence"
            if self.template:
                # Analyze first element as template for all items
                info["item_template"] = type(self.template[0]).__name__

        elif pd is not None and isinstance(self.template, pd.DataFrame):
            info["structure"] = "dataframe"
            info["columns"] = list(self.template.columns)
            info["dtypes"] = {col: str(dtype) for col, dtype in self.template.dtypes.items()}
            info["index_type"] = type(self.template.index).__name__

        elif pd is not None and isinstance(self.template, pd.Series):
            info["structure"] = "series"
            info["dtype"] = str(self.template.dtype)
            info["name"] = self.template.name
            info["index_type"] = type(self.template.index).__name__

        return info

    def deserialize(self, obj: Any) -> Any:
        """Deserialize object using template guidance.

        Args:
            obj: Serialized object to deserialize

        Returns:
            Deserialized object matching template structure
        """
        try:
            return self._deserialize_with_template(obj, self.template)
        except Exception as e:
            if self.strict:
                raise TemplateDeserializationError(
                    f"Failed to deserialize with template {type(self.template).__name__}: {e}"
                ) from e
            elif self.fallback_auto_detect:
                warnings.warn(f"Template deserialization failed, falling back to auto-detection: {e}", stacklevel=2)
                return auto_deserialize(obj, aggressive=True)
            else:
                return obj

    def _deserialize_with_template(self, obj: Any, template: Any) -> Any:
        """Core template-based deserialization logic."""
        # Handle None cases
        if obj is None:
            return None

        # Handle type metadata (highest priority)
        if isinstance(obj, dict) and TYPE_METADATA_KEY in obj:
            return _deserialize_with_type_metadata(obj)

        # Template-guided deserialization based on template type
        if isinstance(template, dict) and isinstance(obj, dict):
            return self._deserialize_dict_with_template(obj, template)

        elif isinstance(template, (list, tuple)) and isinstance(obj, list):
            return self._deserialize_list_with_template(obj, template)

        elif pd is not None and isinstance(template, pd.DataFrame):
            return self._deserialize_dataframe_with_template(obj, template)

        elif pd is not None and isinstance(template, pd.Series):
            return self._deserialize_series_with_template(obj, template)

        elif isinstance(template, datetime) and isinstance(obj, str):
            return self._deserialize_datetime_with_template(obj, template)

        elif isinstance(template, uuid.UUID) and isinstance(obj, str):
            return self._deserialize_uuid_with_template(obj, template)

        else:
            # For basic types or unsupported combinations, apply type coercion
            return self._coerce_to_template_type(obj, template)

    def _deserialize_dict_with_template(self, obj: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize dictionary using template."""
        result = {}

        for key, value in obj.items():
            if key in template:
                # Use template value as guide for this key
                result[key] = self._deserialize_with_template(value, template[key])
            else:
                # Key not in template - use auto-detection or pass through
                if self.fallback_auto_detect:
                    result[key] = auto_deserialize(value, aggressive=True)
                else:
                    result[key] = value

        return result

    def _deserialize_list_with_template(self, obj: List[Any], template: List[Any]) -> List[Any]:
        """Deserialize list using template."""
        if not template:
            # Empty template, use auto-detection
            return [auto_deserialize(item, aggressive=True) for item in obj]

        # Use first item in template as guide for all items
        item_template = template[0]
        return [self._deserialize_with_template(item, item_template) for item in obj]

    def _deserialize_dataframe_with_template(self, obj: Any, template: "pd.DataFrame") -> "pd.DataFrame":
        """Deserialize DataFrame using template structure and dtypes."""
        if pd is None:
            raise ImportError("pandas is required for DataFrame template deserialization")

        # Handle different serialization formats
        if isinstance(obj, list):
            # Records format
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            if "data" in obj and "columns" in obj:
                # Split format
                df = pd.DataFrame(data=obj["data"], columns=obj["columns"])
                if "index" in obj:
                    df.index = obj["index"]
            else:
                # Dict format
                df = pd.DataFrame(obj)
        else:
            raise ValueError(f"Cannot deserialize {type(obj)} to DataFrame")

        # Apply template column types
        for col in template.columns:
            if col in df.columns:
                try:
                    target_dtype = template[col].dtype
                    df[col] = df[col].astype(target_dtype)
                except Exception:
                    # Type conversion failed, keep original
                    warnings.warn(f"Failed to convert column '{col}' to template dtype {target_dtype}", stacklevel=3)

        # Ensure column order matches template
        df = df.reindex(columns=template.columns, fill_value=None)

        return df

    def _deserialize_series_with_template(self, obj: Any, template: "pd.Series") -> "pd.Series":
        """Deserialize Series using template."""
        if pd is None:
            raise ImportError("pandas is required for Series template deserialization")

        if isinstance(obj, dict):
            # Handle Series with metadata
            if "_series_name" in obj:
                name = obj["_series_name"]
                data_dict = {k: v for k, v in obj.items() if k != "_series_name"}
                series = pd.Series(data_dict, name=name)
            elif isinstance(obj, (dict, list)):
                series = pd.Series(obj)
            else:
                series = pd.Series([obj])
        elif isinstance(obj, list):
            series = pd.Series(obj)
        else:
            series = pd.Series([obj])

        # Apply template dtype
        try:
            series = series.astype(template.dtype)
        except Exception:
            warnings.warn(f"Failed to convert Series to template dtype {template.dtype}", stacklevel=3)

        # Set name from template if not already set
        if series.name is None and template.name is not None:
            series.name = template.name

        return series

    def _deserialize_datetime_with_template(self, obj: str, template: datetime) -> datetime:
        """Deserialize datetime string using template."""
        try:
            return datetime.fromisoformat(obj.replace("Z", "+00:00"))
        except ValueError:
            # Try other common formats with dateutil if available
            try:
                import dateutil.parser  # type: ignore  # noqa: F401

                return dateutil.parser.parse(obj)  # type: ignore
            except ImportError:
                # dateutil not available, return as string
                warnings.warn(f"Failed to parse datetime '{obj}' and dateutil not available", stacklevel=3)
                return obj  # Return as string if can't parse

    def _deserialize_uuid_with_template(self, obj: str, template: uuid.UUID) -> uuid.UUID:
        """Deserialize UUID string using template."""
        return uuid.UUID(obj)

    def _coerce_to_template_type(self, obj: Any, template: Any) -> Any:
        """Coerce object to match template type."""
        template_type = type(template)

        if isinstance(obj, template_type):
            return obj

        # Try type coercion
        try:
            if template_type in (int, float, str, bool):
                return template_type(obj)
            elif template_type is Decimal:
                # Handle Decimal conversion explicitly
                if isinstance(obj, str):
                    return Decimal(obj)
                elif isinstance(obj, (int, float)):
                    return Decimal(str(obj))
                else:
                    return Decimal(str(obj))
            else:
                return obj  # Cannot coerce, return as-is
        except (ValueError, TypeError, decimal.InvalidOperation):
            # If coercion fails, return the original object
            return obj


class TemplateDeserializationError(Exception):
    """Raised when template-based deserialization fails."""

    pass


def deserialize_with_template(obj: Any, template: Any, **kwargs: Any) -> Any:
    """Convenience function for template-based deserialization.

    Args:
        obj: Serialized object to deserialize
        template: Template object to guide deserialization
        **kwargs: Additional arguments for TemplateDeserializer

    Returns:
        Deserialized object matching template structure

    Examples:
        >>> import pandas as pd
        >>> template_df = pd.DataFrame({'a': [1], 'b': ['text']})
        >>> serialized_data = [{'a': 2, 'b': 'hello'}, {'a': 3, 'b': 'world'}]
        >>> result = deserialize_with_template(serialized_data, template_df)
        >>> isinstance(result, pd.DataFrame)
        True
        >>> result.dtypes['a']  # Should match template
        int64
    """
    deserializer = TemplateDeserializer(template, **kwargs)
    return deserializer.deserialize(obj)


def infer_template_from_data(data: Any, max_samples: int = 100) -> Any:
    """Infer a template from sample data.

    This function analyzes sample data to create a template that can be used
    for subsequent template-based deserialization.

    Args:
        data: Sample data to analyze (list of records, DataFrame, etc.)
        max_samples: Maximum number of samples to analyze

    Returns:
        Inferred template object

    Examples:
        >>> sample_data = [
        ...     {'name': 'Alice', 'age': 30, 'date': '2023-01-01T10:00:00'},
        ...     {'name': 'Bob', 'age': 25, 'date': '2023-01-02T11:00:00'}
        ... ]
        >>> template = infer_template_from_data(sample_data)
        >>> # template will be a dict with expected types
    """
    if isinstance(data, list) and data:
        # Analyze list of records
        return _infer_template_from_records(data[:max_samples])
    elif pd is not None and isinstance(data, pd.DataFrame):
        # Use DataFrame structure directly as template
        return data.iloc[: min(1, len(data))].copy()
    elif pd is not None and isinstance(data, pd.Series):
        # Use Series structure directly as template
        return data.iloc[: min(1, len(data))].copy()
    elif isinstance(data, dict):
        # Use single dict as template
        return data
    else:
        # Cannot infer meaningful template
        return data


def _infer_template_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer template from list of record dictionaries."""
    if not records:
        return {}

    # Analyze types from all records
    type_counts = {}
    all_keys = set()

    for record in records:
        if isinstance(record, dict):
            all_keys.update(record.keys())
            for key, value in record.items():
                if key not in type_counts:
                    type_counts[key] = {}

                value_type = type(value).__name__
                type_counts[key][value_type] = type_counts[key].get(value_type, 0) + 1

    # Create template with most common type for each key
    template = {}
    for key in all_keys:
        if key in type_counts:
            # Find most common type
            most_common_type = max(type_counts[key].items(), key=lambda x: x[1])[0]

            # Create example value of that type
            if most_common_type == "str":
                template[key] = ""
            elif most_common_type == "int":
                template[key] = 0
            elif most_common_type == "float":
                template[key] = 0.0
            elif most_common_type == "bool":
                template[key] = False
            elif most_common_type == "list":
                template[key] = []
            elif most_common_type == "dict":
                template[key] = {}
            else:
                # Find actual example from records
                for record in records:
                    if isinstance(record, dict) and key in record and type(record[key]).__name__ == most_common_type:
                        template[key] = record[key]
                        break

    return template


def create_ml_round_trip_template(ml_object: Any) -> Dict[str, Any]:
    """Create a template optimized for ML object round-trip serialization.

    This function creates templates specifically designed for machine learning
    workflows where perfect round-trip fidelity is crucial.

    Args:
        ml_object: ML object (model, dataset, etc.) to create template for

    Returns:
        Template dictionary with ML-specific metadata

    Examples:
        >>> import sklearn.linear_model
        >>> model = sklearn.linear_model.LogisticRegression()
        >>> template = create_ml_round_trip_template(model)
        >>> # template will include model structure, parameters, etc.
    """
    template = {
        "__ml_template__": True,
        "object_type": type(ml_object).__name__,
        "module": getattr(ml_object, "__module__", None),
    }

    # Handle pandas objects
    if pd is not None and isinstance(ml_object, pd.DataFrame):
        template.update(
            {
                "structure_type": "dataframe",
                "columns": list(ml_object.columns),
                "dtypes": {col: str(dtype) for col, dtype in ml_object.dtypes.items()},
                "index_name": ml_object.index.name,
                "shape": ml_object.shape,
            }
        )
    elif pd is not None and isinstance(ml_object, pd.Series):
        template.update(
            {
                "structure_type": "series",
                "dtype": str(ml_object.dtype),
                "name": ml_object.name,
                "index_name": ml_object.index.name,
                "length": len(ml_object),
            }
        )

    # Handle numpy arrays
    elif np is not None and isinstance(ml_object, np.ndarray):
        template.update(
            {
                "structure_type": "numpy_array",
                "shape": ml_object.shape,
                "dtype": str(ml_object.dtype),
                "fortran_order": np.isfortran(ml_object),
            }
        )

    # Handle sklearn models
    elif hasattr(ml_object, "get_params"):
        try:
            template.update(
                {
                    "structure_type": "sklearn_model",
                    "parameters": ml_object.get_params(),
                    "fitted": hasattr(ml_object, "classes_") or hasattr(ml_object, "coef_"),
                }
            )
        except Exception:
            pass  # nosec B110

    return template


def deserialize_fast(
    obj: Any, config: Optional["SerializationConfig"] = None, _depth: int = 0, _seen: Optional[Set[int]] = None
) -> Any:
    """High-performance deserialize with ultra-fast basic type handling.

    ULTRA-SIMPLIFIED ARCHITECTURE for maximum basic type performance:
    1. IMMEDIATE basic type handling (zero overhead)
    2. Security checks (only for containers)
    3. Optimized paths (only when needed)

    Args:
        obj: The JSON-compatible object to deserialize
        config: Optional configuration (uses same config as serialization)
        _depth: Current recursion depth (for internal use)
        _seen: Set of object IDs already seen (for internal use)

    Returns:
        Python object with restored types where possible

    Raises:
        DeserializationSecurityError: If security limits are exceeded
    """
    # ==================================================================================
    # PHASE 0: ULTRA-AGGRESSIVE BASIC TYPE FAST PATH (ZERO OVERHEAD)
    # ==================================================================================

    # ULTRA-FAST: Handle the 90% case with minimal type checking
    obj_type = type(obj)

    # Most basic types - return immediately with zero processing
    if obj_type in (_TYPE_INT, _TYPE_BOOL, _TYPE_NONE, _TYPE_FLOAT):
        return obj

    # Short strings - return immediately (covers 95% of string cases)
    if obj_type is _TYPE_STR and len(obj) < 8:
        return obj

    # ==================================================================================
    # PHASE 1: SECURITY CHECKS (ONLY FOR CONTAINERS AND COMPLEX TYPES)
    # ==================================================================================

    # SECURITY CHECK 1: Depth limit enforcement (apply to ALL objects at depth)
    max_depth = config.max_depth if config else MAX_SERIALIZATION_DEPTH
    if _depth > max_depth:
        raise DeserializationSecurityError(
            f"Maximum deserialization depth ({max_depth}) exceeded. Current depth: {_depth}."
        )

    # Additional container-specific security checks
    if isinstance(obj, (dict, list)):
        # SECURITY CHECK 2: Initialize circular reference tracking
        if _seen is None:
            _seen = set()

        # SECURITY CHECK 3: Size limits for containers
        if isinstance(obj, dict) and len(obj) > (config.max_size if config else MAX_OBJECT_SIZE):
            raise DeserializationSecurityError(f"Dictionary size ({len(obj)}) exceeds maximum allowed size.")
        elif isinstance(obj, list) and len(obj) > (config.max_size if config else MAX_OBJECT_SIZE):
            raise DeserializationSecurityError(f"List size ({len(obj)}) exceeds maximum allowed size.")

    # ==================================================================================
    # PHASE 2: OPTIMIZED PROCESSING FOR REMAINING TYPES
    # ==================================================================================

    # Handle remaining string types with optimization
    if obj_type is _TYPE_STR:
        return _deserialize_string_full(obj, config)

    # Handle type metadata (highest priority for complex objects)
    if isinstance(obj, dict) and TYPE_METADATA_KEY in obj:
        return _deserialize_with_type_metadata(obj)

    # Handle containers with optimized processing
    if isinstance(obj, list):
        if _seen is None:
            _seen = set()
        return _process_list_optimized(obj, config, _depth, _seen)

    if isinstance(obj, dict):
        if _seen is None:
            _seen = set()
        return _process_dict_optimized(obj, config, _depth, _seen)

    # Return unknown types as-is
    return obj


def _process_list_optimized(obj: list, config: Optional["SerializationConfig"], _depth: int, _seen: Set[int]) -> list:
    """Optimized list processing with circular reference protection."""
    # SECURITY: Check for circular references
    obj_id = id(obj)
    if obj_id in _seen:
        warnings.warn(f"Circular reference detected in list at depth {_depth}. Breaking cycle.", stacklevel=4)
        return []

    _seen.add(obj_id)
    try:
        # OPTIMIZATION: Use pooled list for memory efficiency
        result = _get_pooled_list()
        try:
            for item in obj:
                deserialized_item = deserialize_fast(item, config, _depth + 1, _seen)
                result.append(deserialized_item)

            # Create final result and return list to pool
            final_result = list(result)
            return final_result
        finally:
            _return_list_to_pool(result)
    finally:
        _seen.discard(obj_id)


def _process_dict_optimized(obj: dict, config: Optional["SerializationConfig"], _depth: int, _seen: Set[int]) -> dict:
    """Optimized dict processing with circular reference protection."""
    # SECURITY: Check for circular references
    obj_id = id(obj)
    if obj_id in _seen:
        warnings.warn(f"Circular reference detected in dict at depth {_depth}. Breaking cycle.", stacklevel=4)
        return {}

    _seen.add(obj_id)
    try:
        # ENHANCED: Check for type metadata first (both new and legacy formats)
        if TYPE_METADATA_KEY in obj or "_type" in obj:
            return _deserialize_with_type_metadata(obj)

        # Auto-detect complex numbers (even without metadata)
        if len(obj) == 2 and "real" in obj and "imag" in obj:
            try:
                return complex(obj["real"], obj["imag"])
            except (TypeError, ValueError):
                pass  # Fall through to normal processing

        # Auto-detect Decimal from string representation (common pattern)
        if len(obj) == 1 and "value" in obj and isinstance(obj["value"], str):
            try:
                # Only try to convert to Decimal if the string looks numeric
                if _looks_like_number(obj["value"]):
                    return Decimal(obj["value"])
            except (TypeError, ValueError, ImportError, decimal.InvalidOperation):
                pass  # Fall through to normal processing

        # Check for special formats
        if _looks_like_split_format(obj):
            return _reconstruct_from_split(obj)
        if _looks_like_dataframe_dict(obj):
            return _reconstruct_dataframe(obj)

        # OPTIMIZATION: Use pooled dict for memory efficiency
        result = _get_pooled_dict()
        try:
            for k, v in obj.items():
                deserialized_value = deserialize_fast(v, config, _depth + 1, _seen)
                result[k] = deserialized_value

            # Create final result and return dict to pool
            final_result = dict(result)
            return final_result
        finally:
            _return_dict_to_pool(result)
    finally:
        _seen.discard(obj_id)


def _deserialize_string_full(s: str, config: Optional["SerializationConfig"]) -> Any:
    """Full string processing with all type detection and aggressive caching."""

    # OPTIMIZATION: Use cached pattern detection first
    pattern_type = _get_cached_string_pattern(s)

    if pattern_type == "plain":
        return s  # Already determined to be plain string

    # For typed patterns, try cached parsed objects first
    if pattern_type in ("uuid", "datetime", "path"):
        cached_result = _get_cached_parsed_object(s, pattern_type)
        if cached_result is not None:
            return cached_result
        elif cached_result is None and f"{pattern_type}:{s}" in _PARSED_OBJECT_CACHE:
            # Cached failure - return as string without retrying
            return s

    # OPTIMIZATION: For uncached or unknown patterns, use optimized detection
    # This path handles cache misses and new patterns

    # Try datetime parsing with optimized detection
    if pattern_type == "datetime" or (pattern_type is None and _looks_like_datetime_optimized(s)):
        try:
            parsed_datetime = datetime.fromisoformat(s.replace("Z", "+00:00"))
            # Cache successful parse
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"datetime:{s}"] = parsed_datetime
            return parsed_datetime
        except ValueError:
            # Cache failure to avoid repeated parsing
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"datetime:{s}"] = None

    # Try UUID parsing with optimized detection
    if pattern_type == "uuid" or (pattern_type is None and _looks_like_uuid_optimized(s)):
        try:
            import uuid as uuid_module  # Fresh import to avoid state issues

            parsed_uuid = uuid_module.UUID(s)
            # Cache successful parse
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"uuid:{s}"] = parsed_uuid
            return parsed_uuid
        except (ValueError, ImportError):
            # Cache failure to avoid repeated parsing
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"uuid:{s}"] = None

    # Try Path detection for common path patterns (always enabled for better round-trips)
    if pattern_type == "path" or (pattern_type is None and _looks_like_path_optimized(s)):
        try:
            from pathlib import Path

            parsed_path = Path(s)
            # Cache successful parse
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"path:{s}"] = parsed_path
            return parsed_path
        except Exception:
            # Cache failure to avoid repeated parsing
            if len(_PARSED_OBJECT_CACHE) < _PARSED_CACHE_SIZE_LIMIT:
                _PARSED_OBJECT_CACHE[f"path:{s}"] = None

    # Return as string if no parsing succeeded
    return s


def _looks_like_datetime_optimized(s: str) -> bool:
    """Optimized datetime detection using character set validation."""
    if len(s) < 10:
        return False

    # Ultra-fast check: YYYY-MM-DD pattern
    return s[4] == "-" and s[7] == "-" and s[:4].isdigit() and s[5:7].isdigit() and s[8:10].isdigit()


def _looks_like_uuid_optimized(s: str) -> bool:
    """Optimized UUID detection using character set validation."""
    if len(s) != 36:
        return False

    # Ultra-fast check: dash positions first (cheapest check)
    if not (s[8] == "-" and s[13] == "-" and s[18] == "-" and s[23] == "-"):
        return False

    # More thorough check: validate all hex segments
    segments = [s[:8], s[9:13], s[14:18], s[19:23], s[24:]]
    return all(all(c in "0123456789abcdefABCDEF" for c in segment) for segment in segments)


def _looks_like_path_optimized(s: str) -> bool:
    """Optimized path detection using quick pattern matching."""
    if not s or len(s) < 2:
        return False

    # Ultra-fast checks for most common patterns
    return (
        s[0] == "/"  # Unix absolute path
        or (len(s) >= 3 and s[1:3] == ":\\")  # Windows drive letter
        or s.startswith("./")  # Relative path
        or s.startswith("../")  # Parent directory
        or "/tmp/" in s  # Common temp directory  # nosec B108
        or (
            s.endswith((".txt", ".py", ".json", ".csv", ".log")) and ("/" in s or s.count(".") >= 1)
        )  # File with extension
        or (s.count("/") >= 1 and not s.startswith("http"))  # Has path separator but not URL
    )


def _looks_like_path(s: str) -> bool:
    """Check if a string looks like a file path (original function kept for compatibility)."""
    if not s or len(s) < 3:
        return False

    # Common path indicators
    path_indicators = [
        s.startswith("/"),  # Unix absolute path
        s.startswith("~/"),  # Unix home directory
        s.startswith("./"),  # Unix relative path
        s.startswith("../"),  # Unix parent directory
        "\\" in s,  # Windows path separators
        ":" in s and len(s) > 2 and s[1:3] == ":\\",  # Windows drive letter
        # NEW: Additional patterns
        "/tmp/" in s,  # Common temp directory  # nosec B108
        "/home/" in s,  # Common home directory
        "/usr/" in s,  # Common system directory
        s.endswith(".txt"),  # Common file extensions
        s.endswith(".py"),
        s.endswith(".json"),
        s.endswith(".csv"),
        s.endswith(".log"),
        # Path-like structure with directory separators
        "/" in s and not s.startswith("http"),  # Has separator but not URL
    ]

    return any(path_indicators)


def _get_cached_string_pattern(s: str) -> Optional[str]:
    """Get cached string pattern type to optimize repeated detection.

    Categories:
    - 'plain': Plain string, no special processing needed
    - 'uuid': UUID pattern detected
    - 'datetime': Datetime pattern detected
    - 'path': Path pattern detected
    - 'unknown': Needs full processing
    """
    s_id = id(s)
    if s_id in _STRING_PATTERN_CACHE:
        return _STRING_PATTERN_CACHE[s_id]

    # Only cache if we haven't hit the limit
    if len(_STRING_PATTERN_CACHE) >= _STRING_CACHE_SIZE_LIMIT:
        return None

    # Determine pattern category using ultra-fast checks
    pattern = None
    s_len = len(s)

    # Quick rejection for obviously plain strings
    if s_len < 8:  # Too short for UUID/datetime
        pattern = "plain"
    # Ultra-fast UUID detection
    elif s_len == 36 and s[8] == "-" and s[13] == "-" and s[18] == "-" and s[23] == "-":
        # Quick character set validation for first few chars
        pattern = "uuid" if all(c in _UUID_CHAR_SET for c in s[:8]) else "plain"
    # Ultra-fast datetime detection (ISO format: YYYY-MM-DD...)
    elif s_len >= 10 and s[4] == "-" and s[7] == "-":
        # Quick character set validation for year
        pattern = "datetime" if s[:4].isdigit() else "plain"
    # Ultra-fast path detection
    elif s[0] == "/" or (s_len >= 3 and s[1:3] == ":\\") or "/tmp/" in s or s.startswith("./"):  # nosec B108
        pattern = "path"
    else:
        pattern = "unknown"  # Needs full processing

    _STRING_PATTERN_CACHE[s_id] = pattern
    return pattern


def _get_cached_parsed_object(s: str, pattern_type: str) -> Any:
    """Get cached parsed object for common strings."""
    cache_key = f"{pattern_type}:{s}"

    if cache_key in _PARSED_OBJECT_CACHE:
        cached_result = _PARSED_OBJECT_CACHE[cache_key]

        # Validate cached result type - if wrong, invalidate cache
        if cached_result is not None:
            if (
                pattern_type == "uuid"
                and not isinstance(cached_result, uuid.UUID)
                or pattern_type == "datetime"
                and not isinstance(cached_result, datetime)
                or pattern_type == "path"
                and not hasattr(cached_result, "as_posix")
            ):
                # Cache corrupted - remove and re-parse
                del _PARSED_OBJECT_CACHE[cache_key]
            else:
                return cached_result
        else:
            return cached_result  # None (failed parse) is valid

    # Only cache if we have space
    if len(_PARSED_OBJECT_CACHE) >= _PARSED_CACHE_SIZE_LIMIT:
        return None  # Don't cache, but proceed with parsing

    # Parse based on pattern type
    parsed_obj = None
    try:
        if pattern_type == "uuid":
            import uuid as uuid_module  # Fresh import to avoid state issues

            parsed_obj = uuid_module.UUID(s)
        elif pattern_type == "datetime":
            from datetime import datetime as datetime_class  # Fresh import

            parsed_obj = datetime_class.fromisoformat(s.replace("Z", "+00:00"))
        elif pattern_type == "path":
            from pathlib import Path

            parsed_obj = Path(s)

        # Cache the result
        if parsed_obj is not None:
            _PARSED_OBJECT_CACHE[cache_key] = parsed_obj

        return parsed_obj
    except Exception:
        # Cache None to avoid repeated parsing attempts
        _PARSED_OBJECT_CACHE[cache_key] = None
        return None


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


def _clear_deserialization_caches() -> None:
    """Clear all internal caches used by the deserialization system.

    This is useful for testing or when you want to ensure fresh state.
    """
    global _STRING_PATTERN_CACHE, _PARSED_OBJECT_CACHE, _RESULT_DICT_POOL, _RESULT_LIST_POOL
    _STRING_PATTERN_CACHE.clear()
    _PARSED_OBJECT_CACHE.clear()
    _RESULT_DICT_POOL.clear()
    _RESULT_LIST_POOL.clear()


def _convert_string_keys_to_int_if_possible(data: Dict[str, Any]) -> Dict[Any, Any]:
    """Convert string keys to integers if they represent valid integers.

    This handles the case where JSON serialization converts integer keys to strings.
    For pandas Series with integer indices, we need to convert them back.
    """
    converted_data = {}
    for key, value in data.items():
        # Try to convert string keys that look like integers back to integers
        if isinstance(key, str) and key.isdigit():
            try:
                int_key = int(key)
                converted_data[int_key] = value
            except ValueError:
                # If conversion fails, keep as string
                converted_data[key] = value
        elif isinstance(key, str) and key.lstrip("-").isdigit():
            # Handle negative integers
            try:
                int_key = int(key)
                converted_data[int_key] = value
            except ValueError:
                # If conversion fails, keep as string
                converted_data[key] = value
        else:
            # Keep non-string keys or non-numeric string keys as-is
            converted_data[key] = value

    return converted_data
