"""Deserialization functionality for datason.

This module provides functions to convert JSON-compatible data back to appropriate
Python objects, including datetime parsing, UUID reconstruction, and pandas types.
"""

import decimal
import uuid
import warnings
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

# Import configuration and security constants
try:
    from .config import CacheScope, SerializationConfig, get_default_config

    _config_available = True
except ImportError:
    _config_available = False
    SerializationConfig = None
    CacheScope = None

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


# OPTIMIZATION: Configurable scoped caches for ultra-fast deserialization
# Using the new cache management system with configurable scopes
try:
    from .cache_manager import clear_caches as clear_scoped_caches
    from .cache_manager import (
        dict_pool,
        list_pool,
    )

    _cache_manager_available = True
except ImportError:
    _cache_manager_available = False

# Always define fallback caches for compatibility
_DESERIALIZATION_TYPE_CACHE: Dict[str, str] = {}  # Maps string patterns to detected types
_STRING_PATTERN_CACHE: Dict[int, str] = {}  # Maps string id to detected pattern type
_PARSED_OBJECT_CACHE: Dict[str, Any] = {}  # Maps string to parsed object
_RESULT_DICT_POOL: List[Dict] = []
_RESULT_LIST_POOL: List[List] = []

# Legacy cache size limits (used when cache manager not available)
_TYPE_CACHE_SIZE_LIMIT = 1000  # Prevent memory growth
_STRING_CACHE_SIZE_LIMIT = 500  # Smaller cache for strings
_PARSED_CACHE_SIZE_LIMIT = 200  # Cache for common UUIDs/datetimes
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


def auto_deserialize(obj: Any, aggressive: bool = False, config: Optional["SerializationConfig"] = None) -> Any:
    """NEW: Intelligent auto-detection deserialization with heuristics.

    Uses pattern recognition and heuristics to automatically detect and restore
    complex data types without explicit configuration.

    Args:
        obj: JSON-compatible object to deserialize
        aggressive: Whether to use aggressive type detection (may have false positives)
        config: Configuration object to control deserialization behavior (NEW)

    Returns:
        Python object with auto-detected types restored

    Examples:
        >>> data = {"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        >>> auto_deserialize(data, aggressive=True)
        {"records": DataFrame(...)}  # May detect as DataFrame

        >>> # NEW: API-compatible UUID handling
        >>> from datason.config import get_api_config
        >>> auto_deserialize("12345678-1234-5678-9012-123456789abc", config=get_api_config())
        "12345678-1234-5678-9012-123456789abc"  # Stays as string
    """
    if obj is None:
        return None

    # Get default config if none provided
    # Note: _config_available and get_default_config are defined at module level
    # through conditional imports at the top of this file
    if config is None and _config_available:
        config = get_default_config()

    # Handle type metadata first
    if isinstance(obj, dict) and TYPE_METADATA_KEY in obj:
        return _deserialize_with_type_metadata(obj)

    # Handle basic types
    if isinstance(obj, (int, float, bool)):
        return obj

    # Handle strings with auto-detection
    if isinstance(obj, str):
        return _auto_detect_string_type(obj, aggressive, config)

    # Handle lists with auto-detection
    if isinstance(obj, list):
        deserialized_list = [auto_deserialize(item, aggressive, config) for item in obj]

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
        return {k: auto_deserialize(v, aggressive, config) for k, v in obj.items()}

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

    SECURITY WARNING: This function may deserialize pickle data for legacy sklearn models.
    Pickle deserialization can execute arbitrary code if the data is untrusted.
    Only use with data from trusted sources.
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
                            if isinstance(item, dict) and ("_type" in item or "__datason_type__" in item):
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
                    if isinstance(value, dict) and "class" in value and "params" in value:
                        # This is the new format: reconstruct the sklearn model from class and params
                        class_name = value["class"]
                        params = value["params"]

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

                    # Handle legacy format for backward compatibility
                    elif isinstance(value, dict) and "_class" in value and "_params" in value:
                        # This is the legacy format: reconstruct the sklearn model from _class and _params
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

                    # Handle legacy pickle format (SECURITY: Pickle deserialization disabled)
                    elif isinstance(value, str):
                        # Legacy pickle data detected - try JSON first, disable pickle for security
                        import base64
                        import json

                        try:
                            # Try JSON format first (safe)
                            json_data = base64.b64decode(value).decode("utf-8")
                            return json.loads(json_data)
                        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                            # Legacy pickle data detected - disabled for security
                            warnings.warn(
                                "Legacy pickle deserialization is disabled for security reasons. "
                                "Please re-serialize your models using a safer format.",
                                UserWarning,
                                stacklevel=3,
                            )
                            return value  # Return original value as fallback
                    elif isinstance(value, dict) and "_pickle_data" in value:
                        # Alternative pickle storage format - try JSON first, disable pickle for security
                        import base64
                        import json

                        try:
                            # Try JSON format first (safe)
                            json_data = base64.b64decode(value["_pickle_data"]).decode("utf-8")
                            return json.loads(json_data)
                        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                            # Legacy pickle data detected - disabled for security
                            warnings.warn(
                                "Legacy pickle deserialization is disabled for security reasons. "
                                "Please re-serialize your models using a safer format.",
                                UserWarning,
                                stacklevel=3,
                            )
                            return value  # Return original value as fallback
                except (ImportError, Exception) as e:
                    warnings.warn(f"Could not reconstruct sklearn model: {e}", stacklevel=2)

            # CatBoost model reconstruction
            elif type_name == "catboost.model":
                try:
                    if isinstance(value, dict) and "class" in value and "params" in value:
                        class_name = value["class"]
                        params = value["params"]

                        # Import CatBoost class dynamically
                        import catboost

                        # Get the class name from the full class path
                        class_name_only = class_name.split(".")[-1]
                        model_class = getattr(catboost, class_name_only)

                        # Create the model with the saved parameters
                        model = model_class(**params)
                        return model
                except Exception as e:
                    warnings.warn(f"Could not reconstruct CatBoost model: {e}", stacklevel=2)

            # Keras model reconstruction
            elif type_name == "keras.model":
                try:
                    if isinstance(value, dict) and "class" in value:
                        # Note: Current Keras serialization stores metadata only
                        # Full reconstruction would require saving the actual model config
                        # For now, we create a basic model structure

                        import keras

                        class_name = value["class"]
                        if "Sequential" in class_name:
                            # Create a basic Sequential model
                            # This is a simplified reconstruction
                            model = keras.Sequential()
                            return model
                        else:
                            # For other model types, return the metadata
                            # This preserves type information for template reconstruction
                            warnings.warn("Keras model reconstruction limited to metadata preservation", stacklevel=2)
                            return value
                except Exception as e:
                    warnings.warn(f"Could not reconstruct Keras model: {e}", stacklevel=2)

            # Optuna study reconstruction - FIXED: Use correct type name (with legacy support)
            elif type_name in ("optuna.Study", "optuna.study"):
                try:
                    if isinstance(value, dict) and "study_name" in value:
                        study_name = value.get("study_name")
                        direction = value.get("direction", "minimize")

                        import optuna

                        # Create a new study with the same configuration
                        # Note: We can't restore trials, but we can restore the study structure
                        direction_obj = (
                            optuna.study.StudyDirection.MINIMIZE
                            if "minimize" in str(direction).lower()
                            else optuna.study.StudyDirection.MAXIMIZE
                        )
                        study = optuna.create_study(study_name=study_name, direction=direction_obj)
                        return study
                except Exception as e:
                    warnings.warn(f"Could not reconstruct Optuna study: {e}", stacklevel=2)

            # Plotly figure reconstruction - FIXED: Use correct type name (with legacy support)
            elif type_name in ("plotly.graph_objects.Figure", "plotly.figure"):
                try:
                    if isinstance(value, dict) and "data" in value and "layout" in value:
                        import plotly.graph_objects as go

                        # Recreate the figure from data and layout
                        fig = go.Figure(data=value["data"], layout=value["layout"])
                        return fig
                except Exception as e:
                    warnings.warn(f"Could not reconstruct Plotly figure: {e}", stacklevel=2)

            # Polars DataFrame reconstruction - FIXED: Use correct type name (with legacy support)
            elif type_name in ("polars.DataFrame", "polars.dataframe"):
                try:
                    if isinstance(value, dict) and "data" in value:
                        data_dict = value["data"]

                        import polars as pl

                        # Recreate the DataFrame from data dict
                        df = pl.DataFrame(data_dict)
                        return df
                except Exception as e:
                    warnings.warn(f"Could not reconstruct Polars DataFrame: {e}", stacklevel=2)

        except Exception as e:
            warnings.warn(f"Failed to reconstruct type {type_name}: {e}", stacklevel=2)

        # Fallback to the original value
        return value

    # NOTE: Legacy '_type' format support removed in v0.8.0
    # Only '__datason_type__' format is now supported for type metadata

    # Not a type metadata object
    return obj


def _auto_detect_string_type(s: str, aggressive: bool = False, config: Optional["SerializationConfig"] = None) -> Any:
    """NEW: Auto-detect the most likely type for a string value.

    Args:
        s: String to analyze
        aggressive: Whether to use aggressive type detection
        config: Configuration to control type conversion behavior
    """
    # Check configuration for UUID handling
    should_parse_uuids = True
    if config is not None:
        # Use parse_uuids config if available
        should_parse_uuids = getattr(config, "parse_uuids", True)
        # Also check uuid_format preference
        uuid_format = getattr(config, "uuid_format", "object")
        if uuid_format == "string":
            should_parse_uuids = False

    # Always try UUID detection first (more specific pattern)
    if should_parse_uuids and _looks_like_uuid(s):
        try:
            import uuid as uuid_module  # Fresh import to avoid state issues

            return uuid_module.UUID(s)
        except (ValueError, ImportError):
            pass

    # Then try datetime detection
    if _looks_like_datetime(s):
        try:
            import re
            from datetime import datetime as datetime_class  # Fresh import

            # Validate ISO 8601 format BEFORE any operations to prevent CodeQL false positive
            # This provides security by validating input format before processing
            iso8601_pattern = r"^\d{4}-\d{2}-\d{2}(?:T| )\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"

            if re.match(iso8601_pattern, s):
                # Only perform operations AFTER validation
                normalized_input = s.replace("Z", "+00:00")
                return datetime_class.fromisoformat(normalized_input)  # Now safe from CodeQL false positive
        except (ValueError, ImportError, re.error):
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
def safe_deserialize(json_str: str, allow_pickle: bool = False, **kwargs: Any) -> Any:
    """Safely deserialize a JSON string, handling parse errors gracefully.

    Args:
        json_str: JSON string to parse and deserialize
        allow_pickle: Whether to allow unsafe pickle deserialization (default: False)
        **kwargs: Arguments passed to deserialize()

    Returns:
        Deserialized Python object, or the original string if parsing fails

    Security Note:
        When allow_pickle=False, this function will refuse to deserialize
        legacy sklearn models that use pickle format, providing safer operation
        at the cost of some backward compatibility.
    """
    import json

    try:
        parsed = json.loads(json_str)

        # Add security configuration
        if not allow_pickle and _contains_pickle_data(parsed):
            raise DeserializationSecurityError(
                "Data contains pickle-serialized objects which are unsafe to deserialize. "
                "Set allow_pickle=True to override this security check, but only with trusted data."
            )

        return deserialize(parsed, **kwargs)
    except (json.JSONDecodeError, TypeError, ValueError):
        return json_str


def _contains_pickle_data(obj: Any) -> bool:
    """Check if an object contains pickle-serialized data (recursive check)."""
    if isinstance(obj, dict):
        # Check for sklearn types with pickle data
        if obj.get("__datason_type__", "").startswith("sklearn.") and (
            isinstance(obj.get("__datason_value__"), str)
            or (isinstance(obj.get("__datason_value__"), dict) and "_pickle_data" in obj.get("__datason_value__", {}))
        ):
            return True

        # Recursively check all values
        for value in obj.values():
            if _contains_pickle_data(value):
                return True
    elif isinstance(obj, list):
        # Recursively check all items
        for item in obj:
            if _contains_pickle_data(item):
                return True

    return False


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

        # NEW: NumPy array support
        elif np is not None and isinstance(template, np.ndarray) and isinstance(obj, list):
            return self._deserialize_numpy_with_template(obj, template)

        # NEW: NumPy scalar support
        elif (
            np is not None
            and hasattr(template, "__class__")
            and hasattr(template.__class__, "__module__")
            and template.__class__.__module__ == "numpy"
        ):
            return self._deserialize_numpy_scalar_with_template(obj, template)

        # NEW: Complex number support
        elif isinstance(template, complex):
            return self._deserialize_complex_with_template(obj, template)

        # NEW: PyTorch tensor support
        elif hasattr(template, "__module__") and template.__module__ and "torch" in template.__module__:
            return self._deserialize_torch_with_template(obj, template)

        # NEW: Sklearn model support
        elif hasattr(template, "get_params") and hasattr(template, "__module__") and "sklearn" in template.__module__:
            return self._deserialize_sklearn_with_template(obj, template)

        # NEW: Path support
        elif isinstance(template, Path) and isinstance(obj, str):
            return self._deserialize_path_with_template(obj, template)

        # NEW: Decimal support
        elif isinstance(template, Decimal):
            return self._deserialize_decimal_with_template(obj, template)

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

    def _deserialize_numpy_with_template(self, obj: List[Any], template: "np.ndarray") -> "np.ndarray":
        """Deserialize list to NumPy array using template."""
        if np is None:
            raise ImportError("numpy is required for NumPy template deserialization")

        try:
            # Create array from list
            result = np.array(obj)

            # Apply template dtype if possible
            if hasattr(template, "dtype"):
                try:
                    result = result.astype(template.dtype)
                except (ValueError, TypeError, AttributeError):
                    # If dtype conversion fails due to incompatible types, keep original
                    pass  # nosec B110 - intentional fallback for dtype conversion

            # Apply template shape if possible
            if hasattr(template, "shape") and result.size == template.size:
                try:
                    result = result.reshape(template.shape)
                except (ValueError, AttributeError):
                    # If reshape fails due to incompatible shapes, keep original shape
                    pass  # nosec B110 - intentional fallback for reshape

            return result
        except Exception as e:
            warnings.warn(f"Failed to convert to NumPy array: {e}", stacklevel=3)
            return obj

    def _deserialize_numpy_scalar_with_template(self, obj: Any, template: Any) -> Any:
        """Deserialize basic Python type to NumPy scalar using template."""
        if np is None:
            return obj

        try:
            # Get the template type
            template_type = type(template)

            # Convert basic Python type to NumPy scalar
            if isinstance(obj, (int, float, bool)):
                return template_type(obj)
            elif isinstance(obj, str):
                # Try to parse string to appropriate type
                return template_type(obj)
            else:
                # For other types, try direct conversion
                return template_type(obj)

        except Exception as e:
            warnings.warn(f"Failed to convert to NumPy scalar type {type(template)}: {e}", stacklevel=3)
            return obj

    def _deserialize_complex_with_template(self, obj: Any, template: complex) -> complex:
        """Deserialize complex number using template."""
        if isinstance(obj, dict):
            # Handle new metadata format: {'__datason_type__': 'complex', '__datason_value__': {'real': 1.0, 'imag': 2.0}}
            if obj.get("__datason_type__") == "complex" and "__datason_value__" in obj:
                value = obj["__datason_value__"]
                if isinstance(value, dict) and "real" in value and "imag" in value:
                    return complex(value["real"], value["imag"])
            # Handle legacy dict format: {'_type': 'complex', 'real': 1.0, 'imag': 2.0}
            elif obj.get("_type") == "complex" and "real" in obj and "imag" in obj or "real" in obj and "imag" in obj:
                return complex(obj["real"], obj["imag"])
        elif isinstance(obj, list) and len(obj) == 2:
            # PHASE 2: Handle new list format [real, imag]
            try:
                return complex(obj[0], obj[1])
            except (ValueError, TypeError, IndexError):
                pass
        elif isinstance(obj, (int, float)):
            return complex(obj)
        elif isinstance(obj, str):
            try:
                return complex(obj)
            except ValueError:
                pass

        # If all else fails, return the original object
        return obj

    def _deserialize_torch_with_template(self, obj: Any, template: Any) -> Any:
        """Deserialize PyTorch tensor using template."""
        if isinstance(obj, dict):
            try:
                import torch

                # Handle new metadata format: {'__datason_type__': 'torch.Tensor', '__datason_value__': {...}}
                if obj.get("__datason_type__") == "torch.Tensor" and "__datason_value__" in obj:
                    value = obj["__datason_value__"]
                    if isinstance(value, dict):
                        data = value.get("data")
                        dtype_str = value.get("dtype")
                        device = value.get("device", "cpu")
                        requires_grad = value.get("requires_grad", False)

                        # Create tensor
                        tensor = torch.tensor(data, device=device, requires_grad=requires_grad)

                        # Apply dtype if specified
                        if dtype_str and hasattr(torch, dtype_str.replace("torch.", "")):
                            try:
                                torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
                                tensor = tensor.to(torch_dtype)
                            except (AttributeError, RuntimeError, TypeError):
                                pass  # nosec B110 - intentional fallback for torch dtype conversion

                        return tensor
                    else:
                        # Simple value format
                        return torch.tensor(value)

                # Handle legacy format: {'_type': 'torch.Tensor', '_data': [...], ...}
                elif obj.get("_type") in ("torch.tensor", "torch.Tensor"):
                    # Handle tensor dict format with _data field
                    if "_data" in obj:
                        data = obj["_data"]
                        dtype_str = obj.get("_dtype")
                        device = obj.get("_device", "cpu")

                        # Create tensor
                        tensor = torch.tensor(data, device=device)

                        # Apply dtype if specified
                        if dtype_str and hasattr(torch, dtype_str.replace("torch.", "")):
                            try:
                                torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
                                tensor = tensor.to(torch_dtype)
                            except (AttributeError, RuntimeError, TypeError):
                                pass  # nosec B110 - intentional fallback for torch dtype conversion

                        return tensor

                    # Handle tensor dict format with data field
                    elif "data" in obj:
                        data = obj["data"]
                        dtype = obj.get("dtype")
                        device = obj.get("device", "cpu")

                        # Create tensor
                        tensor = torch.tensor(data, device=device)

                        # Apply dtype if specified
                        if dtype:
                            try:
                                tensor = tensor.to(getattr(torch, dtype))
                            except (AttributeError, RuntimeError, TypeError):
                                pass  # nosec B110 - intentional fallback for torch dtype conversion

                        return tensor

            except Exception as e:
                warnings.warn(f"Failed to reconstruct PyTorch tensor: {e}", stacklevel=3)
        elif isinstance(obj, list):
            # Simple list → tensor conversion
            try:
                import torch

                return torch.tensor(obj)
            except Exception as e:
                warnings.warn(f"Failed to convert list to PyTorch tensor: {e}", stacklevel=3)

        return obj

    def _deserialize_sklearn_with_template(self, obj: Any, template: Any) -> Any:
        """Deserialize sklearn model using template."""
        if isinstance(obj, dict):
            try:
                # Handle new metadata format: {'__datason_type__': 'sklearn.model', '__datason_value__': {...}}
                if obj.get("__datason_type__") == "sklearn.model" and "__datason_value__" in obj:
                    value = obj["__datason_value__"]
                    if isinstance(value, dict) and "class" in value and "params" in value:
                        # Import the sklearn class dynamically
                        class_name = value["class"]
                        module_path, class_name_only = class_name.rsplit(".", 1)

                        import importlib

                        module = importlib.import_module(module_path)
                        model_class = getattr(module, class_name_only)

                        # Create the model with the saved parameters
                        reconstructed = model_class(**value["params"])

                        # Verify it's the same type as template
                        if type(reconstructed) is type(template):
                            return reconstructed

                # Handle legacy format: {'_type': 'sklearn.model', '_class': '...', '_params': {...}}
                elif "_type" in obj and "_class" in obj and "_params" in obj:
                    # Import the sklearn class dynamically
                    class_name = obj["_class"]
                    module_path, class_name_only = class_name.rsplit(".", 1)

                    import importlib

                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name_only)

                    # Create the model with the saved parameters
                    reconstructed = model_class(**obj["_params"])

                    # Verify it's the same type as template
                    if type(reconstructed) is type(template):
                        return reconstructed

            except Exception as e:
                warnings.warn(f"Failed to reconstruct sklearn model: {e}", stacklevel=3)

        # If reconstruction fails, return the original object
        return obj

    def _deserialize_path_with_template(self, obj: str, template: Path) -> Path:
        """Deserialize Path using template."""
        try:
            return Path(obj)
        except Exception as e:
            warnings.warn(f"Failed to convert to Path: {e}", stacklevel=3)
            return obj

    def _deserialize_decimal_with_template(self, obj: Any, template: Decimal) -> Decimal:
        """Deserialize Decimal using template."""
        if isinstance(obj, dict) and obj.get("_type") == "decimal" and "value" in obj:
            # Handle dict format: {'_type': 'decimal', 'value': '123.456'}
            try:
                return Decimal(obj["value"])
            except (ValueError, TypeError, decimal.InvalidOperation):
                # If decimal conversion fails, continue to fallback conversion
                pass  # nosec B110 - intentional fallback for decimal conversion

        try:
            # Try direct conversion
            if isinstance(obj, str):
                return Decimal(obj)
            elif isinstance(obj, (int, float)):
                return Decimal(str(obj))
        except Exception as e:
            warnings.warn(f"Failed to convert to Decimal: {e}", stacklevel=3)

        return obj

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


def _process_list_optimized(obj: list, config: Optional["SerializationConfig"], _depth: int, _seen: Set[int]) -> Any:
    """Optimized list processing with circular reference protection and smart type detection."""
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

            # METADATA-FIRST: Only do auto-detection if explicitly enabled
            auto_detect = (config and getattr(config, "auto_detect_types", False)) or False

            if auto_detect:
                # Try NumPy array auto-detection
                numpy_array = _try_numpy_array_detection(final_result)
                if numpy_array is not None:
                    return numpy_array

                # Try pandas DataFrame auto-detection (list of records pattern)
                dataframe = _try_dataframe_detection(final_result)
                if dataframe is not None:
                    return dataframe

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

        # METADATA-FIRST: Only do auto-detection if explicitly enabled
        # This avoids false positives in general usage while allowing opt-in
        auto_detect = (config and getattr(config, "auto_detect_types", False)) or False

        if auto_detect:
            # Try pandas DataFrame detection (split format)
            if pd is not None and _looks_like_split_format(obj):
                try:
                    return _reconstruct_from_split(obj)
                except (KeyError, ValueError, TypeError, ImportError):
                    # If pandas reconstruction fails, continue with normal processing
                    pass  # nosec B110 - intentional fallback for pandas reconstruction

            # Try pandas DataFrame detection (column-oriented dict)
            if pd is not None and _looks_like_dataframe_dict(obj):
                try:
                    return _reconstruct_dataframe(obj)
                except (KeyError, ValueError, TypeError, ImportError):
                    # If pandas reconstruction fails, continue with normal processing
                    pass  # nosec B110 - intentional fallback for pandas reconstruction

            # Try pandas Series detection (index-oriented dict)
            if pd is not None:
                series = _try_series_detection(obj)
                if series is not None:
                    return series

        # Check for special formats (legacy - keep for compatibility)
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
    if _cache_manager_available:
        try:
            return dict_pool.get()
        except Exception:
            return {}
    else:
        if _RESULT_DICT_POOL:
            result = _RESULT_DICT_POOL.pop()
            result.clear()  # Ensure it's clean
            return result
        return {}


def _return_dict_to_pool(d: Dict) -> None:
    """Return a dictionary to the pool for reuse."""
    if _cache_manager_available:
        try:
            dict_pool.return_to_pool(d)
        except (AttributeError, RuntimeError, TypeError):
            # If pool return fails due to pool state issues, just skip pooling
            pass  # nosec B110 - intentional fallback for object pool operations
    else:
        if len(_RESULT_DICT_POOL) < _POOL_SIZE_LIMIT:
            d.clear()
            _RESULT_DICT_POOL.append(d)


def _get_pooled_list() -> List:
    """Get a list from the pool or create new one."""
    if _cache_manager_available:
        try:
            return list_pool.get()
        except Exception:
            return []
    else:
        if _RESULT_LIST_POOL:
            result = _RESULT_LIST_POOL.pop()
            result.clear()  # Ensure it's clean
            return result
        return []


def _return_list_to_pool(lst: List) -> None:
    """Return a list to the pool for reuse."""
    if _cache_manager_available:
        try:
            list_pool.return_to_pool(lst)
        except (AttributeError, RuntimeError, TypeError):
            # If pool return fails due to pool state issues, just skip pooling
            pass  # nosec B110 - intentional fallback for object pool operations
    else:
        if len(_RESULT_LIST_POOL) < _POOL_SIZE_LIMIT:
            lst.clear()
            _RESULT_LIST_POOL.append(lst)


def _clear_deserialization_caches() -> None:
    """Clear all internal caches used by the deserialization system.

    This is useful for testing or when you want to ensure fresh state.
    """
    if _cache_manager_available:
        # Use the scoped cache manager
        try:
            clear_scoped_caches()
        except (AttributeError, ImportError, RuntimeError):
            # Fallback to module-level caches if scoped clearing fails
            pass  # nosec B110 - intentional fallback for cache clearing
    else:
        # Use module-level caches
        global _STRING_PATTERN_CACHE, _PARSED_OBJECT_CACHE, _RESULT_DICT_POOL, _RESULT_LIST_POOL
        if "_STRING_PATTERN_CACHE" in globals():
            _STRING_PATTERN_CACHE.clear()
        if "_PARSED_OBJECT_CACHE" in globals():
            _PARSED_OBJECT_CACHE.clear()
        if "_RESULT_DICT_POOL" in globals():
            _RESULT_DICT_POOL.clear()
        if "_RESULT_LIST_POOL" in globals():
            _RESULT_LIST_POOL.clear()


def clear_caches() -> None:
    """Clear all caches - new name for _clear_deserialization_caches."""
    _clear_deserialization_caches()


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


def _try_numpy_array_detection(data: list) -> Optional[Any]:
    """Attempt to auto-detect NumPy arrays from list data.

    Returns numpy array if detection succeeds, None otherwise.
    Uses conservative heuristics to avoid false positives.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    if not data:  # Empty list
        try:
            return np.array(data)
        except Exception:
            return None

    # Conservative detection: only convert if the data looks like a homogeneous numeric array
    if _looks_like_numpy_array(data):
        try:
            return np.array(data)
        except Exception:
            # If numpy conversion fails, return None to fall back to list
            return None

    return None


def _looks_like_numpy_array(data: list) -> bool:
    """Check if a list looks like it should be a NumPy array.

    Uses balanced heuristics targeting known serialization patterns.
    Focuses on cases where we can be confident about the intent.
    """
    if not data:
        return False  # Empty lists stay as Python lists

    # Check for types that definitely shouldn't be arrays
    if any(isinstance(item, (dict, tuple)) for item in data):
        return False

    # Handle numpy arrays that were already converted (during deserialization)
    try:
        import numpy as np

        if any(isinstance(item, np.ndarray) for item in data):
            if all(isinstance(item, (np.ndarray, list)) for item in data):
                # Check if all arrays/lists have the same shape
                shapes = []
                for item in data:
                    if isinstance(item, np.ndarray):
                        shapes.append(item.shape)
                    elif isinstance(item, list):
                        shapes.append((len(item),))
                return len(set(shapes)) == 1
            return False
    except ImportError:
        pass

    # Multi-dimensional arrays: convert if clearly rectangular and numeric
    if all(isinstance(item, list) for item in data):
        if (
            len(data) >= 2  # At least 2 rows
            and len({len(sublist) for sublist in data}) == 1  # Same length
            and len(data[0]) >= 2
        ):  # At least 2 columns
            # Check if all elements are numeric
            all_numeric = True
            for row in data:
                if not all(isinstance(item, (int, float, bool)) for item in row):
                    all_numeric = False
                    break

            if all_numeric:
                return True

    # 1D arrays: Smart detection for specific patterns
    # Target known NumPy serialization patterns while avoiding common Python lists
    elif _is_homogeneous_basic_types(data):
        first_type = type(data[0])

        # Pattern 1: Homogeneous numeric arrays (likely from NumPy)
        if first_type in (int, float) and len(data) >= 3 or first_type is bool and len(data) >= 3:
            return True

        # Pattern 3: String arrays (NumPy pattern)
        elif first_type is str and len(data) >= 3:
            # Simple string array detection
            return True

    return False


def _is_homogeneous_basic_types(data: list) -> bool:
    """Check if a list contains homogeneous basic types suitable for NumPy arrays."""
    if not data:
        return True

    first_type = type(data[0])
    if first_type in (int, float, bool, str, type(None)):
        # Allow mixed int/float (common pattern), but require same type for strings
        if first_type is str:
            return all(isinstance(item, str) for item in data)
        else:
            # For numeric types, allow mixing int/float
            return all(isinstance(item, (int, float, bool, type(None))) for item in data)

    return False


def _try_dataframe_detection(data: list) -> Optional[Any]:
    """Attempt to auto-detect pandas DataFrames from various serialization formats.

    Returns DataFrame if detection succeeds, None otherwise.
    Handles multiple pandas serialization formats:
    - Records: [{"col1": val1, "col2": val2}, ...]
    - Values: [[val1, val2], [val3, val4], ...] (with consistent structure)
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if not data:
        return None

    # Pattern 1: List of records - [{"col1": val1, "col2": val2}, ...]
    if all(isinstance(item, dict) for item in data):
        # Check if all records have the same keys (consistent schema)
        first_keys = set(data[0].keys()) if data else set()
        if all(set(record.keys()) == first_keys for record in data):
            try:
                return pd.DataFrame(data)
            except Exception:
                return None

    # Pattern 2: List of lists (values format) - [[val1, val2], [val3, val4], ...]
    # Only convert if it looks like tabular data (consistent row lengths, reasonable size)
    elif (
        all(isinstance(item, list) for item in data)
        and len(data) >= 2  # At least 2 rows
        and len({len(row) for row in data}) == 1  # All rows same length
        and len(data[0]) >= 2  # At least 2 columns
        and len(data[0]) <= 50
    ):  # Reasonable number of columns
        # Additional check: avoid converting nested arrays that are likely NumPy
        # If all elements are numbers, it's more likely a 2D NumPy array
        all_numeric = True
        for row in data:
            if not all(isinstance(item, (int, float, bool)) for item in row):
                all_numeric = False
                break

        # If it's all numeric and looks array-like, don't convert to DataFrame
        if all_numeric and len(data) == len(data[0]):
            # Square matrix - likely NumPy array
            return None

        try:
            return pd.DataFrame(data)
        except Exception:
            return None

    return None


def _try_series_detection(data: dict) -> Optional[Any]:
    """Attempt to auto-detect pandas Series from dict data.

    Returns Series if detection succeeds, None otherwise.
    Handles Series serialized as index-oriented dicts: {"idx1": val1, "idx2": val2, ...}
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if not data or len(data) < 2:
        return None

    # Pattern: Index-oriented dict that looks like Series data
    # Series are often serialized as {"0": val1, "1": val2, ...} or {"idx1": val1, "idx2": val2}

    # Check if all values are the same basic type (homogeneous data)
    values = list(data.values())
    if not values:
        return None

    # Check for homogeneous value types (common in Series)
    first_type = type(values[0])
    if first_type in (int, float, str, bool, type(None)):
        # Check if most values are the same type
        same_type_count = sum(1 for v in values if type(v) is first_type)
        if same_type_count / len(values) >= 0.8:  # 80% same type threshold
            # Additional heuristics to distinguish from regular dicts
            keys = list(data.keys())

            # Pattern 1: Numeric string indices (common Series pattern)
            if all(isinstance(k, str) and k.isdigit() for k in keys):
                try:
                    # Convert string indices back to integers
                    int_keys = [int(k) for k in keys]
                    return pd.Series(values, index=int_keys)
                except Exception:
                    return None

            # Pattern 2: Sequential integer indices
            elif all(isinstance(k, int) for k in keys) or (
                all(isinstance(k, str) for k in keys)
                and len(keys) >= 3  # At least 3 items
                and all(len(k) <= 20 for k in keys)
            ):
                try:
                    return pd.Series(values, index=keys)
                except Exception:
                    return None

    return None
