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
            else:
                series = pd.Series(obj)
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
            else:
                return obj  # Cannot coerce, return as-is
        except (ValueError, TypeError):
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
