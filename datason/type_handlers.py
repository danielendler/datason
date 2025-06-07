"""Advanced type handling and coercion for datason.

This module provides comprehensive support for converting Python data types
to JSON-compatible formats with configurable coercion strategies.
"""

import decimal
import enum
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

from .config import NanHandling, SerializationConfig, TypeCoercion

logger = logging.getLogger(__name__)


class TypeHandler:
    """Handles type conversion and coercion based on configuration."""

    def __init__(self, config: SerializationConfig) -> None:
        """Initialize with serialization configuration."""
        self.config = config

    def handle_decimal(self, obj: decimal.Decimal) -> Union[float, str]:
        """Handle decimal.Decimal objects.

        Args:
            obj: Decimal object to convert

        Returns:
            Converted value based on configuration
        """
        # Phase 2: Legacy format removed - always convert to float or string
        try:
            return float(obj)
        except (ValueError, OverflowError):
            return str(obj)

    def handle_complex(self, obj: complex) -> Union[List[float], str]:
        """Handle complex numbers.

        Args:
            obj: Complex number to convert

        Returns:
            Converted value based on configuration
        """
        # Phase 2: Legacy format removed - convert to list or string
        if self.config.type_coercion == TypeCoercion.AGGRESSIVE:
            # Convert to list [real, imag]
            return [obj.real, obj.imag]
        # Convert to list format (no more legacy dict format)
        return [obj.real, obj.imag]

    def handle_uuid(self, obj: uuid.UUID) -> str:
        """Handle UUID objects.

        Args:
            obj: UUID object to convert

        Returns:
            String representation
        """
        # Phase 2: Legacy format removed - always convert to string
        return str(obj)

    def handle_path(self, obj: Path) -> str:
        """Handle pathlib.Path objects.

        Args:
            obj: Path object to convert

        Returns:
            String representation
        """
        # Phase 2: Legacy format removed - always convert to string
        return str(obj)

    def handle_enum(self, obj: enum.Enum) -> Union[str, int, float]:
        """Handle Enum objects.

        Args:
            obj: Enum object to convert

        Returns:
            Enum value
        """
        # Phase 2: Legacy format removed - always return enum value
        return obj.value

    def handle_namedtuple(self, obj: tuple) -> Dict[str, Any]:
        """Handle namedtuple objects.

        Args:
            obj: namedtuple instance

        Returns:
            Dict representation of the namedtuple
        """
        if hasattr(obj, "_fields"):
            # This is a namedtuple - Phase 2: Legacy format removed
            return obj._asdict()
        # Regular tuple, handle elsewhere
        raise ValueError("Not a namedtuple")

    def handle_pandas_categorical(self, obj: Any) -> List[Any]:
        """Handle pandas Categorical objects.

        Args:
            obj: pandas Categorical object

        Returns:
            List of values
        """
        if pd is None:
            raise ImportError("pandas not available")

        # Phase 2: Legacy format removed - always convert to list of values
        return obj.tolist()

    def handle_set(self, obj: set) -> List[Any]:
        """Handle set objects.

        Args:
            obj: Set object to convert

        Returns:
            List representation (sets aren't JSON serializable)
        """
        # Convert set to sorted list for deterministic output
        try:
            return sorted(obj)
        except TypeError:
            # If items aren't sortable, just convert to list
            return list(obj)

    def handle_frozenset(self, obj: frozenset) -> List[Any]:
        """Handle frozenset objects.

        Args:
            obj: Frozenset object to convert

        Returns:
            List representation
        """
        return self.handle_set(obj)

    def handle_bytes(self, obj: bytes) -> str:
        """Handle bytes objects.

        Args:
            obj: Bytes object to convert

        Returns:
            String representation
        """
        # Phase 2: Legacy format removed - always convert to string
        try:
            # Try to decode as UTF-8
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            # Fall back to hex representation
            return obj.hex()

    def handle_bytearray(self, obj: bytearray) -> str:
        """Handle bytearray objects."""
        return self.handle_bytes(bytes(obj))

    def handle_range(self, obj: range) -> List[int]:
        """Handle range objects.

        Args:
            obj: Range object to convert

        Returns:
            List of values
        """
        # Phase 2: Legacy format removed - always convert to list
        # For huge ranges, this might be memory intensive, but it's consistent
        return list(obj)

    def handle_nan_value(self, obj: Any) -> Any:
        """Handle NaN/null values according to configuration.

        Args:
            obj: Value that is NaN/null

        Returns:
            Processed value according to nan_handling setting
        """
        if self.config.nan_handling == NanHandling.NULL:
            return None
        if self.config.nan_handling == NanHandling.STRING:
            if hasattr(obj, "__name__"):
                return f"<{obj.__name__}>"
            return str(obj)
        if self.config.nan_handling == NanHandling.KEEP:
            return obj
        # DROP
        # This should be handled at the collection level
        return None

    def is_namedtuple(self, obj: Any) -> bool:
        """Check if an object is a namedtuple.

        Args:
            obj: Object to check

        Returns:
            True if object is a namedtuple
        """
        return (
            isinstance(obj, tuple)
            and hasattr(type(obj), "_fields")
            and hasattr(type(obj), "_field_defaults")
            and callable(getattr(type(obj), "_asdict", None))
        )

    def get_type_handler(self, obj: Any) -> Optional[callable]:
        """Get the appropriate handler function for an object type.

        Args:
            obj: Object to get handler for

        Returns:
            Handler function or None if no specific handler
        """
        # Check custom serializers first
        if self.config.custom_serializers:
            obj_type = type(obj)
            if obj_type in self.config.custom_serializers:
                return self.config.custom_serializers[obj_type]

        # NEW: Skip built-in handlers for types that support type metadata when enabled
        if (
            hasattr(self.config, "include_type_hints")
            and self.config.include_type_hints
            and isinstance(obj, (decimal.Decimal, complex, uuid.UUID, set, range))
        ):
            # Let core handlers handle these types for type metadata
            return None

        # Built-in type handlers
        if isinstance(obj, decimal.Decimal):
            return self.handle_decimal
        if isinstance(obj, complex):
            return self.handle_complex
        if isinstance(obj, uuid.UUID):
            return self.handle_uuid
        if isinstance(obj, Path):
            return self.handle_path
        if isinstance(obj, enum.Enum):
            return self.handle_enum
        if self.is_namedtuple(obj):
            return self.handle_namedtuple
        if isinstance(obj, set):
            return self.handle_set
        if isinstance(obj, frozenset):
            return self.handle_frozenset
        if isinstance(obj, bytes):
            return self.handle_bytes
        if isinstance(obj, bytearray):
            return self.handle_bytearray
        if isinstance(obj, range):
            return self.handle_range
        if pd is not None and isinstance(obj, pd.Categorical):
            return self.handle_pandas_categorical

        return None


def is_nan_like(obj: Any) -> bool:
    """Check if a value is NaN-like (NaN, NaT, None, etc.).

    Args:
        obj: Value to check

    Returns:
        True if value is NaN-like
    """
    if obj is None:
        return True

    # Check for float NaN
    if isinstance(obj, float) and obj != obj:  # NaN != NaN
        return True

    # Check for numpy NaN types
    if np is not None:
        if isinstance(obj, np.floating) and np.isnan(obj):
            return True
        if isinstance(obj, np.datetime64) and np.isnat(obj):
            return True

    # Check for pandas NaT and NA (but not DataFrames/Series)
    if pd is not None and (not hasattr(obj, "__len__") or isinstance(obj, (str, bytes))):
        try:
            if pd.isna(obj):
                return True
        except (ValueError, TypeError):
            # pd.isna might fail on some types
            pass

    return False


def normalize_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types.

    Args:
        obj: Object that might contain numpy types

    Returns:
        Object with numpy types converted to Python types
    """
    if np is None:
        return obj

    # Scalar types
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, np.bytes_):
        return obj.tobytes()

    return obj


def get_object_info(obj: Any) -> Dict[str, Any]:
    """Get detailed information about an object's type and properties.

    This is useful for debugging serialization issues.

    Args:
        obj: Object to analyze

    Returns:
        Dict with type information
    """
    info = {
        "type": type(obj).__name__,
        "module": getattr(type(obj), "__module__", None),
        "mro": [cls.__name__ for cls in type(obj).__mro__],
        "size": None,
        "is_callable": callable(obj),
        "has_dict": hasattr(obj, "__dict__"),
    }

    # Get size for collections
    if hasattr(obj, "__len__"):
        try:
            info["size"] = len(obj)
        except Exception:
            pass  # nosec B110

    # Special handling for different types
    if isinstance(obj, (list, tuple, set, frozenset)) and info["size"] and info["size"] > 0:
        info["sample_types"] = list({type(item).__name__ for item in list(obj)[:5]})
    elif isinstance(obj, dict) and info["size"] and info["size"] > 0:
        sample_keys = list(obj.keys())[:3]
        sample_values = [obj[k] for k in sample_keys]
        info["sample_key_types"] = list({type(k).__name__ for k in sample_keys})
        info["sample_value_types"] = list({type(v).__name__ for v in sample_values})

    return info
