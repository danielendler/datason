"""Configuration module for datason serialization behavior.

This module provides configuration classes and options to customize how
datason serializes different data types. Users can configure:

- Date/time output formats
- NaN/null value handling
- Pandas DataFrame orientations
- Type coercion behavior
- Recursion and size limits
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class DateFormat(Enum):
    """Supported date/time output formats."""

    ISO = "iso"  # ISO 8601 format (default)
    UNIX = "unix"  # Unix timestamp
    UNIX_MS = "unix_ms"  # Unix timestamp in milliseconds
    STRING = "string"  # Human readable string
    CUSTOM = "custom"  # Custom format string


class DataFrameOrient(Enum):
    """Supported pandas DataFrame orientations."""

    RECORDS = "records"  # List of records [{col: val}, ...]
    SPLIT = "split"  # Split into {index: [...], columns: [...], data: [...]}
    INDEX = "index"  # Dict like {index -> {column -> value}}
    COLUMNS = "columns"  # Dict like {column -> {index -> value}}
    VALUES = "values"  # Just the values array
    TABLE = "table"  # Table schema format


class NanHandling(Enum):
    """How to handle NaN/null values."""

    NULL = "null"  # Convert to JSON null (default)
    STRING = "string"  # Convert to string representation
    KEEP = "keep"  # Keep as-is (may cause JSON serialization issues)
    DROP = "drop"  # Remove from collections


class TypeCoercion(Enum):
    """Type coercion behavior."""

    STRICT = "strict"  # Raise errors on unknown types
    SAFE = "safe"  # Convert unknown types to safe representations (default)
    AGGRESSIVE = "aggressive"  # Try harder conversions, may lose precision


@dataclass
class SerializationConfig:
    """Configuration for datason serialization behavior.

    Attributes:
        date_format: How to format datetime objects
        custom_date_format: Custom strftime format when date_format is CUSTOM
        dataframe_orient: Pandas DataFrame orientation
        nan_handling: How to handle NaN/null values
        type_coercion: Type coercion behavior
        preserve_decimals: Whether to preserve decimal.Decimal precision
        preserve_complex: Whether to preserve complex numbers as dict
        max_depth: Maximum recursion depth (security)
        max_size: Maximum collection size (security)
        max_string_length: Maximum string length (security)
        custom_serializers: Dict of type -> serializer function
        sort_keys: Whether to sort dictionary keys in output
        ensure_ascii: Whether to ensure ASCII output only
    """

    # Date/time formatting
    date_format: DateFormat = DateFormat.ISO
    custom_date_format: Optional[str] = None

    # DataFrame formatting
    dataframe_orient: DataFrameOrient = DataFrameOrient.RECORDS

    # Value handling
    nan_handling: NanHandling = NanHandling.NULL
    type_coercion: TypeCoercion = TypeCoercion.SAFE

    # Precision control
    preserve_decimals: bool = True
    preserve_complex: bool = True

    # Security limits
    max_depth: int = 1000
    max_size: int = 10_000_000
    max_string_length: int = 1_000_000

    # Extensibility
    custom_serializers: Optional[Dict[type, Callable[[Any], Any]]] = None

    # Output formatting
    sort_keys: bool = False
    ensure_ascii: bool = False


# Global default configuration
_default_config = SerializationConfig()


def get_default_config() -> SerializationConfig:
    """Get the global default configuration."""
    return _default_config


def set_default_config(config: SerializationConfig) -> None:
    """Set the global default configuration."""
    global _default_config  # noqa: PLW0603
    _default_config = config


def reset_default_config() -> None:
    """Reset the global configuration to defaults."""
    global _default_config  # noqa: PLW0603
    _default_config = SerializationConfig()


# Preset configurations for common use cases
def get_ml_config() -> SerializationConfig:
    """Get configuration optimized for ML workflows.

    Returns:
        Configuration with aggressive type coercion and tensor-friendly settings
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX_MS,
        dataframe_orient=DataFrameOrient.RECORDS,
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.AGGRESSIVE,
        preserve_decimals=False,  # ML often doesn't need exact decimal precision
        preserve_complex=False,  # ML typically converts complex to real
        sort_keys=True,  # Consistent output for ML pipelines
    )


def get_api_config() -> SerializationConfig:
    """Get configuration optimized for API responses.

    Returns:
        Configuration with clean, consistent output for web APIs
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,
        dataframe_orient=DataFrameOrient.RECORDS,
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.SAFE,
        preserve_decimals=True,
        preserve_complex=True,
        sort_keys=True,
        ensure_ascii=True,  # Safe for all HTTP clients
    )


def get_strict_config() -> SerializationConfig:
    """Get configuration with strict type checking.

    Returns:
        Configuration that raises errors on unknown types
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,
        dataframe_orient=DataFrameOrient.RECORDS,
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.STRICT,
        preserve_decimals=True,
        preserve_complex=True,
    )


def get_performance_config() -> SerializationConfig:
    """Get configuration optimized for performance.

    Returns:
        Configuration with minimal processing for maximum speed
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX,  # Fastest date format
        dataframe_orient=DataFrameOrient.VALUES,  # Fastest DataFrame format
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.SAFE,
        preserve_decimals=False,  # Skip decimal preservation for speed
        preserve_complex=False,  # Skip complex preservation for speed
        sort_keys=False,  # Don't sort for speed
    )
