"""Configuration module for datason serialization behavior.

This module provides configuration classes and options to customize how
datason serializes different data types. Users can configure:

- Date/time output formats
- NaN/null value handling
- Pandas DataFrame orientations
- Type coercion behavior
- Recursion and size limits
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional


class DateFormat(Enum):
    """Supported date/time output formats."""

    ISO = "iso"  # ISO 8601 format (default)
    UNIX = "unix"  # Unix timestamp
    UNIX_MS = "unix_ms"  # Unix timestamp in milliseconds
    STRING = "string"  # Human readable string
    CUSTOM = "custom"  # Custom format string


class DataFrameOrient(Enum):
    """Supported pandas DataFrame orientations.

    Based on pandas.DataFrame.to_dict() valid orientations.
    """

    RECORDS = "records"  # List of records [{col: val}, ...]
    SPLIT = "split"  # Split into {index: [...], columns: [...], data: [...]}
    INDEX = "index"  # Dict like {index -> {column -> value}}
    DICT = "dict"  # Dict like {column -> {index -> value}} (pandas default)
    LIST = "list"  # Dict like {column -> [values]}
    SERIES = "series"  # Dict like {column -> Series(values)}
    TIGHT = "tight"  # Tight format with index/columns/data
    VALUES = "values"  # Just the values array


class OutputType(Enum):
    """How to output different data types."""

    JSON_SAFE = "json_safe"  # Convert to JSON-safe primitives (default)
    OBJECT = "object"  # Keep as Python objects


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


class CacheScope(Enum):
    """Cache scope behavior for performance optimization."""

    OPERATION = "operation"  # Cache cleared after each serialize/deserialize call (default, safest)
    REQUEST = "request"  # Cache persists within a single request/context (context-local)
    PROCESS = "process"  # Cache persists for the entire process (global, fastest but potential cross-contamination)
    DISABLED = "disabled"  # No caching at all (slowest but most predictable)


@dataclass
class SerializationConfig:
    """Configuration for datason serialization behavior.

    Attributes:
        date_format: How to format datetime objects
        custom_date_format: Custom strftime format when date_format is CUSTOM
        dataframe_orient: Pandas DataFrame orientation
        datetime_output: How to output datetime objects
        series_output: How to output pandas Series
        dataframe_output: How to output pandas DataFrames (overrides orient for object output)
        numpy_output: How to output numpy arrays
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
        check_if_serialized: Skip processing if object is already JSON-safe
        include_type_hints: Include type metadata for perfect round-trip deserialization
        redact_fields: Field patterns to redact (e.g., ["password", "api_key", "*.secret"])
        redact_patterns: Regex patterns to redact (e.g., credit card numbers)
        redact_large_objects: Auto-redact objects >10MB
        redaction_replacement: Replacement text for redacted content
        include_redaction_summary: Include summary of what was redacted
        audit_trail: Track all redaction operations for compliance
    """

    # Date/time formatting
    date_format: DateFormat = DateFormat.ISO
    custom_date_format: Optional[str] = None

    # NEW: UUID handling configuration (addressing FastAPI/Pydantic integration feedback)
    uuid_format: str = "object"  # "object" (uuid.UUID) or "string" (str)
    parse_uuids: bool = True  # Whether to auto-convert UUID strings to UUID objects

    # DataFrame formatting
    dataframe_orient: DataFrameOrient = DataFrameOrient.RECORDS

    # NEW: Output type control (addressing user feedback)
    datetime_output: OutputType = OutputType.JSON_SAFE
    series_output: OutputType = OutputType.JSON_SAFE
    dataframe_output: OutputType = OutputType.JSON_SAFE
    numpy_output: OutputType = OutputType.JSON_SAFE

    # Value handling
    nan_handling: NanHandling = NanHandling.NULL
    type_coercion: TypeCoercion = TypeCoercion.SAFE

    # Precision control
    preserve_decimals: bool = True
    preserve_complex: bool = True

    # Security limits
    max_depth: int = 50  # SECURITY FIX: Changed from 1000 to 50 to match MAX_SERIALIZATION_DEPTH
    max_size: int = 100_000  # SECURITY FIX: Reduced from 10_000_000 to 100_000 to prevent size bomb attacks
    max_string_length: int = 1_000_000

    # Extensibility
    custom_serializers: Optional[Dict[type, Callable[[Any], Any]]] = None

    # Output formatting
    sort_keys: bool = False
    ensure_ascii: bool = False

    # NEW: Performance optimization (addressing user feedback)
    check_if_serialized: bool = False

    # NEW: Type metadata for round-trip serialization
    include_type_hints: bool = False

    # NEW: Auto-detection of complex types (experimental)
    auto_detect_types: bool = False

    # NEW: Production Safety & Redaction (v0.5.5)
    redact_fields: Optional[List[str]] = None  # Field patterns to redact (e.g., ["password", "api_key", "*.secret"])
    redact_patterns: Optional[List[str]] = None  # Regex patterns to redact (e.g., credit card numbers)
    redact_large_objects: bool = False  # Auto-redact objects >10MB
    redaction_replacement: str = "<REDACTED>"  # Replacement text for redacted content
    include_redaction_summary: bool = False  # Include summary of what was redacted
    audit_trail: bool = False  # Track all redaction operations for compliance

    # NEW: Configurable Caching System
    cache_scope: CacheScope = CacheScope.OPERATION  # Default to operation-scoped for safety
    cache_size_limit: int = 1000  # Maximum number of entries per cache type
    cache_warn_on_limit: bool = True  # Warn when cache reaches size limit
    cache_metrics_enabled: bool = False  # Enable cache hit/miss metrics collection


# Global default configuration
_default_config = SerializationConfig()

# Context variables for request-scoped caching
_cache_scope_context: ContextVar[CacheScope] = ContextVar("cache_scope", default=CacheScope.OPERATION)


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
        include_type_hints=True,  # Enable type metadata for ML objects
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
        # NEW: Keep UUIDs as strings for API compatibility (Pydantic/FastAPI)
        uuid_format="string",
        parse_uuids=False,
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


# NEW: Cache Management Functions and Context Manager


def get_cache_scope() -> CacheScope:
    """Get the current cache scope from context or default config."""
    try:
        return _cache_scope_context.get()
    except LookupError:
        return _default_config.cache_scope


def set_cache_scope(scope: CacheScope) -> None:
    """Set the cache scope in the current context."""
    _cache_scope_context.set(scope)


@contextmanager
def cache_scope(scope: CacheScope) -> Generator[None, None, None]:
    """Context manager for temporarily setting cache scope.

    Args:
        scope: The cache scope to use within this context

    Example:
        >>> with cache_scope(CacheScope.PROCESS):
        ...     # All serialize/deserialize operations in this block
        ...     # will use process-level caching
        ...     result = serialize(data)
        # Cache automatically managed according to the specified scope
    """
    # Clear existing caches when entering different scope
    try:
        from . import deserializers

        if hasattr(deserializers, "clear_caches"):
            deserializers.clear_caches()
    except (AttributeError, ImportError):
        # Ignore cache clearing errors during scope changes
        pass  # nosec B110 - intentional fallback for cache clearing

    # Set new scope
    token = _cache_scope_context.set(scope)
    try:
        yield
    finally:
        # Restore previous scope and clear if needed
        _cache_scope_context.reset(token)
        if scope == CacheScope.OPERATION:
            # Clear caches when exiting operation scope
            try:
                if hasattr(deserializers, "clear_caches"):
                    deserializers.clear_caches()
            except (AttributeError, ImportError):
                # Ignore cache clearing errors during scope exit
                pass  # nosec B110 - intentional fallback for cache clearing
