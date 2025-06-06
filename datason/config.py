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


def get_financial_config() -> SerializationConfig:
    """Get configuration optimized for financial ML workflows.

    Returns:
        Configuration with precise decimal handling and timestamp consistency
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX_MS,  # Precise timestamps for trading
        dataframe_orient=DataFrameOrient.RECORDS,  # Standard format for financial data
        nan_handling=NanHandling.NULL,  # Clean handling of missing market data
        type_coercion=TypeCoercion.SAFE,  # Preserve financial precision
        preserve_decimals=True,  # Critical for monetary values
        preserve_complex=False,  # Financial data typically real-valued
        sort_keys=True,  # Consistent output for financial reports
        ensure_ascii=True,  # Safe for financial system integration
        check_if_serialized=True,  # Performance for high-frequency data
    )


def get_time_series_config() -> SerializationConfig:
    """Get configuration optimized for time series analysis workflows.

    Returns:
        Configuration optimized for temporal data and chronological ordering
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,  # Standard temporal format
        dataframe_orient=DataFrameOrient.SPLIT,  # Efficient for time series data
        nan_handling=NanHandling.NULL,  # Handle missing temporal observations
        type_coercion=TypeCoercion.SAFE,  # Preserve temporal precision
        preserve_decimals=True,  # Important for measurement precision
        preserve_complex=False,  # Time series typically real-valued
        sort_keys=True,  # Maintain temporal ordering
        datetime_output=OutputType.JSON_SAFE,  # Standardized time representation
    )


def get_inference_config() -> SerializationConfig:
    """Get configuration optimized for ML model inference workflows.

    Returns:
        Configuration with minimal overhead for production model serving
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX,  # Fast timestamp format
        dataframe_orient=DataFrameOrient.VALUES,  # Minimal overhead format
        nan_handling=NanHandling.NULL,  # Clean inference inputs
        type_coercion=TypeCoercion.AGGRESSIVE,  # Maximum inference compatibility
        preserve_decimals=False,  # Speed over precision for inference
        preserve_complex=False,  # Inference typically real-valued
        sort_keys=False,  # Skip sorting for speed
        check_if_serialized=True,  # Maximum performance
        include_type_hints=False,  # Minimal metadata for speed
    )


def get_research_config() -> SerializationConfig:
    """Get configuration optimized for research and experimentation workflows.

    Returns:
        Configuration that preserves maximum information for research reproducibility
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,  # Human-readable timestamps
        dataframe_orient=DataFrameOrient.RECORDS,  # Standard research format
        nan_handling=NanHandling.NULL,  # Clean research data
        type_coercion=TypeCoercion.SAFE,  # Preserve research data fidelity
        preserve_decimals=True,  # Maintain precision for analysis
        preserve_complex=True,  # Keep complex numbers for research
        sort_keys=True,  # Consistent output for reproducibility
        include_type_hints=True,  # Maximum metadata for reproducibility
    )


def get_logging_config() -> SerializationConfig:
    """Get configuration optimized for production logging workflows.

    Returns:
        Configuration that is safe and efficient for production logging
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,  # Standard log timestamp format
        dataframe_orient=DataFrameOrient.RECORDS,  # Readable log format
        nan_handling=NanHandling.STRING,  # Explicit NaN representation in logs
        type_coercion=TypeCoercion.SAFE,  # Safe logging without errors
        preserve_decimals=False,  # Simplified logging format
        preserve_complex=False,  # Keep logs simple
        sort_keys=True,  # Consistent log structure
        ensure_ascii=True,  # Safe for all logging systems
        max_string_length=1000,  # Prevent log bloat
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


# Preset configurations with cache-aware settings


def get_batch_processing_config() -> SerializationConfig:
    """Get configuration optimized for batch processing workflows.

    Returns:
        Configuration with process-level caching for maximum performance
        in homogeneous batch workloads
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX,  # Fast format for batch processing
        dataframe_orient=DataFrameOrient.VALUES,  # Efficient format
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.AGGRESSIVE,
        preserve_decimals=False,  # Speed over precision
        preserve_complex=False,
        sort_keys=False,  # Skip sorting for speed
        check_if_serialized=True,  # Maximum performance
        cache_scope=CacheScope.PROCESS,  # Use process-level caching
        cache_size_limit=5000,  # Larger cache for batch processing
        cache_warn_on_limit=True,
        cache_metrics_enabled=True,  # Monitor cache performance
    )


def get_web_api_config() -> SerializationConfig:
    """Get configuration optimized for web API workflows.

    Returns:
        Configuration with request-scoped caching for multi-tenant safety
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,
        dataframe_orient=DataFrameOrient.RECORDS,
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.SAFE,
        preserve_decimals=True,
        preserve_complex=True,
        sort_keys=True,
        ensure_ascii=True,
        cache_scope=CacheScope.REQUEST,  # Safe for multi-tenant applications
        cache_size_limit=1000,  # Moderate cache size
        cache_warn_on_limit=True,
        cache_metrics_enabled=False,  # Reduce overhead in API contexts
    )


def get_realtime_config() -> SerializationConfig:
    """Get configuration optimized for real-time/streaming workflows.

    Returns:
        Configuration with operation-scoped caching for maximum predictability
    """
    return SerializationConfig(
        date_format=DateFormat.UNIX,  # Fast timestamps
        dataframe_orient=DataFrameOrient.VALUES,  # Minimal overhead
        nan_handling=NanHandling.NULL,
        type_coercion=TypeCoercion.SAFE,
        preserve_decimals=False,  # Speed over precision
        preserve_complex=False,
        sort_keys=False,  # No sorting for speed
        check_if_serialized=True,
        cache_scope=CacheScope.OPERATION,  # Most predictable for real-time
        cache_size_limit=500,  # Small cache to prevent latency spikes
        cache_warn_on_limit=False,  # Avoid warnings in real-time contexts
        cache_metrics_enabled=False,  # Minimal overhead
    )


def get_development_config() -> SerializationConfig:
    """Get configuration optimized for development and debugging.

    Returns:
        Configuration with caching disabled for maximum predictability during development
    """
    return SerializationConfig(
        date_format=DateFormat.ISO,  # Human-readable
        dataframe_orient=DataFrameOrient.RECORDS,  # Easy to inspect
        nan_handling=NanHandling.STRING,  # Explicit in output
        type_coercion=TypeCoercion.SAFE,
        preserve_decimals=True,  # Preserve all information
        preserve_complex=True,
        sort_keys=True,  # Consistent output for debugging
        ensure_ascii=False,  # Allow unicode for debugging
        include_type_hints=True,  # Maximum information
        cache_scope=CacheScope.DISABLED,  # No caching for predictable behavior
        cache_warn_on_limit=True,
        cache_metrics_enabled=True,  # Useful for development insights
    )
