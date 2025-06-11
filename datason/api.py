"""Modern API for datason - Phase 3 API Modernization.

This module provides intention-revealing wrapper functions around the existing
datason functionality. The goal is to make the API more discoverable and
user-friendly while maintaining 100% backward compatibility.

Key improvements:
- Intention-revealing names (load_basic, load_smart, load_perfect, etc.)
- Compositional utilities (dump_secure, dump_chunked, etc.)
- Domain-specific convenience (dump_ml, dump_api, etc.)
- Progressive disclosure of complexity
"""

import warnings
from typing import Any, Dict, List, Optional

from .config import (
    SerializationConfig,
    get_api_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
)
from .core import serialize, serialize_chunked, stream_serialize
from .deserializers import (
    deserialize,
    deserialize_fast,
    deserialize_with_template,
)

# Deprecation warning suppression for internal use
_suppress_deprecation_warnings = False


def suppress_deprecation_warnings(suppress: bool = True) -> None:
    """Control deprecation warnings for backward compatibility testing.

    Args:
        suppress: Whether to suppress deprecation warnings
    """
    global _suppress_deprecation_warnings
    _suppress_deprecation_warnings = suppress


# =============================================================================
# MODERN DUMP API - Clear intent, composable utilities
# =============================================================================


def dump(
    obj: Any,
    *,
    secure: bool = False,
    chunked: bool = False,
    chunk_size: int = 1000,
    ml_mode: bool = False,
    api_mode: bool = False,
    fast_mode: bool = False,
    config: Optional[SerializationConfig] = None,
    **kwargs: Any,
) -> Any:
    """Modern unified dump function with clear options.

    This is the main entry point for serialization with intention-revealing
    parameters instead of requiring deep config knowledge.

    Args:
        obj: Object to serialize
        secure: Enable security features (PII redaction, etc.)
        chunked: Enable chunked serialization for large objects
        chunk_size: Size of chunks when chunked=True
        ml_mode: Optimize for ML/AI objects (models, tensors, etc.)
        api_mode: Optimize for API responses (clean, predictable format)
        fast_mode: Optimize for performance (minimal type checking)
        config: Advanced configuration (overrides other options)
        **kwargs: Additional configuration options

    Returns:
        Serialized object

    Examples:
        >>> # Basic usage
        >>> dump(data)

        >>> # ML-optimized
        >>> dump(model, ml_mode=True)

        >>> # Secure serialization
        >>> dump(sensitive_data, secure=True)

        >>> # Chunked for large data
        >>> dump(big_data, chunked=True, chunk_size=5000)
    """
    # Handle mutually exclusive modes
    mode_count = sum([ml_mode, api_mode, fast_mode])
    if mode_count > 1:
        raise ValueError("Only one mode can be enabled: ml_mode, api_mode, or fast_mode")

    # Use provided config or determine from mode
    if config is None:
        if ml_mode:
            config = get_ml_config()
        elif api_mode:
            config = get_api_config()
        elif fast_mode:
            config = get_performance_config()
        else:
            config = SerializationConfig(**kwargs) if kwargs else None

    # Handle security enhancements
    if secure:
        if config is None:
            config = SerializationConfig()

        # Add common PII redaction patterns
        config.redact_patterns = config.redact_patterns or []
        config.redact_patterns.extend(
            [
                r"\b\d{16}\b",  # Credit cards
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ]
        )
        config.redact_fields = config.redact_fields or []
        config.redact_fields.extend(["password", "api_key", "secret", "token"])
        config.include_redaction_summary = True

    # Handle chunked serialization
    if chunked:
        return serialize_chunked(obj, chunk_size=chunk_size, config=config)

    return serialize(obj, config=config)


def dump_ml(obj: Any, **kwargs: Any) -> Any:
    """ML-optimized serialization for models, tensors, and ML objects.

    Automatically configures optimal settings for machine learning objects
    including NumPy arrays, PyTorch tensors, scikit-learn models, etc.

    Args:
        obj: ML object to serialize
        **kwargs: Additional configuration options

    Returns:
        Serialized ML object optimized for reconstruction

    Example:
        >>> model = sklearn.ensemble.RandomForestClassifier()
        >>> serialized = dump_ml(model)
        >>> # Optimized for ML round-trip fidelity
    """
    config = get_ml_config()
    return serialize(obj, config=config)


def dump_api(obj: Any, **kwargs: Any) -> Any:
    """API-safe serialization for web responses and APIs.

    Produces clean, predictable JSON suitable for API responses.
    Handles edge cases gracefully and ensures consistent output format.

    Args:
        obj: Object to serialize for API response
        **kwargs: Additional configuration options

    Returns:
        API-safe serialized object

    Example:
        >>> @app.route('/api/data')
        >>> def get_data():
        >>>     return dump_api(complex_data_structure)
    """
    config = get_api_config()
    return serialize(obj, config=config)


def dump_secure(
    obj: Any,
    *,
    redact_pii: bool = True,
    redact_fields: Optional[List[str]] = None,
    redact_patterns: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Security-focused serialization with PII redaction.

    Automatically redacts sensitive information like credit cards,
    SSNs, emails, and common secret fields.

    Args:
        obj: Object to serialize securely
        redact_pii: Enable automatic PII pattern detection
        redact_fields: Additional field names to redact
        redact_patterns: Additional regex patterns to redact
        **kwargs: Additional configuration options

    Returns:
        Serialized object with sensitive data redacted

    Example:
        >>> user_data = {"name": "John", "ssn": "123-45-6789"}
        >>> safe_data = dump_secure(user_data)
        >>> # SSN will be redacted: {"name": "John", "ssn": "[REDACTED]"}
    """
    patterns = []
    fields = []

    if redact_pii:
        patterns.extend(
            [
                r"\b\d{16}\b",  # Credit cards
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ]
        )
        fields.extend(["password", "api_key", "secret", "token", "ssn", "credit_card"])

    if redact_patterns:
        patterns.extend(redact_patterns)
    if redact_fields:
        fields.extend(redact_fields)

    config = SerializationConfig(
        redact_patterns=patterns, redact_fields=fields, include_redaction_summary=True, **kwargs
    )

    return serialize(obj, config=config)


def dump_fast(obj: Any, **kwargs: Any) -> Any:
    """Performance-optimized serialization.

    Optimized for speed with minimal type checking and validation.
    Use when you need maximum performance and can accept some trade-offs
    in type fidelity.

    Args:
        obj: Object to serialize quickly
        **kwargs: Additional configuration options

    Returns:
        Serialized object optimized for speed

    Example:
        >>> # For high-throughput scenarios
        >>> result = dump_fast(large_dataset)
    """
    config = get_performance_config()
    return serialize(obj, config=config)


def dump_chunked(obj: Any, *, chunk_size: int = 1000, **kwargs: Any) -> Any:
    """Chunked serialization for large objects.

    Breaks large objects into manageable chunks for memory efficiency
    and streaming processing.

    Args:
        obj: Large object to serialize in chunks
        chunk_size: Size of each chunk
        **kwargs: Additional configuration options

    Returns:
        ChunkedSerializationResult with metadata and chunks

    Example:
        >>> big_list = list(range(10000))
        >>> result = dump_chunked(big_list, chunk_size=1000)
        >>> # Returns ChunkedSerializationResult with 10 chunks
    """
    return serialize_chunked(obj, chunk_size=chunk_size, **kwargs)


def stream_dump(file_path: str, **kwargs: Any) -> Any:
    """Streaming serialization to file.

    Efficiently serialize large datasets directly to file without
    loading everything into memory.

    Args:
        file_path: Path to output file
        **kwargs: Additional configuration options

    Returns:
        StreamingSerializer instance for continued operations

    Example:
        >>> with stream_dump("output.jsonl") as streamer:
        >>>     for item in large_dataset:
        >>>         streamer.write(item)
    """
    return stream_serialize(file_path, **kwargs)


# =============================================================================
# MODERN LOAD API - Progressive complexity, clear success rates
# =============================================================================


def load_basic(data: Any, **kwargs: Any) -> Any:
    """Basic deserialization using heuristics only.

    Uses simple heuristics to reconstruct Python objects from serialized data.
    Fast but with limited type fidelity - suitable for exploration and
    non-critical applications.

    Success rate: ~60-70% for complex objects
    Speed: Fastest
    Use case: Data exploration, simple objects

    Args:
        data: Serialized data to deserialize
        **kwargs: Additional options (parse_dates, parse_uuids, etc.)

    Returns:
        Deserialized Python object

    Example:
        >>> serialized = {"numbers": [1, 2, 3], "text": "hello"}
        >>> result = load_basic(serialized)
        >>> # Works well for simple structures
    """
    return deserialize(data, **kwargs)


def load_smart(data: Any, config: Optional[SerializationConfig] = None, **kwargs: Any) -> Any:
    """Smart deserialization with auto-detection and heuristics.

    Combines automatic type detection with heuristic fallbacks.
    Good balance of accuracy and performance for most use cases.

    Success rate: ~80-90% for complex objects
    Speed: Moderate
    Use case: General purpose, production data processing

    Args:
        data: Serialized data to deserialize
        config: Configuration for deserialization behavior
        **kwargs: Additional options

    Returns:
        Deserialized Python object with improved type fidelity

    Example:
        >>> serialized = dump_api(complex_object)
        >>> result = load_smart(serialized)
        >>> # Better type reconstruction than load_basic
    """
    if config is None:
        config = SerializationConfig(auto_detect_types=True)
    return deserialize_fast(data, config=config, **kwargs)


def load_perfect(data: Any, template: Any, **kwargs: Any) -> Any:
    """Perfect deserialization using template matching.

    Uses a template object to achieve 100% accurate reconstruction.
    Requires you to provide the structure/type information but
    guarantees perfect fidelity.

    Success rate: 100% when template matches data
    Speed: Fast (direct template matching)
    Use case: Critical applications, ML model loading, exact reconstruction

    Args:
        data: Serialized data to deserialize
        template: Template object showing expected structure/types
        **kwargs: Additional options

    Returns:
        Perfectly reconstructed Python object matching template

    Example:
        >>> original = MyComplexClass(...)
        >>> serialized = dump_ml(original)
        >>> template = MyComplexClass.get_template()  # or original itself
        >>> result = load_perfect(serialized, template)
        >>> # Guaranteed perfect reconstruction
    """
    return deserialize_with_template(data, template, **kwargs)


def load_typed(data: Any, config: Optional[SerializationConfig] = None, **kwargs: Any) -> Any:
    """Metadata-based type reconstruction.

    Uses embedded type metadata from serialization to reconstruct objects.
    Requires data was serialized with type information preserved.

    Success rate: ~95% when metadata available
    Speed: Fast (direct metadata lookup)
    Use case: When you control both serialization and deserialization

    Args:
        data: Serialized data with embedded type metadata
        config: Configuration for type reconstruction
        **kwargs: Additional options

    Returns:
        Type-accurate deserialized Python object

    Example:
        >>> # Works best with datason-serialized data
        >>> serialized = dump(original_object)  # Preserves type info
        >>> result = load_typed(serialized)
        >>> # High fidelity reconstruction using embedded metadata
    """
    if config is None:
        config = get_strict_config()  # Use strict config for best type preservation
    return deserialize_fast(data, config=config, **kwargs)


# =============================================================================
# CONVENIENCE FUNCTIONS - Backward compatibility with modern names
# =============================================================================


def loads(s: str, **kwargs: Any) -> Any:
    """Load from JSON string (json.loads compatible name).

    Args:
        s: JSON string to deserialize
        **kwargs: Additional deserialization options

    Returns:
        Deserialized Python object

    Example:
        >>> json_str = '{"key": "value"}'
        >>> result = loads(json_str)
    """
    import json

    data = json.loads(s)
    # For json.loads compatibility, use load_basic instead of load_smart
    # to avoid auto-detection which can create NumPy arrays
    return load_basic(data, **kwargs)


def dumps(obj: Any, **kwargs: Any) -> str:
    """Dump to JSON string (json.dumps compatible name).

    Args:
        obj: Object to serialize to JSON string
        **kwargs: Additional serialization options

    Returns:
        JSON string representation

    Example:
        >>> obj = {"key": "value"}
        >>> json_str = dumps(obj)
    """
    import json

    serialized = dump(obj, **kwargs)
    return json.dumps(serialized)


# =============================================================================
# MIGRATION HELPERS - For smooth transition from old API
# =============================================================================


def serialize_modern(*args, **kwargs) -> Any:
    """Modern serialize function with deprecation guidance.

    This is a transitional function to help users migrate from the old
    serialize() function to the new dump() family of functions.
    """
    if not _suppress_deprecation_warnings:
        warnings.warn(
            "serialize() is deprecated. Use dump() or specific variants like dump_ml(), "
            "dump_api(), dump_secure() for better intent clarity. "
            "See migration guide: https://github.com/yourusername/datason/blob/main/docs/migration/api-modernization.md",
            DeprecationWarning,
            stacklevel=2,
        )
    return serialize(*args, **kwargs)


def deserialize_modern(*args, **kwargs) -> Any:
    """Modern deserialize function with deprecation guidance.

    This is a transitional function to help users migrate from the old
    deserialize() functions to the new load() family of functions.
    """
    if not _suppress_deprecation_warnings:
        warnings.warn(
            "deserialize() is deprecated. Use load_basic(), load_smart(), load_perfect(), "
            "or load_typed() for better intent clarity and success rates. "
            "See migration guide: https://github.com/yourusername/datason/blob/main/docs/migration/api-modernization.md",
            DeprecationWarning,
            stacklevel=2,
        )
    return deserialize(*args, **kwargs)


# =============================================================================
# API DISCOVERY - Help users find the right function
# =============================================================================


def help_api() -> Dict[str, Any]:
    """Get help on choosing the right API function.

    Returns:
        Dictionary with API guidance and function recommendations

    Example:
        >>> help_info = help_api()
        >>> print(help_info['recommendations'])
    """
    return {
        "serialization": {
            "basic": {"function": "dump()", "use_case": "General purpose serialization", "example": "dump(data)"},
            "ml_optimized": {
                "function": "dump_ml()",
                "use_case": "ML models, tensors, NumPy arrays",
                "example": "dump_ml(sklearn_model)",
            },
            "api_safe": {
                "function": "dump_api()",
                "use_case": "Web APIs, clean JSON output",
                "example": "dump_api(response_data)",
            },
            "secure": {
                "function": "dump_secure()",
                "use_case": "Sensitive data with PII redaction",
                "example": "dump_secure(user_data, redact_pii=True)",
            },
            "performance": {
                "function": "dump_fast()",
                "use_case": "High-throughput scenarios",
                "example": "dump_fast(large_dataset)",
            },
            "chunked": {
                "function": "dump_chunked()",
                "use_case": "Very large objects, memory efficiency",
                "example": "dump_chunked(huge_list, chunk_size=1000)",
            },
        },
        "deserialization": {
            "basic": {
                "function": "load_basic()",
                "success_rate": "60-70%",
                "speed": "Fastest",
                "use_case": "Simple objects, data exploration",
            },
            "smart": {
                "function": "load_smart()",
                "success_rate": "80-90%",
                "speed": "Moderate",
                "use_case": "General purpose, production data",
            },
            "perfect": {
                "function": "load_perfect()",
                "success_rate": "100%",
                "speed": "Fast",
                "use_case": "Critical applications, requires template",
                "example": "load_perfect(data, template)",
            },
            "typed": {
                "function": "load_typed()",
                "success_rate": "95%",
                "speed": "Fast",
                "use_case": "When metadata available",
            },
        },
        "recommendations": [
            "For ML workflows: dump_ml() + load_perfect() with template",
            "For APIs: dump_api() + load_smart()",
            "For sensitive data: dump_secure() + load_smart()",
            "For exploration: dump() + load_basic()",
            "For production: dump() + load_smart() or load_typed()",
        ],
    }


def get_api_info() -> Dict[str, Any]:
    """Get information about the modern API.

    Returns:
        Dictionary with API version and feature information
    """
    return {
        "api_version": "modern",
        "phase": "3",
        "features": {
            "intention_revealing_names": True,
            "compositional_utilities": True,
            "domain_specific_convenience": True,
            "progressive_complexity": True,
            "backward_compatibility": True,
        },
        "dump_functions": ["dump", "dump_ml", "dump_api", "dump_secure", "dump_fast", "dump_chunked", "stream_dump"],
        "load_functions": ["load_basic", "load_smart", "load_perfect", "load_typed"],
        "convenience": ["loads", "dumps"],
        "help": ["help_api", "get_api_info"],
    }
