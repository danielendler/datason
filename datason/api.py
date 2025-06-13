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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .config import (
    SerializationConfig,
    get_api_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
)
from .core_new import (
    deserialize_chunked_file,
    serialize,
    serialize_chunked,
    stream_serialize,
)
from .deserializers_new import (
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
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit cards with dashes
                r"\b\d{16}\b",  # Credit cards without dashes
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
    from copy import deepcopy

    # Make a deep copy to avoid circular reference issues
    try:
        obj_copy = deepcopy(obj)
    except Exception:
        # If deepcopy fails (some objects can't be copied), use the original
        obj_copy = obj

    # Use ML-optimized config
    config = get_ml_config()

    # Add extra config settings from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Use higher max_depth to avoid circular reference warnings
    config.max_depth = 10000

    # Directly call serialize, avoiding any wrappers that might add circular refs
    from .core_new import serialize

    return serialize(obj_copy, config=config)


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
    from copy import deepcopy

    # Make a deep copy to avoid circular reference issues
    try:
        obj_copy = deepcopy(obj)
    except Exception:
        # If deepcopy fails (some objects can't be copied), use the original
        obj_copy = obj

    # Use API-optimized config
    config = get_api_config()

    # Add extra config settings from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Use higher max_depth to avoid circular reference warnings
    config.max_depth = 10000

    # Directly call serialize, avoiding any wrappers that might add circular refs
    from .core_new import serialize

    return serialize(obj_copy, config=config)


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
    from copy import deepcopy

    # Make a deep copy to avoid circular reference issues
    try:
        obj_copy = deepcopy(obj)
    except Exception:
        # If deepcopy fails (some objects can't be copied), use the original
        obj_copy = obj

    # Create secure config with redaction settings
    patterns = []
    fields = []

    if redact_pii:
        patterns.extend(
            [
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit cards with dashes
                r"\b\d{16}\b",  # Credit cards without dashes
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ]
        )
        fields.extend(["password", "api_key", "secret", "token", "ssn", "credit_card"])

    if redact_patterns:
        patterns.extend(redact_patterns)
    if redact_fields:
        fields.extend(redact_fields)

    # Remove include_redaction_summary from kwargs if present to avoid duplicate
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "include_redaction_summary"}

    config = SerializationConfig(
        redact_patterns=patterns,
        redact_fields=fields,
        include_redaction_summary=True,
        max_depth=10000,  # High max_depth to avoid circular reference warnings
        **kwargs_clean,
    )

    # Directly call serialize, avoiding any wrappers that might add circular refs
    from .core_new import serialize

    return serialize(obj_copy, config=config)


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


def serialize_modern(*args: Any, **kwargs: Any) -> Any:
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


def deserialize_modern(*args: Any, **kwargs: Any) -> Any:
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
        "file_operations": {
            "save_ml": {
                "function": "save_ml()",
                "use_case": "ML models/data to JSON/JSONL files",
                "examples": [
                    "save_ml(model, 'model.json')    # Single JSON object",
                    "save_ml(model, 'model.jsonl')   # Multiple JSONL objects",
                    "save_ml(model, 'model.txt', format='json')  # Explicit format",
                ],
            },
            "save_secure": {
                "function": "save_secure()",
                "use_case": "Secure JSON/JSONL with redaction",
                "examples": [
                    "save_secure(data, 'secure.json', redact_pii=True)",
                    "save_secure(data, 'secure.jsonl', redact_pii=True)",
                ],
            },
            "load_file": {
                "function": "load_smart_file()",
                "use_case": "Smart loading from JSON/JSONL files",
                "examples": [
                    "list(load_smart_file('data.json'))",
                    "list(load_smart_file('data.jsonl'))",
                    "list(load_smart_file('data.txt', format='json'))",
                ],
            },
        },
        "recommendations": [
            "For ML workflows: save_ml() + load_perfect_file() with template",
            "For APIs: save_api() + load_smart_file()",
            "For sensitive data: save_secure() + load_smart_file()",
            "For exploration: dump() + load_basic()",
            "For production: save_ml() + load_smart_file()",
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
            "file_operations": True,
        },
        "dump_functions": ["dump", "dump_ml", "dump_api", "dump_secure", "dump_fast", "dump_chunked", "stream_dump"],
        "load_functions": ["load_basic", "load_smart", "load_perfect", "load_typed"],
        "file_functions": ["save_ml", "save_secure", "save_api", "load_smart_file", "load_perfect_file"],
        "convenience": ["loads", "dumps"],
        "help": ["help_api", "get_api_info"],
    }


# =============================================================================
# FILE I/O OPERATIONS - Modern API with JSONL Integration
# =============================================================================


def _detect_file_format(path: Union[str, Path], format: Optional[str] = None) -> str:
    """Detect file format from extension or explicit parameter."""
    if format:
        return format

    path_obj = Path(path)
    suffixes = path_obj.suffixes

    # Handle .json.gz, .jsonl.gz etc.
    if suffixes and suffixes[-1] == ".gz" and len(suffixes) > 1:
        ext = suffixes[-2]
    elif suffixes:
        ext = suffixes[-1]
    else:
        ext = ""

    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    elif ext == ".json":
        return "json"
    else:
        # Default to jsonl for compatibility
        return "jsonl"


def _save_to_file(
    serialized_data: Any,
    path: Union[str, Path],
    config: Optional[SerializationConfig] = None,
    format: Optional[str] = None,
) -> None:
    """Core file writing utility supporting both JSON and JSONL formats."""
    detected_format = _detect_file_format(path, format)

    with stream_serialize(path, config=config, format=detected_format) as stream:
        if isinstance(serialized_data, (list, tuple)):
            for item in serialized_data:
                stream.write(item)
        else:
            stream.write(serialized_data)


def _load_from_file(
    path: Union[str, Path], config: Optional[SerializationConfig] = None, format: Optional[str] = None
) -> Iterator[Any]:
    """Core file reading utility supporting both JSON and JSONL formats."""
    detected_format = _detect_file_format(path, format)
    return deserialize_chunked_file(path, format=detected_format)


def save_ml(obj: Any, path: Union[str, Path], *, format: Optional[str] = None, **kwargs: Any) -> None:
    """Save ML-optimized data to JSON or JSONL file.

    Combines ML-specific serialization with file I/O, preserving
    ML types like NumPy arrays, PyTorch tensors, etc.

    Args:
        obj: ML object or data to save
        path: Output file path (.json for single object, .jsonl for multiple objects)
        format: Explicit format ('json' or 'jsonl'), auto-detected from extension if None
        **kwargs: Additional ML configuration options

    Examples:
        >>> import numpy as np
        >>> data = [{"weights": np.array([1, 2, 3]), "epoch": 1}]
        >>>
        >>> # Save as JSONL (multiple objects, one per line)
        >>> save_ml(data, "training.jsonl")
        >>> save_ml(data, "training.json", format="jsonl")  # Force JSONL
        >>>
        >>> # Save as JSON (single array object)
        >>> save_ml(data, "training.json")
        >>> save_ml(data, "training.jsonl", format="json")  # Force JSON
    """
    import gzip
    import json
    from pathlib import Path

    # Get ML-optimized config
    config = get_ml_config()

    # Apply any additional config options
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Detect format
    path_obj = Path(path)
    detected_format = _detect_file_format(path_obj, format)

    # Check for compression
    is_compressed = path_obj.suffix == ".gz" or (len(path_obj.suffixes) > 1 and path_obj.suffixes[-1] == ".gz")

    # Pre-serialize the object - this already applies the ML-specific serialization
    serialized = dump_ml(obj, **kwargs)

    # Open file with appropriate compression
    def open_func(mode):
        if is_compressed:
            return gzip.open(path_obj, mode, encoding="utf-8")
        else:
            return path_obj.open(mode)

    # Write to file in appropriate format, don't re-serialize
    with open_func("wt") as f:
        if detected_format == "jsonl":
            # JSONL: Write each item on a separate line
            if isinstance(serialized, (list, tuple)):
                for item in serialized:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            else:
                json.dump(serialized, f, ensure_ascii=False)
                f.write("\n")
        else:
            # JSON: Write as single object
            json.dump(serialized, f, ensure_ascii=False)


def save_secure(
    obj: Any,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    redact_pii: bool = True,
    redact_fields: Optional[List[str]] = None,
    redact_patterns: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """Save data to JSON/JSONL file with security features.

    Automatically redacts sensitive information before saving.

    Args:
        obj: Data to save securely
        path: Output file path
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        redact_pii: Enable automatic PII pattern detection
        redact_fields: Additional field names to redact
        redact_patterns: Additional regex patterns to redact
        **kwargs: Additional security options

    Examples:
        >>> user_data = [{"name": "John", "ssn": "123-45-6789"}]
        >>>
        >>> # Save as JSONL (auto-detected)
        >>> save_secure(user_data, "users.jsonl", redact_pii=True)
        >>>
        >>> # Save as JSON (auto-detected)
        >>> save_secure(user_data, "users.json", redact_pii=True)
    """
    import gzip
    import json
    from pathlib import Path

    # Create secure config with redaction settings (same logic as dump_secure)
    patterns = []
    fields = []

    if redact_pii:
        patterns.extend(
            [
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit cards with dashes
                r"\b\d{16}\b",  # Credit cards without dashes
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ]
        )
        fields.extend(["password", "api_key", "secret", "token", "ssn", "credit_card"])

    if redact_patterns:
        patterns.extend(redact_patterns)
    if redact_fields:
        fields.extend(redact_fields)

    # Apply secure configuration
    secure_kwargs = {"redact_patterns": patterns, "redact_fields": fields, "include_redaction_summary": True, **kwargs}

    # Pre-serialize with redaction applied - this already handles the secure serialization
    secure_serialized = dump_secure(obj, **secure_kwargs)

    # Detect format
    path_obj = Path(path)
    detected_format = _detect_file_format(path_obj, format)

    # Check for compression
    is_compressed = path_obj.suffix == ".gz" or (len(path_obj.suffixes) > 1 and path_obj.suffixes[-1] == ".gz")

    # Open file with appropriate compression
    def open_func(mode):
        if is_compressed:
            return gzip.open(path_obj, mode, encoding="utf-8")
        else:
            return path_obj.open(mode)

    # Write to file in appropriate format, don't re-serialize
    with open_func("wt") as f:
        if detected_format == "jsonl":
            # JSONL: Write each item on a separate line
            if isinstance(secure_serialized, (list, tuple)):
                for item in secure_serialized:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            else:
                json.dump(secure_serialized, f, ensure_ascii=False)
                f.write("\n")
        else:
            # JSON: Write as single object
            json.dump(secure_serialized, f, ensure_ascii=False)


def save_api(obj: Any, path: Union[str, Path], *, format: Optional[str] = None, **kwargs: Any) -> None:
    """Save API-safe data to JSON/JSONL file.

    Produces clean, predictable output suitable for API data exchange.

    Args:
        obj: Data to save for API use
        path: Output file path
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: Additional API configuration options

    Examples:
        >>> api_data = [{"status": "success", "data": [1, 2, 3]}]
        >>>
        >>> # Save as single JSON object
        >>> save_api(api_data, "responses.json")
        >>>
        >>> # Save as JSONL (one response per line)
        >>> save_api(api_data, "responses.jsonl")
    """
    import gzip
    import json
    from pathlib import Path

    # Get API-optimized config and serialize the data
    api_serialized = dump_api(obj, **kwargs)

    # Detect format
    path_obj = Path(path)
    detected_format = _detect_file_format(path_obj, format)

    # Check for compression
    is_compressed = path_obj.suffix == ".gz" or (len(path_obj.suffixes) > 1 and path_obj.suffixes[-1] == ".gz")

    # Open file with appropriate compression
    def open_func(mode):
        if is_compressed:
            return gzip.open(path_obj, mode, encoding="utf-8")
        else:
            return path_obj.open(mode)

    # Write to file in appropriate format, don't re-serialize
    with open_func("wt") as f:
        if detected_format == "jsonl":
            # JSONL: Write each item on a separate line
            if isinstance(api_serialized, (list, tuple)):
                for item in api_serialized:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            else:
                json.dump(api_serialized, f, ensure_ascii=False)
                f.write("\n")
        else:
            # JSON: Write as single object
            json.dump(api_serialized, f, ensure_ascii=False)


def save_chunked(
    obj: Any, path: Union[str, Path], *, chunk_size: int = 1000, format: Optional[str] = None, **kwargs: Any
) -> None:
    """Save large data to JSON/JSONL file using chunked serialization.

    Memory-efficient saving for large datasets.

    Args:
        obj: Large dataset to save
        path: Output file path
        chunk_size: Size of each chunk
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: Additional chunking options

    Example:
        >>> large_dataset = list(range(100000))
        >>> save_chunked(large_dataset, "large.jsonl", chunk_size=5000)
        >>> save_chunked(large_dataset, "large.json", chunk_size=5000)  # JSON array format
    """
    detected_format = _detect_file_format(path, format)
    chunked_result = dump_chunked(obj, chunk_size=chunk_size, **kwargs)
    chunked_result.save_to_file(path, format=detected_format)


def load_smart_file(path: Union[str, Path], *, format: Optional[str] = None, **kwargs: Any) -> Iterator[Any]:
    """Load data from JSON/JSONL file using smart deserialization.

    Good balance of accuracy and performance for most use cases.
    Success rate: ~80-90% for complex objects.

    Args:
        path: Input file path
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: Additional deserialization options

    Returns:
        Iterator of deserialized objects

    Examples:
        >>> # Load from JSONL (yields each line)
        >>> for item in load_smart_file("data.jsonl"):
        ...     process(item)
        >>>
        >>> # Load from JSON (yields each item in array, or single item)
        >>> for item in load_smart_file("data.json"):
        ...     process(item)
        >>>
        >>> # Or load all at once
        >>> data = list(load_smart_file("data.jsonl"))
    """
    config = SerializationConfig(auto_detect_types=True)
    for raw_item in _load_from_file(path, config, format):
        yield load_smart(raw_item, config, **kwargs)


def load_perfect_file(
    path: Union[str, Path], template: Any, *, format: Optional[str] = None, **kwargs: Any
) -> Iterator[Any]:
    """Load data from JSON/JSONL file using perfect template-based deserialization.

    Uses template for 100% accurate reconstruction.
    Success rate: 100% when template matches data.

    Args:
        path: Input file path
        template: Template object showing expected structure/types
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: Additional template options

    Returns:
        Iterator of perfectly reconstructed objects

    Examples:
        >>> template = {"weights": np.array([0.0]), "epoch": 0}
        >>>
        >>> # Perfect loading from JSONL
        >>> for item in load_perfect_file("training.jsonl", template):
        ...     assert isinstance(item["weights"], np.ndarray)
        >>>
        >>> # Perfect loading from JSON
        >>> for item in load_perfect_file("training.json", template):
        ...     assert isinstance(item["weights"], np.ndarray)
    """
    for raw_item in _load_from_file(path, format=format):
        yield load_perfect(raw_item, template, **kwargs)


def load_basic_file(path: Union[str, Path], *, format: Optional[str] = None, **kwargs: Any) -> Iterator[Any]:
    """Load data from JSON/JSONL file using basic deserialization.

    Fast but with limited type fidelity - suitable for exploration.
    Success rate: ~60-70% for complex objects.

    Args:
        path: Input file path
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: Additional options

    Returns:
        Iterator of deserialized objects

    Example:
        >>> for item in load_basic_file("simple.json"):
        ...     print(item)  # Quick exploration
    """
    for raw_item in _load_from_file(path, format=format):
        yield load_basic(raw_item, **kwargs)


def stream_save_ml(path: Union[str, Path], *, format: Optional[str] = None, **kwargs: Any) -> Any:
    """Create ML-optimized streaming serializer for JSON/JSONL files.

    Memory-efficient ML data streaming with type preservation.

    Args:
        path: Output file path
        format: Explicit format ('json' or 'jsonl'), auto-detected if None
        **kwargs: ML serialization options

    Returns:
        StreamingSerializer configured for ML data

    Examples:
        >>> # Stream to JSONL (one object per line)
        >>> with stream_save_ml("training.jsonl") as stream:
        ...     for epoch_data in training_loop():
        ...         stream.write(epoch_data)
        >>>
        >>> # Stream to JSON (array format)
        >>> with stream_save_ml("training.json") as stream:
        ...     for epoch_data in training_loop():
        ...         stream.write(epoch_data)
    """
    config = get_ml_config()
    detected_format = _detect_file_format(path, format)
    return stream_serialize(path, config=config, format=detected_format, **kwargs)
