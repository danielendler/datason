"""datason - A comprehensive serialization package for Python.

This package provides intelligent serialization that handles complex data types
with ease, perfect for ML/AI workflows and data science applications.

NEW: Configurable caching system for optimized performance across different workflows.
"""

# Test codecov upload after permissions and configuration fixes

import sys
import warnings
from typing import Any

# Python version compatibility check
if sys.version_info < (3, 8):  # noqa: UP036
    raise RuntimeError(
        f"datason requires Python 3.8 or higher. Your Python version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Warn for EOL Python versions
if sys.version_info < (3, 9):
    warnings.warn(
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor} which reached end-of-life. "
        f"Consider upgrading to Python 3.9+ for better performance and security.",
        DeprecationWarning,
        stacklevel=2,
    )

# NEW: Modern API (Phase 3) - Enhanced defaults with JSON compatibility
from .api import (
    dump,  # Enhanced default (returns dict, smart features)
    dump_api,
    dump_chunked,
    dump_fast,
    dump_json,  # JSON compatibility (exact stdlib behavior)
    dump_ml,
    dump_secure,
    dumps,  # Enhanced default (returns dict, smart features)
    dumps_json,  # JSON compatibility (returns string, exact stdlib behavior)
    get_api_info,
    help_api,
    load,  # Enhanced default (smart parsing, datetime support)
    load_basic,
    # NEW: File I/O operations integrated with modern API
    load_basic_file,
    load_json,  # JSON compatibility (exact stdlib behavior)
    load_perfect,
    load_perfect_file,
    load_smart,
    load_smart_file,
    load_typed,
    loads,  # Enhanced default (smart parsing, datetime support)
    loads_json,  # JSON compatibility (exact stdlib behavior)
    save_api,
    save_chunked,
    save_ml,
    save_secure,
    save_string,
    stream_dump,
    stream_load,  # Streaming deserialization for large files
    stream_save_ml,
    suppress_deprecation_warnings,
)

# Note: serialize function from api.py not imported as it's not used in __init__.py
from .converters import safe_float, safe_int
from .core_new import (
    ChunkedSerializationResult,
    SecurityError,
    StreamingSerializer,
    deserialize_chunked_file,
    estimate_memory_usage,
    serialize_chunked,
    stream_serialize,
)
from .core_new import (
    serialize as _serialize_core,  # Make core serialize less visible
)
from .data_utils import convert_string_method_votes

# Removed simple file_io - now using modern API with file operations
from .datetime_utils import (
    convert_pandas_timestamps,
    ensure_dates,
    ensure_timestamp,
    serialize_datetimes,
)
from .deserializers_new import (
    TemplateDeserializationError,
    TemplateDeserializer,
    auto_deserialize,
    clear_caches,  # noqa: F401
    create_ml_round_trip_template,
    deserialize,
    deserialize_to_pandas,
    deserialize_with_template,
    infer_template_from_data,
    parse_datetime_string,
    parse_uuid_string,
    safe_deserialize,
)
from .serializers import serialize_detection_details

# Configuration system (new)
try:
    from .config import (
        CacheScope,
        DataFrameOrient,
        DateFormat,
        NanHandling,
        OutputType,
        SerializationConfig,
        TypeCoercion,
        cache_scope,  # noqa: F401
        get_accel_mode,  # noqa: F401
        get_api_config,  # noqa: F401
        get_cache_scope,  # noqa: F401
        get_default_config,  # noqa: F401
        get_ml_config,  # noqa: F401
        get_performance_config,  # noqa: F401
        get_strict_config,  # noqa: F401
        reset_default_config,  # noqa: F401
        set_accel_mode,  # noqa: F401
        set_cache_scope,  # noqa: F401
        set_default_config,
    )

    _config_available = True
except ImportError:
    _config_available = False
    SerializationConfig = None
    DateFormat = None
    DataFrameOrient = None
    OutputType = None
    NanHandling = None
    TypeCoercion = None
    CacheScope = None

# ML/AI serializers (optional - only available if ML libraries are installed)
try:
    import importlib

    # Test if ml_serializers module is available
    importlib.import_module(".ml_serializers", package="datason")
    from . import ml_serializers  # Make the module accessible as datason.ml_serializers  # noqa: F401

    _ml_available = True
except ImportError:
    _ml_available = False

# ML Type Handlers (new unified architecture) - Import to trigger auto-registration
try:
    from . import ml_type_handlers  # noqa: F401  # Import to trigger handler registration

    _ml_type_handlers_available = True
except ImportError:
    _ml_type_handlers_available = False

# Pickle Bridge (new in v0.3.0) - Zero dependencies, always available
try:
    import importlib

    # Test if pickle_bridge module is available
    importlib.import_module(".pickle_bridge", package="datason")
    _pickle_bridge_available = True
except ImportError:
    _pickle_bridge_available = False

# Validation helpers (always available, dependencies imported lazily)
# Always import datetime_utils module for tests
from . import (
    datetime_utils,  # noqa: F401
    validation,  # noqa: F401
)

# Cache management functions
from .cache_manager import (
    clear_all_caches,  # noqa: F401
    get_cache_metrics,  # noqa: F401
    operation_scope,  # noqa: F401
    request_scope,  # noqa: F401
    reset_cache_metrics,  # noqa: F401
)

# Integrity utilities (always available)
from .integrity import (  # noqa: F401
    canonicalize,
    hash_and_redact,
    hash_json,
    hash_object,
    sign_object,
    verify_json,
    verify_object,
    verify_signature,
)
from .validation import serialize_marshmallow, serialize_pydantic  # noqa: F401


def _get_version() -> str:
    """Get version from pyproject.toml or fallback to a default."""
    import os
    import re

    # Get the project root directory (parent of datason package)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pyproject_path = os.path.join(project_root, "pyproject.toml")

    try:
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
            # Use regex to find version = "x.y.z" in the project section
            match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except (FileNotFoundError, OSError):
        pass

    # Fallback version if pyproject.toml is not found or version not found
    return "0.5.0"


__version__ = "0.12.0"
__author__ = "datason Contributors"
__license__ = "MIT"
__description__ = "Python serialization of complex data types for JSON with configurable caching"

__all__ = [  # noqa: RUF022
    "SecurityError",
    # Enhanced DataSON API (default recommended usage)
    "dump",  # Enhanced file writing with smart features
    "dumps",  # Enhanced serialization returning dict
    "load",  # Enhanced file reading with smart parsing
    "loads",  # Enhanced string parsing with smart features
    "serialize",  # Enhanced serialization (returns dict)
    # JSON Compatibility API (for stdlib replacement)
    "dump_json",  # Exact json.dump() behavior
    "dumps_json",  # Exact json.dumps() behavior (returns string)
    "load_json",  # Exact json.load() behavior
    "loads_json",  # Exact json.loads() behavior
    # NEW: Chunked processing and streaming (v0.4.0)
    "serialize_chunked",
    "ChunkedSerializationResult",
    "stream_serialize",
    "StreamingSerializer",
    "deserialize_chunked_file",
    "estimate_memory_usage",
    # Data conversion utilities
    "convert_pandas_timestamps",
    "convert_string_method_votes",
    "safe_float",
    "safe_int",
    # Deserialization
    "auto_deserialize",
    "deserialize",
    "deserialize_to_pandas",
    "parse_datetime_string",
    "parse_uuid_string",
    "safe_deserialize",
    # NEW: Template-based deserialization (v0.4.5)
    "deserialize_with_template",
    "TemplateDeserializer",
    "TemplateDeserializationError",
    "infer_template_from_data",
    "create_ml_round_trip_template",
    # Date/time utilities
    "ensure_dates",
    "ensure_timestamp",
    "serialize_datetimes",
    # Serializers
    "serialize_detection_details",
    # NEW: Modern API (Phase 3) - Intention-revealing wrapper functions
    "dump",
    "dump_ml",
    "dump_api",
    "dump_secure",
    "dump_fast",
    "dump_chunked",
    "stream_dump",
    # File I/O Operations - Modern API Integration
    "save_ml",
    "save_secure",
    "save_api",
    "save_chunked",
    "save_string",
    "load_smart_file",
    "load_perfect_file",
    "load_basic_file",
    "stream_save_ml",
    # Progressive Loading
    "load_basic",
    "load_smart",
    "load_perfect",
    "load_typed",
    "stream_load",
    "loads",
    "dumps",
    "help_api",
    "get_api_info",
    "suppress_deprecation_warnings",
]

# Add configuration exports if available
if _config_available:
    from .type_handlers import (  # noqa: F401
        TypeHandler,
        get_object_info,
        is_nan_like,
        normalize_numpy_types,
    )

    __all__.extend(
        [  # noqa: RUF022
            # Configuration classes (already imported above)
            "SerializationConfig",
            "DateFormat",
            "DataFrameOrient",
            "OutputType",
            "NanHandling",
            "TypeCoercion",
            "CacheScope",
            # Configuration functions
            "get_default_config",
            "set_default_config",
            "reset_default_config",
            # Preset configurations
            "get_ml_config",
            "get_api_config",
            "get_strict_config",
            "get_performance_config",
            # Cache management
            "cache_scope",
            "get_cache_scope",
            "set_cache_scope",
            "clear_caches",
            "clear_all_caches",
            "get_cache_metrics",
            "reset_cache_metrics",
            "operation_scope",
            "request_scope",
            # Type handling
            "TypeHandler",
            "is_nan_like",
            "normalize_numpy_types",
            "get_object_info",
        ]
    )

# Integrity utilities always available
__all__.extend(
    [
        "canonicalize",
        "hash_and_redact",
        "hash_json",
        "hash_object",
        "sign_object",
        "verify_json",
        "verify_object",
        "verify_signature",
    ]
)

# Add ML serializers to __all__ if available
if _ml_available:
    from .ml_serializers import (  # noqa: F401
        detect_and_serialize_ml_object,
        get_ml_library_info,
        serialize_huggingface_tokenizer,
        serialize_pil_image,
        serialize_pytorch_tensor,
        serialize_scipy_sparse,
        serialize_sklearn_model,
        serialize_tensorflow_tensor,
    )

    __all__.extend(
        [
            "detect_and_serialize_ml_object",
            "get_ml_library_info",
            "serialize_huggingface_tokenizer",
            "serialize_pil_image",
            "serialize_pytorch_tensor",
            "serialize_scipy_sparse",
            "serialize_sklearn_model",
            "serialize_tensorflow_tensor",
        ]
    )

# Add Pickle Bridge to __all__ if available
if _pickle_bridge_available:
    from .pickle_bridge import (  # noqa: F401  # nosec B403
        PickleBridge,
        PickleSecurityError,
        convert_pickle_directory,
        from_pickle,
        get_ml_safe_classes,
    )

    __all__.extend(
        [
            "PickleBridge",
            "PickleSecurityError",
            "convert_pickle_directory",
            "from_pickle",
            "get_ml_safe_classes",
        ]
    )

# Always expose validation helpers
__all__.extend(["serialize_pydantic", "serialize_marshmallow"])


# Convenience functions for quick access
def configure(config: "SerializationConfig") -> None:
    """Set the global default configuration.

    Args:
        config: Configuration to set as default

    Example:
        >>> import datason
        >>> datason.configure(datason.get_ml_config())
        >>> # Now all serialize() calls use ML config by default
    """
    if _config_available:
        set_default_config(config)
    else:
        raise ImportError("Configuration system not available")


def serialize_with_config(obj: Any, **kwargs: Any) -> Any:
    """Serialize with quick configuration options.

    Args:
        obj: Object to serialize
        **kwargs: Configuration options (date_format, nan_handling, etc.)

    Returns:
        Serialized object

    Example:
        >>> datason.serialize_with_config(data, date_format='unix', sort_keys=True)
    """
    if not _config_available:
        return _serialize_core(obj)

    # Convert string options to enums
    if "date_format" in kwargs and isinstance(kwargs["date_format"], str):
        kwargs["date_format"] = DateFormat(kwargs["date_format"])
    if "nan_handling" in kwargs and isinstance(kwargs["nan_handling"], str):
        kwargs["nan_handling"] = NanHandling(kwargs["nan_handling"])
    if "type_coercion" in kwargs and isinstance(kwargs["type_coercion"], str):
        kwargs["type_coercion"] = TypeCoercion(kwargs["type_coercion"])
    if "dataframe_orient" in kwargs and isinstance(kwargs["dataframe_orient"], str):
        kwargs["dataframe_orient"] = DataFrameOrient(kwargs["dataframe_orient"])

    config = SerializationConfig(**kwargs)
    return _serialize_core(obj, config=config)


# Backward compatibility: Provide serialize() function with deprecation warning
def serialize(obj: Any, config: Any = None, **kwargs: Any) -> Any:
    """Serialize an object (DEPRECATED - use dump/dumps instead).

    DEPRECATION WARNING: Direct use of serialize() is discouraged.
    Use the clearer API functions instead:
    - dump(obj, file) - write to file (like json.dump)
    - dumps(obj) - convert to string (like json.dumps)
    - serialize_enhanced(obj, **options) - enhanced serialization with clear options

    Args:
        obj: Object to serialize
        config: Optional configuration
        **kwargs: Additional options

    Returns:
        Serialized object
    """
    import warnings

    warnings.warn(
        "serialize() is deprecated. Use dump/dumps for JSON compatibility or "
        "serialize_enhanced() for advanced features. Direct serialize() will be "
        "removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _serialize_core(obj, config, **kwargs)


# Add convenience functions to __all__ if config is available
if _config_available:
    __all__.extend(
        [
            "configure",
            "serialize_with_config",
            "serialize",
            "set_accel_mode",
            "get_accel_mode",
        ]
    )

# Add redaction exports if available (v0.5.5)
try:
    from .redaction import (  # noqa: F401
        RedactionEngine,
        create_financial_redaction_engine,
        create_healthcare_redaction_engine,
        create_minimal_redaction_engine,
    )

    _redaction_available = True
except ImportError:
    _redaction_available = False

# Add data utilities (v0.5.5)
try:
    from .utils import (  # noqa: F401
        UtilityConfig,
        UtilitySecurityError,
        deep_compare,
        enhance_data_types,
        enhance_numpy_array,
        enhance_pandas_dataframe,
        extract_temporal_features,
        find_data_anomalies,
        get_available_utilities,
        get_default_utility_config,
        normalize_data_structure,
        standardize_datetime_formats,
    )

    _utils_available = True
except ImportError:
    _utils_available = False

# Add redaction exports to __all__ if available (v0.5.5)
if _redaction_available:
    __all__.extend(
        [
            "RedactionEngine",
            "create_financial_redaction_engine",
            "create_healthcare_redaction_engine",
            "create_minimal_redaction_engine",
        ]
    )

# Add utility exports to __all__ if available (v0.5.5)
if _utils_available:
    __all__.extend(
        [
            "deep_compare",
            "find_data_anomalies",
            "enhance_data_types",
            "normalize_data_structure",
            "standardize_datetime_formats",
            "extract_temporal_features",
            "get_available_utilities",
            "UtilityConfig",
            "UtilitySecurityError",
            "get_default_utility_config",
            "enhance_pandas_dataframe",
            "enhance_numpy_array",
        ]
    )


def get_version() -> str:
    """Get the current version of datason."""
    return __version__


def get_info() -> dict:
    """Get information about the datason package."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __author__,
        "description": __description__,
        "config_available": _config_available,
        "cache_system": "configurable" if _config_available else "legacy",
    }
