"""datason - A comprehensive Python package for intelligent serialization.

datason solves the fundamental problem of serializing complex Python objects
that standard `json.dumps()` cannot handle. Perfect for AI/ML workflows, data science,
and modern Python applications that deal with:

• PyTorch/TensorFlow tensors and models
• Pandas DataFrames and Series
• NumPy arrays and scientific computing objects
• Datetime objects with timezone support
• UUIDs, custom classes, and complex nested structures

Example:
    ```python
    import datason
    import torch
    import pandas as pd

    data = {
        'model_output': torch.tensor([0.9, 0.1, 0.8]),
        'dataset': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        'timestamp': datetime.now()
    }

    # One line to serialize anything
    result = datason.serialize(data)

    # Or with custom configuration
    from datason.config import get_ml_config
    result = datason.serialize(data, config=get_ml_config())
    ```

The package provides intelligent serialization that:
• Preserves type information for complex objects
• Handles circular references and edge cases gracefully
• Maintains compatibility with standard JSON format
• Optimizes performance for large datasets
• Provides extensible architecture for custom types

Key functions:
- serialize(): Convert any Python object to JSON-compatible format
- deserialize(): Reconstruct original objects from serialized data
- register_serializer(): Add custom serialization logic

Supports 20+ data types out of the box with optional dependencies for:
- pandas, numpy (data science)
- torch, tensorflow, jax (machine learning)
- scikit-learn, scipy (scientific computing)
"""

from typing import Any

from .converters import safe_float, safe_int
from .core import SecurityError, serialize
from .data_utils import convert_string_method_votes
from .datetime_utils import (
    convert_pandas_timestamps,
    ensure_dates,
    ensure_timestamp,
    serialize_datetimes,
)
from .deserializers import (
    deserialize,
    deserialize_to_pandas,
    parse_datetime_string,
    parse_uuid_string,
    safe_deserialize,
)
from .serializers import serialize_detection_details

# Configuration system (new)
try:
    from .config import (
        DataFrameOrient,
        DateFormat,
        NanHandling,
        SerializationConfig,
        TypeCoercion,
        get_api_config,
        get_default_config,
        get_ml_config,
        get_performance_config,
        get_strict_config,
        reset_default_config,
        set_default_config,
    )
    from .type_handlers import (
        TypeHandler,
        get_object_info,
        is_nan_like,
        normalize_numpy_types,
    )

    _config_available = True
except ImportError:
    _config_available = False

# ML/AI serializers (optional - only available if ML libraries are installed)
try:
    from .ml_serializers import (
        detect_and_serialize_ml_object,
        get_ml_library_info,
        serialize_huggingface_tokenizer,
        serialize_pil_image,
        serialize_pytorch_tensor,
        serialize_scipy_sparse,
        serialize_sklearn_model,
        serialize_tensorflow_tensor,
    )

    _ml_available = True
except ImportError:
    _ml_available = False

__version__ = "0.2.0"
__author__ = "datason Contributors"
__license__ = "MIT"

__all__ = [  # noqa: RUF022
    "SecurityError",
    # Core serialization
    "serialize",
    # Data conversion utilities
    "convert_pandas_timestamps",
    "convert_string_method_votes",
    "safe_float",
    "safe_int",
    # Deserialization
    "deserialize",
    "deserialize_to_pandas",
    "parse_datetime_string",
    "parse_uuid_string",
    "safe_deserialize",
    # Date/time utilities
    "ensure_dates",
    "ensure_timestamp",
    "serialize_datetimes",
    # Serializers
    "serialize_detection_details",
]

# Add configuration exports if available
if _config_available:
    __all__.extend(
        [  # noqa: RUF022
            # Configuration classes
            "SerializationConfig",
            "DateFormat",
            "DataFrameOrient",
            "NanHandling",
            "TypeCoercion",
            # Configuration functions
            "get_default_config",
            "set_default_config",
            "reset_default_config",
            # Preset configurations
            "get_ml_config",
            "get_api_config",
            "get_strict_config",
            "get_performance_config",
            # Type handling
            "TypeHandler",
            "is_nan_like",
            "normalize_numpy_types",
            "get_object_info",
        ]
    )

# Add ML serializers to __all__ if available
if _ml_available:
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
        return serialize(obj)

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
    return serialize(obj, config=config)


# Add convenience functions to __all__ if config is available
if _config_available:
    __all__.extend(["configure", "serialize_with_config"])
