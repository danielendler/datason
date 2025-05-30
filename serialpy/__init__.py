"""SerialPy - A comprehensive Python package for intelligent serialization.

SerialPy solves the fundamental problem of serializing complex Python objects
that standard `json.dumps()` cannot handle. Perfect for AI/ML workflows, data science,
and modern Python applications that deal with:

• PyTorch/TensorFlow tensors and models
• Pandas DataFrames and Series
• NumPy arrays and scientific computing objects
• Datetime objects with timezone support
• UUIDs, custom classes, and complex nested structures

Example:
    ```python
    import serialpy
    import torch
    import pandas as pd

    data = {
        'model_output': torch.tensor([0.9, 0.1, 0.8]),
        'dataset': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        'timestamp': datetime.now()
    }

    # One line to serialize anything
    result = serialpy.serialize(data)
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

__version__ = "0.1.0"
__author__ = "SerialPy Contributors"
__license__ = "MIT"

__all__ = [
    # Alphabetically sorted for RUF022 compliance
    "convert_pandas_timestamps",
    "convert_string_method_votes",
    "deserialize",
    "deserialize_to_pandas",
    "ensure_dates",
    "ensure_timestamp",
    "parse_datetime_string",
    "parse_uuid_string",
    "safe_deserialize",
    "safe_float",
    "safe_int",
    "SecurityError",
    "serialize",
    "serialize_datetimes",
    "serialize_detection_details",
]

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
