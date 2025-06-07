# 🚀 Quick Start Guide

Get up and running with datason in minutes! This guide will walk you through installation, basic usage, and key features using both our traditional comprehensive API and the new modern intention-revealing API.

## Installation

### Basic Installation

Install datason using pip:

```bash
pip install datason
```

### With Optional Dependencies

For full ML/AI support, install with optional dependencies:

```bash
# For ML libraries (PyTorch, TensorFlow, scikit-learn)
pip install datason[ml]

# For data science (pandas, numpy extras)
pip install datason[data]

# For all optional dependencies
pip install datason[all]
```

### Development Installation

For development or latest features:

```bash
pip install git+https://github.com/danielendler/datason.git
```

## First Steps

### 1. Basic Serialization

=== "Modern API (Recommended)"

    Start with the new intention-revealing functions:

    ```python
    import datason as ds
    from datetime import datetime

    # Simple data
    data = {
        "name": "Alice",
        "age": 30,
        "active": True,
        "joined": datetime.now(),
        "scores": [95, 87, 92]
    }

    # Serialize using modern API - clear intent
    serialized = ds.dump(data)
    print(type(serialized))  # <class 'dict'>

    # Deserialize back - smart detection preserves types!
    restored = ds.load_smart(serialized)
    print(type(restored["joined"]))  # <class 'datetime.datetime'>
    print(restored["joined"] == data["joined"])  # True

    # For exploration/debugging, use basic loader (fastest)
    quick_restore = ds.load_basic(serialized)

    # For critical applications, use perfect loader with template
    template = {
        "name": str,
        "age": int,
        "active": bool,
        "joined": datetime,
        "scores": [int]
    }
    perfect_restore = ds.load_perfect(serialized, template)
    ```

=== "Traditional API"

    Start with traditional comprehensive functions:

    ```python
    import datason as ds
    from datetime import datetime

    # Simple data
    data = {
        "name": "Alice",
        "age": 30,
        "active": True,
        "joined": datetime.now(),
        "scores": [95, 87, 92]
    }

    # Serialize to JSON-compatible format
    serialized = ds.serialize(data)
    print(type(serialized))  # <class 'dict'>

    # Deserialize back - types are preserved!
    restored = ds.deserialize(serialized)
    print(type(restored["joined"]))  # <class 'datetime.datetime'>
    print(restored["joined"] == data["joined"])  # True
    ```

### 2. Complex Data Types

datason automatically handles complex types with both APIs:

=== "Modern API (Recommended)"

    ```python
    import pandas as pd
    import numpy as np

    # Complex data with DataFrames and arrays
    complex_data = {
        "dataframe": pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, 92.1]
        }),
        "numpy_array": np.array([1, 2, 3, 4, 5]),
        "metadata": {
            "created": datetime.now(),
            "version": 1.0
        }
    }

    # ML-optimized serialization for this type of data
    result = ds.dump_ml(complex_data)

    # Smart deserialization (80-90% success rate, production-ready)
    restored = ds.load_smart(result)
    print(type(restored["dataframe"]))      # <class 'pandas.core.frame.DataFrame'>
    print(type(restored["numpy_array"]))    # <class 'numpy.ndarray'>
    print(restored["dataframe"].shape)      # (3, 2)

    # For 100% reliability, use template-based deserialization
    template = {
        "dataframe": pd.DataFrame,
        "numpy_array": np.ndarray,
        "metadata": {"created": datetime, "version": float}
    }
    perfect_restored = ds.load_perfect(result, template)
    ```

=== "Traditional API"

    ```python
    import pandas as pd
    import numpy as np

    # Complex data with DataFrames and arrays
    complex_data = {
        "dataframe": pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, 92.1]
        }),
        "numpy_array": np.array([1, 2, 3, 4, 5]),
        "metadata": {
            "created": datetime.now(),
            "version": 1.0
        }
    }

    # Serialize complex data
    result = ds.serialize(complex_data)

    # Deserialize - everything is restored correctly
    restored = ds.deserialize(result)
    print(type(restored["dataframe"]))      # <class 'pandas.core.frame.DataFrame'>
    print(type(restored["numpy_array"]))    # <class 'numpy.ndarray'>
    print(restored["dataframe"].shape)      # (3, 2)
    ```

### 3. Configuration & Specialization

=== "Modern API (Recommended)"

    ```python
    # Domain-specific functions with automatic optimization
    user_data = {"name": "John", "email": "john@example.com", "ssn": "123-45-6789"}

    # Security-focused with PII redaction
    secure_result = ds.dump_secure(user_data, redact_pii=True)

    # API-safe clean output for web endpoints
    api_data = {"status": "success", "data": [1, 2, 3], "errors": None}
    api_result = ds.dump_api(api_data)  # Removes null values, optimizes structure

    # High-performance for throughput scenarios
    fast_result = ds.dump_fast(complex_data)  # Minimal type checking

    # Memory-efficient for large datasets  
    large_data = {"big_array": np.random.random((10000, 100))}
    chunked_result = ds.dump_chunked(large_data, chunk_size=1000)

    # Composable options
    sensitive_ml_data = {"model": trained_model, "user_data": user_data}
    result = ds.dump(
        sensitive_ml_data,
        secure=True,     # Enable PII redaction
        ml_mode=True,    # Optimize for ML objects
        chunked=True     # Memory-efficient processing
    )
    ```

=== "Traditional API"

    ```python
    # Use preset configurations for different use cases

    # Machine Learning configuration
    ml_config = ds.get_ml_config()
    ml_result = ds.serialize(complex_data, config=ml_config)

    # API-optimized configuration  
    api_config = ds.get_api_config()
    api_result = ds.serialize(complex_data, config=api_config)

    # Performance-optimized configuration
    perf_config = ds.get_performance_config()
    perf_result = ds.serialize(complex_data, config=perf_config)
    ```

## 🎯 Modern API Features Tour

### Progressive Complexity for Deserialization

The modern API provides clear progression from exploration to production:

```python
json_data = '{"values": [1, 2, 3], "created": "2024-01-01T12:00:00", "metadata": {"version": 1.0}}'

# Level 1: Basic (60-70% success rate) - Fastest, for exploration
basic_result = ds.load_basic(json_data)
print("Basic result:", basic_result)

# Level 2: Smart (80-90% success rate) - Production-ready
smart_result = ds.load_smart(json_data)  
print("Smart result:", smart_result)
print("Date restored:", type(smart_result.get("created")))

# Level 3: Perfect (100% success rate) - Critical applications
template = {
    "values": [int],
    "created": datetime,
    "metadata": {"version": float}
}
perfect_result = ds.load_perfect(json_data, template)
print("Perfect result:", perfect_result)

# Level 4: Typed (95% success rate) - When metadata available
# (Used when data was serialized with type information)
typed_result = ds.load_typed(json_data)
```

### API Discovery & Help

The modern API includes built-in discovery to help you choose the right function:

```python
# Get personalized recommendations
ds.help_api()

# Example output:
# 🎯 datason API Guide
#
# For your use case, consider:
#
# SERIALIZATION (Dump Functions):
# • dump() - General purpose serialization
# • dump_ml() - ML models, tensors, NumPy arrays  
# • dump_api() - Web APIs, clean JSON output
# • dump_secure() - Sensitive data with PII redaction
#
# DESERIALIZATION (Load Functions):
# • load_basic() - 60-70% success, fastest (exploration)
# • load_smart() - 80-90% success, moderate speed (production)
# • load_perfect() - 100% success, requires template (critical)

# Get detailed API information
api_info = ds.get_api_info()
print("Available dump functions:", api_info['dump_functions'])
print("Usage recommendations:", api_info['recommendations'])
```

### JSON Module Compatibility

```python
# Drop-in replacement for json module
data = {"key": "value", "timestamp": datetime.now()}

# Like json.dumps() but with intelligent type handling
json_string = ds.dumps(data)
print(json_string)  # Datetime is properly serialized

# Like json.loads() but with type restoration
restored_data = ds.loads(json_string)
print(type(restored_data["timestamp"]))  # <class 'datetime.datetime'>
```

## Advanced Features Tour

### Intelligent Type Detection

datason automatically detects and handles various data types with both APIs:

=== "Modern API"

    ```python
    import torch
    from sklearn.ensemble import RandomForestClassifier

    # Mixed ML data types
    ml_data = {
        "pytorch_tensor": torch.tensor([1, 2, 3, 4, 5]),
        "sklearn_model": RandomForestClassifier(),
        "pandas_df": pd.DataFrame({"x": [1, 2, 3]}),
        "numpy_array": np.array([[1, 2], [3, 4]]),
        "python_datetime": datetime.now(),
        "nested_dict": {
            "inner": {
                "values": [1, 2, 3],
                "timestamp": datetime.now()
            }
        }
    }

    # ML-optimized serialization
    serialized = ds.dump_ml(ml_data)

    # Smart deserialization preserves all types
    restored = ds.load_smart(serialized)

    # Types are preserved
    print(type(restored["pytorch_tensor"]))  # <class 'torch.Tensor'>
    print(type(restored["sklearn_model"]))   # <class 'sklearn.ensemble._forest.RandomForestClassifier'>
    ```

=== "Traditional API"

    ```python
    import torch
    from sklearn.ensemble import RandomForestClassifier

    # Mixed ML data types
    ml_data = {
        "pytorch_tensor": torch.tensor([1, 2, 3, 4, 5]),
        "sklearn_model": RandomForestClassifier(),
        "pandas_df": pd.DataFrame({"x": [1, 2, 3]}),
        "numpy_array": np.array([[1, 2], [3, 4]]),
        "python_datetime": datetime.now(),
        "nested_dict": {
            "inner": {
                "values": [1, 2, 3],
                "timestamp": datetime.now()
            }
        }
    }

    # All types are automatically handled
    serialized = ds.serialize(ml_data, config=ds.get_ml_config())
    restored = ds.deserialize(serialized)

    # Types are preserved
    print(type(restored["pytorch_tensor"]))  # <class 'torch.Tensor'>
    print(type(restored["sklearn_model"]))   # <class 'sklearn.ensemble._forest.RandomForestClassifier'>
    ```

### Data Privacy & Redaction

=== "Modern API"

    ```python
    # Data with sensitive information
    sensitive_data = {
        "user": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789",
            "password": "secret123"
        },
        "transaction": {
            "amount": 1500.00,
            "card_number": "4532-1234-5678-9012"
        }
    }

    # Security-focused serialization with automatic PII redaction
    secure_result = ds.dump_secure(sensitive_data, redact_pii=True)
    print("Secure result:", secure_result)
    # Sensitive fields are automatically redacted

    # Custom redaction patterns
    custom_secure = ds.dump_secure(
        sensitive_data,
        redact_fields=["custom_field"],
        redact_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"]  # Credit card pattern
    )
    ```

=== "Traditional API"

    ```python
    # Data with sensitive information
    sensitive_data = {
        "user": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789",
            "password": "secret123"
        },
        "transaction": {
            "amount": 1500.00,
            "card_number": "4532-1234-5678-9012"
        }
    }

    # Create redaction engine
    engine = ds.create_financial_redaction_engine()

    # Redact sensitive data
    redacted = engine.process_object(sensitive_data)
    print(redacted)
    # Sensitive fields are now "<REDACTED>"

    # Serialize safely
    safe_data = ds.serialize(redacted, config=ds.get_api_config())
    ```

### Large Data Handling

=== "Modern API"

    ```python
    # Large dataset
    large_data = {
        "images": [np.random.random((512, 512, 3)) for _ in range(100)],
        "features": pd.DataFrame(np.random.random((10000, 50))),
    }

    # Memory-efficient chunked processing
    chunked_result = ds.dump_chunked(large_data, chunk_size=1000)

    # Streaming to file for very large data
    with open('large_data.json', 'w') as f:
        ds.stream_dump(large_data, f)

    # Composable memory-efficient ML processing
    result = ds.dump(large_data, ml_mode=True, chunked=True)
    ```

=== "Traditional API"

    ```python
    # Large dataset
    large_data = {
        "images": [np.random.random((512, 512, 3)) for _ in range(100)],
        "features": pd.DataFrame(np.random.random((10000, 50))),
    }

    # Check memory usage
    memory_estimate = ds.estimate_memory_usage(large_data)
    print(f"Estimated memory: {memory_estimate / (1024*1024):.1f} MB")

    # Use chunked processing for large data
    if memory_estimate > 50 * 1024 * 1024:  # > 50MB
        chunked_result = ds.serialize_chunked(
            large_data,
            chunk_size=10 * 1024 * 1024  # 10MB chunks
        )
    else:
        result = ds.serialize(large_data, config=ds.get_ml_config())
    ```

## Next Steps

Now that you've mastered the basics with both APIs, explore more advanced features:

1. **[Examples Gallery](examples/index.md)** - Real-world usage patterns
2. **[ML/AI Integration](../features/ml-ai/index.md)** - Deep dive into machine learning support  
3. **[Configuration Guide](../features/configuration/index.md)** - Customize behavior for your needs
4. **[Template System](../features/template-deserialization/index.md)** - Enforce data structures
5. **[Data Privacy](../features/redaction.md)** - Comprehensive security features

## Migration from Other Libraries

### From json module

```python
import datason as ds

# Instead of:
# import json
# json_str = json.dumps(data)
# data = json.loads(json_str)

# Use datason for intelligent type handling:
json_str = ds.dumps(data)  # Handles datetime, numpy, etc.
data = ds.loads(json_str)  # Restores original types
```

### From pickle

```python
# Instead of pickle (security concerns, not human-readable):
# import pickle
# data = pickle.loads(pickle.dumps(obj))

# Use datason for secure, readable serialization:
serialized = ds.dump(obj)  # JSON-compatible, human-readable
restored = ds.load_smart(serialized)  # Type-aware restoration
```

Ready to dive deeper? Check out our [Examples Gallery](examples/index.md) for comprehensive real-world usage patterns!
