# 📋 Core Functions

The main serialization and deserialization functions that form the core of datason's traditional API.

## 🎯 Overview

The core functions provide comprehensive, configuration-based serialization with maximum control and flexibility.

| Function | Purpose | Best For |
|----------|---------|----------|
| `serialize()` | Main serialization function | Custom configurations |
| `deserialize()` | Main deserialization function | Structured data restoration |
| `auto_deserialize()` | Automatic type detection | Quick data exploration |
| `safe_deserialize()` | Error-resilient deserialization | Untrusted data sources |

## 📦 Detailed Function Documentation

### serialize()

The primary serialization function with full configuration support.

::: datason.serialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Configuration Example:**
```python
import datason as ds
from datetime import datetime
import pandas as pd

# Basic serialization
data = {"values": [1, 2, 3], "timestamp": datetime.now()}
result = ds.serialize(data)

# With custom configuration
config = ds.SerializationConfig(
    include_type_info=True,
    compress_arrays=True,
    date_format=ds.DateFormat.ISO_8601,
    nan_handling=ds.NanHandling.NULL
)

complex_data = {
    "dataframe": pd.DataFrame({"x": [1, 2, 3]}),
    "timestamp": datetime.now(),
    "metadata": {"version": 1.0}
}

result = ds.serialize(complex_data, config=config)
```

### deserialize()

The primary deserialization function with configuration support.

::: datason.deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Deserialization Example:**
```python
# Basic deserialization
restored_data = ds.deserialize(serialized_result)

# With custom configuration for specific type handling
config = ds.SerializationConfig(
    strict_types=True,
    preserve_numpy_arrays=True,
    datetime_parsing=True
)

restored_data = ds.deserialize(serialized_result, config=config)
print(type(restored_data["dataframe"]))  # <class 'pandas.core.frame.DataFrame'>
```

### auto_deserialize()

Automatic type detection and intelligent deserialization.

::: datason.auto_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Auto-Detection Example:**
```python
# Automatically detect and restore types from JSON
json_data = '{"timestamp": "2024-01-01T12:00:00", "values": [1, 2, 3]}'

# Intelligent type detection
auto_restored = ds.auto_deserialize(json_data)
print(type(auto_restored["timestamp"]))  # <class 'datetime.datetime'>

# Works with complex nested structures
complex_json = ds.serialize({
    "df": pd.DataFrame({"x": [1, 2, 3]}),
    "date": datetime.now(),
    "array": np.array([1, 2, 3])
})

auto_complex = ds.auto_deserialize(complex_json)
```

### safe_deserialize()

Error-resilient deserialization for untrusted or malformed data.

::: datason.safe_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Safe Processing Example:**
```python
# Handle potentially malformed data
untrusted_data = '{"timestamp": "invalid-date", "values": [1, "bad", 3]}'

try:
    # Regular deserialization might fail
    result = ds.deserialize(untrusted_data)
except Exception as e:
    # Safe deserialization provides fallbacks
    safe_result = ds.safe_deserialize(untrusted_data)
    print("Safely processed:", safe_result)

# With custom error handling
safe_result = ds.safe_deserialize(
    untrusted_data,
    fallback_values={"timestamp": None, "values": []},
    skip_invalid=True
)
```

## 🔧 Configuration System Integration

The core functions work seamlessly with datason's configuration system:

### Preset Configurations

```python
# Use predefined configurations for common scenarios
ml_config = ds.get_ml_config()
ml_result = ds.serialize(ml_data, config=ml_config)

api_config = ds.get_api_config()
api_result = ds.serialize(api_data, config=api_config)

strict_config = ds.get_strict_config()
strict_result = ds.serialize(data, config=strict_config)

performance_config = ds.get_performance_config()
fast_result = ds.serialize(data, config=performance_config)
```

### Custom Configuration

```python
# Build custom configurations
custom_config = ds.SerializationConfig(
    # Type handling
    include_type_info=True,
    strict_types=False,
    preserve_numpy_arrays=True,

    # Performance
    compress_arrays=True,
    optimize_memory=True,

    # Data handling
    date_format=ds.DateFormat.TIMESTAMP,
    nan_handling=ds.NanHandling.STRING,
    dataframe_orient=ds.DataFrameOrient.RECORDS,

    # Security
    redact_patterns=["ssn", "password"],
    max_depth=100
)

result = ds.serialize(data, config=custom_config)
```

## 🔄 Error Handling Patterns

### Graceful Degradation

```python
def robust_serialize(data):
    """Serialize with multiple fallback strategies."""
    try:
        # Try with full configuration
        return ds.serialize(data, config=ds.get_ml_config())
    except MemoryError:
        # Fall back to chunked processing
        return ds.serialize_chunked(data)
    except SecurityError:
        # Fall back to safe mode
        safe_config = ds.SerializationConfig(secure_mode=True)
        return ds.serialize(data, config=safe_config)
    except Exception:
        # Last resort: safe deserialization
        return ds.safe_deserialize(data)
```

### Validation and Recovery

```python
def validate_and_deserialize(serialized_data):
    """Validate data before deserialization."""
    try:
        # First attempt: auto deserialization
        result = ds.auto_deserialize(serialized_data)
        return result
    except ValueError:
        # Second attempt: safe deserialization
        return ds.safe_deserialize(serialized_data)
```

## 📊 Performance Considerations

### Function Performance Characteristics

| Function | Speed | Reliability | Features |
|----------|-------|-------------|----------|
| `serialize()` | ⚡⚡ | 🛡️🛡️🛡️ | ⭐⭐⭐ |
| `deserialize()` | ⚡⚡ | 🛡️🛡️🛡️ | ⭐⭐⭐ |
| `auto_deserialize()` | ⚡ | 🛡️🛡️ | ⭐⭐ |
| `safe_deserialize()` | ⚡ | 🛡️🛡️🛡️🛡️ | ⭐ |

### Optimization Tips

```python
# Reuse configurations for better performance
config = ds.get_ml_config()
for batch in data_batches:
    result = ds.serialize(batch, config=config)

# Use appropriate function for your needs
if data_is_trusted:
    result = ds.deserialize(data)  # Fastest
else:
    result = ds.safe_deserialize(data)  # Most reliable
```

## 🔗 Related Documentation

- **[Configuration System](configuration.md)** - Detailed configuration options
- **[Chunked & Streaming](chunked-streaming.md)** - Large data processing
- **[Template System](template-system.md)** - Data validation
- **[Modern API](modern-api.md)** - Compare with intention-revealing functions
