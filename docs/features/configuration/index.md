# Configuration System

The datason configuration system provides fine-grained control over serialization behavior through a comprehensive set of options and preset configurations optimized for different use cases.

## Overview

Instead of a one-size-fits-all approach, datason allows you to configure serialization behavior through the `SerializationConfig` class and convenient preset functions.

```python
import datason
from datason.config import SerializationConfig, get_ml_config

# Use presets for common scenarios
ml_config = get_ml_config()
result = datason.serialize(data, config=ml_config)

# Or create custom configurations
custom_config = SerializationConfig(
    date_format=DateFormat.UNIX_MS,
    nan_handling=NanHandling.DROP,
    sort_keys=True
)
result = datason.serialize(data, config=custom_config)
```

## 🎛️ Configuration Options

### Date/Time Formatting

Control how datetime objects are serialized with the `date_format` option:

```python
from datason.config import DateFormat, SerializationConfig
from datetime import datetime

dt = datetime(2024, 1, 15, 10, 30, 45)

# ISO format (default) - human readable
config = SerializationConfig(date_format=DateFormat.ISO)
# Result: "2024-01-15T10:30:45"

# Unix timestamp - numeric, compact
config = SerializationConfig(date_format=DateFormat.UNIX)
# Result: 1705316445.0

# Unix milliseconds - JavaScript compatible
config = SerializationConfig(date_format=DateFormat.UNIX_MS)
# Result: 1705316445000

# String representation - human readable
config = SerializationConfig(date_format=DateFormat.STRING)
# Result: "2024-01-15 10:30:45"

# Custom format - your pattern
config = SerializationConfig(
    date_format=DateFormat.CUSTOM,
    custom_date_format="%Y-%m-%d %H:%M"
)
# Result: "2024-01-15 10:30"
```

### NaN/Null Handling

Configure how NaN, None, and missing values are processed:

```python
from datason.config import NanHandling
import numpy as np

data = {"values": [1, 2, float('nan'), 4, None]}

# Convert to JSON null (default)
config = SerializationConfig(nan_handling=NanHandling.NULL)
# Result: {"values": [1, 2, null, 4, null]}

# Convert to string representation
config = SerializationConfig(nan_handling=NanHandling.STRING)
# Result: {"values": [1, 2, "nan", 4, "None"]}

# Keep original values (may not be JSON compatible)
config = SerializationConfig(nan_handling=NanHandling.KEEP)
# Result: {"values": [1, 2, NaN, 4, null]}

# Drop NaN values from collections
config = SerializationConfig(nan_handling=NanHandling.DROP)
# Result: {"values": [1, 2, 4]}
```

### Type Coercion Strategies

Control how aggressively types are converted:

```python
from datason.config import TypeCoercion
import decimal
import uuid

data = {
    "amount": decimal.Decimal("123.45"),
    "id": uuid.uuid4(),
    "complex": 3+4j
}

# Strict - preserve all type information
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
# Result: {"amount": {"_type": "decimal", "value": "123.45", ...}, ...}

# Safe - convert when safe to do so
config = SerializationConfig(type_coercion=TypeCoercion.SAFE)  # Default
# Result: {"amount": 123.45, "id": "uuid-string", "complex": "3+4j"}

# Aggressive - convert to simple types when possible
config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
# Result: {"amount": 123.45, "id": "uuid-string", "complex": [3, 4]}
```

### Pandas DataFrame Orientation

Control how DataFrames are serialized:

```python
from datason.config import DataFrameOrient
import pandas as pd

df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Records format (default) - list of row objects
config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
# Result: [{"A": 1, "B": 3}, {"A": 2, "B": 4}]

# Split format - separate index/columns/data
config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
# Result: {"index": [0, 1], "columns": ["A", "B"], "data": [[1, 3], [2, 4]]}

# Values format - raw data only
config = SerializationConfig(dataframe_orient=DataFrameOrient.VALUES)
# Result: [[1, 3], [2, 4]]

# Index format - index as keys
config = SerializationConfig(dataframe_orient=DataFrameOrient.INDEX)
# Result: {0: {"A": 1, "B": 3}, 1: {"A": 2, "B": 4}}

# Columns format - columns as keys
config = SerializationConfig(dataframe_orient=DataFrameOrient.COLUMNS)
# Result: {"A": {0: 1, 1: 2}, "B": {0: 3, 1: 4}}

# Table format - complete DataFrame representation
config = SerializationConfig(dataframe_orient=DataFrameOrient.TABLE)
# Result: {"schema": {...}, "data": [...]}
```

### Custom Serializers

Register custom handlers for your own types:

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

def serialize_person(person):
    return {
        "_type": "Person",
        "name": person.name,
        "age": person.age,
        "is_adult": person.age >= 18
    }

config = SerializationConfig(
    custom_serializers={Person: serialize_person}
)

data = {"user": Person("Alice", 25)}
result = datason.serialize(data, config=config)
# Result: {"user": {"_type": "Person", "name": "Alice", "age": 25, "is_adult": true}}
```

### Performance and Security Options

Control resource usage and optimization behavior:

```python
config = SerializationConfig(
    # Security limits
    max_depth=500,              # Maximum nesting depth
    max_size=1_000_000,         # Maximum collection size
    max_string_length=100_000,  # Maximum string length

    # Performance options
    sort_keys=True,             # Sort dictionary keys for consistent output
    preserve_decimals=True,     # Keep decimal precision vs convert to float
    preserve_complex=True,      # Preserve complex number metadata
)
```

## 🎯 Preset Configurations

For common use cases, datason provides optimized preset configurations:

### ML/AI Configuration
```python
from datason.config import get_ml_config

config = get_ml_config()
# Optimized for machine learning workflows:
# - Unix timestamp dates (numeric, efficient)
# - Aggressive type coercion (simple types)
# - NULL NaN handling (JSON compatibility)
# - DataFrame records format (standard ML format)
```

### API Configuration
```python
from datason.config import get_api_config

config = get_api_config()
# Optimized for REST API responses:
# - ISO date format (human readable)
# - Safe type coercion (predictable)
# - Sorted keys (consistent output)
# - DataFrame records format (web standard)
```

### Strict Configuration
```python
from datason.config import get_strict_config

config = get_strict_config()
# Optimized for type preservation:
# - Strict type coercion (no data loss)
# - Preserve decimals and complex numbers
# - String NaN handling (explicit)
# - DataFrame table format (complete metadata)
```

### Performance Configuration
```python
from datason.config import get_performance_config

config = get_performance_config()
# Optimized for speed:
# - Unix date format (fast)
# - Aggressive coercion (minimal processing)
# - Drop NaN values (smaller output)
# - DataFrame values format (minimal structure)
```

## 🔧 Global Configuration

Set a default configuration for all serialization operations:

```python
import datason
from datason.config import get_ml_config

# Set global default
datason.configure(get_ml_config())

# Now all serialize() calls use ML config by default
result = datason.serialize(data)  # Uses ML config

# Override for specific calls
result = datason.serialize(data, config=get_api_config())
```

## ⚡ Convenience Functions

Quick configuration without creating config objects:

```python
# Quick configuration with keyword arguments
result = datason.serialize_with_config(
    data,
    date_format='unix_ms',
    nan_handling='drop',
    sort_keys=True
)

# Equivalent to:
config = SerializationConfig(
    date_format=DateFormat.UNIX_MS,
    nan_handling=NanHandling.DROP,
    sort_keys=True
)
result = datason.serialize(data, config=config)
```

## 📊 Configuration Examples by Use Case

### Data Science Pipeline
```python
# For data analysis and scientific computing
config = SerializationConfig(
    date_format=DateFormat.ISO,         # Human readable dates
    dataframe_orient=DataFrameOrient.TABLE,  # Complete DataFrame info
    nan_handling=NanHandling.STRING,    # Explicit NaN representation
    type_coercion=TypeCoercion.STRICT,  # Preserve all type information
    preserve_decimals=True,             # Keep numeric precision
    preserve_complex=True               # Preserve complex numbers
)
```

### Microservices API
```python
# For service-to-service communication
config = SerializationConfig(
    date_format=DateFormat.UNIX_MS,     # JavaScript compatible
    dataframe_orient=DataFrameOrient.RECORDS,  # Standard format
    nan_handling=NanHandling.NULL,      # JSON compatible
    type_coercion=TypeCoercion.SAFE,    # Predictable conversion
    sort_keys=True,                     # Consistent output
    max_depth=100,                      # Security limit
    max_size=10_000                     # Memory protection
)
```

### ML Model Serving
```python
# For model inference and results
config = SerializationConfig(
    date_format=DateFormat.UNIX,        # Efficient numeric format
    dataframe_orient=DataFrameOrient.VALUES,  # Raw data only
    nan_handling=NanHandling.DROP,      # Clean output
    type_coercion=TypeCoercion.AGGRESSIVE,  # Simple types
    max_depth=50,                       # Fast processing
    custom_serializers={
        ModelResult: custom_model_serializer
    }
)
```

### Development and Debugging
```python
# For development and troubleshooting
config = SerializationConfig(
    date_format=DateFormat.STRING,      # Human readable
    nan_handling=NanHandling.STRING,    # Explicit representation
    type_coercion=TypeCoercion.STRICT,  # No information loss
    sort_keys=True,                     # Consistent for diffs
    preserve_decimals=True,             # Exact values
    preserve_complex=True               # Complete information
)
```

## 🔍 Advanced Configuration

### Dynamic Configuration
```python
def get_config_for_environment():
    import os

    if os.getenv('ENV') == 'production':
        return get_performance_config()
    elif os.getenv('ENV') == 'development':
        return get_strict_config()
    else:
        return get_api_config()

config = get_config_for_environment()
```

### Configuration Inheritance
```python
# Start with a base configuration
base_config = get_ml_config()

# Create variations
debug_config = SerializationConfig(
    **base_config.__dict__,
    nan_handling=NanHandling.STRING,    # Override for debugging
    sort_keys=True                      # Add sorting
)
```

### Conditional Custom Serializers
```python
def smart_model_serializer(obj):
    """Serialize models differently based on size."""
    if hasattr(obj, 'coef_') and len(obj.coef_) > 1000:
        # Large model - serialize metadata only
        return {"_type": "large_model", "_class": type(obj).__name__}
    else:
        # Small model - full serialization
        return {"_type": "model", "params": obj.get_params()}

config = SerializationConfig(
    custom_serializers={
        type(my_model): smart_model_serializer
    }
)
```

## 📚 Best Practices

### 1. Choose the Right Preset
Start with a preset configuration that matches your primary use case, then customize as needed.

### 2. Profile Your Configuration
Use the benchmarking tools to measure performance impact of different configuration options.

### 3. Document Your Configuration
When using custom configurations in production, document the choices and reasoning.

### 4. Version Your Configuration
Keep configuration objects in version control and treat them as part of your API contract.

### 5. Test Edge Cases
Ensure your configuration handles edge cases in your data (NaN, None, circular references, large objects).

## 🔗 Related Documentation

- **[Advanced Types →](../advanced-types/index.md)** - Supported Python types
- **[Performance Guide →](../performance/index.md)** - Optimization strategies
- **[ML/AI Integration →](../ml-ai/index.md)** - ML-specific configuration
- **[Pandas Integration →](../pandas/index.md)** - DataFrame handling options
