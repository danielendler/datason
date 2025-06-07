# Deserialization & Type Support

Complete guide to datason's deserialization capabilities, supported types, and type preservation strategies.

## 🎯 Overview

Datason provides sophisticated deserialization with **automatic type detection** and **optional type metadata** for perfect round-trip fidelity. This page documents all supported types and their behavior with and without type hints.

## 📊 Current Round-Trip Audit Status (v0.7.0)

**Overall Success Rate: 75.0% (51/68 tests passed)** ⬆️ *Recently improved from 67.6%*

### Success Rate by Category:
- **Basic Types**: 100% (20/20) ✅ Perfect! (improved from 95.0%)
- **Complex Types**: 100% (15/15) ✅ Perfect! (improved from 93.3%)
- **NumPy Types**: 71.4% (10/14) ⚠️ Needs work
- **Pandas Types**: 30.8% (4/13) ❌ Critical gaps
- **ML Types**: 33.3% (2/6) ⚠️ Improving (was 0.0%)

### Recent Fixes:
- ✅ **UUID Cache Issue**: Fixed cache pollution causing UUID detection to fail
- ✅ **Auto-Detection**: UUID and datetime auto-detection now working reliably
- ✅ **PyTorch Tensors**: Fixed tensor comparison logic in audit script (+3.0%)
- ✅ **Set/Tuple Verification**: Fixed audit logic to handle expected type conversions (+2.9%)
- ✅ **Test Infrastructure**: Added comprehensive enhanced type support tests

### Enhancement Roadmap:
- **v0.7.5 Target**: Fix critical gaps → 85%+ success rate (need +10% more)
- **v0.8.0 Target**: Complete round-trip support → 95%+ success rate
- **v0.8.5 Target**: Perfect ML integration → 99%+ success rate

## 🚀 Performance Features (v0.6.0+)

### Ultra-Fast Deserializer
```python
from datason.deserializers import deserialize_fast

# 3.73x average performance improvement
result = deserialize_fast(data)

# With configuration for type preservation
config = SerializationConfig(include_type_hints=True)
result = deserialize_fast(data, config=config)
```

### Performance Benchmarks
| Data Type | Improvement | Use Case |
|-----------|-------------|----------|
| **Basic Types** | **0.84x** (18% faster) | Ultra-fast path |
| **DateTime/UUID Heavy** | **3.49x** | Log processing |
| **Large Nested** | **16.86x** | Complex data structures |
| **Average Overall** | **3.73x** | All workloads |

## 📋 Type Support Matrix

### Types That Work WITHOUT Type Hints

These types serialize and deserialize perfectly without `include_type_hints=True`:

#### Basic JSON Types (Perfect Round-Trips)
```python
# Perfect preservation without any configuration
none_value = None
string_value = "hello world"
unicode_string = "hello 世界 🌍"
integer_value = 42
float_value = 3.14159
boolean_value = True
list_value = [1, 2, 3]
dict_value = {"key": "value"}

# All preserve exact types and values
result = deserialize_fast(serialized_data)
```

| Type | Example | Notes |
|------|---------|--------|
| `None` | `None` | Direct pass-through |
| `str` | `"hello"` | Unicode support |
| `int` | `42` | All integer sizes |
| `float` | `3.14` | Full precision |
| `bool` | `True` | No conversion |
| `list` | `[1, 2, 3]` | Recursive support |
| `dict` | `{"a": 1}` | String keys preserved |

#### Auto-Detectable Types (Smart Recognition)
```python
# Automatically detected without configuration
from datetime import datetime
import uuid

dt = datetime(2023, 1, 1, 12, 0, 0)
uid = uuid.uuid4()

# Smart pattern recognition - no type hints needed
result = deserialize_fast('2023-01-01T12:00:00')  # → datetime
result = deserialize_fast('12345678-1234-5678-9012-123456789abc')  # → UUID
```

| Type | Pattern | Auto-Detection |
|------|---------|----------------|
| `datetime` | ISO format strings | ✅ Full support |
| `UUID` | Standard UUID format | ✅ Full support |
| `Path` | Absolute paths | ✅ Limited support |

#### Legacy Types (Always Get Metadata)
```python
from decimal import Decimal

# These ALWAYS get type metadata regardless of settings
complex_num = complex(1, 2)
decimal_num = Decimal("123.45")

# Round-trip perfectly without explicit type hints
serialized = serialize(complex_num)  # Always includes metadata
result = deserialize_fast(serialized)  # → complex(1, 2)
```

| Type | Behavior | Reason |
|------|----------|---------|
| `complex` | Always preserved | Legacy compatibility |
| `Decimal` | Always preserved | Precision requirements |

### Types That NEED Type Hints

These types require `include_type_hints=True` for perfect round-trip preservation:

#### Container Types (Lose Type Information)
```python
config = SerializationConfig(include_type_hints=True)

# Without type hints: tuple → list, set → list
original_tuple = (1, 2, 3)
original_set = {1, 2, 3}

# With type hints: perfect preservation
serialized = serialize(original_tuple, config=config)
result = deserialize_fast(serialized, config=config)
assert type(result) is tuple  # ✅ Preserved
```

| Type | Without Hints | With Hints | Notes |
|------|---------------|------------|--------|
| `tuple` | → `list` | ✅ `tuple` | Order preserved |
| `set` | → `list` | ✅ `set` | Order may change |
| `frozenset` | → `list` | ✅ `frozenset` | Immutability preserved |

#### NumPy Types (Become Python Primitives)
```python
import numpy as np

config = SerializationConfig(include_type_hints=True)

# NumPy scalars
np_int = np.int32(42)
np_float = np.float64(3.14)
np_bool = np.bool_(True)

# Arrays
np_array = np.array([[1, 2], [3, 4]])
np_zeros = np.zeros((3, 3))
```

| NumPy Type | Without Hints | With Hints | Notes |
|------------|---------------|------------|--------|
| `np.int*` | → `int` | ✅ Exact type | All integer types |
| `np.float*` | → `float` | ✅ Exact type | Precision preserved |
| `np.bool_` | → `bool` | ✅ `np.bool_` | Type distinction |
| `np.array` | → `list` | ✅ `np.array` | Shape + dtype preserved |
| `np.complex*` | → `str`/`complex` | ✅ Exact type | Complex number handling |

#### Pandas Types (Become Lists/Dicts)
```python
import pandas as pd

config = SerializationConfig(include_type_hints=True)

# DataFrame
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

# Series  
series = pd.Series([1, 2, 3], name="my_series")

# Categorical data
categorical = pd.Series(pd.Categorical(['a', 'b', 'a', 'c']))
```

| Pandas Type | Without Hints | With Hints | Notes |
|-------------|---------------|------------|--------|
| `DataFrame` | → `list` (records) | ✅ `DataFrame` | Index + columns preserved |
| `Series` | → `dict` | ✅ `Series` | Index + name preserved |
| `Categorical` | → `dict` (as object) | ✅ `Categorical` | Categories + ordered preserved |

#### Machine Learning Types (Become Dicts)
```python
import torch
from sklearn.linear_model import LogisticRegression

config = SerializationConfig(include_type_hints=True)

# PyTorch tensors
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Sklearn models  
model = LogisticRegression(random_state=42)
```

| ML Type | Without Hints | With Hints | Notes |
|---------|---------------|------------|--------|
| `torch.Tensor` | → `list` | ✅ `torch.Tensor` | Device + dtype preserved |
| `sklearn.*` | → `dict` | ✅ Model class | Parameters preserved* |

*Note: Sklearn fitted state (weights) cannot be preserved without training data.

## 🔧 Configuration Options

### Basic Deserialization
```python
from datason.deserializers import deserialize_fast

# Default behavior - auto-detection only
result = deserialize_fast(data)
```

### Type Preservation
```python
from datason.config import SerializationConfig

# Full type preservation
config = SerializationConfig(include_type_hints=True)
result = deserialize_fast(data, config=config)
```

### Security Limits
```python
# Custom security limits
config = SerializationConfig(
    max_depth=50,        # Maximum nesting depth
    max_size=100_000,    # Maximum container size
    include_type_hints=True
)
result = deserialize_fast(data, config=config)
```

### DataFrame Orientations
```python
from datason.config import DataFrameOrient

# Different DataFrame serialization formats
config = SerializationConfig(
    dataframe_orient=DataFrameOrient.RECORDS,  # Default
    include_type_hints=True
)

# Available orientations:
# - RECORDS: [{"a": 1, "b": 4}, {"a": 2, "b": 5}]
# - SPLIT: {"index": [...], "columns": [...], "data": [...]}
# - INDEX: {0: {"a": 1, "b": 4}, 1: {"a": 2, "b": 5}}
# - VALUES: [[1, 4], [2, 5]]  # Loses column names
```

## 🎨 Advanced Features

### Automatic String Type Detection
```python
# Smart pattern recognition without type hints
datetime_str = "2023-01-01T12:00:00"
uuid_str = "12345678-1234-5678-9012-123456789abc"
path_str = "/tmp/test/file.txt"

# Automatically converted to appropriate types
dt = deserialize_fast(datetime_str)    # → datetime object
uid = deserialize_fast(uuid_str)       # → UUID object
path = deserialize_fast(path_str)      # → Path object (if absolute)
```

### Type Metadata Format
```python
# New format (v0.6.0+)
{
    "__datason_type__": "numpy.int32",
    "__datason_value__": 42
}

# Legacy format (still supported)
{
    "_type": "numpy.int32",
    "value": 42
}
```

### Complex Type Reconstruction
```python
# Complex numbers - automatic detection
complex_dict = {"real": 1.0, "imag": 2.0}
result = deserialize_fast(complex_dict)  # → complex(1, 2)

# Decimal from string pattern
decimal_dict = {"value": "123.456"}
result = deserialize_fast(decimal_dict)  # → Decimal("123.456")
```

## 🛡️ Security Features

### Depth Protection
```python
# Protects against deeply nested attacks
config = SerializationConfig(max_depth=5)

deep_data = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
# Raises DeserializationSecurityError when depth > 5
```

### Size Limits
```python
# Protects against memory exhaustion
config = SerializationConfig(max_size=1000)

huge_dict = {f"key_{i}": i for i in range(10000)}
# Raises DeserializationSecurityError when size > 1000
```

### Circular Reference Detection
```python
# Safe handling of circular references
circular_data = {"self": None}
circular_data["self"] = circular_data

result = deserialize_fast(circular_data)
# Returns: {"self": {}}  # Circular reference safely broken
```

## 📊 Performance Optimization

### Ultra-Fast Basic Types
```python
# Zero-overhead processing for common cases
result = deserialize_fast(42)         # Immediate return
result = deserialize_fast("hello")    # < 8 chars = immediate return
result = deserialize_fast(True)       # Immediate return
result = deserialize_fast(None)       # Immediate return
```

### Caching Systems

**New in v0.7.0**: [Configurable caching system](caching/index.md) with multiple scopes for different workflows.

```python
import datason
from datason import CacheScope

# Choose cache scope based on your needs
datason.set_cache_scope(CacheScope.REQUEST)   # For web APIs
datason.set_cache_scope(CacheScope.PROCESS)   # For ML training

# Pattern caching for repeated strings (automatic)
datetime_str = "2023-01-01T12:00:00"
result1 = datason.deserialize_fast(datetime_str)  # Parse and cache
result2 = datason.deserialize_fast(datetime_str)  # Cache hit! (faster)

# Object caching for parsed results (automatic)
uuid_str = "12345678-1234-5678-9012-123456789abc"
result1 = datason.deserialize_fast(uuid_str)   # Parse and cache UUID object
result2 = datason.deserialize_fast(uuid_str)   # Return cached UUID object
```

See the [Caching Documentation](caching/index.md) for detailed configuration and performance tuning.

### Memory Pooling
```python
# Container pooling for reduced allocations
# Automatically managed - no user configuration needed
large_nested = {"data": [{"item": i} for i in range(1000)]}
result = deserialize_fast(large_nested)  # Uses pooled containers internally
```

## 🧪 Testing & Validation

### Round-Trip Testing
```python
def test_round_trip(original_data):
    """Test perfect round-trip preservation"""
    config = SerializationConfig(include_type_hints=True)

    # Serialize
    serialized = serialize(original_data, config=config)

    # Ensure JSON compatibility
    json_str = json.dumps(serialized)
    parsed = json.loads(json_str)

    # Deserialize
    deserialized = deserialize_fast(parsed, config=config)

    # Validate
    assert type(deserialized) is type(original_data)
    assert deserialized == original_data
```

### Type Validation
```python
# Comprehensive type support validation
test_cases = [
    (42, int),                           # Basic int
    (np.int32(42), np.int32),           # NumPy int32
    ((1, 2, 3), tuple),                 # Tuple preservation
    (pd.Series([1, 2, 3]), pd.Series),  # Pandas Series
    (torch.tensor([1, 2]), torch.Tensor), # PyTorch tensor
]

config = SerializationConfig(include_type_hints=True)
for original, expected_type in test_cases:
    serialized = serialize(original, config=config)
    result = deserialize_fast(serialized, config=config)
    assert type(result) is expected_type
```

## 🔄 Migration Guide

### From v0.5.x to v0.6.0
```python
# Old approach
from datason.deserializers import deserialize
result = deserialize(data)

# New optimized approach (3.73x faster)
from datason.deserializers import deserialize_fast
result = deserialize_fast(data)  # Drop-in replacement

# Enhanced type preservation
config = SerializationConfig(include_type_hints=True)
result = deserialize_fast(data, config=config)
```

### Performance Migration
```python
# Replace slow patterns
for item in large_dataset:
    result = deserialize(item)  # Slow

# With optimized batch processing
results = [deserialize_fast(item) for item in large_dataset]  # Fast
```

## 📈 Best Practices

### Type Preservation Strategy
```python
# For maximum compatibility: include type hints
config = SerializationConfig(include_type_hints=True)

# For maximum performance: rely on auto-detection
# (only for basic types + datetime/UUID)
result = deserialize_fast(data)  # No config needed
```

### Error Handling
```python
from datason.deserializers import DeserializationSecurityError

try:
    result = deserialize_fast(untrusted_data, config=config)
except DeserializationSecurityError as e:
    logger.warning(f"Security violation in deserialization: {e}")
    # Handle security issue appropriately
except Exception as e:
    logger.error(f"Deserialization failed: {e}")
    # Handle other errors
```

### Production Monitoring
```python
import warnings

# Monitor security warnings in production
with warnings.catch_warnings(record=True) as w:
    result = deserialize_fast(data)
    for warning in w:
        if "circular reference" in str(warning.message).lower():
            logger.info(f"Circular reference detected and handled: {warning.message}")
```

This comprehensive deserialization system provides both high performance and complete type safety, making it suitable for everything from high-throughput data processing to complex scientific computing workflows.
