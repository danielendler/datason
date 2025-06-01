# Core Serialization

The foundation of datason's intelligent serialization, providing basic JSON compatibility, safety features, and high-performance serialization for standard Python types.

## 🎯 Overview

Core serialization handles the fundamental building blocks of data serialization:

- **Standard JSON Types**: Direct support for `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **Safety Features**: Circular reference detection, depth limits, size limits
- **Performance**: Optimized paths for already-serialized data
- **Error Handling**: Graceful fallbacks for unsupported types

## 📦 Core Functions

### `serialize()`

The main serialization function that intelligently converts Python objects to JSON-serializable formats.

```python
import datason

# Basic usage
data = {"users": [1, 2, 3], "active": True}
result = datason.serialize(data)

# With configuration
config = datason.get_strict_config()
result = datason.serialize(data, config=config)
```

**Function Signature:**
```python
def serialize(obj: Any, config: Optional[SerializationConfig] = None) -> Any:
    """Serialize Python objects to JSON-compatible format."""
```

**Parameters:**
- `obj`: Any Python object to serialize
- `config`: Optional configuration to control serialization behavior

**Returns:** JSON-serializable representation of the input object

### `safe_serialize()`

Enhanced serialization with additional safety checks and error handling.

```python
# Safe serialization with automatic fallbacks
result = datason.safe_serialize(complex_data)
```

## 🛡️ Safety Features

### Circular Reference Detection

Prevents infinite loops when objects reference themselves:

```python
# Create circular reference
data = {"self": None}
data["self"] = data

# datason handles this safely
result = datason.serialize(data)
# Returns: {"self": "<Circular Reference Detected>"}
```

### Resource Limits

Protects against resource exhaustion attacks:

```python
# Configurable limits
config = datason.SerializationConfig(
    max_depth=1000,        # Maximum nesting depth
    max_string_length=1_000_000,  # Maximum string length
    max_sequence_length=10_000_000  # Maximum list/dict size
)
```

### Input Validation

Type checking and safe handling of all input types:

```python
class CustomObject:
    def __init__(self):
        self.data = "sensitive"

    def __dict__(self):
        raise RuntimeError("Access denied")

# datason handles safely
obj = CustomObject()
result = datason.serialize(obj)
# Returns: Safe string representation, no error exposure
```

## ⚡ Performance Features

### Early Detection

Skip processing for JSON-compatible data:

```python
# Already JSON-compatible data is passed through efficiently
json_data = {"string": "value", "number": 42, "boolean": True}
result = datason.serialize(json_data)  # Fast path
```

### Optimized Type Handling

Different optimization strategies based on data types:

```python
# String optimization
large_text = "x" * 100_000
result = datason.serialize(large_text)  # Memory-efficient handling

# Numeric optimization  
numbers = list(range(10_000))
result = datason.serialize(numbers)  # Vectorized processing
```

## 🔧 Error Handling

### Graceful Fallbacks

When objects can't be serialized normally, datason provides safe fallbacks:

```python
class UnserializableObject:
    def __init__(self):
        self._data = lambda x: x  # Function objects can't be serialized

obj = UnserializableObject()
result = datason.serialize(obj)
# Returns: String representation with type information
```

### Security Error Handling

Specific handling for security-related issues:

```python
try:
    result = datason.serialize(potentially_malicious_data)
except datason.SecurityError as e:
    print(f"Security issue detected: {e}")
    # Handle security violation appropriately
```

## 📊 Supported Types

### Native JSON Types
| Type | Example | Notes |
|------|---------|--------|
| `str` | `"hello"` | Direct pass-through |
| `int` | `42` | Direct pass-through |
| `float` | `3.14` | Direct pass-through |
| `bool` | `True` | Direct pass-through |
| `None` | `None` | Converted to `null` |
| `list` | `[1, 2, 3]` | Recursive serialization |
| `dict` | `{"key": "value"}` | Recursive serialization |

### Extended Support
| Type | Example | Serialization |
|------|---------|---------------|
| `tuple` | `(1, 2, 3)` | → `[1, 2, 3]` |
| `set` | `{1, 2, 3}` | → `[1, 2, 3]` |
| `frozenset` | `frozenset([1, 2])` | → `[1, 2]` |
| `bytes` | `b"data"` | → Base64 string |
| `bytearray` | `bytearray(b"data")` | → Base64 string |

## 🚀 Usage Examples

### Basic Serialization

```python
import datason

# Simple data structures
user_data = {
    "id": 12345,
    "name": "Alice",
    "active": True,
    "tags": ["python", "datascience"],
    "metadata": None
}

result = datason.serialize(user_data)
print(result)
# Output: Same structure, ready for JSON.dumps()
```

### Complex Nested Data

```python
# Deeply nested structures
complex_data = {
    "users": [
        {"id": i, "profile": {"settings": {"theme": "dark"}}}
        for i in range(1000)
    ],
    "metadata": {
        "version": "1.0",
        "timestamps": {
            "created": "2024-01-01",
            "updated": "2024-06-01"
        }
    }
}

result = datason.serialize(complex_data)
# Handles deep nesting efficiently
```

### Error-Prone Data

```python
# Data that might cause issues
problematic_data = {
    "numbers": [1, 2, float('inf'), float('nan')],
    "circular": None,
    "large_string": "x" * 10_000_000
}
problematic_data["circular"] = problematic_data

# datason handles gracefully
result = datason.serialize(problematic_data)
# Safe result with appropriate fallbacks
```

## 🔍 Configuration Options

Core serialization can be customized through configuration:

```python
from datason import SerializationConfig

# Custom configuration
config = SerializationConfig(
    # Safety limits
    max_depth=500,
    max_string_length=100_000,

    # Behavior options
    sort_keys=True,
    ensure_ascii=False,

    # Error handling
    strict_mode=False,
    fallback_to_string=True
)

result = datason.serialize(data, config=config)
```

## 🛠️ Advanced Usage

### Custom Serializers

Register custom handlers for specific types:

```python
def serialize_custom_type(obj):
    return {"type": "custom", "data": str(obj)}

config = datason.SerializationConfig(
    custom_serializers={MyCustomType: serialize_custom_type}
)

result = datason.serialize(custom_obj, config=config)
```

### Monitoring Performance

Track serialization performance:

```python
import time

start = time.perf_counter()
result = datason.serialize(large_dataset)
duration = time.perf_counter() - start

print(f"Serialized {len(result)} objects in {duration:.3f}s")
```

### Security Monitoring

Monitor for security warnings:

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    result = datason.serialize(untrusted_data)

    if w:
        for warning in w:
            print(f"Security warning: {warning.message}")
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - Customize serialization behavior
- **[Advanced Types](../advanced-types/index.md)** - Extended type support  
- **[Security](../../SECURITY.md)** - Security considerations and best practices
- **[Performance](../performance/index.md)** - Optimization techniques

## 🚀 Next Steps

- **[Advanced Types →](../advanced-types/index.md)** - Handle complex Python types
- **[Configuration →](../configuration/index.md)** - Customize behavior for your use case
- **[ML/AI Integration →](../ml-ai/index.md)** - Work with machine learning objects
