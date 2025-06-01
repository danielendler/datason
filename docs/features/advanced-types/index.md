# Advanced Type Handling

datason extends beyond basic JSON types to support Python's rich type system through intelligent type detection and configurable coercion strategies.

## Overview

While standard JSON supports only 6 basic types (`str`, `int`, `float`, `bool`, `null`, `array`, `object`), Python applications use dozens of specialized types. datason bridges this gap with the `TypeHandler` system.

```python
import datason
from datason.config import TypeCoercion, SerializationConfig
import decimal
import uuid
from pathlib import Path

# Complex data with advanced types
data = {
    "id": uuid.uuid4(),
    "amount": decimal.Decimal("123.456"),
    "path": Path("/data/file.txt"),
    "complex_num": 3+4j,
    "coordinates": {1, 2, 3},  # set
    "range_data": range(10, 100, 5)
}

# Automatic handling
result = datason.serialize(data)
```

## 🎯 Supported Advanced Types

### Numeric Types

#### `decimal.Decimal`
High-precision decimal arithmetic with configurable handling:

```python
import decimal

amount = decimal.Decimal("123.456789")

# Preserve precision (strict mode)
config = SerializationConfig(
    type_coercion=TypeCoercion.STRICT,
    preserve_decimals=True
)
result = datason.serialize({"amount": amount}, config=config)
# Result: {
#   "amount": {
#     "_type": "decimal",
#     "value": "123.456789",
#     "precision": 9,
#     "scale": 6
#   }
# }

# Convert to float (safe mode, default)
config = SerializationConfig(preserve_decimals=False)
result = datason.serialize({"amount": amount}, config=config)
# Result: {"amount": 123.456789}
```

#### `complex`
Complex numbers with real and imaginary parts:

```python
num = 3+4j

# Preserve as object (strict mode)
config = SerializationConfig(
    type_coercion=TypeCoercion.STRICT,
    preserve_complex=True
)
result = datason.serialize({"num": num}, config=config)
# Result: {"num": {"_type": "complex", "real": 3.0, "imag": 4.0}}

# Convert to list (aggressive mode)
config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
result = datason.serialize({"num": num}, config=config)
# Result: {"num": [3.0, 4.0]}

# Convert to string (safe mode, default)
result = datason.serialize({"num": num})
# Result: {"num": "(3+4j)"}
```

### Identifier Types

#### `uuid.UUID`
Universally unique identifiers with metadata preservation:

```python
import uuid

id_val = uuid.uuid4()

# Strict mode - full metadata
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"id": id_val}, config=config)
# Result: {
#   "id": {
#     "_type": "uuid",
#     "hex": "550e8400e29b41d4a716446655440000",
#     "version": 4,
#     "variant": "specified in RFC 4122"
#   }
# }

# Safe mode - string representation (default)
result = datason.serialize({"id": id_val})
# Result: {"id": "550e8400-e29b-41d4-a716-446655440000"}
```

### Path Types

#### `pathlib.Path`
File system paths with rich metadata:

```python
from pathlib import Path

path = Path("/data/analysis/results.csv")

# Strict mode - full path information
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"path": path}, config=config)
# Result: {
#   "path": {
#     "_type": "path",
#     "path": "/data/analysis/results.csv",
#     "is_absolute": true,
#     "exists": false,
#     "suffix": ".csv",
#     "stem": "results"
#   }
# }

# Safe mode - string representation (default)
result = datason.serialize({"path": path})
# Result: {"path": "/data/analysis/results.csv"}
```

### Enumeration Types

#### `enum.Enum`
Enumerated constants with class information:

```python
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

status = Status.PENDING

# Strict mode - full enum information
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"status": status}, config=config)
# Result: {
#   "status": {
#     "_type": "enum",
#     "class": "mymodule.Status",
#     "name": "PENDING",
#     "value": "pending"
#   }
# }

# Safe mode - enum value (default)
result = datason.serialize({"status": status})
# Result: {"status": "pending"}
```

### Collection Types

#### `set` and `frozenset`
Unordered collections converted to sorted lists:

```python
data = {
    "tags": {"python", "datason", "serialization"},
    "immutable_tags": frozenset({"dev", "prod"})
}

result = datason.serialize(data)
# Result: {
#   "tags": ["datason", "python", "serialization"],  # sorted
#   "immutable_tags": ["dev", "prod"]  # sorted
# }

# Note: Sets with non-sortable elements become unsorted lists
mixed_set = {1, "a", 3.14}
result = datason.serialize({"mixed": mixed_set})
# Result: {"mixed": [1, "a", 3.14]}  # order not guaranteed
```

#### `range`
Range objects with start/stop/step information:

```python
r = range(10, 100, 5)

# Strict mode - range metadata
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"range": r}, config=config)
# Result: {
#   "range": {
#     "_type": "range",
#     "start": 10,
#     "stop": 100,
#     "step": 5
#   }
# }

# Safe mode - expanded list (if reasonable size)
result = datason.serialize({"range": r})
# Result: {"range": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]}

# Large ranges preserve metadata to avoid memory issues
large_range = range(0, 1_000_000)
result = datason.serialize({"large": large_range})
# Result: {
#   "large": {
#     "_type": "range",
#     "start": 0,
#     "stop": 1000000,
#     "step": 1,
#     "_note": "Range too large to expand"
#   }
# }
```

#### `namedtuple`
Named tuples with field information:

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
point = Point(10, 20)

# Strict mode - includes type information
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"point": point}, config=config)
# Result: {
#   "point": {
#     "x": 10,
#     "y": 20,
#     "_type": "namedtuple",
#     "_class": "mymodule.Point"
#   }
# }

# Safe mode - dictionary without metadata (default)
result = datason.serialize({"point": point})
# Result: {"point": {"x": 10, "y": 20}}
```

### Binary Types

#### `bytes` and `bytearray`
Binary data with encoding strategies:

```python
# Text bytes
text_bytes = "Hello, 世界".encode('utf-8')

# Strict mode - hex with metadata
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"data": text_bytes}, config=config)
# Result: {
#   "data": {
#     "_type": "bytes",
#     "data": "48656c6c6f2c20e4b896e7958c",
#     "length": 13
#   }
# }

# Safe mode - UTF-8 decode if possible, hex otherwise
result = datason.serialize({"data": text_bytes})
# Result: {"data": "Hello, 世界"}

# Binary data (non-UTF-8)
binary_bytes = bytes([0xFF, 0xFE, 0x00, 0x01])
result = datason.serialize({"binary": binary_bytes})
# Result: {"binary": "fffe0001"}  # hex representation

# ByteArray handled similarly
byte_array = bytearray([65, 66, 67])
result = datason.serialize({"array": byte_array})
# Result: {"array": "ABC"}  # UTF-8 decoded
```

## 🎛️ Type Coercion Strategies

### TypeCoercion.STRICT
Preserves maximum type information for perfect reconstruction:

```python
from datason.config import TypeCoercion

config = SerializationConfig(type_coercion=TypeCoercion.STRICT)

data = {
    "decimal": decimal.Decimal("123.45"),
    "uuid": uuid.uuid4(),
    "complex": 3+4j,
    "path": Path("/tmp/file.txt")
}

result = datason.serialize(data, config=config)
# All types include "_type" metadata and full information
```

**Use when:**
- Perfect reconstruction is required
- Debugging complex data structures
- Research/scientific computing with precision requirements

### TypeCoercion.SAFE (Default)
Balances compatibility with information preservation:

```python
config = SerializationConfig(type_coercion=TypeCoercion.SAFE)

data = {
    "decimal": decimal.Decimal("123.45"),
    "uuid": uuid.uuid4(),
    "complex": 3+4j,
    "path": Path("/tmp/file.txt")
}

result = datason.serialize(data, config=config)
# Result: {
#   "decimal": 123.45,                    # float
#   "uuid": "uuid-string",                # string
#   "complex": "(3+4j)",                  # string
#   "path": "/tmp/file.txt"               # string
# }
```

**Use when:**
- Building APIs with broad compatibility
- Working with multiple systems/languages
- Need human-readable output

### TypeCoercion.AGGRESSIVE
Converts to simplest possible JSON-compatible types:

```python
config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)

data = {
    "decimal": decimal.Decimal("123.45"),
    "complex": 3+4j,
    "range": range(3),
    "set": {1, 2, 3}
}

result = datason.serialize(data, config=config)
# Result: {
#   "decimal": 123.45,                    # float
#   "complex": [3.0, 4.0],                # list [real, imag]
#   "range": [0, 1, 2],                   # expanded list
#   "set": [1, 2, 3]                      # sorted list
# }
```

**Use when:**
- Maximum performance is required
- Working with systems that only understand basic JSON
- ML pipelines where simple types are preferred

## 🔧 Custom Type Handlers

Register your own serialization logic for custom types:

```python
from dataclasses import dataclass
from datason.config import SerializationConfig

@dataclass
class DatabaseRecord:
    id: int
    created_at: datetime
    data: dict

def serialize_db_record(record):
    """Custom serializer for database records."""
    return {
        "_type": "db_record",
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "data": record.data,
        "_table": "records"
    }

config = SerializationConfig(
    custom_serializers={
        DatabaseRecord: serialize_db_record
    }
)

record = DatabaseRecord(123, datetime.now(), {"key": "value"})
result = datason.serialize({"record": record}, config=config)
```

### Multiple Custom Handlers
```python
config = SerializationConfig(
    custom_serializers={
        MyModel: serialize_ml_model,
        MyUser: serialize_user,
        MyConfig: serialize_config,
    }
)
```

### Conditional Handlers
```python
def smart_serializer(obj):
    """Different serialization based on object state."""
    if obj.is_large():
        return {"_type": "large_object", "summary": obj.summary()}
    else:
        return {"_type": "small_object", "data": obj.data}

config = SerializationConfig(
    custom_serializers={MyClass: smart_serializer}
)
```

## 🧪 Advanced Type Examples

### Working with Pandas Categorical
```python
import pandas as pd

# Categorical data
cat_data = pd.Categorical(['A', 'B', 'A', 'C'], categories=['A', 'B', 'C'], ordered=True)

# Strict mode - preserve category metadata
config = SerializationConfig(type_coercion=TypeCoercion.STRICT)
result = datason.serialize({"categories": cat_data}, config=config)
# Result: {
#   "categories": {
#     "_type": "categorical",
#     "categories": ["A", "B", "C"],
#     "codes": [0, 1, 0, 2],
#     "ordered": true
#   }
# }

# Safe mode - convert to list of values
result = datason.serialize({"categories": cat_data})
# Result: {"categories": ["A", "B", "A", "C"]}
```

### Nested Advanced Types
```python
from collections import OrderedDict

nested_data = {
    "metadata": {
        "id": uuid.uuid4(),
        "created": datetime.now(),
        "precision_value": decimal.Decimal("99.999999")
    },
    "coordinates": {
        "complex_points": [1+2j, 3+4j, 5+6j],
        "paths": [Path("/data/file1.txt"), Path("/data/file2.txt")]
    },
    "tags": {"machine-learning", "python", "serialization"}
}

# All advanced types handled recursively
result = datason.serialize(nested_data)
```

### Type Coercion with Pandas Integration
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ids": [uuid.uuid4() for _ in range(3)],
    "amounts": [decimal.Decimal(f"{i}.99") for i in range(3)],
    "complex_values": [1+2j, 3+4j, 5+6j]
})

# Aggressive coercion simplifies DataFrame contents
config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
result = datason.serialize({"data": df}, config=config)
# UUIDs → strings, Decimals → floats, Complex → [real, imag] lists
```

## 🚀 Performance Considerations

### Type Detection Overhead
```python
# Type handlers add minimal overhead
import time

simple_data = {"values": [1, 2, 3, 4, 5]}
complex_data = {"uuid": uuid.uuid4(), "decimal": decimal.Decimal("123.45")}

# Simple data - no type handler overhead
start = time.time()
datason.serialize(simple_data)
simple_time = time.time() - start

# Complex data - type handler processing
start = time.time()
datason.serialize(complex_data)
complex_time = time.time() - start

# Overhead is typically < 10% for most workloads
```

### Large Collection Handling
```python
# Sets and ranges have special handling for large sizes
large_set = set(range(100_000))
large_range = range(1_000_000)

# Efficient handling without memory explosion
result = datason.serialize({
    "large_set": large_set,      # Converted efficiently
    "large_range": large_range   # Metadata only
})
```

### Custom Serializer Performance
```python
def fast_serializer(obj):
    """Optimized for speed."""
    return {"id": obj.id, "key": obj.key}  # Only essential fields

def detailed_serializer(obj):
    """Comprehensive but slower."""
    return {
        "id": obj.id,
        "all_fields": obj.__dict__,
        "metadata": obj.get_metadata(),
        "computed": obj.expensive_computation()
    }

# Choose based on performance vs information needs
config = SerializationConfig(
    custom_serializers={MyClass: fast_serializer}
)
```

## 🔍 Utility Functions

datason provides utility functions for type analysis and handling:

### `is_nan_like()`
```python
from datason.type_handlers import is_nan_like
import numpy as np
import pandas as pd

# Check various NaN-like values
is_nan_like(None)                  # True
is_nan_like(float('nan'))          # True
is_nan_like(np.nan)                # True
is_nan_like(pd.NaT)                # True
is_nan_like(pd.NA)                 # True
is_nan_like(42)                    # False
```

### `normalize_numpy_types()`
```python
from datason.type_handlers import normalize_numpy_types
import numpy as np

# Convert numpy types to Python natives
np_int = np.int64(42)
np_float = np.float32(3.14)
np_bool = np.bool_(True)

normalized_int = normalize_numpy_types(np_int)      # int(42)
normalized_float = normalize_numpy_types(np_float)  # float(3.14)
normalized_bool = normalize_numpy_types(np_bool)    # bool(True)
```

### `get_object_info()`
```python
from datason.type_handlers import get_object_info

# Analyze objects for debugging
info = get_object_info(my_complex_object)
# Returns: {
#   "type": "MyClass",
#   "module": "mymodule",
#   "mro": ["MyClass", "BaseClass", "object"],
#   "size": 42,
#   "is_callable": False,
#   "has_dict": True,
#   "sample_types": ["str", "int", "float"]
# }
```

## 📚 Best Practices

### 1. Start with Safe Coercion
Begin with `TypeCoercion.SAFE` (default) and adjust based on your needs.

### 2. Profile Your Types
Use `get_object_info()` to understand your data structure before choosing coercion strategies.

### 3. Document Custom Serializers
Custom serializers should include documentation about their behavior and any data transformations.

### 4. Handle Edge Cases
Test your type handlers with edge cases like empty collections, None values, and circular references.

### 5. Consider Reconstruction
If you need to deserialize back to original objects, use `TypeCoercion.STRICT` and preserve type metadata.

## 🔗 Related Documentation

- **[Configuration System →](../configuration/index.md)** - Control type coercion behavior
- **[Pandas Integration →](../pandas/index.md)** - Pandas-specific type handling
- **[Performance Guide →](../performance/index.md)** - Optimize type handling for speed
- **[Core Serialization →](../core/index.md)** - Basic type support
