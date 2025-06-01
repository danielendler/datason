# datason Features Overview

datason provides intelligent serialization through a layered architecture of features, from core JSON compatibility to advanced ML/AI object handling and configurable behavior.

## 🎯 Feature Categories

### [Core Serialization](core/index.md)
The foundation layer providing basic JSON compatibility and safety features.

- **Basic Types**: `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **Security**: Circular reference detection, depth limits, size limits
- **Performance**: Optimization for already-serialized data
- **Error Handling**: Graceful fallbacks for unsupported types

### [Advanced Types](advanced-types/index.md)
Extended support for Python's rich type system and specialized objects.

- **Built-in Types**: `complex`, `decimal.Decimal`, `uuid.UUID`, `pathlib.Path`
- **Collections**: `set`, `frozenset`, `namedtuple`, `range`, `bytes`
- **Enums**: Support for `enum.Enum` and custom enumeration classes
- **Type Coercion**: Configurable strategies from strict to aggressive

### [Date/Time Handling](datetime/index.md)
Comprehensive support for temporal data with timezone awareness.

- **Formats**: ISO, Unix timestamp, Unix milliseconds, custom patterns
- **Types**: `datetime`, `date`, `time`, `timedelta`
- **Pandas Integration**: `pd.Timestamp`, `pd.NaT`, `pd.DatetimeIndex`
- **Timezone Support**: Aware and naive datetime handling

### [ML/AI Integration](ml-ai/index.md)
Native support for machine learning and scientific computing objects.

- **PyTorch**: Tensors, models, parameters
- **TensorFlow**: Tensors, variables, SavedModel metadata  
- **Scikit-learn**: Fitted models, pipelines, transformers
- **NumPy**: Arrays, scalars, dtypes
- **JAX**: Arrays and computation graphs
- **PIL/Pillow**: Images with format preservation

### [Pandas Integration](pandas/index.md)
Deep integration with the pandas ecosystem for data science workflows.

- **DataFrames**: Configurable orientation (records, split, index, columns, values, table)
- **Series**: Index preservation and metadata handling
- **Index Types**: RangeIndex, DatetimeIndex, MultiIndex
- **Categorical**: Category metadata and ordering
- **NaN Handling**: Configurable strategies for missing data

### [Configuration System](configuration/index.md)
Fine-grained control over serialization behavior with preset configurations.

- **Presets**: ML, API, Strict, Performance optimized configurations
- **Date Formats**: 5 different datetime serialization formats
- **NaN Handling**: 4 strategies for missing/null values
- **Type Coercion**: 3 levels from strict type preservation to aggressive conversion
- **Custom Serializers**: Register handlers for custom types

### [Pickle Bridge](pickle-bridge/index.md)
Secure migration of legacy ML pickle files to portable JSON format.

- **Security-First**: Class whitelisting prevents arbitrary code execution
- **Zero Dependencies**: Uses only Python standard library
- **ML Coverage**: 54 safe classes covering 95%+ of common pickle files
- **Bulk Processing**: Directory-level conversion with statistics tracking
- **Production Ready**: File size limits, error handling, monitoring

### [Performance Features](performance/index.md)
Optimizations for speed and memory efficiency in production environments.

- **Early Detection**: Skip processing for JSON-compatible data
- **Memory Streaming**: Handle large datasets without full memory loading
- **Configurable Limits**: Prevent resource exhaustion attacks
- **Benchmarking**: Built-in performance measurement tools

## 🚀 Quick Feature Matrix

| Feature Category | Basic | Advanced | Enterprise |
|------------------|-------|----------|------------|
| **Core Types** | ✅ JSON types | ✅ + Python types | ✅ + Custom types |
| **ML/AI Objects** | ❌ | ✅ Common libraries | ✅ + Custom models |
| **Configuration** | ❌ | ✅ Presets | ✅ + Full control |
| **Pickle Bridge** | ❌ | ✅ Safe conversion | ✅ + Bulk migration |
| **Performance** | ✅ Basic | ✅ Optimized | ✅ + Monitoring |
| **Data Science** | ❌ | ✅ Pandas/NumPy | ✅ + Advanced |

## 📖 Usage Patterns

### Simple Usage (Core Features)
```python
import datason

# Works out of the box
data = {"users": [1, 2, 3], "timestamp": datetime.now()}
result = datason.serialize(data)
```

### Configured Usage (Advanced Features)
```python
import datason
from datason.config import get_ml_config

# Optimized for ML workflows
config = get_ml_config()
result = datason.serialize(ml_data, config=config)
```

### Custom Usage (Enterprise Features)
```python
import datason
from datason.config import SerializationConfig, DateFormat, TypeCoercion

# Full control over behavior
config = SerializationConfig(
    date_format=DateFormat.UNIX_MS,
    type_coercion=TypeCoercion.AGGRESSIVE,
    preserve_decimals=True,
    custom_serializers={MyClass: my_serializer}
)
result = datason.serialize(data, config=config)
```

### Pickle Bridge Usage (ML Migration)
```python
import datason

# Convert legacy pickle files safely
result = datason.from_pickle("legacy_model.pkl")

# Bulk migration with security controls
stats = datason.convert_pickle_directory(
    source_dir="old_models/",
    target_dir="json_models/",
    safe_classes=datason.get_ml_safe_classes()
)
```

## 🛣️ Feature Roadmap

### ✅ Available Now
- Core serialization with safety features
- Advanced Python type support
- ML/AI object integration
- Configuration system with presets
- Pandas deep integration
- Performance optimizations
- **Pickle Bridge for legacy ML migration**

### 🔄 In Development
- Schema validation
- Compression support
- Streaming serialization
- Plugin architecture
- Type hints integration

### 🔮 Planned
- GraphQL integration
- Protocol Buffers support
- Arrow format compatibility
- Cloud storage adapters
- Real-time synchronization

## 📚 Learn More

Each feature category has detailed documentation with examples, best practices, and performance considerations:

- **[Core Serialization →](core/index.md)** - Start here for basic usage
- **[Configuration System →](configuration/index.md)** - Control serialization behavior  
- **[ML/AI Integration →](ml-ai/index.md)** - Work with ML frameworks
- **[Performance Guide →](performance/index.md)** - Optimize for production
- **[Migration Guide →](migration/index.md)** - Upgrade from other serializers
