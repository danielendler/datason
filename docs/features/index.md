# datason Features Overview

datason provides intelligent serialization through a layered architecture of features, from core JSON compatibility to advanced ML/AI object handling and configurable behavior.

## 🎯 Feature Categories

### [Core Serialization](core/index.md)
The foundation layer providing basic JSON compatibility and safety features.

- **Basic Types**: `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **Security**: Circular reference detection, depth limits, size limits
- **Performance**: Optimization for already-serialized data
- **Error Handling**: Graceful fallbacks for unsupported types

### [Deserialization & Type Support](deserialization.md) 🆕 v0.6.0
Ultra-fast deserialization with comprehensive type support and intelligent auto-detection.

- **Performance**: 3.73x average improvement, 16.86x on large nested data
- **Type Matrix**: Complete documentation of 133+ supported types
- **Auto-Detection**: Smart recognition of datetime, UUID, and numeric patterns
- **Type Preservation**: Optional metadata for perfect round-trip fidelity
- **Security**: Depth/size limits with zero performance impact

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

### [File Operations](file-operations.md) 🆕 v0.11.0
Complete JSON/JSONL file I/O integrated as first-class citizens in the modern API.

- **Dual Format Support**: Both JSON (.json) and JSONL (.jsonl) with auto-detection
- **Progressive API**: Basic → Smart → Perfect loading complexity
- **Full Integration**: All datason features work with files (ML, security, streaming, compression)
- **Domain-Specific**: Specialized `save_ml()`, `save_secure()`, `save_api()` functions
- **Auto-Compression**: Automatic .gz compression detection and handling

### [Chunked Processing & Streaming](chunked-processing/index.md) 🆕 v0.4.0
Memory-efficient handling of large datasets that exceed available RAM.

- **Chunked Serialization**: Break large objects into manageable pieces
- **Streaming**: Continuous data writing without memory accumulation
- **Memory Estimation**: Automatic optimization recommendations
- **File Formats**: JSONL and JSON array support for chunked data

### [Template-Based Deserialization](template-deserialization/index.md) 🆕 v0.4.5
Type-guided reconstruction with ML-optimized round-trip fidelity.

- **Template Guidance**: Use reference objects to ensure consistent types
- **Auto-Inference**: Generate templates from sample data
- **ML Templates**: Specialized templates for machine learning workflows
- **Type Validation**: Consistent data structure validation

### [Data Utilities with Security Patterns](data-utilities/index.md) 🆕 v0.5.5
Comprehensive data analysis and transformation tools with consistent security protection.

- **Deep Comparison**: Advanced object comparison with tolerance and security limits
- **Anomaly Detection**: Identify large strings, collections, and suspicious patterns
- **Type Enhancement**: Smart type inference and conversion with safety checks
- **Structure Normalization**: Flatten or transform data structures securely
- **Datetime Processing**: Standardize formats and extract temporal features
- **Pandas/NumPy Integration**: Enhanced DataFrame and array processing with limits
- **Configurable Security**: Environment-specific configurations for different trust levels

### [Data Integrity & Verification](../integrity.md) 🆕 v0.10.0
Cryptographic hashing and signature utilities for tamper detection.

- **Canonicalization**: Deterministic JSON for stable hashing
- **Hash & Verify**: Object and JSON hashing with algorithm validation
- **Digital Signatures**: Ed25519 sign/verify support
- **Redaction Integration**: Optional PII removal before hashing

### [ML/AI Integration](ml-ai/index.md)
Native support for machine learning and scientific computing objects.

- **PyTorch**: Tensors, models, parameters
- **TensorFlow**: Tensors, variables, SavedModel metadata
- **Scikit-learn**: Fitted models, pipelines, transformers
- **NumPy**: Arrays, scalars, dtypes
- **JAX**: Arrays and computation graphs
- **PIL/Pillow**: Images with format preservation

### [Model Serving Integration](model-serving/index.md)
Guides for BentoML, Ray Serve, Streamlit, Gradio, MLflow, and Seldon/KServe.

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
- **Domain-Specific**: Financial, Time Series, Inference, Research, Logging presets
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

### [Configurable Caching System](caching/index.md) 🆕 v0.7.0
Intelligent caching that adapts to different workflow requirements with multiple cache scopes.

- **Multiple Scopes**: Operation, Request, Process, and Disabled caching modes
- **Performance**: 50-200% speed improvements for repeated operations
- **ML-Optimized**: Perfect for training loops and data analytics
- **Context Managers**: Easy scope management and isolation
- **Metrics & Monitoring**: Built-in cache performance tracking
- **Object Pooling**: Memory-efficient object reuse with automatic cleanup

### [Performance Features](performance/index.md)
Optimizations for speed and memory efficiency in production environments.

- **Early Detection**: Skip processing for JSON-compatible data
- **Memory Streaming**: Handle large datasets without full memory loading
- **Configurable Limits**: Prevent resource exhaustion attacks
- **Benchmarking**: Built-in performance measurement tools
- **Intelligent Caching**: Context-aware caching for maximum performance

## 🚀 Quick Feature Matrix

| Feature Category | Basic | Advanced | Enterprise |
|------------------|-------|----------|------------|
| **Core Types** | ✅ JSON types | ✅ + Python types | ✅ + Custom types |
| **Large Data** | ❌ | ✅ Chunked/Streaming | ✅ + Memory optimization |
| **Type Safety** | ❌ | ✅ Template validation | ✅ + ML round-trip |
| **ML/AI Objects** | ❌ | ✅ Common libraries | ✅ + Custom models |
| **Configuration** | ❌ | ✅ Presets | ✅ + Full control |
| **Pickle Bridge** | ❌ | ✅ Safe conversion | ✅ + Bulk migration |
| **Performance** | ✅ Basic | ✅ Optimized | ✅ + Monitoring |
| **Caching** | ❌ | ✅ Operation scope | ✅ + All scopes + Metrics |
| **Data Science** | ❌ | ✅ Pandas/NumPy | ✅ + Advanced |
| **Data Utilities** | ❌ | ✅ Basic tools | ✅ + Security patterns |

## 📖 Usage Patterns

### Simple Usage (Core Features)
```python
import datason

# Works out of the box
data = {"users": [1, 2, 3], "timestamp": datetime.now()}
result = datason.serialize(data)
```

### Data Analysis & Transformation (v0.5.5)
```python
import datason

# Compare complex data structures with tolerance
obj1 = {"users": [{"score": 85.5}], "metadata": {"version": "1.0"}}
obj2 = {"users": [{"score": 85.6}], "metadata": {"version": "1.0"}}
comparison = datason.deep_compare(obj1, obj2, tolerance=1e-1)

# Detect anomalies and security issues
messy_data = {"large_text": "x" * 50000, "items": list(range(5000))}
anomalies = datason.find_data_anomalies(messy_data)

# Smart type enhancement with security
raw_data = {"id": "123", "score": "85.5", "active": "true"}
enhanced, report = datason.enhance_data_types(raw_data)
# enhanced["id"] is now int(123), not string

# Configurable security for different environments
from datason.utils import UtilityConfig
api_config = UtilityConfig(max_depth=10, max_object_size=10_000)
result = datason.find_data_anomalies(untrusted_data, config=api_config)
```

### Large Data Processing (v0.4.0)
```python
import datason

# Memory-efficient processing of large datasets
large_data = create_huge_dataset()  # Multi-GB dataset

# Process in chunks without memory overflow
result = datason.serialize_chunked(large_data, chunk_size=10000)

# Stream data to file
with datason.stream_serialize("output.jsonl") as stream:
    for item in continuous_data_source():
        stream.write(item)
```

### Type-Safe Deserialization (v0.4.5)
```python
import datason
from datason.deserializers import deserialize_with_template

# Ensure consistent types with templates
template = {"user_id": 0, "created": datetime.now(), "active": True}
data = {"user_id": "123", "created": "2023-01-01T10:00:00", "active": "true"}

# Template ensures proper type conversion
result = deserialize_with_template(data, template)
# result["user_id"] is int(123), not string
```

### Configured Usage (Advanced Features)
```python
import datason
from datason.config import get_ml_config

# Optimized for ML workflows
config = get_ml_config()
result = datason.serialize(ml_data, config=config)
```

### High-Performance Caching (v0.7.0)
```python
import datason
from datason import CacheScope

# ML training with maximum performance
datason.set_cache_scope(CacheScope.PROCESS)
for epoch in range(num_epochs):
    for batch in training_data:
        # Repeated datetime/UUID patterns cached automatically
        parsed_batch = datason.deserialize_fast(batch)  # 150-200% faster!
        train_step(parsed_batch)

# Web API with request-scoped caching
def api_handler(request_data):
    with datason.request_scope():
        # Cache shared within request, cleared between requests
        return process_api_request(request_data)

# Monitor cache performance
metrics = datason.get_cache_metrics()
print(f"Cache hit rate: {metrics[CacheScope.PROCESS].hit_rate:.1%}")
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
- Pickle Bridge for legacy ML migration
- **🆕 Chunked processing & streaming (v0.4.0)**
- **🆕 Template-based deserialization (v0.4.5)**
- **🆕 Data utilities with security patterns (v0.5.5)**

### 🔄 In Development
- Schema validation
- Compression support
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
- **[Chunked Processing →](chunked-processing/index.md)** - 🆕 Handle large datasets efficiently
- **[Template Deserialization →](template-deserialization/index.md)** - 🆕 Type-safe reconstruction
- **[Data Utilities →](data-utilities/index.md)** - 🆕 Analysis & transformation with security
- **[Configuration System →](configuration/index.md)** - Control serialization behavior  
- **[ML/AI Integration →](ml-ai/index.md)** - Work with ML frameworks
- **[Performance Guide →](performance/index.md)** - Optimize for production
- **[Migration Guide →](migration/index.md)** - Upgrade from other serializers
