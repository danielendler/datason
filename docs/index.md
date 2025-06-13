# 🚀 datason Documentation

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

## 🎯 **NEW: UUID + Pydantic Compatibility Solved!** ⭐

**The #1 issue when integrating with FastAPI, Django, and Pydantic is now solved:**

```python
import datason
from datason.config import get_api_config
from pydantic import BaseModel

# ❌ Problem: UUIDs become objects, breaking Pydantic validation
data = {"user_id": "12345678-1234-5678-9012-123456789abc"}
result = datason.auto_deserialize(data)  # UUID object - fails Pydantic!

# ✅ Solution: Use API config to keep UUIDs as strings
api_config = get_api_config()
result = datason.auto_deserialize(data, config=api_config)  # UUID string - works!

class User(BaseModel):
    user_id: str  # ✅ Now works perfectly!

user = User(**result)  # Success! 🎉
```

**Perfect for:** FastAPI APIs, Django REST Framework, Flask JSON endpoints, any Pydantic application

[**📖 Read the complete API integration guide →**](features/api-integration.md)

---

## 🎯 Two Powerful Approaches

=== "Modern API - Intention-Revealing"

    ```python
    import datason as ds

    # 🎯 Clear function names that tell you exactly what they do
    user_data = {"name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"}

    # Security-focused with automatic PII redaction
    secure_data = ds.dump_secure(user_data, redact_pii=True)

    # ML-optimized for models and tensors
    import torch
    model_data = {"model": torch.nn.Linear(10, 1), "weights": torch.randn(10, 1)}
    ml_data = ds.dump_ml(model_data)

    # API-safe clean JSON for web endpoints
    api_response = ds.dump_api({"status": "success", "data": [1, 2, 3]})

    # 📈 Progressive complexity for deserialization
    json_data = '{"values": [1, 2, 3], "metadata": {"created": "2024-01-01T12:00:00"}}'

    # Basic: Fast exploration (60-70% success rate)
    basic = ds.load_basic(json_data)

    # Smart: Production-ready (80-90% success rate)  
    smart = ds.load_smart(json_data)

    # Perfect: Template-based (100% success rate)
    from datetime import datetime
    template = {"values": [int], "metadata": {"created": datetime}}
    perfect = ds.load_perfect(json_data, template)

    # 🔍 Built-in help and discovery
    ds.help_api()  # Get personalized recommendations
    ```

=== "Traditional API - UUID Compatible"

    ```python
    import datason as ds
    from datason.config import get_api_config
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Complex data that "just works"
    data = {
        'user_id': "12345678-1234-5678-9012-123456789abc",  # UUID string
        'dataframe': pd.DataFrame({'A': [1, 2, 3], 'B': [4.5, 5.5, 6.5]}),
        'timestamp': datetime.now(),
        'array': np.array([1, 2, 3, 4, 5]),
        'nested': {'values': [1, 2, 3], 'metadata': {'created': datetime.now()}}
    }

    # Use API config for Pydantic/FastAPI compatibility
    api_config = get_api_config()
    json_data = ds.serialize(data, config=api_config)

    # Deserialize back - UUIDs stay as strings, other types preserved!
    restored = ds.deserialize(json_data, config=api_config)
    assert type(restored['user_id']) == str  # ✅ UUID stays as string
    assert type(restored['dataframe']) == pd.DataFrame
    assert type(restored['array']) == np.ndarray

    # Perfect for FastAPI/Pydantic:
    from pydantic import BaseModel
    class DataModel(BaseModel):
        user_id: str  # Works perfectly!
        # ... other fields

    model = DataModel(**restored)  # ✅ Success!
    ```

## ✨ Key Features

### 🌐 **Web Framework Integration** ⭐ **NEW**
- **FastAPI + Pydantic**: Perfect UUID string compatibility with `get_api_config()`
- **Django REST Framework**: Seamless model serialization with proper UUID handling
- **Flask APIs**: Clean JSON output with consistent type handling
- **Production Ready**: Used in real financial and ML applications

### 🧠 **Intelligent & Automatic**
- **Smart Type Detection**: Automatically handles pandas DataFrames, NumPy arrays, datetime objects, and more
- **Bidirectional**: Serialize to JSON and deserialize back to original objects with type preservation
- **Zero Configuration**: Works out of the box with sensible defaults

### 🚀 **ML/AI Optimized**
- **ML Library Support**: PyTorch tensors, TensorFlow objects, scikit-learn models, Hugging Face tokenizers
- **Large Data Handling**: Chunked processing for memory-efficient serialization
- **Template Deserialization**: Consistent data structure enforcement for ML pipelines

### 🛡️ **Enterprise Ready**
- **Data Privacy**: Comprehensive redaction engine for sensitive data (PII, financial, healthcare)
- **Security**: Safe deserialization with configurable security policies
- **Audit Trail**: Complete logging and compliance tracking
- **Performance**: Optimized for speed with minimal overhead

### 🔧 **Highly Configurable**
- **Multiple Presets**: ML, API, financial, healthcare, research configurations
- **Fine-grained Control**: Custom serializers, type handlers, and processing rules
- **Extensible**: Easy to add custom serializers for your own types

### 🎯 **Modern API Design** *(New!)*
- **Intention-Revealing Names**: Functions clearly indicate purpose and expected outcomes
- **Progressive Complexity**: Clear path from basic exploration to production-critical applications  
- **Domain-Specific Optimizations**: Specialized functions for ML, API, and security use cases
- **Compositional Utilities**: Combine features like `secure + chunked + ml_mode`
- **Built-in Discovery**: `help_api()` and `get_api_info()` for self-documentation

## 🎯 Quick Navigation

=== "👨‍💻 For Developers"

    **Getting Started**

    - [🚀 Quick Start Guide](user-guide/quick-start.md) - Get up and running in 5 minutes
    - [🌐 **API Integration Guide**](features/api-integration.md) ⭐ **FastAPI/Django/Flask integration**
    - [🎯 Modern API Guide](user-guide/modern-api-guide.md) - Complete guide to intention-revealing functions
    - [💡 Examples Gallery](user-guide/examples/index.md) - Common use cases and patterns
    - [🔧 Configuration Guide](features/configuration/index.md) - Customize behavior for your needs

    **Core Features**

    - [📊 Data Types Support](features/advanced-types/index.md) - All supported types and conversion
    - [🤖 ML/AI Integration](features/ml-ai/index.md) - Machine learning library support
    - [🔐 Data Privacy & Redaction](features/redaction.md) - Protect sensitive information
    - [⚡ Performance & Chunking](features/performance/index.md) - Handle large datasets efficiently

    **Advanced Usage**

    - [🎯 Template Deserialization](features/template-deserialization/index.md) - Enforce data structures
    - [🔄 Pickle Bridge](features/pickle-bridge/index.md) - Migrate from legacy pickle files
    - [🔍 Type Detection](features/core/index.md) - How automatic detection works

=== "🤖 For AI Systems"

    **Integration Guides**

    - [🤖 AI Integration Guide](ai-guide/overview.md) - How to integrate datason in AI systems
    - [🌐 **API Integration**](features/api-integration.md) ⭐ **Pydantic/FastAPI compatibility**
    - [📦 Pydantic & Marshmallow Integration](features/pydantic-marshmallow-integration.md) - Serialize validated objects
    - [📝 API Reference](api/index.md) - Complete API documentation with examples
    - [🔧 Configuration Presets](features/configuration/index.md) - Pre-built configs for common AI use cases

    **Automation & Tooling**

    - [⚙️ Auto-Detection Capabilities](features/core/index.md) - What datason can detect automatically
    - [🔌 Custom Serializers](AI_USAGE_GUIDE.md) - Extend for custom types
    - [📊 Schema Inference](features/template-deserialization/index.md) - Automatic schema generation

    **Deployment**

    - [🚀 Production Deployment](BUILD_PUBLISH.md) - Best practices for production
    - [🔍 Monitoring & Logging](CI_PERFORMANCE.md) - Track serialization performance
    - [🛡️ Security Considerations](community/security.md) - Security best practices

## 🚀 **Quick Start: Web API Integration**

Perfect for FastAPI, Django, Flask developers:

```python
# 1. Install
pip install datason

# 2. Import and configure
import datason
from datason.config import get_api_config

# 3. Set up once, use everywhere
API_CONFIG = get_api_config()

# 4. Process any data - UUIDs stay as strings!
data = {"user_id": "12345678-1234-5678-9012-123456789abc", "name": "John"}
result = datason.auto_deserialize(data, config=API_CONFIG)

# 5. Works with Pydantic!
from pydantic import BaseModel
class User(BaseModel):
    user_id: str
    name: str

user = User(**result)  # ✅ Perfect!
```

[**📖 Complete integration guide →**](features/api-integration.md) | [**🏃‍♂️ Quick examples →**](user-guide/examples/index.md)

## 🎯 Modern API Functions

The new modern API provides intention-revealing function names with progressive complexity:

### 🔹 Serialization (Dump Functions)

| Function | Purpose | Use Case | Features |
|----------|---------|----------|----------|
| `dump()` | General-purpose serialization | Most scenarios | Composable options |
| `dump_ml()` | ML-optimized | Models, tensors, NumPy | ML library support |
| `dump_api()` | API-safe | Web endpoints | Clean JSON output |
| `dump_secure()` | Security-focused | Sensitive data | PII redaction |
| `dump_fast()` | Performance-optimized | High-throughput | Minimal overhead |
| `dump_chunked()` | Memory-efficient | Large objects | Chunked processing |
| `stream_dump()` | File streaming | Very large data | Direct to file |

### 🔹 Deserialization (Load Functions)

| Function | Success Rate | Speed | Use Case |
|----------|-------------|-------|----------|
| `load_basic()` | 60-70% | Fastest | Simple objects, exploration |
| `load_smart()` | 80-90% | Moderate | General purpose, production |
| `load_perfect()` | 100% | Fast | Critical apps (needs template) |
| `load_typed()` | 95% | Fast | When metadata available |

### 🔹 Utility & Discovery

```python
# Get personalized API recommendations
ds.help_api()

# Explore available functions and features
api_info = ds.get_api_info()
print(api_info['dump_functions'])    # List all dump functions
print(api_info['recommendations'])   # Usage recommendations

# JSON module compatibility
data_str = ds.dumps({"key": "value"})  # Like json.dumps()
data_obj = ds.loads(data_str)          # Like json.loads()
```

## 📚 Documentation Sections

### 📖 User Guide
Comprehensive guides for getting started and using datason effectively.

- **[Quick Start](user-guide/quick-start.md)** - Installation and first steps
- **[Examples Gallery](user-guide/examples/index.md)** - Code examples for every feature

### 🔧 Features
Detailed documentation for all datason features.

- **[Features Overview](features/index.md)** - Complete feature overview
- **[Core Serialization](features/core/index.md)** - Core serialization functionality
- **[ML/AI Integration](features/ml-ai/index.md)** - PyTorch, TensorFlow, scikit-learn support
- **[Data Privacy & Redaction](features/redaction.md)** - PII protection and compliance
- **[Performance & Chunking](features/performance/index.md)** - Memory-efficient processing
- **[Template System](features/template-deserialization/index.md)** - Structure enforcement
- **[Pickle Bridge](features/pickle-bridge/index.md)** - Legacy pickle migration

### 🤖 AI Developer Guide  
Specialized documentation for AI systems and automated workflows.

- **[AI Integration Overview](ai-guide/overview.md)** - Integration patterns for AI systems

### 📋 API Reference
Complete API documentation with examples.

- **[API Overview](api/index.md)** - Complete API documentation with examples

### 🔬 Advanced Topics
In-depth technical documentation.

- **[Performance Benchmarks](advanced/benchmarks.md)** - Performance analysis and comparisons
- **[Core Strategy](core-serialization-strategy.md)** - Internal design and architecture
- **[Performance Improvements](performance-improvements.md)** - Optimization techniques

### 👥 Community & Development
Resources for contributors and the community.

- **[Contributing Guide](community/contributing.md)** - How to contribute to datason
- **[Release Notes](community/changelog.md)** - Version history and changes
- **[Roadmap](community/roadmap.md)** - Future development plans
- **[Security Policy](community/security.md)** - Security practices and reporting

## 🚀 Quick Examples

### Basic Serialization

```python
import datason as ds

# Simple data
data = {"numbers": [1, 2, 3], "text": "hello world"}
serialized = ds.serialize(data)
restored = ds.deserialize(serialized)

# Modern API equivalent
serialized_modern = ds.dump(data)  # Same result, clearer intent
restored_modern = ds.load_smart(serialized_modern)
```

### ML Workflow Example

```python
import torch
import datason as ds

# ML model and data
model = torch.nn.Linear(10, 1)
data = {"model": model, "weights": torch.randn(10, 1)}

# Traditional API with ML config
config = ds.get_ml_config()
result = ds.serialize(data, config=config)

# Modern API - intention is clear
result_modern = ds.dump_ml(data)  # Optimized for ML automatically
```

### Security Example

```python
# Sensitive user data
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "ssn": "123-45-6789",
    "password": "secret123"
}

# Modern API - security by design
secure_result = ds.dump_secure(user_data, redact_pii=True)
# PII fields are automatically redacted

# Traditional API equivalent
redaction_engine = ds.create_financial_redaction_engine()
redacted_data = redaction_engine.process_object(user_data)
result = ds.serialize(redacted_data, config=ds.get_api_config())
```

## 🔗 External Links

- **[GitHub Repository](https://github.com/danielendler/datason)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/datason/)** - Package downloads
- **[Issue Tracker](https://github.com/danielendler/datason/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/danielendler/datason/discussions)** - Community Q&A

## 📄 License

datason is released under the [MIT License](https://github.com/danielendler/datason/blob/main/LICENSE).
