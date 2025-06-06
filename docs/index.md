# 🚀 datason Documentation

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

```python
import datason as ds
import pandas as pd
import numpy as np
from datetime import datetime

# Complex data that "just works"
data = {
    'dataframe': pd.DataFrame({'A': [1, 2, 3], 'B': [4.5, 5.5, 6.5]}),
    'timestamp': datetime.now(),
    'array': np.array([1, 2, 3, 4, 5]),
    'nested': {'values': [1, 2, 3], 'metadata': {'created': datetime.now()}}
}

# Serialize to JSON-compatible format
json_data = ds.serialize(data)

# Deserialize back to original objects - types preserved!
restored = ds.deserialize(json_data)
assert type(restored['dataframe']) == pd.DataFrame
assert type(restored['array']) == np.ndarray
```

## ✨ Key Features

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

## 🎯 Quick Navigation

=== "👨‍💻 For Developers"

    **Getting Started**

    - [🚀 Quick Start Guide](user-guide/quick-start.md) - Get up and running in 5 minutes
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

## 🚀 Quick Start

### Installation

```bash
pip install datason
```

### Basic Usage

```python
import datason as ds

# Simple data
data = {"numbers": [1, 2, 3], "text": "hello world"}
serialized = ds.serialize(data)
restored = ds.deserialize(serialized)

# Complex data with configuration
import pandas as pd
from datetime import datetime

complex_data = {
    "df": pd.DataFrame({"A": [1, 2, 3]}),
    "timestamp": datetime.now(),
    "metadata": {"version": 1.0}
}

# Use ML-optimized configuration
config = ds.get_ml_config()
result = ds.serialize(complex_data, config=config)
```

## 🔗 External Links

- **[GitHub Repository](https://github.com/danielendler/datason)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/datason/)** - Package downloads
- **[Issue Tracker](https://github.com/danielendler/datason/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/danielendler/datason/discussions)** - Community Q&A

## 📄 License

datason is released under the [MIT License](https://github.com/danielendler/datason/blob/main/LICENSE).
