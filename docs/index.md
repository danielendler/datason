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
    - [💡 Basic Examples](user-guide/examples/basic.md) - Common use cases and patterns
    - [� Configuration Guide](user-guide/configuration.md) - Customize behavior for your needs
    
    **Core Features**
    
    - [📊 Data Types Support](features/data-types.md) - All supported types and conversion
    - [🤖 ML/AI Integration](features/ml-ai.md) - Machine learning library support
    - [� Data Privacy & Redaction](features/redaction.md) - Protect sensitive information
    - [⚡ Performance & Chunking](features/performance.md) - Handle large datasets efficiently
    
    **Advanced Usage**
    
    - [🎯 Template Deserialization](features/template-deserialization.md) - Enforce data structures
    - [🔄 Pickle Bridge](features/pickle-bridge.md) - Migrate from legacy pickle files
    - [🔍 Type Detection](features/type-detection.md) - How automatic detection works

=== "🤖 For AI Systems"

    **Integration Guides**
    
    - [🤖 AI Integration Guide](ai-guide/overview.md) - How to integrate datason in AI systems
    - [📝 API Reference](api/index.md) - Complete API documentation with examples
    - [🔧 Configuration Presets](ai-guide/presets.md) - Pre-built configs for common AI use cases
    
    **Automation & Tooling**
    
    - [⚙️ Auto-Detection Capabilities](ai-guide/auto-detection.md) - What datason can detect automatically
    - [🔌 Custom Serializers](ai-guide/custom-serializers.md) - Extend for custom types
    - [📊 Schema Inference](ai-guide/schema-inference.md) - Automatic schema generation
    
    **Deployment**
    
    - [🚀 Production Deployment](ai-guide/deployment.md) - Best practices for production
    - [🔍 Monitoring & Logging](ai-guide/monitoring.md) - Track serialization performance
    - [🛡️ Security Considerations](ai-guide/security.md) - Security best practices

## 📚 Documentation Sections

### 📖 User Guide
Comprehensive guides for getting started and using datason effectively.

- **[Quick Start](user-guide/quick-start.md)** - Installation and first steps
- **[Examples Gallery](user-guide/examples/index.md)** - Code examples for every feature
- **[Configuration](user-guide/configuration.md)** - Customize behavior and presets
- **[Migration Guide](user-guide/migration.md)** - Upgrade from older versions

### 🔧 Features
Detailed documentation for all datason features.

- **[Data Types & Conversion](features/data-types.md)** - Supported types and conversion rules
- **[ML/AI Integration](features/ml-ai.md)** - PyTorch, TensorFlow, scikit-learn support
- **[Data Privacy & Redaction](features/redaction.md)** - PII protection and compliance
- **[Performance & Chunking](features/performance.md)** - Memory-efficient processing
- **[Template System](features/template-deserialization.md)** - Structure enforcement
- **[Pickle Bridge](features/pickle-bridge.md)** - Legacy pickle migration

### 🤖 AI Developer Guide  
Specialized documentation for AI systems and automated workflows.

- **[AI Integration Overview](ai-guide/overview.md)** - Integration patterns for AI systems
- **[Configuration Presets](ai-guide/presets.md)** - ML, research, and inference configs
- **[Auto-Detection](ai-guide/auto-detection.md)** - Automatic type and schema detection
- **[Custom Extensions](ai-guide/custom-serializers.md)** - Extend for domain-specific types

### 📋 API Reference
Complete API documentation with examples.

- **[Core Functions](api/core.md)** - serialize(), deserialize(), and main functions
- **[Configuration Classes](api/config.md)** - SerializationConfig and presets
- **[ML Serializers](api/ml.md)** - Machine learning library serializers
- **[Redaction Engine](api/redaction.md)** - Privacy and security features
- **[Utilities](api/utils.md)** - Helper functions and data utilities

### 🔬 Advanced Topics
In-depth technical documentation.

- **[Performance Benchmarks](advanced/benchmarks.md)** - Performance analysis and comparisons
- **[Security Model](advanced/security.md)** - Security architecture and best practices  
- **[Extensibility](advanced/extensibility.md)** - Plugin system and custom handlers
- **[Architecture](advanced/architecture.md)** - Internal design and data flow

### 👥 Community & Development
Resources for contributors and the community.

- **[Contributing Guide](community/contributing.md)** - How to contribute to datason
- **[Development Setup](community/development.md)** - Set up development environment
- **[Release Notes](community/changelog.md)** - Version history and changes
- **[Roadmap](community/roadmap.md)** - Future development plans

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
