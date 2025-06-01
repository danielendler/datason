# Welcome to datason

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

```python
import datason as ds
import pandas as pd
from datetime import datetime

# Works with complex data out of the box
data = {
    'dataframe': pd.DataFrame({'A': [1, 2, 3]}),
    'timestamp': datetime.now(),
    'nested': {'values': [1, 2, 3]}
}

json_data = ds.serialize(data)
# Works perfectly! 🎉
```

## ✨ Key Features

- **🧠 Intelligent Type Detection**: Automatically handles pandas DataFrames, NumPy arrays, datetime objects, and more
- **🔄 Bidirectional**: Serialize to JSON and deserialize back to original objects  
- **🚀 ML/AI Optimized**: Special support for PyTorch tensors, TensorFlow objects, and scikit-learn models
- **🛡️ Type Safety**: Preserves data types and structure integrity
- **⚡ High Performance**: Optimized for speed with minimal overhead
- **🔧 Configurable**: Multiple presets and fine-grained control
- **🔒 Secure**: Safe handling with protection against malicious data
- **📦 Zero Dependencies**: Core functionality works without additional packages

## 🚀 Quick Start

```bash
pip install datason
```

```python
import datason

# Simple serialization
data = {"numbers": [1, 2, 3], "text": "hello"}
json_ready = datason.serialize(data)

# With configuration
config = datason.get_ml_config()
result = datason.serialize(complex_ml_data, config=config)

# Pickle Bridge (NEW in v0.3.0)
json_data = datason.from_pickle("legacy_model.pkl")
```

## 📚 Documentation

### 🎯 **User Guides**
- **[Features Overview](features/)** - Complete feature guide and examples
- **[Tooling Guide](TOOLING_GUIDE.md)** - Development tools and workflows
- **[Feature Matrix](FEATURE_MATRIX.md)** - Capability comparison
- **[AI/ML Usage](AI_USAGE_GUIDE.md)** - Machine learning integration
- **[Benchmarking](BENCHMARKING.md)** - Performance metrics and optimization

### 🔧 **Development**
- **[Contributing](CONTRIBUTING.md)** - How to contribute to datason
- **[Security](SECURITY.md)** - Security policies and reporting
- **[CI/CD Guide](CI_PIPELINE_GUIDE.md)** - Continuous integration workflows
- **[Release Management](RELEASE_MANAGEMENT.md)** - Release process and versioning

### 📊 **Reference**
- **[Changelog](CHANGELOG.md)** - Version history and changes
- **[Roadmap](ROADMAP.md)** - Future development plans

## 🤝 Community & Support

- **GitHub**: [danielendler/datason](https://github.com/danielendler/datason)
- **Issues**: [Report bugs](https://github.com/danielendler/datason/issues)
- **Discussions**: [Ask questions](https://github.com/danielendler/datason/discussions)
- **PyPI**: [Package index](https://pypi.org/project/datason/)

## 📄 License

MIT License - see [LICENSE](https://github.com/danielendler/datason/blob/main/LICENSE) file for details.
