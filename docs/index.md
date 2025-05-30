# SerialPy Documentation

Welcome to SerialPy, a comprehensive Python package for intelligent serialization that handles complex data types with ease.

## What is SerialPy?

SerialPy solves the fundamental problem of serializing complex Python objects that standard `json` can't handle. Perfect for AI/ML workflows, data science, and modern Python applications.

```python
import serialpy as sp
import torch
import pandas as pd
from datetime import datetime

# 🔥 Works with ANY Python object
data = {
    'model_results': torch.tensor([0.9, 0.1, 0.8]),
    'dataframe': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
    'timestamp': datetime.now(),
    'metrics': {'accuracy': 0.95, 'loss': 0.05}
}

# ✅ One line solution
json_data = sp.serialize(data)
# Works perfectly! 🎉
```

## 📊 SerialPy vs Standard JSON

| Feature | SerialPy | Standard JSON |
|---------|----------|---------------|
| **Basic types** | ✅ | ✅ |
| **Datetime objects** | ✅ | ❌ |
| **Pandas DataFrames** | ✅ | ❌ |
| **NumPy arrays** | ✅ | ❌ |
| **ML models** | ✅ | ❌ |
| **Circular references** | ✅ Protected | ❌ Crashes |
| **Performance** | ⚡ 40x faster tooling | Standard |
| **Security** | 🛡️ Enterprise-grade | Basic |

**SerialPy helps you achieve:**
- 🤖 **AI/ML Data Serialization**: Native support for PyTorch, TensorFlow, scikit-learn
- 📊 **Data Science Workflows**: Handle pandas DataFrames, NumPy arrays seamlessly
- 🌐 **Cross-Platform APIs**: JSON output readable by any programming language
- 🛡️ **Production Safety**: Built-in protection against circular references and edge cases
- ⚡ **High Performance**: Optimized for large datasets and real-time applications

## Quick Start

```bash
pip install serialpy
```

## Features

- **🤖 AI/ML Ready**: Native support for PyTorch, TensorFlow, scikit-learn
- **📊 Data Science**: Handles pandas DataFrames, NumPy arrays, time series
- **⚡ High Performance**: Optimized for speed and memory efficiency
- **🛡️ Safe & Reliable**: Handles edge cases, circular references gracefully
- **🔄 Bidirectional**: Serialize and deserialize with type preservation

## Documentation Sections

- [AI/ML Usage Guide](AI_USAGE_GUIDE.md) - Comprehensive guide for AI/ML workflows
- [Feature Matrix](FEATURE_MATRIX.md) - Complete list of supported types
- [Benchmarking](BENCHMARKING.md) - Performance analysis and comparisons
- [Build & Publish](BUILD_PUBLISH.md) - Development and release guide
- [Tooling Guide](TOOLING_GUIDE.md) - Development tools and workflows

## Project Information

- [Contributing Guide](CONTRIBUTING.md) - How to contribute to SerialPy
- [Security Policy](SECURITY.md) - Security guidelines and reporting
- [Changelog](CHANGELOG.md) - Version history and release notes

## Links

- **🏠 Homepage**: [GitHub Repository](https://github.com/danielendler/SerialPy)
- **📚 Documentation**: [Read the Docs](https://serialpy.readthedocs.io/)
- **🐛 Issues**: [Bug Reports](https://github.com/danielendler/SerialPy/issues)
- **💬 Discussions**: [Community Forum](https://github.com/danielendler/SerialPy/discussions)

---

SerialPy is open source and welcomes contributions from the community.

For questions, issues, or feature requests, please visit our [GitHub repository](https://github.com/danielendler/SerialPy).
