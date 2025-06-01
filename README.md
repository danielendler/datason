# 🚀 datason

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

[![PyPI version](https://img.shields.io/pypi/v/datason.svg)](https://pypi.org/project/datason/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datason)](https://pypi.org/project/datason/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/datason)](https://pypi.org/project/datason/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/danielendler/datason)](https://github.com/danielendler/datason/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/danielendler/datason?style=social)](https://github.com/danielendler/datason)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/danielendler/datason/graph/badge.svg?token=UYL9LvVb8O)](https://codecov.io/github/danielendler/datason)
[![CI Status](https://img.shields.io/github/actions/workflow/status/danielendler/datason/ci.yml?branch=main)](https://github.com/danielendler/datason/actions)

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

## ✨ Features

- 🧠 **Intelligent Type Detection**: Automatically handles pandas DataFrames, NumPy arrays, datetime objects, and more
- 🔄 **Bidirectional**: Serialize to JSON and deserialize back to original objects
- 🚀 **ML/AI Optimized**: Special support for PyTorch tensors, TensorFlow objects, and scikit-learn models  
- 🛡️ **Type Safety**: Preserves data types and structure integrity
- ⚡ **High Performance**: Optimized for speed with minimal overhead
- 🔌 **Extensible**: Easy to add custom serializers for your own types
- 📦 **Zero Dependencies**: Core functionality works without additional packages

## 🐍 Python Version Support

datason officially supports **Python 3.8+** and is actively tested on:

- ✅ **Python 3.8** - Minimum supported version (core functionality)
- ✅ **Python 3.9** - Full compatibility  
- ✅ **Python 3.10** - Full compatibility
- ✅ **Python 3.11** - Full compatibility (primary development version)
- ✅ **Python 3.12** - Latest stable version

### Compatibility Testing

We maintain compatibility through:
- **Automated CI testing** on all supported Python versions with strategic coverage:
  - **Python 3.8**: Core functionality validation (minimal dependencies)
  - **Python 3.9**: Data science focus (pandas integration)
  - **Python 3.10**: ML focus (scikit-learn, scipy)
  - **Python 3.11**: Full test suite (primary development version)
  - **Python 3.12**: Full test suite (latest stable)
- **Core functionality tests** ensuring basic serialization works on Python 3.8+
- **Dependency compatibility checks** for optional ML/data science libraries
- **Runtime version validation** with helpful error messages

> **Note**: While core functionality works on Python 3.8, some optional dependencies (like latest ML frameworks) may require newer Python versions. The package will still work - you'll just have fewer optional features available.

### Python 3.8 Limitations

Python 3.8 users should be aware:
- ✅ **Core serialization** - Full support
- ✅ **Basic types** - datetime, UUID, decimal, etc.
- ✅ **Pandas/NumPy** - Basic DataFrame and array serialization
- ⚠️ **Advanced ML libraries** - Some may require Python 3.9+
- ⚠️ **Latest features** - Some newer configuration options may have limited support

We recommend Python 3.9+ for the best experience with all features.

## 🏃‍♂️ Quick Start

### Installation

```bash
pip install datason
```

### Basic Usage

```python
import datason as ds
from datetime import datetime
import pandas as pd
import numpy as np

# Complex nested data structure
data = {
    "timestamp": datetime.now(),
    "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
    "array": np.array([1, 2, 3, 4, 5]),
    "nested": {
        "values": [1, 2, {"inner": datetime.now()}]
    }
}

# Serialize to JSON-compatible format
serialized = ds.serialize(data)
print(serialized)

# Deserialize back to original objects
restored = ds.deserialize(serialized)
print(restored)
```

## 📚 Documentation

For full documentation, examples, and API reference, visit: https://datason.readthedocs.io

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
