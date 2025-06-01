# 🚀 datason

**A comprehensive Python package for intelligent serialization that handles complex data types with ease**

[![PyPI version](https://badge.fury.io/py/datason.svg)](https://badge.fury.io/py/datason)
[![Python Support](https://img.shields.io/pypi/pyversions/datason.svg)](https://pypi.org/project/datason/)
[![Downloads](https://pepy.tech/badge/datason)](https://pepy.tech/project/datason)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

datason transforms complex Python objects into JSON-serializable formats and back with intelligence. Perfect for ML/AI workflows, data science, and any application dealing with complex nested data structures.

## ✨ Features

- 🧠 **Intelligent Type Detection**: Automatically handles pandas DataFrames, NumPy arrays, datetime objects, and more
- 🔄 **Bidirectional**: Serialize to JSON and deserialize back to original objects
- 🚀 **ML/AI Optimized**: Special support for PyTorch tensors, TensorFlow objects, and scikit-learn models  
- 🛡️ **Type Safety**: Preserves data types and structure integrity
- ⚡ **High Performance**: Optimized for speed with minimal overhead
- 🔌 **Extensible**: Easy to add custom serializers for your own types
- 📦 **Zero Dependencies**: Core functionality works without additional packages

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
