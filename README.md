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
- 🎯 **Modern API**: Intention-revealing function names with progressive complexity

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

### Traditional API - Comprehensive & Configurable

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

### Modern API - Intention-Revealing & Progressive

```python
import datason as ds

# 🎯 Clear intentions with domain-specific functions
user_data = {"name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"}

# Security-focused with automatic PII redaction
secure_data = ds.dump_secure(user_data, redact_pii=True)

# ML-optimized for models and tensors
import torch
model_data = {"model": torch.nn.Linear(10, 1), "weights": torch.randn(10, 1)}
ml_serialized = ds.dump_ml(model_data)

# API-safe clean JSON for web endpoints
api_response = ds.dump_api({"status": "success", "data": [1, 2, 3]})

# 📈 Progressive complexity for deserialization
json_data = '{"values": [1, 2, 3], "metadata": {"created": "2024-01-01T12:00:00"}}'

# Basic: Fast exploration (60-70% success rate)
basic_result = ds.load_basic(json_data)

# Smart: Production-ready (80-90% success rate)  
smart_result = ds.load_smart(json_data)

# Perfect: Template-based (100% success rate)
template = {"values": [int], "metadata": {"created": datetime}}
perfect_result = ds.load_perfect(json_data, template)

# 🔍 API Discovery
ds.help_api()  # Interactive guidance for choosing the right function
```

### Composable Options

```python
# Combine features for specific needs
large_sensitive_ml_data = {
    "model": trained_model,
    "user_data": {"email": "user@example.com", "preferences": {...}},
    "large_dataset": huge_numpy_array
}

# Secure + ML-optimized + Memory-efficient
result = ds.dump(
    large_sensitive_ml_data,
    secure=True,           # Enable PII redaction
    ml_mode=True,         # Optimize for ML objects
    chunked=True          # Memory-efficient processing
)
```

## 📚 Documentation

For full documentation, examples, and API reference, visit: https://datason.readthedocs.io

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
