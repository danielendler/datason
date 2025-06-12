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
- 🚀 **ML/AI Optimized**: Production-ready support for 10+ ML frameworks with unified architecture
- 🛡️ **Type Safety**: Preserves data types and structure integrity with **guaranteed round-trip** serialization
- ⚡ **High Performance**: Sub-millisecond serialization optimized for ML workloads
- 🔌 **Extensible**: Easy to add custom serializers for your own types
- 📦 **Zero Dependencies**: Core functionality works without additional packages
- 🎯 **Simple & Direct API**: Intention-revealing functions (`dump_api`, `dump_ml`, `dump_secure`, `dump_fast`) with automatic optimization
- 📈 **Progressive Loading**: Choose your success rate - `load_basic` (60-70%), `load_smart` (80-90%), `load_perfect` (100%)
- 🏗️ **Production Ready**: Enterprise-grade ML serving with monitoring, A/B testing, and security

## 🤖 ML Framework Support

datason provides **production-ready integration** for major ML frameworks with consistent serialization:

### **Core ML Libraries**
- 🐼 **Pandas** - DataFrames with schema preservation
- 🔢 **NumPy** - Arrays with dtype and shape preservation  
- 🔥 **PyTorch** - Tensors with exact dtype/shape reconstruction
- 🧠 **TensorFlow/Keras** - Models with architecture and weights
- 🌲 **Scikit-learn** - Fitted models with parameters

### **Advanced ML Frameworks**
- 🚀 **CatBoost** - Models with fitted state and parameter extraction
- 📊 **Optuna** - Studies with trial history and hyperparameter tracking
- 📈 **Plotly** - Interactive figures with data, layout, and configuration
- ⚡ **Polars** - High-performance DataFrames with schema preservation
- 🎯 **XGBoost** - Gradient boosting models (via scikit-learn interface)

### **ML Serving Platforms**
- 🍱 **BentoML** - Production services with A/B testing and monitoring
- ☀️ **Ray Serve** - Scalable deployment with autoscaling
- 🔬 **MLflow** - Model registry integration with experiment tracking
- 🎨 **Streamlit** - Interactive dashboards with real-time data
- 🎭 **Gradio** - ML demos with consistent data handling
- ⚡ **FastAPI** - Custom APIs with validation and rate limiting
- ☸️ **Seldon Core/KServe** - Kubernetes-native model serving

> **Universal Pattern**: All frameworks use the same `get_api_config()` for consistent UUID and datetime handling across your entire ML pipeline.

## 🐍 Python Version Support

datason officially supports **Python 3.8+** and is actively tested on:

- ✅ **Python 3.8** - Minimum supported version (core functionality)
- ✅ **Python 3.9** - Full compatibility  
- ✅ **Python 3.10** - Full compatibility
- ✅ **Python 3.11** - Full compatibility (primary development version)
- ✅ **Python 3.12** - Full compatibility
- ✅ **Python 3.13** - Latest stable version (core features only; many ML libraries still releasing wheels)

### Compatibility Testing

We maintain compatibility through:
- **Automated CI testing** on all supported Python versions with strategic coverage:
  - **Python 3.8**: Core functionality validation (minimal dependencies)
  - **Python 3.9**: Data science focus (pandas integration)
  - **Python 3.10**: ML focus (scikit-learn, scipy)
  - **Python 3.11**: Full test suite (primary development version)
  - **Python 3.12**: Full test suite
  - **Python 3.13**: Core serialization tests only (latest stable)
- **Core functionality tests** ensuring basic serialization works on Python 3.8+
- **Dependency compatibility checks** for optional ML/data science libraries
- **Runtime version validation** with helpful error messages

> **Note**: While core functionality works on Python 3.8, some optional dependencies (like latest ML frameworks) may require newer Python versions. The package will still work - you'll just have fewer optional features available.
>
> **Python 3.13 Caution**: Many machine learning libraries have not yet released official 3.13 builds. Datason runs on Python 3.13, but only with core serialization features until those libraries catch up.

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

### Production ML Serving - Simple & Direct

```python
import datason as ds
import uuid
from datetime import datetime

# ML prediction data with UUIDs and complex types
prediction_data = {
    "request_id": uuid.uuid4(),
    "timestamp": datetime.now(),
    "features": {"feature1": 1.0, "feature2": 2.0},
    "model_version": "1.0.0"
}

# Simple, direct API with automatic optimizations
api_response = ds.dump_api(prediction_data)  # Perfect for web APIs
# ✅ UUIDs become strings automatically - no more Pydantic errors!

# ML-optimized serialization
import torch
model_data = {"model": torch.nn.Linear(10, 1), "weights": torch.randn(10, 1)}
ml_serialized = ds.dump_ml(model_data)  # Automatic ML optimization

# Security-focused with automatic PII redaction
user_data = {"name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"}
secure_data = ds.dump_secure(user_data)  # Automatic PII redaction

# Works across ALL ML frameworks with same simple pattern
import bentoml
from bentoml.io import JSON

@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    features = ds.load_smart(input_data)  # 80-90% success rate
    prediction = model.predict(features)
    return ds.dump_api({"prediction": prediction})  # Clean API response
```

### Simple & Direct API

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

# Simple API with automatic optimization
api_data = ds.dump_api(data)        # Web APIs (UUIDs as strings, clean JSON)
ml_data = ds.dump_ml(data)          # ML optimized (framework detection)
secure_data = ds.dump_secure(data)  # Security focused (PII redaction)
fast_data = ds.dump_fast(data)      # Performance optimized

# Progressive loading - choose your success rate
basic_result = ds.load_basic(api_data)    # 60-70% success, fastest
smart_result = ds.load_smart(api_data)    # 80-90% success, balanced
perfect_result = ds.load_perfect(api_data, template=data)  # 100% with template

# Traditional API still available
serialized = ds.serialize(data)
restored = ds.deserialize(serialized)
```

### Advanced Options - Composable & Flexible

```python
import datason as ds

# Use the main dump() function with options for complex scenarios
large_sensitive_ml_data = {
    "model": trained_model,
    "user_data": {"email": "user@example.com", "preferences": {...}},
    "large_dataset": huge_numpy_array
}

# Combine multiple optimizations
result = ds.dump(
    large_sensitive_ml_data,
    secure=True,           # Enable PII redaction
    ml_mode=True,         # Optimize for ML objects
    chunked=True          # Memory-efficient processing
)

# Or use specialized functions for simple cases
api_data = ds.dump_api(response_data)      # Web API optimized
ml_data = ds.dump_ml(model_data)           # ML optimized
secure_data = ds.dump_secure(sensitive_data)  # Security focused
fast_data = ds.dump_fast(performance_data) # Speed optimized

# Progressive loading with clear success rates
basic_result = ds.load_basic(json_data)    # 60-70% success, fastest
smart_result = ds.load_smart(json_data)    # 80-90% success, balanced
perfect_result = ds.load_perfect(json_data, template)  # 100% with template

# API discovery and help
help_info = ds.help_api()  # Get guidance on function selection
```



## 🏗️ Production Architecture

datason provides a **complete ML serving architecture** with visual documentation:

- **🎯 Universal Integration Pattern**: Single configuration works across all frameworks
- **📊 Comprehensive Monitoring**: Prometheus metrics, health checks, and observability
- **🔒 Enterprise Security**: Input validation, rate limiting, and PII redaction
- **⚡ Performance Optimized**: Sub-millisecond serialization with caching support
- **🔄 A/B Testing**: Framework for testing multiple model versions
- **📈 Production Examples**: Ready-to-deploy BentoML, Ray Serve, and FastAPI services

### Quick Architecture Overview

```mermaid
graph LR
    A[Client Apps] --> B[API Gateway]
    B --> C[ML Services<br/>BentoML/Ray/FastAPI]
    C --> D[Models<br/>CatBoost/Keras/etc]
    C --> E[Cache<br/>Redis]
    C --> F[DB<br/>PostgreSQL]

    style C fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style D fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
```

> **See Full Documentation**: Complete architecture diagrams and production patterns in `docs/features/model-serving/`

## 📚 Documentation

### **Core Documentation**
For full documentation, examples, and API reference, visit: https://datason.readthedocs.io

### **ML Serving Guides**
- 🏗️ **[Architecture Overview](docs/features/model-serving/architecture-overview.md)** - Complete system architecture with Mermaid diagrams
- 🚀 **[Model Serving Integration](docs/features/model-serving/index.md)** - Production-ready examples for all major frameworks
- 🎯 **[Production Patterns](docs/features/model-serving/production-patterns.md)** - Advanced deployment strategies and best practices

### **Production Examples**
- 🍱 **[Advanced BentoML Integration](examples/advanced_bentoml_integration.py)** - Enterprise service with A/B testing and monitoring
- 📊 **[Production ML Serving Guide](examples/production_ml_serving_guide.py)** - Complete implementation with security and observability

> **Quick Start**: Run `python examples/production_ml_serving_guide.py` to see all features in action!

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
