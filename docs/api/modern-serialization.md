# 📤 Modern API: Serialization Functions

Intention-revealing dump functions for different use cases and optimization needs.

## 🎯 Function Overview

| Function | Purpose | Best For |
|----------|---------|----------|
| `dump()` | General-purpose with composable options | Flexible workflows |
| `dump_ml()` | ML-optimized for models and tensors | Data science |
| `dump_api()` | Clean JSON for web APIs | Web development |
| `dump_secure()` | Security-focused with PII redaction | Sensitive data |
| `dump_fast()` | Performance-optimized | High-throughput |
| `dump_chunked()` | Memory-efficient for large data | Big datasets |
| `stream_dump()` | Direct file streaming | Very large files |

## 📦 Detailed Function Documentation

### dump()

General-purpose serialization with composable options.

::: datason.dump
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Composable Options Example:**
```python
import datason as ds
import torch
import pandas as pd

# Basic usage
data = {"values": [1, 2, 3], "timestamp": datetime.now()}
result = ds.dump(data)

# Composable options for specific needs
ml_data = {"model": torch.nn.Linear(10, 1), "df": pd.DataFrame({"x": [1, 2, 3]})}

# Combine security + ML optimization + chunked processing
secure_ml_result = ds.dump(
    ml_data,
    secure=True,    # Enable PII redaction
    ml_mode=True,   # Optimize for ML objects
    chunked=True    # Memory-efficient processing
)
```

### dump_ml()

ML-optimized serialization for models, tensors, and NumPy arrays.

::: datason.dump_ml
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**ML Workflow Example:**
```python
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ml_data = {
    "pytorch_model": torch.nn.Linear(10, 1),
    "sklearn_model": RandomForestClassifier(),
    "tensor": torch.randn(100, 10),
    "numpy_array": np.random.random((100, 10)),
}

# Automatically optimized for ML objects
result = ds.dump_ml(ml_data)
```

### dump_api()

API-safe serialization for clean JSON output.

::: datason.dump_api
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Web API Example:**
```python
# Web API response data
api_data = {
    "status": "success",
    "data": [1, 2, 3],
    "errors": None,        # Will be removed
    "timestamp": datetime.now(),
    "metadata": {"version": "1.0"}
}

# Clean JSON output, removes null values
clean_result = ds.dump_api(api_data)
```

### dump_secure()

Security-focused serialization with PII redaction.

::: datason.dump_secure
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Security Example:**
```python
# Sensitive user data
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "ssn": "123-45-6789",
    "password": "secret123",
    "credit_card": "4532-1234-5678-9012"
}

# Automatic PII redaction
secure_result = ds.dump_secure(user_data, redact_pii=True)

# Custom redaction patterns
custom_result = ds.dump_secure(
    user_data,
    redact_fields=["internal_id"],
    redact_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"]
)
```

### dump_fast()

Performance-optimized for high-throughput scenarios.

::: datason.dump_fast
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**High-Throughput Example:**
```python
# Large batch processing
batch_data = [{"id": i, "value": random.random()} for i in range(10000)]

# Minimal overhead, optimized for speed
fast_result = ds.dump_fast(batch_data)
```

### dump_chunked()

Memory-efficient chunked serialization for large objects.

::: datason.dump_chunked
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Large Dataset Example:**
```python
# Very large dataset
large_data = {
    "images": [np.random.random((512, 512, 3)) for _ in range(1000)],
    "features": np.random.random((100000, 200))
}

# Process in memory-efficient chunks
chunked_result = ds.dump_chunked(large_data, chunk_size=1000)
```

### stream_dump()

Direct file streaming for very large data.

::: datason.stream_dump
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**File Streaming Example:**
```python
# Stream directly to file
huge_data = {"massive_array": np.random.random((1000000, 100))}

with open('large_output.json', 'w') as f:
    ds.stream_dump(huge_data, f)
```

## 🔄 Choosing the Right Function

### Decision Tree

1. **Need security/PII redaction?** → Use `dump_secure()`
2. **Working with ML models/tensors?** → Use `dump_ml()`
3. **Building web APIs?** → Use `dump_api()`
4. **Processing very large data?** → Use `dump_chunked()` or `stream_dump()`
5. **Need maximum speed?** → Use `dump_fast()`
6. **Want flexibility?** → Use `dump()` with options

### Performance Comparison

| Function | Speed | Memory Usage | Features |
|----------|-------|--------------|----------|
| `dump_fast()` | ⚡⚡⚡ | 🧠🧠 | Minimal |
| `dump()` | ⚡⚡ | 🧠🧠 | Composable |
| `dump_api()` | ⚡⚡ | 🧠🧠 | Clean output |
| `dump_ml()` | ⚡ | 🧠🧠🧠 | ML optimized |
| `dump_secure()` | ⚡ | 🧠🧠🧠 | Security features |
| `dump_chunked()` | ⚡ | 🧠 | Memory efficient |

## 🎨 Composable Patterns

### Combining Features

```python
# Security + ML + Performance
secure_ml_fast = ds.dump(
    ml_model_data,
    secure=True,
    ml_mode=True,
    fast=True
)

# API + Security
secure_api = ds.dump_api(api_data, secure=True)

# ML + Chunked for large models
large_ml = ds.dump_ml(huge_model, chunked=True)
```

## 🔗 Related Documentation

- **[Deserialization Functions](modern-deserialization.md)** - Load functions
- **[Utility Functions](modern-utilities.md)** - Helper functions
- **[Data Privacy](data-privacy.md)** - Security and redaction details
- **[ML Integration](ml-integration.md)** - Machine learning support
