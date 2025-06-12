# 📥 Modern API: Deserialization Functions

Progressive complexity load functions for different accuracy and performance needs, including file I/O variants.

## 🎯 Progressive Complexity Approach

| Function | Success Rate | Speed | Best For |
|----------|-------------|-------|----------|
| `load_basic()` | 60-70% | ⚡⚡⚡ | Quick exploration |
| `load_smart()` | 80-90% | ⚡⚡ | Production use |
| `load_perfect()` | 100% | ⚡ | Mission-critical |
| `load_typed()` | 95% | ⚡⚡ | Metadata-driven |
| **FILE OPERATIONS** | | | |
| `load_smart_file()` | 80-90% | ⚡⚡ | File-based production |
| `load_perfect_file()` | 100% | ⚡ | File-based critical |

## 📦 Detailed Function Documentation

### load_basic()

Fast, basic deserialization for exploration and testing.

::: datason.load_basic
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Quick Exploration Example:**
```python
# Fast loading for data exploration
json_data = '{"values": [1, 2, 3], "timestamp": "2024-01-01T12:00:00"}'
basic_data = ds.load_basic(json_data)
# Basic types only, minimal processing
```

### load_smart()

Intelligent deserialization with good accuracy for production use.

::: datason.load_smart
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Production Example:**
```python
# Intelligent type detection for production
smart_data = ds.load_smart(json_data)
print(type(smart_data["timestamp"]))  # <class 'datetime.datetime'>
```

### load_perfect()

Perfect accuracy deserialization using templates for mission-critical applications.

::: datason.load_perfect
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Mission-Critical Example:**
```python
# Define expected structure
template = {
    "values": [int],
    "timestamp": datetime,
    "metadata": {"version": float}
}

# 100% reliable restoration
perfect_data = ds.load_perfect(json_data, template)
```

### load_typed()

High-accuracy deserialization using embedded type metadata.

::: datason.load_typed
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Metadata-Driven Example:**
```python
# Use embedded type information
typed_data = ds.load_typed(data_with_types)
# Uses metadata for accurate restoration
```

## 🗃️ File Operations Functions

### load_smart_file()

Smart file loading with automatic format detection and good accuracy.

::: datason.load_smart_file
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**File-Based Production Example:**
```python
# Automatic format detection (.json, .jsonl, .gz)
data = ds.load_smart_file("experiment.json")
jsonl_data = ds.load_smart_file("training_logs.jsonl")
compressed_data = ds.load_smart_file("model.json.gz")

# Smart type reconstruction for production use
ml_data = ds.load_smart_file("model_checkpoint.json")
```

### load_perfect_file()

Perfect file loading using templates for mission-critical applications.

::: datason.load_perfect_file
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**File-Based Critical Example:**
```python
import torch
import numpy as np

# Define expected ML structure
ml_template = {
    "model": torch.nn.Linear(10, 1),
    "weights": torch.randn(100, 50),
    "features": np.random.random((1000, 20)),
    "metadata": {"accuracy": 0.0}
}

# 100% reliable ML reconstruction from file
perfect_ml = ds.load_perfect_file("experiment.json", ml_template)

# Works with JSONL files too
for item in ds.load_perfect_file("training_log.jsonl", ml_template):
    process_item(item)
```



## 🔄 Choosing the Right Load Function

### Decision Matrix

```python
# Choose based on your needs:

# Exploration phase - speed matters most
data = ds.load_basic(json_string)

# Development/testing - good balance
data = ds.load_smart(json_string)  

# Production - reliability critical
data = ds.load_perfect(json_string, template)

# Has embedded types - leverage metadata
data = ds.load_typed(json_string)
```

## 🔗 Related Documentation

- **[Serialization Functions](modern-serialization.md)** - Corresponding dump functions
- **[Template System](template-system.md)** - Creating templates for perfect loading
- **[Modern API Overview](modern-api.md)** - Complete modern API guide
