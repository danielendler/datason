# 🔧 Modern API: Utility Functions

Helper functions for JSON compatibility, API discovery, and assistance.

## 🎯 Function Overview

| Function | Purpose | Best For |
|----------|---------|----------|
| `dumps()` / `loads()` | JSON module compatibility | Drop-in replacement |
| `help_api()` | Interactive guidance | Learning and discovery |
| `get_api_info()` | API metadata | Programmatic access |

## 📦 Detailed Function Documentation

### dumps() / loads()

JSON module compatibility with intelligent type handling.

::: datason.dumps
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.loads
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**JSON Compatibility Example:**
```python
import datason as ds
from datetime import datetime
import numpy as np

# Drop-in replacement for json.dumps/loads
data = {
    "timestamp": datetime.now(),
    "array": np.array([1, 2, 3]),
    "value": 42.5
}

# Like json.dumps() but with type intelligence
json_string = ds.dumps(data)

# Like json.loads() but with type restoration
restored = ds.loads(json_string)
print(type(restored["timestamp"]))  # <class 'datetime.datetime'>
print(type(restored["array"]))      # <class 'numpy.ndarray'>
```

### help_api()

Interactive API guidance and recommendations.

::: datason.help_api
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Interactive Guidance Example:**
```python
# Get personalized function recommendations
ds.help_api()

# Output example:
# 🎯 datason API Guide
#
# SERIALIZATION (Dump Functions):
# • dump() - General purpose serialization
# • dump_ml() - ML models, tensors, NumPy arrays
# • dump_api() - Web APIs, clean JSON output
# • dump_secure() - Sensitive data with PII redaction
#
# DESERIALIZATION (Load Functions):
# • load_basic() - 60-70% success, fastest (exploration)
# • load_smart() - 80-90% success, moderate speed (production)
# • load_perfect() - 100% success, requires template (critical)

# Get help for specific categories
ds.help_api("dump")      # Focus on serialization
ds.help_api("load")      # Focus on deserialization
ds.help_api("security")  # Focus on security features
```

### get_api_info()

Comprehensive API metadata and feature information.

::: datason.get_api_info
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**API Metadata Example:**
```python
# Get comprehensive API information
api_info = ds.get_api_info()

print("Available functions:")
print("Dump functions:", api_info['dump_functions'])
print("Load functions:", api_info['load_functions'])
print("Utility functions:", api_info['utility_functions'])

print("\nRecommendations:")
print("For ML workflows:", api_info['recommendations']['ml'])
print("For web APIs:", api_info['recommendations']['web'])
print("For security:", api_info['recommendations']['security'])

# Explore capabilities
print("\nFeatures:")
print("Supported types:", api_info['features']['supported_types'])
print("Security features:", api_info['features']['security'])
print("Performance features:", api_info['features']['performance'])

# Version and compatibility info
print("\nVersion info:")
print("API version:", api_info['version']['api'])
print("Package version:", api_info['version']['package'])
print("Compatibility:", api_info['compatibility'])
```

## 🔍 API Discovery Workflow

### Learning the API

```python
# Step 1: Get overview
ds.help_api()

# Step 2: Get detailed information
info = ds.get_api_info()

# Step 3: Explore specific areas
ds.help_api("ml")        # ML-specific guidance
ds.help_api("security")  # Security-specific guidance

# Step 4: Use the functions
data = ds.dump_ml(model_data)    # Based on guidance
loaded = ds.load_smart(data)     # Progressive complexity
```

### Integration Patterns

```python
# JSON module migration
import json
import datason as ds

# Replace json with datason for better type handling
# Old: json.dumps(data)
# New: ds.dumps(data)

# Old: json.loads(json_string)  
# New: ds.loads(json_string)

# Benefits: automatic type preservation
original = {"date": datetime.now(), "array": np.array([1, 2, 3])}
restored = ds.loads(ds.dumps(original))
# Types are preserved automatically!
```

## 📚 Development Workflow Integration

### Interactive Development

```python
# In Jupyter notebooks or interactive development
def explore_data(data):
    # Get recommendations
    ds.help_api()

    # Serialize with appropriate function
    if "model" in str(type(data)):
        return ds.dump_ml(data)
    elif "sensitive" in data:
        return ds.dump_secure(data)
    else:
        return ds.dump(data)

# Programmatic API selection
info = ds.get_api_info()
best_function = info['recommendations']['for_data_type'](type(my_data))
```

### Documentation and Onboarding

```python
# For new team members
def onboard_developer():
    print("Welcome to datason!")
    ds.help_api()

    print("\nDetailed API information:")
    info = ds.get_api_info()
    print(f"Available functions: {len(info['all_functions'])}")
    print(f"Supported data types: {info['features']['supported_types']}")

    print("\nTry these examples:")
    print("ds.dump_ml(model_data)  # For ML workflows")
    print("ds.dump_api(response)   # For web APIs")
    print("ds.load_smart(json)     # For production loading")
```

## 🔗 Related Documentation

- **[Modern API Overview](modern-api.md)** - Complete modern API guide
- **[Serialization Functions](modern-serialization.md)** - Dump functions
- **[Deserialization Functions](modern-deserialization.md)** - Load functions
- **[Quick Start Guide](../user-guide/quick-start.md)** - Getting started tutorial
