# ⚙️ Configuration System

Configuration classes and preset functions for customizing serialization behavior, with **special focus on UUID handling and web framework compatibility**.

## 🎯 Overview

The configuration system provides comprehensive control over datason's serialization behavior through the `SerializationConfig` class and preset configurations. **Most importantly, it solves UUID compatibility issues with Pydantic, FastAPI, Django, and other web frameworks.**

## 🚀 Quick Start: UUID + Pydantic Compatibility

**The #1 use case**: Fix UUID compatibility with Pydantic models in FastAPI/Django:

```python
import datason
from datason.config import get_api_config

# ❌ Problem: Default behavior converts UUIDs to objects
data = {"user_id": "12345678-1234-5678-9012-123456789abc"}
result = datason.auto_deserialize(data)  # user_id becomes UUID object

# ✅ Solution: Use API config to keep UUIDs as strings
api_config = get_api_config()
result = datason.auto_deserialize(data, config=api_config)  # user_id stays string

# Now works with Pydantic!
from pydantic import BaseModel
class User(BaseModel):
    user_id: str  # ✅ Works perfectly!

user = User(**result)  # Success! 🎉
```

## 📦 SerializationConfig Class

Main configuration class for all serialization options.

### Key UUID Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `uuid_format` | `str` | `"object"` | `"object"` converts to `uuid.UUID`, `"string"` keeps as `str` |
| `parse_uuids` | `bool` | `True` | Whether to attempt UUID string parsing at all |

::: datason.SerializationConfig
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## 🔧 Configuration Presets

Pre-built configurations for common scenarios.

### get_api_config() ⭐ **Most Used**

**Perfect for web APIs, Pydantic models, and framework integration.**

- ✅ Keeps UUIDs as strings (Pydantic compatible)
- ✅ ISO datetime format
- ✅ Consistent JSON output
- ✅ Safe for HTTP clients

::: datason.get_api_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Use Cases:**
- FastAPI + Pydantic applications
- Django REST Framework APIs
- Flask JSON endpoints
- Any web API requiring string UUIDs

### get_ml_config()

**Optimized for machine learning and data processing workflows.**

- 🔬 Converts UUIDs to objects (for ML processing)
- 🔬 Rich type preservation
- 🔬 Optimized for scientific computing

::: datason.get_ml_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Use Cases:**
- Machine learning pipelines
- Data science notebooks
- Scientific computing
- Internal data processing

### get_strict_config()

**Enhanced security and validation for production systems.**

::: datason.get_strict_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### get_performance_config()

**Optimized for high-performance scenarios.**

::: datason.get_performance_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## 🌐 Framework-Specific Usage

### FastAPI Integration

```python
from fastapi import FastAPI
from datason.config import get_api_config
import datason

app = FastAPI()
API_CONFIG = get_api_config()  # Set once, use everywhere

@app.post("/users/")
async def create_user(user_data: dict):
    processed = datason.auto_deserialize(user_data, config=API_CONFIG)
    return User(**processed)  # Works with Pydantic!
```

### Django Integration

```python
from datason.config import get_api_config
import datason

class UserAPIView(View):
    def __init__(self):
        self.api_config = get_api_config()

    def post(self, request):
        data = json.loads(request.body)
        processed = datason.auto_deserialize(data, config=self.api_config)
        user = User.objects.create(**processed)
        return JsonResponse(user.to_dict())
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from datason.config import get_api_config
import datason

app = Flask(__name__)
API_CONFIG = get_api_config()

@app.route('/api/users/', methods=['POST'])
def create_user():
    processed = datason.auto_deserialize(request.json, config=API_CONFIG)
    # UUIDs are now strings, compatible with database operations
    return jsonify(processed)
```

## 🛠️ Custom Configuration Examples

### Strict API Configuration

```python
from datason.config import SerializationConfig

strict_api_config = SerializationConfig(
    uuid_format="string",      # Keep UUIDs as strings
    parse_uuids=False,         # Don't auto-convert to UUID objects
    max_size=1_000_000,       # 1MB payload limit
    max_depth=10,             # Prevent deep nesting attacks
    sort_keys=True,           # Consistent JSON output
    ensure_ascii=True         # Safe for all HTTP clients
)
```

### Database JSON Field Configuration

```python
json_field_config = SerializationConfig(
    uuid_format="string",      # Store UUIDs as strings in JSON
    preserve_decimals=True,    # Keep precision in JSON
    max_depth=20,             # Allow deeper nesting in JSON fields
)
```

## 📊 Configuration Comparison

| Use Case | Preset | UUID Format | Parse UUIDs | Best For |
|----------|---------|-------------|-------------|----------|
| **Web APIs** | `get_api_config()` | `"string"` | `False` | FastAPI, Django, Flask |
| **ML Workflows** | `get_ml_config()` | `"object"` | `True` | Data science, ML pipelines |
| **High Performance** | `get_performance_config()` | `"object"` | `True` | Speed-critical applications |
| **Security Critical** | `get_strict_config()` | `"string"` | `False` | Production APIs with limits |

## 🚨 Common Pitfalls

### ❌ Inconsistent Configuration

```python
# Don't mix configurations!
result1 = datason.auto_deserialize(data1)  # Default config
result2 = datason.auto_deserialize(data2, config=get_api_config())  # Different!
# UUIDs will be different types!
```

### ✅ Consistent Configuration

```python
# Use consistent configuration throughout your app
API_CONFIG = get_api_config()
result1 = datason.auto_deserialize(data1, config=API_CONFIG)
result2 = datason.auto_deserialize(data2, config=API_CONFIG)
# All UUIDs are consistently strings
```

## 🔗 Related Documentation

- **[API Integration Guide](../features/api-integration.md)** - Framework-specific integration patterns
- **[Core Functions](core-functions.md)** - Using configurations with serialize/deserialize
- **[Modern API](modern-api.md)** - Compare with intention-revealing approach
- **[Examples](../user-guide/examples/index.md)** - Real-world integration examples
