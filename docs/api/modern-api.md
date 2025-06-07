# 🚀 Modern API Overview

The Modern API provides intention-revealing function names with progressive complexity disclosure, designed for clarity and ease of use.

## 🎯 Design Philosophy

The Modern API follows these principles:

1. **Intention-Revealing Names** - Function names clearly indicate what they do
2. **Progressive Complexity** - Choose your level of sophistication  
3. **Domain-Specific Optimization** - Built-in optimizations for specific use cases
4. **Composable Options** - Combine features as needed
5. **100% Backward Compatible** - Works alongside the traditional API

## 📦 Function Categories

### Serialization Functions (Dump)
- `dump()` - General-purpose with composable options
- `dump_ml()` - ML-optimized for models and tensors
- `dump_api()` - Clean JSON for web APIs
- `dump_secure()` - Security-focused with PII redaction
- `dump_fast()` - Performance-optimized
- `dump_chunked()` - Memory-efficient for large data
- `stream_dump()` - Direct file streaming

### Deserialization Functions (Load) - Progressive Complexity
- `load_basic()` - 60-70% accuracy, fastest (exploration)
- `load_smart()` - 80-90% accuracy, balanced (production)
- `load_perfect()` - 100% accuracy, requires template (critical)
- `load_typed()` - 95% accuracy, uses embedded metadata

### Utility Functions
- `dumps()` / `loads()` - JSON module compatibility
- `help_api()` - Interactive guidance
- `get_api_info()` - API metadata and capabilities

## 🔄 Progressive Complexity in Action

```python
import datason as ds

# Phase 1: Quick exploration
data = ds.load_basic(json_string)  # Fast, good enough for exploration

# Phase 2: Production reliability  
data = ds.load_smart(json_string)  # Better accuracy for production

# Phase 3: Mission-critical accuracy
template = {"name": str, "age": int, "scores": [float]}
data = ds.load_perfect(json_string, template)  # 100% reliable
```

## 🎨 Composable Design

Modern API functions support composable options:

```python
# Combine multiple optimizations
secure_ml_data = ds.dump(
    model_data,
    secure=True,      # Enable security features
    ml_mode=True,     # Optimize for ML objects  
    chunked=True      # Memory-efficient processing
)

# Or use specialized functions
secure_ml_data = ds.dump_secure(model_data, ml_mode=True)
```

## 🔍 API Discovery

The Modern API includes built-in discovery tools:

```python
# Get interactive guidance
ds.help_api()

# Get comprehensive API information
info = ds.get_api_info()
print("Available functions:", info['dump_functions'])
print("Recommendations:", info['recommendations'])
```

## 📊 When to Use Modern API

**✅ Recommended for:**
- New projects starting fresh
- Clear, readable code requirements
- Progressive complexity needs
- Domain-specific optimizations
- Built-in security requirements

**Example Use Cases:**
```python
# Data science workflow
model_data = ds.dump_ml({"model": model, "metrics": metrics})

# Web API responses  
clean_response = ds.dump_api({"data": results, "status": "success"})

# Secure data handling
safe_data = ds.dump_secure(user_data, redact_pii=True)

# Large dataset processing
chunked_data = ds.dump_chunked(massive_dataset, chunk_size=1000)
```

## 🔗 Next Steps

- **[Serialization Functions](modern-serialization.md)** - Detailed dump function documentation
- **[Deserialization Functions](modern-deserialization.md)** - Detailed load function documentation
- **[Utility Functions](modern-utilities.md)** - Helper and discovery functions
- **[Complete Reference](complete-reference.md)** - Auto-generated documentation

## 📚 Related Documentation

- **[Traditional API](core-functions.md)** - Compare with configuration-based approach
- **[Quick Start Guide](../user-guide/quick-start.md)** - Getting started tutorial
