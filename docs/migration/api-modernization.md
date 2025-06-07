# API Modernization Migration Guide

Complete guide for migrating from datason's traditional API to the modern API introduced in v0.8.0.

## 🎯 Overview

In datason v0.8.0, we introduced a **Modern API** with intention-revealing function names and progressive complexity disclosure. This guide helps you migrate from the traditional configuration-based API to the new streamlined approach.

## 🚀 Why Migrate?

### Before (Traditional API)
```python
from datason import serialize, deserialize, SerializationConfig, get_ml_config

# Complex configuration setup
config = SerializationConfig(
    include_type_hints=True,
    compress_arrays=True,
    secure_mode=True
)

# Intent unclear from function name
result = serialize(data, config=config)
restored = deserialize(result)

# Domain-specific configurations require knowledge
ml_config = get_ml_config()
ml_result = serialize(model, config=ml_config)
```

### After (Modern API)
```python
import datason as ds

# Clear intent from function names
result = ds.dump_secure(data)           # Security-focused
ml_result = ds.dump_ml(model)           # ML-optimized
api_result = ds.dump_api(response)      # API-friendly

# Progressive complexity for loading
basic_data = ds.load_basic(json_data)   # 60-70% success, fast
smart_data = ds.load_smart(json_data)   # 80-90% success, balanced
perfect_data = ds.load_perfect(json_data, template)  # 100% success
```

## 📋 Complete Migration Reference

### Serialization Functions

| Traditional API | Modern API | Intent |
|----------------|------------|--------|
| `serialize(data)` | `dump(data)` | General purpose |
| `serialize(data, config=get_ml_config())` | `dump_ml(data)` | ML-optimized |
| `serialize(data, config=get_api_config())` | `dump_api(data)` | Clean JSON output |
| `serialize(data, config=get_performance_config())` | `dump_fast(data)` | Performance optimized |
| `serialize_chunked(data)` | `dump_chunked(data)` | Memory efficient |
| `stream_serialize(file)` | `stream_dump(file)` | File streaming |

### Deserialization Functions

| Traditional API | Modern API | Success Rate | Intent |
|----------------|------------|--------------|--------|
| `deserialize(data)` | `load_basic(data)` | 60-70% | Quick exploration |
| `auto_deserialize(data)` | `load_smart(data)` | 80-90% | Production use |
| `deserialize_fast(data)` | `load_typed(data)` | 95% | Metadata-driven |
| `deserialize_with_template(data, template)` | `load_perfect(data, template)` | 100% | Mission-critical |

### Utility Functions

| Traditional API | Modern API | Purpose |
|----------------|------------|---------|
| `serialize_to_json(data)` | `dumps(data)` | JSON string output |
| `deserialize_from_json(json_str)` | `loads(json_str)` | JSON string input |
| No equivalent | `help_api()` | Interactive guidance |
| No equivalent | `get_api_info()` | API metadata |

## 🔄 Step-by-Step Migration

### Step 1: Replace Basic Serialization

```python
# OLD
from datason import serialize, deserialize
result = serialize(data)
restored = deserialize(result)

# NEW  
import datason as ds
result = ds.dump(data)
restored = ds.load_smart(result)  # Better than load_basic for production
```

### Step 2: Replace Configuration-Based Calls

```python
# OLD - ML Configuration
from datason import serialize, get_ml_config
config = get_ml_config()
ml_result = serialize(model_data, config=config)

# NEW - Domain-Specific Function
import datason as ds
ml_result = ds.dump_ml(model_data)  # Same result, clearer intent
```

```python
# OLD - API Configuration
from datason import serialize, get_api_config
config = get_api_config()
api_result = serialize(response_data, config=config)

# NEW - API-Specific Function
import datason as ds
api_result = ds.dump_api(response_data)  # Cleaner, more discoverable
```

### Step 3: Replace Security-Focused Serialization

```python
# OLD - Manual Security Configuration
from datason import serialize, SerializationConfig
config = SerializationConfig(
    redact_fields=['password', 'api_key'],
    redact_patterns=[r'\b\d{16}\b'],  # Credit cards
    include_redaction_summary=True
)
secure_result = serialize(sensitive_data, config=config)

# NEW - Built-in Security Function
import datason as ds
secure_result = ds.dump_secure(
    sensitive_data,
    redact_pii=True,  # Automatic PII detection
    redact_fields=['password', 'api_key']
)
```

### Step 4: Progressive Complexity for Deserialization

```python
# OLD - One-size-fits-all approach
from datason import deserialize, auto_deserialize
basic_result = deserialize(data)  # Unclear capability
auto_result = auto_deserialize(data)  # Unclear success rate

# NEW - Progressive complexity with clear expectations
import datason as ds

# Choose based on your reliability needs:
exploration_data = ds.load_basic(data)      # 60-70% success, fastest
production_data = ds.load_smart(data)       # 80-90% success, balanced  
critical_data = ds.load_perfect(data, template)  # 100% success, requires template
metadata_data = ds.load_typed(data)         # 95% success, uses embedded types
```

### Step 5: JSON Module Compatibility

```python
# OLD - Manual JSON handling
import json
from datason import serialize, deserialize

data = {"timestamp": datetime.now(), "array": np.array([1, 2, 3])}
json_data = serialize(data)
json_str = json.dumps(json_data)

parsed = json.loads(json_str)
restored = deserialize(parsed)

# NEW - Direct JSON string functions
import datason as ds

data = {"timestamp": datetime.now(), "array": np.array([1, 2, 3])}
json_str = ds.dumps(data)  # Direct to JSON string
restored = ds.loads(json_str)  # Direct from JSON string
```

## 🔧 Advanced Migration Patterns

### Composable Options (NEW in Modern API)

```python
# OLD - Complex configuration objects
config = SerializationConfig(
    ml_mode=True,
    secure_mode=True,
    chunked=True,
    max_chunk_size=1000
)
result = serialize(complex_data, config=config)

# NEW - Composable function calls
import datason as ds

# Option 1: Use the most specific function
if data_is_sensitive_ml:
    result = ds.dump_secure(complex_data, ml_mode=True, chunked=True)

# Option 2: Chain operations  
secure_result = ds.dump_secure(complex_data)
chunked_result = ds.dump_chunked(secure_result, chunk_size=1000)

# Option 3: Use main dump() with mode flags
result = ds.dump(complex_data, ml_mode=True, secure=True, chunked=True)
```

### Error Handling and Fallbacks (NEW)

```python
# OLD - Limited fallback options
try:
    result = deserialize(data)
except Exception:
    result = auto_deserialize(data, aggressive=False)

# NEW - Progressive fallback strategy
import datason as ds

def robust_load(data):
    """Try progressively simpler approaches."""

    # Try typed first (95% success rate)
    try:
        return ds.load_typed(data)
    except (TypeError, ValueError):
        pass

    # Fall back to smart loading (80-90% success)
    try:
        return ds.load_smart(data)
    except (TypeError, ValueError):
        pass

    # Last resort: basic loading (60-70% success)
    return ds.load_basic(data)
```

### API Discovery (NEW)

```python
# NEW - Built-in guidance system
import datason as ds

# Get personalized recommendations
ds.help_api()
# Output: Function recommendations based on common use cases

# Programmatic API exploration
api_info = ds.get_api_info()
print("Available dump functions:", api_info['dump_functions'])
print("Usage recommendations:", api_info['recommendations'])

# Use programmatically
if 'dump_ml' in api_info['dump_functions']:
    result = ds.dump_ml(ml_data)
```

## ⚠️ Breaking Changes

### None! 100% Backward Compatible

The modern API is **completely additive**. All existing code continues to work:

```python
# This still works exactly as before
from datason import serialize, deserialize, SerializationConfig

config = SerializationConfig(include_type_hints=True)
result = serialize(data, config=config)
restored = deserialize(result)

# But you can gradually adopt the new API
import datason as ds
better_result = ds.dump(data, with_types=True)  # Equivalent to above
```

### Deprecation Timeline

- **v0.8.0+**: Modern API available, traditional API fully supported
- **v0.9.0+**: Soft deprecation warnings for traditional API (suppressible)
- **v1.0.0+**: Traditional API remains but documentation focuses on modern API
- **v2.0.0+**: Traditional API may be moved to legacy module

## 🎯 Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

```python
# Week 1: Replace basic serialization
result = ds.dump(data)          # Instead of serialize(data)
restored = ds.load_smart(result) # Instead of deserialize(result)

# Week 2: Replace domain-specific usage  
ml_result = ds.dump_ml(model)   # Instead of serialize(model, get_ml_config())
api_result = ds.dump_api(data)  # Instead of serialize(data, get_api_config())

# Week 3: Add progressive complexity
basic_data = ds.load_basic(data)    # For exploration
production_data = ds.load_smart(data)  # For production
critical_data = ds.load_perfect(data, template)  # For mission-critical
```

### Strategy 2: Full Migration

```python
# Replace all imports at once
# OLD
from datason import (
    serialize, deserialize, auto_deserialize,
    SerializationConfig, get_ml_config, get_api_config
)

# NEW
import datason as ds
# Everything available under ds.* namespace with clear names
```

### Strategy 3: Side-by-Side (Testing)

```python
# Run both APIs in parallel during migration
import datason as ds
from datason import serialize, deserialize

# Compare results
old_result = serialize(data)
new_result = ds.dump(data)
assert old_result == new_result  # Should be identical

old_restored = deserialize(old_result)  
new_restored = ds.load_smart(new_result)
assert old_restored == new_restored  # Should be equivalent
```

## 🔍 Troubleshooting

### Common Migration Issues

#### 1. Import Changes
```python
# PROBLEM: Old imports not working
from datason.api import dump  # ❌ Not the main import

# SOLUTION: Use main package import
import datason as ds
result = ds.dump(data)  # ✅ Correct way
```

#### 2. Function Name Confusion
```python
# PROBLEM: Which load function should I use?
result = ds.load_???(data)  # Which one?

# SOLUTION: Use the decision matrix
if use_case == "exploration":
    result = ds.load_basic(data)      # 60-70% success, fastest
elif use_case == "production":
    result = ds.load_smart(data)      # 80-90% success, reliable  
elif use_case == "critical":
    result = ds.load_perfect(data, template)  # 100% success
elif data_has_metadata:
    result = ds.load_typed(data)      # 95% success, metadata-driven
```

#### 3. Performance Concerns
```python
# PROBLEM: Worried about performance regression
old_time = time_serialize_with_config()
new_time = time_dump_function()

# SOLUTION: Modern API has zero overhead (thin wrappers)
assert new_time <= old_time * 1.05  # Should be equivalent or better
```

### Getting Help

```python
# Built-in help system
ds.help_api()  # Interactive guidance

# Community support
# - GitHub Issues: https://github.com/yourusername/datason/issues
# - Discussions: https://github.com/yourusername/datason/discussions
# - Documentation: https://datason.readthedocs.io/

# Migration-specific help
api_info = ds.get_api_info()
print("Migration guide:", api_info['migration_guide'])
```

## 🏆 Benefits After Migration

### 1. **Improved Discoverability**
```python
# Before: How do I serialize for ML?
# Answer: You need to know about get_ml_config()

# After: How do I serialize for ML?  
ds.dump_ml(model)  # Obvious from the name!
```

### 2. **Progressive Learning Curve**
```python
# Beginner: Start simple
result = ds.dump(data)
restored = ds.load_basic(result)

# Intermediate: Add reliability
result = ds.dump_api(data)  # API-optimized
restored = ds.load_smart(result)  # Better accuracy

# Advanced: Maximum control
result = ds.dump_secure(data, redact_pii=True)  # Security-focused
restored = ds.load_perfect(result, template)    # 100% accuracy
```

### 3. **Self-Documenting Code**
```python
# Before: Unclear intent
result = serialize(data, config=some_config)

# After: Crystal clear intent
result = ds.dump_secure(sensitive_data)  # Obviously for security
result = ds.dump_ml(model_data)          # Obviously for ML
result = ds.dump_api(response_data)      # Obviously for APIs
```

### 4. **Better Error Messages**
```python
# Modern API provides more helpful error messages
try:
    result = ds.load_perfect(data, template)
except TemplateValidationError as e:
    print(f"Template mismatch: {e.details}")
    print(f"Suggestion: Try ds.load_smart() for automatic type detection")
```

## 🚀 Next Steps

1. **Choose your migration strategy** (gradual, full, or side-by-side)
2. **Start with basic functions** (`dump()`, `load_smart()`)
3. **Gradually adopt domain-specific functions** (`dump_ml()`, `dump_api()`, etc.)
4. **Leverage progressive complexity** for deserialization
5. **Use built-in help** (`ds.help_api()`, `ds.get_api_info()`)

The modern API makes datason more discoverable, learnable, and enjoyable to use while maintaining 100% compatibility with existing code. Welcome to the future of datason! 🎉
