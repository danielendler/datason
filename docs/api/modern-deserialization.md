# 📥 Modern API: Deserialization Functions

Progressive complexity load functions for different accuracy and performance needs.

## 🎯 Progressive Complexity Approach

| Function | Success Rate | Speed | Best For |
|----------|-------------|-------|----------|
| `load_basic()` | 60-70% | ⚡⚡⚡ | Quick exploration |
| `load_smart()` | 80-90% | ⚡⚡ | Production use |
| `load_perfect()` | 100% | ⚡ | Mission-critical |
| `load_typed()` | 95% | ⚡⚡ | Metadata-driven |

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
