# 📋 API Reference

Complete API documentation for datason with two powerful approaches to data serialization.

## 🚀 Two Powerful Approaches

datason provides two complementary APIs designed for different use cases:

=== "Modern API (Recommended)"

    **Intention-revealing function names with progressive complexity**

    ```python
    import datason as ds

    # Clear intent - what you want to achieve
    secure_data = ds.dump_secure(sensitive_data)    # Security-first
    ml_data = ds.dump_ml(model_data)                # ML-optimized  
    api_data = ds.dump_api(response_data)           # Clean web APIs

    # Progressive complexity - choose your level
    basic_data = ds.load_basic(json_data)           # 60-70% accuracy, fast
    smart_data = ds.load_smart(json_data)           # 80-90% accuracy, balanced
    perfect_data = ds.load_perfect(json_data)       # 100% accuracy, thorough
    ```

=== "Traditional API (Comprehensive)"

    **Comprehensive configuration with maximum control**

    ```python
    import datason as ds

    # Maximum configurability
    config = ds.SerializationConfig(
        include_type_info=True,
        compress_arrays=True,
        secure_mode=True,
        ml_mode=True
    )

    # Full control over every aspect
    result = ds.serialize(data, config=config)
    restored = ds.deserialize(result)
    ```

## 📖 API Documentation Sections

### Modern API Functions
- **[Modern API Overview](modern-api.md)** - Intention-revealing functions with progressive complexity
- **[Serialization Functions](modern-serialization.md)** - dump(), dump_ml(), dump_api(), dump_secure(), etc.
- **[Deserialization Functions](modern-deserialization.md)** - load_basic(), load_smart(), load_perfect(), load_typed()
- **[Utility Functions](modern-utilities.md)** - dumps/loads, help_api(), get_api_info()

### Traditional API Functions  
- **[Core Functions](core-functions.md)** - serialize(), deserialize(), auto_deserialize(), safe_deserialize()
- **[Configuration System](configuration.md)** - SerializationConfig, presets, and customization
- **[Chunked & Streaming](chunked-streaming.md)** - Large data processing and memory management
- **[Template System](template-system.md)** - Data validation and structure enforcement

### Specialized Features
- **[ML Integration](ml-integration.md)** - Machine learning library support
- **[Data Privacy](data-privacy.md)** - Redaction engines and security features
- **[Type System](type-system.md)** - Advanced type handling and conversion
- **[Utilities](utilities.md)** - Helper functions and data processing tools

### Reference
- **[Exceptions](exceptions.md)** - Error handling and custom exceptions
- **[Enums & Constants](enums-constants.md)** - Configuration enums and constants
- **[Complete API Reference](complete-reference.md)** - Auto-generated documentation for all functions

## 🎯 Quick Start Examples

### JSON Module Drop-in Replacement

```python
import datason as ds

# Like json.dumps() but with type intelligence
data = {"timestamp": datetime.now(), "array": np.array([1, 2, 3])}
json_string = ds.dumps(data)

# Like json.loads() but with type restoration  
restored = ds.loads(json_string)
print(type(restored["timestamp"]))  # <class 'datetime.datetime'>
print(type(restored["array"]))      # <class 'numpy.ndarray'>
```

### Progressive Complexity Example

```python
import datason as ds

# Start simple, add complexity as needed
data = ds.load_basic(json_data)      # Fast exploration
data = ds.load_smart(json_data)      # Production use
data = ds.load_perfect(json_data)    # Critical accuracy

# Or combine features
secure_ml_data = ds.dump_secure(model_data, ml_mode=True)
```

## 🔗 Getting Started

- **New to datason?** Start with the [Quick Start Guide](../user-guide/quick-start.md)
- **Need examples?** Browse the [Examples Gallery](../user-guide/examples/index.md)
- **Looking for specific functions?** Use the [Complete API Reference](complete-reference.md)

## 📚 Related Documentation

- **[User Guide](../user-guide/quick-start.md)** - Getting started guide
- **[Features](../features/configuration/index.md)** - Detailed feature documentation  
- **[Examples](../user-guide/examples/index.md)** - Real-world usage patterns
