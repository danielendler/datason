# 📋 API Reference

Complete API documentation for datason - the **perfect drop-in replacement** for Python's JSON module with enhanced features.

## 🎯 **JSON Module Drop-in Replacement**

**Zero migration effort** - datason works exactly like Python's `json` module with optional enhanced features:

=== "JSON Compatibility Mode"

    **Perfect drop-in replacement for Python's json module**

    ```python
    # Your existing code works unchanged
    import datason.json as json

    # Exact same API as stdlib json
    data = json.loads('{"timestamp": "2024-01-01T00:00:00Z"}')
    # Returns: {'timestamp': '2024-01-01T00:00:00Z'}  # String (exact json behavior)

    json_string = json.dumps({"key": "value"}, indent=2)
    # All json.dumps() parameters work exactly the same
    ```

=== "Enhanced Mode (Smart Defaults)"

    **Same API with intelligent enhancements automatically enabled**

    ```python
    # Just import datason for enhanced features
    import datason

    # Smart datetime parsing automatically enabled
    data = datason.loads('{"timestamp": "2024-01-01T00:00:00Z"}')
    # Returns: {'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)}

    # Enhanced dict output for chaining and inspection
    result = datason.dumps({"timestamp": datetime.now()})
    # Returns: dict with enhanced type handling
    ```

## 🚀 Advanced APIs

For specialized use cases, datason provides advanced APIs with progressive complexity:

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

### JSON Module Replacement
- **[JSON Drop-in Replacement](json-replacement.md)** - ⭐ Perfect compatibility with Python's json module plus enhanced features

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
- **[Integrity Functions](integrity.md)** - Hashing and signature utilities
- **[Type System](type-system.md)** - Advanced type handling and conversion
- **[Utilities](utilities.md)** - Helper functions and data processing tools

### Reference
- **[Exceptions](exceptions.md)** - Error handling and custom exceptions
- **[Enums & Constants](enums-constants.md)** - Configuration enums and constants
- **[Complete API Reference](complete-reference.md)** - Auto-generated documentation for all functions

## 🎯 Quick Start Examples

### 🔄 Perfect JSON Module Replacement

```python
# Option 1: Perfect compatibility (zero risk migration)
import datason.json as json

# Works exactly like Python's json module
data = json.loads('{"timestamp": "2024-01-01T00:00:00Z"}')
output = json.dumps({"key": "value"}, indent=2)

# Option 2: Enhanced features (smart datetime parsing)
import datason

# Same API, automatic enhancements
data = datason.loads('{"timestamp": "2024-01-01T00:00:00Z"}')
# Returns: {'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)}

output = datason.dumps({"timestamp": datetime.now()})
# Returns: dict with enhanced type handling
```

### 🚀 Advanced Features (When You Need Them)

```python
import datason as ds

# ML-optimized serialization
ml_data = ds.dump_ml({"model": pytorch_model, "data": numpy_arrays})

# Security-focused with PII redaction
secure_data = ds.dump_secure({"name": "Alice", "email": "alice@email.com"})

# Progressive loading accuracy
basic_data = ds.load_basic(json_data)      # 60-70% accuracy, fast
smart_data = ds.load_smart(json_data)      # 80-90% accuracy, balanced  
perfect_data = ds.load_perfect(json_data)  # 100% accuracy, comprehensive
```

## 🔗 Getting Started

- **New to datason?** Start with the [Quick Start Guide](../user-guide/quick-start.md)
- **Need examples?** Browse the [Examples Gallery](../user-guide/examples/index.md)
- **Looking for specific functions?** Use the [Complete API Reference](complete-reference.md)

## 📚 Related Documentation

- **[User Guide](../user-guide/quick-start.md)** - Getting started guide
- **[Features](../features/configuration/index.md)** - Detailed feature documentation  
- **[Examples](../user-guide/examples/index.md)** - Real-world usage patterns
