# ⚙️ Configuration System

Configuration classes and preset functions for customizing serialization behavior in the traditional API.

## 🎯 Overview

The configuration system provides comprehensive control over datason's serialization behavior through the `SerializationConfig` class and preset configurations.

## 📦 SerializationConfig Class

Main configuration class for all serialization options.

::: datason.SerializationConfig
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## 🔧 Configuration Presets

Pre-built configurations for common scenarios.

### get_ml_config()

::: datason.get_ml_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### get_api_config()

::: datason.get_api_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### get_strict_config()

::: datason.get_strict_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### get_performance_config()

::: datason.get_performance_config
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## 🔗 Related Documentation

- **[Core Functions](core-functions.md)** - Using configurations with serialize/deserialize
- **[Modern API](modern-api.md)** - Compare with intention-revealing approach
