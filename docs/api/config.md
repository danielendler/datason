## Configuration System

Configuration classes and preset functions for customizing serialization behavior.

### SerializationConfig

::: datason.SerializationConfig
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### Configuration Presets

```python
import datason as ds

# Available preset configurations
configs = {
    "ml": ds.get_ml_config(),              # Machine learning workflows
    "api": ds.get_api_config(),            # REST API endpoints
    "strict": ds.get_strict_config(),      # Strict type validation
    "performance": ds.get_performance_config(),  # Speed optimized
    "financial": ds.get_financial_config(),      # Financial data
    "research": ds.get_research_config(),        # Research workflows
    "inference": ds.get_inference_config(),      # Production inference
}
```

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

## Template Deserialization

Functions for enforcing consistent data structures.

### TemplateDeserializer

::: datason.TemplateDeserializer
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### deserialize_with_template()

::: datason.deserialize_with_template
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### infer_template_from_data()

::: datason.infer_template_from_data
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### create_ml_round_trip_template()

::: datason.create_ml_round_trip_template
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
