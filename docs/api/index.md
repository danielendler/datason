# 📋 API Reference

Complete API documentation for datason with examples and auto-generated documentation from source code.

## Core Functions

The main serialization and deserialization functions that form the core of datason.

### serialize()

::: datason.serialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### deserialize()

::: datason.deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### auto_deserialize()

::: datason.auto_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### safe_deserialize()

::: datason.safe_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Chunked & Streaming Processing

Functions for handling large datasets efficiently.

### serialize_chunked()

::: datason.serialize_chunked
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### ChunkedSerializationResult

::: datason.ChunkedSerializationResult
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### StreamingSerializer

::: datason.StreamingSerializer
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
      members:
        - serialize_async
        - stream_serialize

### estimate_memory_usage()

::: datason.estimate_memory_usage
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

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

## ML Library Integration

Specialized serializers for machine learning libraries.

### ML Serialization Functions

::: datason.detect_and_serialize_ml_object
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.serialize_pytorch_tensor
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.serialize_tensorflow_tensor
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.serialize_sklearn_model
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.serialize_huggingface_tokenizer
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### get_ml_library_info()

::: datason.get_ml_library_info
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Redaction & Privacy

Privacy protection and sensitive data redaction.

### RedactionEngine

::: datason.RedactionEngine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
      members:
        - __init__
        - process_object
        - redact_text
        - get_redaction_summary
        - get_audit_trail

### Pre-built Redaction Engines

::: datason.create_minimal_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.create_financial_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.create_healthcare_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Pickle Bridge

Legacy pickle file migration and conversion.

### PickleBridge

::: datason.PickleBridge
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### from_pickle()

::: datason.from_pickle
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### convert_pickle_directory()

::: datason.convert_pickle_directory
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Data Utilities

Helper functions for data processing and analysis.

### Data Enhancement

::: datason.enhance_data_types
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.enhance_pandas_dataframe
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.enhance_numpy_array
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### Data Analysis

::: datason.deep_compare
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.find_data_anomalies
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.normalize_data_structure
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### Date/Time Utilities

::: datason.standardize_datetime_formats
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.extract_temporal_features
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.ensure_timestamp
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Type Handling

Low-level type detection and conversion utilities.

### TypeHandler

::: datason.TypeHandler
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### Type Utilities

::: datason.get_object_info
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.is_nan_like
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.normalize_numpy_types
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Exceptions

Custom exception classes used by datason.

### SecurityError

::: datason.SecurityError
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### TemplateDeserializationError

::: datason.TemplateDeserializationError
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### PickleSecurityError

::: datason.PickleSecurityError
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### UtilitySecurityError

::: datason.UtilitySecurityError
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Constants & Enums

Configuration enums and constants.

### DateFormat

::: datason.DateFormat
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### NanHandling

::: datason.NanHandling
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### DataFrameOrient

::: datason.DataFrameOrient
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### TypeCoercion

::: datason.TypeCoercion
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Quick Reference

### Common Usage Patterns

```python
import datason as ds
import pandas as pd
import numpy as np
from datetime import datetime

# Basic serialization
data = {"values": [1, 2, 3], "timestamp": datetime.now()}
serialized = ds.serialize(data)
restored = ds.deserialize(serialized)

# With configuration
config = ds.get_ml_config()
ml_data = {"model": model, "features": pd.DataFrame(data)}
result = ds.serialize(ml_data, config=config)

# Chunked processing for large data
large_data = {"arrays": [np.random.random((1000, 1000)) for _ in range(100)]}
chunked = ds.serialize_chunked(large_data, chunk_size=10*1024*1024)

# Template enforcement
template = ds.infer_template_from_data(sample_data)
validated = ds.deserialize_with_template(new_data, template)

# Privacy protection
engine = ds.create_financial_redaction_engine()
safe_data = engine.process_object(sensitive_data)
```

### Error Handling

```python
try:
    result = ds.serialize(complex_data)
except ds.SecurityError as e:
    print(f"Security violation: {e}")
except MemoryError as e:
    # Fall back to chunked processing
    result = ds.serialize_chunked(complex_data)
except Exception as e:
    # Generic error handling
    result = ds.safe_serialize(complex_data)
```

### Performance Tips

```python
# For repeated operations, reuse configuration
config = ds.get_ml_config()
for batch in data_batches:
    result = ds.serialize(batch, config=config)

# Estimate memory before processing
memory_estimate = ds.estimate_memory_usage(large_data)
if memory_estimate > threshold:
    use_chunked_processing()

# Monitor performance
import time
start = time.time()
result = ds.serialize(data)
duration = time.time() - start
```
