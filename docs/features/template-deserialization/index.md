# Template-Based Deserialization (v0.4.5)

Template-based deserialization in datason v0.4.5 provides type-guided reconstruction of complex data structures. This feature ensures consistent data types and structures when deserializing, making it ideal for ML pipelines, API contracts, and data validation scenarios.

## Overview

Traditional deserialization relies on heuristics to guess data types. Template-based deserialization uses a reference object (template) to guide the process, ensuring:

- **Consistent type reconstruction**
- **Validation of data structure**
- **ML-optimized round-trip fidelity**
- **Automatic type coercion with templates**

## Key Features

### 1. Template-Guided Deserialization

Use existing objects as templates to guide deserialization:

```python
import datason
from datason.deserializers import TemplateDeserializer
from datetime import datetime

# Define template with expected structure and types
template = {
    'user_id': 0,
    'name': '',
    'created': datetime.now(),
    'active': True,
    'score': 0.0
}

# Serialized data with string representations
serialized_data = {
    'user_id': '123',
    'name': 'Alice',
    'created': '2023-12-25T10:30:45',
    'active': 'true',
    'score': '95.5'
}

# Deserialize with template guidance
deserializer = TemplateDeserializer(template)
result = deserializer.deserialize(serialized_data)

# Result has correct types:
# result['user_id'] is int(123)
# result['created'] is datetime object
# result['active'] is bool(True)
# result['score'] is float(95.5)
```

### 2. Automatic Template Inference

Generate templates from sample data:

```python
from datason.deserializers import infer_template_from_data

# Sample data for template inference
sample_data = [
    {'name': 'Alice', 'age': 30, 'created': '2023-01-01T10:00:00'},
    {'name': 'Bob', 'age': 25, 'created': '2023-01-02T11:00:00'},
    {'name': 'Charlie', 'age': 35, 'created': '2023-01-03T12:00:00'}
]

# Infer template from samples
template = infer_template_from_data(sample_data)

# Use inferred template for consistent deserialization
deserializer = TemplateDeserializer(template)
new_data = {'name': 'Diana', 'age': '28', 'created': '2023-01-04T13:00:00'}
result = deserializer.deserialize(new_data)
```

### 3. ML-Optimized Templates

Create templates specifically for machine learning workflows:

```python
from datason.deserializers import create_ml_round_trip_template
import pandas as pd

# Training data
training_df = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [10, 20, 30],
    'target': ['class_a', 'class_b', 'class_a']
})

# Create ML-optimized template
ml_template = create_ml_round_trip_template(training_df)

# Template includes ML-specific metadata:
# - DataFrame structure and dtypes
# - Shape information
# - Index details
# - ML-specific optimization flags

print(ml_template['__ml_template__'])  # True
print(ml_template['structure_type'])   # 'dataframe'
print(ml_template['dtypes'])          # Column type mapping
```

### 4. Convenience Functions

Simple template-based deserialization:

```python
from datason.deserializers import deserialize_with_template

# One-line template deserialization
template = {'id': 0, 'value': 0.0, 'name': ''}
data = {'id': '42', 'value': '3.14', 'name': 'test'}

result = deserialize_with_template(data, template)
# Automatically applies template and returns typed result
```

## Supported Template Types

### Basic Data Types
```python
template = {
    'int_field': 0,
    'float_field': 0.0,
    'str_field': '',
    'bool_field': True
}

# Handles automatic type coercion
data = {
    'int_field': '42',      # str → int
    'float_field': '3.14',  # str → float
    'str_field': 'hello',   # str → str
    'bool_field': 'true'    # str → bool
}
```

### DateTime and UUID Templates
```python
from datetime import datetime
from uuid import uuid4

template = {
    'timestamp': datetime.now(),
    'unique_id': uuid4()
}

data = {
    'timestamp': '2023-12-25T10:30:45',
    'unique_id': '12345678-1234-5678-9012-123456789abc'
}

result = deserialize_with_template(data, template)
# result['timestamp'] is datetime object
# result['unique_id'] is UUID object
```

### DataFrame Templates
```python
import pandas as pd

# Template DataFrame with specific dtypes
template_df = pd.DataFrame({
    'id': pd.Series([1], dtype='int32'),
    'value': pd.Series([0.0], dtype='float64'),
    'category': pd.Series([''], dtype='category')
})

# Data to deserialize
data_records = [
    {'id': 1, 'value': 10.5, 'category': 'A'},
    {'id': 2, 'value': 20.3, 'category': 'B'}
]

result = deserialize_with_template(data_records, template_df)
# result is DataFrame with template's dtypes preserved
```

### Nested Structure Templates
```python
template = {
    'user': {
        'id': 0,
        'profile': {
            'name': '',
            'created': datetime.now()
        }
    },
    'data': [{'key': '', 'value': 0.0}]
}

nested_data = {
    'user': {
        'id': '123',
        'profile': {
            'name': 'Alice',
            'created': '2023-01-01T10:00:00'
        }
    },
    'data': [
        {'key': 'metric1', 'value': '42.5'},
        {'key': 'metric2', 'value': '24.8'}
    ]
}

result = deserialize_with_template(nested_data, template)
# Recursive template application to nested structures
```

## Configuration Options

### Strict vs Flexible Mode
```python
template = {'expected': 0, 'known': ''}

# Strict mode: Only process template fields
strict_deserializer = TemplateDeserializer(template, strict=True)

# Flexible mode: Allow extra fields
flexible_deserializer = TemplateDeserializer(template, strict=False)

data_with_extra = {
    'expected': '42',
    'known': 'value',
    'extra_field': 'unexpected'
}

# Flexible mode preserves extra fields
result = flexible_deserializer.deserialize(data_with_extra)
# result includes 'extra_field'
```

### Fallback Auto-Detection
```python
# Enable automatic type detection for unknown fields
deserializer = TemplateDeserializer(
    template,
    strict=False,
    fallback_auto_detect=True
)

# Auto-detection applied to fields not in template
data = {
    'known_field': '42',
    'datetime_field': '2023-01-01T10:00:00',  # Auto-detected as datetime
    'uuid_field': '12345678-1234-5678-9012-123456789abc'  # Auto-detected as UUID
}
```

## Advanced Usage

### Custom Type Coercion
```python
class CustomTemplateDeserializer(TemplateDeserializer):
    def _coerce_to_template_type(self, obj, template):
        # Custom coercion logic
        if isinstance(template, MyCustomType):
            return MyCustomType.from_string(obj)
        return super()._coerce_to_template_type(obj, template)

# Use custom deserializer
custom_template = {'special_field': MyCustomType()}
deserializer = CustomTemplateDeserializer(custom_template)
```

### Template Validation
```python
def validate_template_compatibility(data, template):
    """Check if data is compatible with template."""
    try:
        deserializer = TemplateDeserializer(template, strict=True)
        deserializer.deserialize(data)
        return True
    except Exception:
        return False

# Validate before processing
if validate_template_compatibility(user_data, user_template):
    result = deserialize_with_template(user_data, user_template)
else:
    # Handle incompatible data
    result = fallback_processing(user_data)
```

## Real-World Examples

### API Response Validation
```python
import datason
from datason.deserializers import TemplateDeserializer

# Define API response template
api_response_template = {
    'status': '',
    'data': {
        'user_id': 0,
        'username': '',
        'created_at': datetime.now(),
        'is_active': True
    },
    'metadata': {
        'request_id': uuid4(),
        'timestamp': datetime.now()
    }
}

def process_api_response(response_json):
    """Process API response with template validation."""
    deserializer = TemplateDeserializer(api_response_template)

    try:
        validated_response = deserializer.deserialize(response_json)
        return validated_response
    except Exception as e:
        raise ValueError(f"Invalid API response format: {e}")

# Usage
response_data = fetch_api_response()
validated_data = process_api_response(response_data)
```

### ML Pipeline Data Consistency
```python
import datason
from datason.deserializers import create_ml_round_trip_template

def create_ml_pipeline(training_data, model_config):
    """Create ML pipeline with template-based consistency."""

    # Create template from training data structure
    data_template = create_ml_round_trip_template(training_data)

    # Save template with model
    model_artifacts = {
        'model': model_config,
        'data_template': data_template,
        'training_metadata': {
            'created': datetime.now(),
            'version': '1.0'
        }
    }

    return model_artifacts

def predict_with_template(model_artifacts, new_data):
    """Make predictions with template validation."""

    # Ensure new data matches training data structure
    template = model_artifacts['data_template']
    validated_data = deserialize_with_template(new_data, template)

    # Proceed with prediction using validated data
    predictions = model_artifacts['model'].predict(validated_data)

    return predictions

# Usage
artifacts = create_ml_pipeline(training_df, trained_model)
predictions = predict_with_template(artifacts, new_samples)
```

### Configuration File Processing
```python
from datason.deserializers import infer_template_from_data

def process_config_files(config_dir):
    """Process configuration files with template inference."""

    config_files = list(Path(config_dir).glob("*.json"))

    # Load sample configs to infer template
    sample_configs = []
    for config_file in config_files[:5]:  # Sample first 5
        with config_file.open() as f:
            sample_configs.append(json.load(f))

    # Infer common template
    config_template = infer_template_from_data(sample_configs)

    # Process all configs with template
    processed_configs = []
    for config_file in config_files:
        with config_file.open() as f:
            config_data = json.load(f)

        # Apply template for consistency
        processed_config = deserialize_with_template(config_data, config_template)
        processed_configs.append(processed_config)

    return processed_configs

# Usage
configs = process_config_files("./app_configs/")
```

### Time Series Data Normalization
```python
def normalize_time_series_data(data_sources):
    """Normalize time series data from multiple sources."""

    # Template for time series records
    time_series_template = {
        'timestamp': datetime.now(),
        'sensor_id': '',
        'value': 0.0,
        'unit': '',
        'quality': 1.0
    }

    normalized_data = []

    for source_data in data_sources:
        # Apply template to ensure consistent structure
        for record in source_data:
            normalized_record = deserialize_with_template(record, time_series_template)
            normalized_data.append(normalized_record)

    return normalized_data

# Usage
sensor_data = [
    [{'timestamp': '2023-01-01T10:00:00', 'sensor_id': 'temp_01', 'value': '23.5'}],
    [{'timestamp': '2023-01-01T10:01:00', 'sensor_id': 'temp_02', 'value': '24.1'}]
]
normalized = normalize_time_series_data(sensor_data)
```

## Performance Considerations

### Template Caching
```python
# Cache template analysis for repeated use
template = {'id': 0, 'name': '', 'value': 0.0}
deserializer = TemplateDeserializer(template)  # Analysis done once

# Reuse deserializer for multiple operations
results = []
for data_item in large_dataset:
    result = deserializer.deserialize(data_item)  # Fast subsequent calls
    results.append(result)
```

### Batch Processing
```python
def batch_template_deserialize(data_list, template):
    """Efficiently process large batches with templates."""
    deserializer = TemplateDeserializer(template)

    # Process in chunks for memory efficiency
    chunk_size = 1000
    results = []

    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i:i+chunk_size]
        chunk_results = [deserializer.deserialize(item) for item in chunk]
        results.extend(chunk_results)

    return results

# Usage
large_dataset = load_large_dataset()
template = infer_template_from_data(large_dataset[:10])
processed_data = batch_template_deserialize(large_dataset, template)
```

## Error Handling

### Template Mismatch Errors
```python
from datason.deserializers import TemplateDeserializationError

try:
    result = deserialize_with_template(incompatible_data, template)
except TemplateDeserializationError as e:
    print(f"Template mismatch: {e}")
    # Handle with fallback processing
    result = auto_deserialize(incompatible_data)
```

### Graceful Degradation
```python
def robust_template_deserialize(data, template):
    """Template deserialization with graceful fallback."""
    try:
        # Try template-based deserialization
        return deserialize_with_template(data, template, strict=True)
    except Exception:
        # Fall back to flexible mode
        try:
            return deserialize_with_template(data, template, strict=False)
        except Exception:
            # Final fallback to auto-detection
            return datason.auto_deserialize(data)

# Usage
result = robust_template_deserialize(uncertain_data, expected_template)
```

## Best Practices

### 1. Template Design
```python
# Good: Specific templates with example values
good_template = {
    'user_id': 0,              # int expected
    'email': '',               # string expected
    'created': datetime.now(), # datetime expected
    'active': True             # boolean expected
}

# Avoid: Generic templates
avoid_template = {
    'user_id': None,    # Ambiguous type
    'email': None,      # Ambiguous type
    'created': None,    # Ambiguous type
    'active': None      # Ambiguous type
}
```

### 2. Template Inference Sample Size
```python
# Use representative sample for inference
sample_size = min(100, len(full_dataset))
sample_data = full_dataset[:sample_size]
template = infer_template_from_data(sample_data)
```

### 3. Validation Strategy
```python
# Validate critical fields explicitly
def validate_critical_fields(data, template):
    critical_fields = ['user_id', 'timestamp', 'amount']

    for field in critical_fields:
        if field not in data:
            raise ValueError(f"Missing critical field: {field}")

        expected_type = type(template[field])
        if not isinstance(data[field], (str, expected_type)):
            raise ValueError(f"Invalid type for {field}")

# Use before template deserialization
validate_critical_fields(user_data, user_template)
result = deserialize_with_template(user_data, user_template)
```

### 4. ML Template Persistence
```python
# Save ML templates with models
def save_model_with_template(model, training_data, model_path):
    """Save model with data template for consistency."""

    # Create ML template
    data_template = create_ml_round_trip_template(training_data)

    # Package model with template
    model_package = {
        'model': model,
        'data_template': data_template,
        'metadata': {
            'created': datetime.now(),
            'datason_version': datason.__version__
        }
    }

    # Save complete package
    with open(model_path, 'w') as f:
        json.dump(datason.serialize(model_package), f)

# Load and use with template
def load_model_with_template(model_path):
    """Load model and apply data template."""

    with open(model_path) as f:
        model_package = datason.deserialize(json.load(f))

    return model_package['model'], model_package['data_template']
```

## API Reference

### Core Functions

- **`deserialize_with_template(obj, template, **kwargs)`**
  - Convenience function for template-based deserialization
  - Returns deserialized object matching template structure

- **`infer_template_from_data(data, max_samples=100)`**
  - Infer template from sample data
  - Returns template object

- **`create_ml_round_trip_template(ml_object)`**
  - Create ML-optimized template
  - Returns template with ML-specific metadata

### Classes

- **`TemplateDeserializer(template, strict=True, fallback_auto_detect=True)`**
  - Template-based deserializer class
  - Methods: `deserialize(obj)`, `_analyze_template()`

- **`TemplateDeserializationError`**
  - Exception for template deserialization failures

### Template Types Supported

- Basic types: `int`, `float`, `str`, `bool`
- Date/time: `datetime`, `date`, `time`
- Identifiers: `UUID`
- Collections: `list`, `dict`, `tuple`
- DataFrames: `pandas.DataFrame`, `pandas.Series`
- NumPy: `numpy.ndarray`
- Nested structures: recursive templates

## See Also

- [Chunked Processing](../chunked-processing/index.md) - Memory-efficient large data handling
- [Configuration Guide](../configuration/index.md) - Domain-specific configurations
- [Core Features](../core/index.md) - Basic serialization and auto-detection
- [ML Integration](../ml-ai/index.md) - Machine learning workflows
