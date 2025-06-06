# 🚀 Quick Start Guide

Get up and running with datason in minutes! This guide will walk you through installation, basic usage, and key features.

## Installation

### Basic Installation

Install datason using pip:

```bash
pip install datason
```

### With Optional Dependencies

For full ML/AI support, install with optional dependencies:

```bash
# For ML libraries (PyTorch, TensorFlow, scikit-learn)
pip install datason[ml]

# For data science (pandas, numpy extras)
pip install datason[data]

# For all optional dependencies
pip install datason[all]
```

### Development Installation

For development or latest features:

```bash
pip install git+https://github.com/danielendler/datason.git
```

## First Steps

### 1. Basic Serialization

Start with simple data types:

```python
import datason as ds
from datetime import datetime

# Simple data
data = {
    "name": "Alice",
    "age": 30,
    "active": True,
    "joined": datetime.now(),
    "scores": [95, 87, 92]
}

# Serialize to JSON-compatible format
serialized = ds.serialize(data)
print(type(serialized))  # <class 'dict'>

# Deserialize back - types are preserved!
restored = ds.deserialize(serialized)
print(type(restored["joined"]))  # <class 'datetime.datetime'>
print(restored["joined"] == data["joined"])  # True
```

### 2. Complex Data Types

datason automatically handles complex types:

```python
import pandas as pd
import numpy as np

# Complex data with DataFrames and arrays
complex_data = {
    "dataframe": pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 87.2, 92.1]
    }),
    "numpy_array": np.array([1, 2, 3, 4, 5]),
    "metadata": {
        "created": datetime.now(),
        "version": 1.0
    }
}

# Serialize complex data
result = ds.serialize(complex_data)

# Deserialize - everything is restored correctly
restored = ds.deserialize(result)
print(type(restored["dataframe"]))      # <class 'pandas.core.frame.DataFrame'>
print(type(restored["numpy_array"]))    # <class 'numpy.ndarray'>
print(restored["dataframe"].shape)      # (3, 2)
```

### 3. Configuration

Use preset configurations for different use cases:

```python
# Machine Learning configuration
ml_config = ds.get_ml_config()
ml_result = ds.serialize(complex_data, config=ml_config)

# API-optimized configuration  
api_config = ds.get_api_config()
api_result = ds.serialize(complex_data, config=api_config)

# Performance-optimized configuration
perf_config = ds.get_performance_config()
perf_result = ds.serialize(complex_data, config=perf_config)
```

## Key Features Tour

### Intelligent Type Detection

datason automatically detects and handles various data types:

```python
import torch
from sklearn.ensemble import RandomForestClassifier

# Mixed ML data types
ml_data = {
    "pytorch_tensor": torch.tensor([1, 2, 3, 4, 5]),
    "sklearn_model": RandomForestClassifier(),
    "pandas_df": pd.DataFrame({"x": [1, 2, 3]}),
    "numpy_array": np.array([[1, 2], [3, 4]]),
    "python_datetime": datetime.now(),
    "nested_dict": {
        "inner": {
            "values": [1, 2, 3],
            "timestamp": datetime.now()
        }
    }
}

# All types are automatically handled
serialized = ds.serialize(ml_data, config=ds.get_ml_config())
restored = ds.deserialize(serialized)

# Types are preserved
print(type(restored["pytorch_tensor"]))  # <class 'torch.Tensor'>
print(type(restored["sklearn_model"]))   # <class 'sklearn.ensemble._forest.RandomForestClassifier'>
```

### Data Privacy & Redaction

Protect sensitive information automatically:

```python
# Data with sensitive information
sensitive_data = {
    "user": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "ssn": "123-45-6789",
        "password": "secret123"
    },
    "transaction": {
        "amount": 1500.00,
        "card_number": "4532-1234-5678-9012"
    }
}

# Create redaction engine
engine = ds.create_financial_redaction_engine()

# Redact sensitive data
redacted = engine.process_object(sensitive_data)
print(redacted)
# Sensitive fields are now "<REDACTED>"

# Serialize safely
safe_data = ds.serialize(redacted, config=ds.get_api_config())
```

### Large Data Handling

Process large datasets efficiently:

```python
# Large dataset
large_data = {
    "images": [np.random.random((512, 512, 3)) for _ in range(100)],
    "features": pd.DataFrame(np.random.random((10000, 50))),
}

# Check memory usage
memory_estimate = ds.estimate_memory_usage(large_data)
print(f"Estimated memory: {memory_estimate / (1024*1024):.1f} MB")

# Use chunked processing for large data
if memory_estimate > 50 * 1024 * 1024:  # > 50MB
    chunked_result = ds.serialize_chunked(
        large_data,
        chunk_size=10 * 1024 * 1024  # 10MB chunks
    )
    print(f"Split into {len(chunked_result.chunks)} chunks")
else:
    result = ds.serialize(large_data)
```

### Template-based Validation

Ensure consistent data structures:

```python
# Define expected structure
sample_data = {
    "features": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
    "labels": np.array([0, 1, 0]),
    "metadata": {"version": "v1.0"}
}

# Create template
template = ds.infer_template_from_data(sample_data)

# Validate new data against template
def validate_data(new_data):
    try:
        validated = ds.deserialize_with_template(new_data, template)
        return {"status": "valid", "data": validated}
    except ds.TemplateDeserializationError as e:
        return {"status": "error", "message": str(e)}

# Test validation
test_data = ds.serialize(sample_data)  # Valid data
result = validate_data(test_data)
print(result["status"])  # "valid"
```

## Common Patterns

### 1. API Endpoints

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
api_config = ds.get_api_config()

@app.route('/data', methods=['POST'])
def process_data():
    try:
        # Deserialize request data
        input_data = ds.deserialize(request.json)

        # Process data (your logic here)
        result = process_your_data(input_data)

        # Serialize response
        response = ds.serialize(result, config=api_config)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
```

### 2. ML Pipelines

```python
def ml_pipeline_step(input_data, model):
    # Deserialize input
    data = ds.deserialize(input_data) if isinstance(input_data, dict) else input_data

    # Process with model
    features = data["features"]
    predictions = model.predict(features)

    # Package results
    result = {
        "predictions": predictions,
        "metadata": {
            "timestamp": datetime.now(),
            "model_version": "v1.0"
        }
    }

    # Serialize for next step
    return ds.serialize(result, config=ds.get_ml_config())
```

### 3. Data Storage

```python
import json

def save_complex_data(data, filename):
    """Save complex data as JSON file."""
    serialized = ds.serialize(data, config=ds.get_default_config())

    with open(filename, 'w') as f:
        json.dump(serialized, f, indent=2)

def load_complex_data(filename):
    """Load complex data from JSON file."""
    with open(filename, 'r') as f:
        serialized = json.load(f)

    return ds.deserialize(serialized)

# Usage
save_complex_data(complex_data, "my_data.json")
loaded_data = load_complex_data("my_data.json")
```

## Error Handling

Always handle errors gracefully:

```python
def robust_serialization(data):
    try:
        return ds.serialize(data, config=ds.get_ml_config())
    except ds.SecurityError as e:
        print(f"Security error: {e}")
        return None
    except MemoryError:
        print("Memory limit exceeded, trying chunked processing...")
        return ds.serialize_chunked(data)
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Fallback to safe serialization
        return ds.safe_serialize(data)

result = robust_serialization(your_data)
```

## Performance Tips

1. **Reuse configurations**: Create config once, use multiple times
```python
config = ds.get_ml_config()
for batch in data_batches:
    result = ds.serialize(batch, config=config)
```

2. **Check memory usage**: Estimate before processing large data
```python
if ds.estimate_memory_usage(data) > threshold:
    use_chunked_processing()
```

3. **Choose right config**: Use appropriate preset for your use case
```python
# Fast API responses
api_config = ds.get_api_config()

# ML model training
ml_config = ds.get_ml_config()

# Data analysis
research_config = ds.get_research_config()
```

## Next Steps

Now that you're up and running, explore these topics:

- **[Examples Gallery](examples/index.md)** - Comprehensive examples for every feature
- **[Configuration Guide](../features/configuration/index.md)** - Customize behavior for your needs  
- **[ML/AI Integration](../features/ml-ai/index.md)** - Deep dive into ML library support
- **[Data Privacy & Redaction](../features/redaction.md)** - Protect sensitive information
- **[API Reference](../api/index.md)** - Complete function documentation

## Getting Help

- **[GitHub Issues](https://github.com/danielendler/datason/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/danielendler/datason/discussions)** - Community Q&A
- **[Examples](https://github.com/danielendler/datason/tree/main/examples)** - Complete code examples
