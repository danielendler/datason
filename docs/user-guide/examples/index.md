# 💡 Examples Gallery

Comprehensive collection of examples showcasing datason features and use cases.

## Quick Start Examples

### Basic Usage
Simple serialization and deserialization examples for getting started.

```python
import datason as ds
from datetime import datetime
import pandas as pd
import numpy as np

# Basic data types
simple_data = {
    "numbers": [1, 2, 3, 4, 5],
    "text": "Hello, world!",
    "timestamp": datetime.now(),
    "boolean": True,
    "nested": {"inner": {"value": 42}}
}

# Serialize and deserialize
serialized = ds.serialize(simple_data)
restored = ds.deserialize(serialized)

print("Original types preserved:", type(restored["timestamp"]))
# Output: Original types preserved: <class 'datetime.datetime'>
```

### Complex Data Types
Working with pandas DataFrames, NumPy arrays, and nested structures.

```python
# Complex data with various types
complex_data = {
    "dataframe": pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [95.5, 87.2, 92.1]
    }),
    "numpy_array": np.array([[1, 2, 3], [4, 5, 6]]),
    "metadata": {
        "created": datetime.now(),
        "version": 1.0,
        "tags": ["data", "science", "ml"]
    }
}

# Serialize with ML configuration
config = ds.get_ml_config()
result = ds.serialize(complex_data, config=config)

# Deserialize back
restored = ds.deserialize(result)
print("DataFrame shape:", restored["dataframe"].shape)
print("Array dtype:", restored["numpy_array"].dtype)
```

## Machine Learning Examples

### PyTorch Integration
Working with PyTorch tensors and models.

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Create model and data
model = SimpleModel()
tensor_data = torch.randn(5, 10)
model_output = model(tensor_data)

ml_data = {
    "model_state": model.state_dict(),
    "input_tensor": tensor_data,
    "output_tensor": model_output,
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
}

# Serialize ML data
config = ds.get_ml_config()
serialized = ds.serialize(ml_data, config=config)

# Deserialize and verify
restored = ds.deserialize(serialized)
print("Input tensor shape:", restored["input_tensor"].shape)
print("Model parameters:", list(restored["model_state"].keys()))
```

### Scikit-learn Models
Serializing trained scikit-learn models with data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create and train model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Package model with metadata
sklearn_data = {
    "model": model,
    "training_data": {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    },
    "model_info": {
        "accuracy": model.score(X_test, y_test),
        "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
        "trained_at": datetime.now()
    }
}

# Serialize sklearn model
result = ds.serialize(sklearn_data, config=ds.get_ml_config())
```

## Data Privacy & Security

### Basic Redaction
Protecting sensitive information in datasets.

```python
# Sample data with sensitive information
sensitive_data = {
    "customers": [
        {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789",
            "phone": "555-123-4567",
            "account_balance": 10000.50
        },
        {
            "name": "Jane Smith",
            "email": "jane.smith@company.com",
            "ssn": "987-65-4321",
            "phone": "555-987-6543",
            "account_balance": 25000.75
        }
    ],
    "api_config": {
        "api_key": "secret-key-12345",
        "database_url": "postgresql://user:pass@host:5432/db"
    }
}

# Create and apply redaction
engine = ds.create_financial_redaction_engine()
redacted_data = engine.process_object(sensitive_data)

print("Redacted data:", redacted_data)
# Sensitive fields are now "<REDACTED>"

# Get audit trail
audit_trail = engine.get_audit_trail()
for entry in audit_trail:
    print(f"Redacted: {entry['target']} at {entry['timestamp']}")
```

### Healthcare Data Redaction
HIPAA-compliant data protection for healthcare.

```python
# Healthcare data
patient_data = {
    "patients": [
        {
            "patient_id": "P001",
            "name": "Alice Johnson",
            "dob": "1985-03-15",
            "ssn": "555-44-3333",
            "phone": "555-111-2222",
            "email": "alice@email.com",
            "diagnosis": "Hypertension",
            "treatment": "Medication therapy"
        }
    ],
    "medical_records": pd.DataFrame({
        "patient_id": ["P001", "P002"],
        "visit_date": ["2024-01-15", "2024-01-16"],
        "diagnosis_code": ["I10", "E11"],
        "provider": ["Dr. Smith", "Dr. Jones"]
    })
}

# Apply healthcare redaction
healthcare_engine = ds.create_healthcare_redaction_engine()
redacted_healthcare = healthcare_engine.process_object(patient_data)

# Serialize with privacy protection
config = ds.get_api_config()
safe_serialized = ds.serialize(redacted_healthcare, config=config)
```

## Large-Scale Data Processing

### Chunked Processing
Handling large datasets efficiently.

```python
# Create large dataset
large_dataset = {
    "images": [np.random.random((512, 512, 3)) for _ in range(100)],
    "features": pd.DataFrame(np.random.random((10000, 50))),
    "metadata": {
        "total_samples": 10000,
        "created": datetime.now(),
        "dataset_version": "v2.1"
    }
}

# Estimate memory usage
memory_estimate = ds.estimate_memory_usage(large_dataset)
print(f"Estimated memory usage: {memory_estimate / (1024*1024):.1f} MB")

# Use chunked processing for large data
config = ds.get_performance_config()
chunked_result = ds.serialize_chunked(
    large_dataset,
    chunk_size=10 * 1024 * 1024,  # 10MB chunks
    config=config
)

print(f"Data split into {len(chunked_result.chunks)} chunks")
print(f"Total size: {chunked_result.total_size} bytes")

# Process chunks individually
for i, chunk_info in enumerate(chunked_result.chunks):
    chunk_data = ds.deserialize(chunk_info["data"])
    print(f"Chunk {i}: {chunk_info['size']} bytes")
```

### Streaming Data
Real-time data processing with streaming.

```python
import asyncio
from datason import StreamingSerializer

async def stream_processing_example():
    """Example of streaming data processing."""

    config = ds.get_performance_config()
    streaming_serializer = StreamingSerializer(config=config)

    # Simulate streaming data
    data_stream = [
        {"batch_id": i, "data": np.random.random((100, 10))}
        for i in range(10)
    ]

    for batch in data_stream:
        # Stream serialize each batch
        async for chunk in streaming_serializer.serialize_async(batch):
            # Process chunk (e.g., send to downstream service)
            print(f"Processing chunk of size: {len(chunk)}")
            await asyncio.sleep(0.1)  # Simulate processing time

# Run streaming example
# asyncio.run(stream_processing_example())
```

## Template-based Deserialization

### Schema Enforcement
Ensuring consistent data structures in ML pipelines.

```python
# Define expected data structure
sample_training_data = {
    "features": pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0]
    }),
    "labels": np.array([0, 1, 0]),
    "metadata": {
        "timestamp": datetime.now(),
        "data_version": "v1.0"
    }
}

# Create template from sample
template = ds.infer_template_from_data(sample_training_data)

# Function to validate incoming data
def validate_training_data(raw_data):
    try:
        validated = ds.deserialize_with_template(raw_data, template)
        return {"status": "valid", "data": validated}
    except ds.TemplateDeserializationError as e:
        return {"status": "error", "message": str(e)}

# Test with valid data
valid_data = ds.serialize(sample_training_data)
result = validate_training_data(valid_data)
print("Validation result:", result["status"])

# Test with invalid data
invalid_data = {"features": "not a dataframe", "labels": [1, 2, 3]}
result = validate_training_data(invalid_data)
print("Invalid data result:", result["status"])
```

### ML Round-trip Templates
Ensuring data consistency in ML workflows.

```python
# Create ML round-trip template
ml_template = ds.create_ml_round_trip_template(
    input_types=[pd.DataFrame, np.ndarray],
    output_types=[torch.Tensor],
    preserve_metadata=True
)

# Use template in ML pipeline
def ml_pipeline_step(input_data):
    # Validate input structure
    validated_input = ds.deserialize_with_template(input_data, ml_template)

    # Process data
    features = validated_input["features"]  # Guaranteed to be DataFrame
    processed = features.values  # Convert to numpy

    # Create output with consistent structure
    output = {
        "processed_features": processed,
        "metadata": validated_input["metadata"]
    }

    return ds.serialize(output, config=ds.get_ml_config())
```

## Configuration Examples

### Domain-specific Configurations
Using preset configurations for different domains.

```python
# Financial data configuration
financial_config = ds.get_financial_config()
financial_data = {
    "transactions": pd.DataFrame({
        "amount": [100.50, 250.75, 1500.00],
        "currency": ["USD", "EUR", "USD"],
        "timestamp": [datetime.now()] * 3
    }),
    "risk_metrics": {"var": 0.05, "volatility": 0.12}
}
financial_result = ds.serialize(financial_data, config=financial_config)

# Research configuration
research_config = ds.get_research_config()
research_data = {
    "experiment_id": "EXP-2024-001",
    "results": np.random.random((1000, 50)),
    "parameters": {"learning_rate": 0.001, "batch_size": 32},
    "reproducibility": {
        "random_seed": 42,
        "library_versions": {"numpy": "1.24.0", "pandas": "2.0.0"}
    }
}
research_result = ds.serialize(research_data, config=research_config)

# API configuration for web services
api_config = ds.get_api_config()
api_data = {
    "user_id": 12345,
    "preferences": {"theme": "dark", "notifications": True},
    "last_login": datetime.now()
}
api_result = ds.serialize(api_data, config=api_config)
```

### Custom Configuration
Creating custom configurations for specific needs.

```python
from datason import SerializationConfig, DateFormat, NanHandling

# Custom configuration for IoT data
iot_config = SerializationConfig(
    date_format=DateFormat.UNIX,        # Unix timestamps for efficiency
    nan_handling=NanHandling.NULL,      # Convert NaN to null
    sort_keys=False,                    # Preserve sensor data order
    ensure_ascii=False,                 # Support international characters
    preserve_numpy_types=True,          # Keep sensor data precision
    include_metadata=True,              # Track device information
    chunk_size=5 * 1024 * 1024         # 5MB chunks for streaming
)

# IoT sensor data
iot_data = {
    "device_id": "SENSOR-001",
    "readings": pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1min"),
        "temperature": np.random.normal(20, 5, 1000),
        "humidity": np.random.normal(50, 10, 1000),
        "pressure": np.random.normal(1013, 20, 1000)
    }),
    "device_info": {
        "location": {"lat": 40.7128, "lon": -74.0060},
        "firmware": "v2.1.3",
        "last_calibration": datetime.now()
    }
}

# Serialize with custom config
iot_result = ds.serialize(iot_data, config=iot_config)
```

## Legacy Migration

### Pickle to JSON Migration
Converting legacy pickle files to JSON format.

```python
# Convert individual pickle file
try:
    # Load legacy pickle file
    json_data = ds.from_pickle("legacy_model.pkl")

    # Save as JSON
    with open("converted_model.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print("Successfully converted pickle to JSON")

except ds.PickleSecurityError as e:
    print(f"Security error: {e}")
except Exception as e:
    print(f"Conversion error: {e}")

# Batch convert directory of pickle files
conversion_results = ds.convert_pickle_directory(
    input_dir="legacy_models/",
    output_dir="converted_models/",
    file_pattern="*.pkl"
)

for result in conversion_results:
    if result["status"] == "success":
        print(f"Converted: {result['input_file']} -> {result['output_file']}")
    else:
        print(f"Failed: {result['input_file']} - {result['error']}")
```

## Error Handling & Monitoring

### Robust Error Handling
Implementing comprehensive error handling.

```python
def robust_serialization(data, description=""):
    """Robust serialization with comprehensive error handling."""

    try:
        # Try optimal configuration first
        result = ds.serialize(data, config=ds.get_ml_config())
        return {"status": "success", "data": result, "method": "ml_config"}

    except ds.SecurityError as e:
        # Handle security violations
        print(f"Security error in {description}: {e}")
        return {"status": "error", "type": "security", "message": str(e)}

    except MemoryError as e:
        # Fall back to chunked processing
        print(f"Memory limit exceeded for {description}, using chunked processing")
        try:
            chunked_result = ds.serialize_chunked(data)
            return {"status": "success", "data": chunked_result, "method": "chunked"}
        except Exception as chunked_error:
            return {"status": "error", "type": "memory", "message": str(chunked_error)}

    except Exception as e:
        # Generic fallback
        print(f"Unexpected error in {description}: {e}")
        try:
            safe_result = ds.safe_serialize(data)
            return {"status": "success", "data": safe_result, "method": "safe"}
        except Exception as safe_error:
            return {"status": "error", "type": "generic", "message": str(safe_error)}

# Usage examples
test_data = {"large_array": np.random.random((10000, 1000))}
result = robust_serialization(test_data, "large dataset test")
print(f"Serialization result: {result['status']} using {result.get('method', 'N/A')}")
```

### Performance Monitoring
Tracking serialization performance.

```python
import time
from typing import Dict, List

class SerializationProfiler:
    def __init__(self):
        self.metrics: List[Dict] = []

    def profile_serialization(self, data, config=None, description=""):
        """Profile serialization performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            result = ds.serialize(data, config=config)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)

        end_time = time.time()
        end_memory = self._get_memory_usage()

        metrics = {
            "description": description,
            "duration": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "data_size": len(str(data)),
            "success": success,
            "error": error_message,
            "timestamp": datetime.now()
        }

        self.metrics.append(metrics)
        return result, metrics

    def _get_memory_usage(self):
        """Get current memory usage (simplified)."""
        import psutil
        return psutil.Process().memory_info().rss

    def get_performance_summary(self):
        """Get performance summary statistics."""
        if not self.metrics:
            return {}

        successful_runs = [m for m in self.metrics if m["success"]]

        return {
            "total_runs": len(self.metrics),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(self.metrics),
            "avg_duration": sum(m["duration"] for m in successful_runs) / len(successful_runs) if successful_runs else 0,
            "max_duration": max(m["duration"] for m in successful_runs) if successful_runs else 0,
            "avg_memory_delta": sum(m["memory_delta"] for m in successful_runs) / len(successful_runs) if successful_runs else 0
        }

# Usage
profiler = SerializationProfiler()

# Profile different data types
test_cases = [
    ({"simple": [1, 2, 3]}, "simple data"),
    (pd.DataFrame(np.random.random((1000, 10))), "medium dataframe"),
    ({"array": np.random.random((5000, 100))}, "large array")
]

for data, description in test_cases:
    result, metrics = profiler.profile_serialization(data, description=description)
    print(f"{description}: {metrics['duration']:.3f}s, Success: {metrics['success']}")

# Get overall performance summary
summary = profiler.get_performance_summary()
print("Performance Summary:", summary)
```

## Production Examples

### API Integration
Using datason in web APIs.

```python
from flask import Flask, request, jsonify
import datason as ds

app = Flask(__name__)

# Configure for API use
API_CONFIG = ds.get_api_config()

@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    """ML prediction endpoint with datason serialization."""

    try:
        # Get input data
        input_data = request.json

        # Deserialize and validate
        features = ds.deserialize(input_data["features"])

        # Run ML model (placeholder)
        predictions = run_ml_model(features)

        # Serialize results
        result = {
            "predictions": predictions,
            "metadata": {
                "timestamp": datetime.now(),
                "model_version": "v1.2.0"
            }
        }

        serialized_result = ds.serialize(result, config=API_CONFIG)

        return jsonify({
            "status": "success",
            "data": serialized_result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

def run_ml_model(features):
    """Placeholder ML model."""
    import numpy as np
    return np.random.random(len(features)).tolist()

# if __name__ == '__main__':
#     app.run(debug=True)
```

## More Examples

For complete, runnable examples, see the examples directory:

- **[Basic Usage](https://github.com/danielendler/datason/blob/main/examples/basic_usage.py)** - Fundamental serialization patterns
- **[AI/ML Examples](https://github.com/danielendler/datason/blob/main/examples/ai_ml_examples.py)** - Machine learning library integration
- **[Advanced ML](https://github.com/danielendler/datason/blob/main/examples/advanced_ml_examples.py)** - Complex ML workflows
- **[Chunked & Template Demo](https://github.com/danielendler/datason/blob/main/examples/chunked_and_template_demo.py)** - Large data processing
- **[Pickle Bridge Demo](https://github.com/danielendler/datason/blob/main/examples/pickle_bridge_demo.py)** - Legacy migration
- **[Security Patterns](https://github.com/danielendler/datason/blob/main/examples/security_patterns_demo.py)** - Privacy and security
- **[Auto-Detection Demo](https://github.com/danielendler/datason/blob/main/examples/auto_detection_and_metadata_demo.py)** - Automatic type detection

Each example file contains detailed comments and demonstrates best practices for different use cases.
