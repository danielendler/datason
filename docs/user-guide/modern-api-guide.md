# 🎯 Modern API Guide

Welcome to datason's modern API - a collection of intention-revealing functions designed for clarity, progressive complexity, and domain-specific optimization. This guide will help you master the modern API and understand when and how to use each function.

## 🌟 Why Modern API?

The modern API addresses common pain points with traditional serialization:

- **🎯 Clear Intent**: Function names tell you exactly what they do
- **📈 Progressive Complexity**: Start simple, add complexity as needed
- **🔧 Domain-Specific**: Optimized functions for ML, API, security use cases
- **🧩 Composable**: Mix and match features for your specific needs
- **🔍 Self-Documenting**: Built-in help and discovery

## 🔹 Serialization Functions (Dump Functions)

### dump() - The Universal Function

The `dump()` function is your Swiss Army knife - it can handle any scenario with composable options:

```python
import datason as ds
import pandas as pd
import torch

# Basic usage
data = {"values": [1, 2, 3], "timestamp": datetime.now()}
result = ds.dump(data)

# Composable options for specific needs
complex_data = {
    "model": torch.nn.Linear(10, 1),
    "user_data": {"email": "user@example.com", "ssn": "123-45-6789"},
    "large_df": pd.DataFrame(np.random.random((10000, 50)))
}

# Combine multiple features
result = ds.dump(
    complex_data,
    secure=True,     # Enable PII redaction
    ml_mode=True,    # Optimize for ML objects
    chunked=True,    # Memory-efficient processing
    fast_mode=True   # Performance optimization
)
```

**When to use:**
- General-purpose serialization
- When you need multiple features combined
- As a starting point before moving to specialized functions

### dump_ml() - ML-Optimized

Perfect for machine learning workflows with automatic optimization for ML objects:

```python
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ML objects are automatically optimized
ml_data = {
    "pytorch_model": torch.nn.Linear(100, 10),
    "sklearn_model": RandomForestClassifier(),
    "tensor": torch.randn(1000, 100),
    "numpy_features": np.random.random((1000, 50)),
    "training_config": {"lr": 0.001, "epochs": 100}
}

result = ds.dump_ml(ml_data)
```

**Features:**
- Optimized tensor serialization
- Model state preservation
- NumPy array compression
- Training metadata handling

**When to use:**
- PyTorch/TensorFlow models
- NumPy arrays and tensors
- ML training pipelines
- Model checkpoints

### dump_api() - API-Safe

Clean, web-safe JSON output optimized for REST APIs:

```python
# API response with mixed data
api_response = {
    "status": "success",
    "data": [1, 2, 3],
    "errors": None,           # Will be removed
    "timestamp": datetime.now(),
    "pagination": {
        "page": 1,
        "total": 100,
        "has_more": True
    }
}

clean_result = ds.dump_api(api_response)
# Result: Clean JSON, null values removed, optimized structure
```

**Features:**
- Removes null/None values
- Optimizes nested structures
- Ensures JSON compatibility
- Minimal payload size

**When to use:**
- REST API endpoints
- Web service responses
- JSON data for frontend
- Configuration files

### dump_secure() - Security-Focused

Automatic PII redaction and security-focused serialization:

```python
# Sensitive user data
user_data = {
    "profile": {
        "name": "John Doe",
        "email": "john@example.com",
        "ssn": "123-45-6789",
        "phone": "+1-555-123-4567"
    },
    "account": {
        "password": "secret123",
        "api_key": "sk-abc123def456",
        "credit_card": "4532-1234-5678-9012"
    }
}

# Automatic PII redaction
secure_result = ds.dump_secure(user_data, redact_pii=True)

# Custom redaction patterns
custom_secure = ds.dump_secure(
    user_data,
    redact_fields=["internal_id", "session_token"],
    redact_patterns=[
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit cards
        r"sk-[a-zA-Z0-9]{20,}",          # API keys
    ]
)
```

**Features:**
- Automatic PII detection and redaction
- Custom redaction patterns
- Field-based redaction
- Audit trail generation

**When to use:**
- User data with PII
- Financial information
- Healthcare data
- Compliance requirements

### dump_fast() - Performance-Optimized

High-throughput serialization with minimal overhead:

```python
# Large batch processing
batch_data = []
for i in range(10000):
    batch_data.append({
        "id": i,
        "value": random.random(),
        "category": f"cat_{i % 10}"
    })

# Optimized for speed
fast_result = ds.dump_fast(batch_data)
```

**Features:**
- Minimal type checking
- Optimized algorithms
- Reduced memory allocations
- Streamlined processing

**When to use:**
- High-volume data processing
- Real-time systems
- Performance-critical paths
- Simple data structures

### dump_chunked() - Memory-Efficient

Handle very large objects without memory exhaustion:

```python
# Very large dataset
large_data = {
    "images": [np.random.random((512, 512, 3)) for _ in range(1000)],
    "features": pd.DataFrame(np.random.random((100000, 200))),
    "metadata": {"size": "huge", "format": "research"}
}

# Process in memory-efficient chunks
chunked_result = ds.dump_chunked(large_data, chunk_size=1000)

# Chunked result is a generator - process piece by piece
for chunk in chunked_result:
    # Process each chunk independently
    process_chunk(chunk)
```

**Features:**
- Memory-efficient processing
- Streaming capabilities
- Configurable chunk size
- Generator-based output

**When to use:**
- Very large datasets
- Memory-constrained environments
- Streaming applications
- ETL pipelines

### stream_dump() - File Streaming

Direct streaming to files for extremely large data:

```python
# Massive dataset that won't fit in memory
huge_data = {
    "sensor_data": generate_sensor_readings(1000000),
    "images": generate_image_batch(10000),
    "metadata": {"source": "sensors", "duration": "24h"}
}

# Stream directly to file
with open('massive_dataset.json', 'w') as f:
    ds.stream_dump(huge_data, f)

# For compressed output
import gzip
with gzip.open('massive_dataset.json.gz', 'wt') as f:
    ds.stream_dump(huge_data, f)
```

**Features:**
- Direct file output
- No memory buffering
- Supports any file-like object
- Works with compression

**When to use:**
- Extremely large datasets
- Direct file output
- Memory-constrained systems
- Archive creation

## 🔹 Deserialization Functions (Load Functions)

The load functions provide progressive complexity - start with basic exploration and move to production-ready functions as needed.

### load_basic() - Fast Exploration (60-70% Success Rate)

Quick and dirty deserialization for data exploration:

```python
# Simple JSON data
json_data = '''
{
    "values": [1, 2, 3, 4, 5],
    "timestamp": "2024-01-01T12:00:00",
    "metadata": {"version": 1.0}
}
'''

# Fast basic loading - minimal processing
basic_result = ds.load_basic(json_data)
print(basic_result)
# Note: timestamp remains as string, minimal type conversion
```

**Features:**
- Fastest loading
- Basic type conversion
- Minimal error handling
- Good for exploration

**When to use:**
- Data exploration and debugging
- Simple JSON structures
- Quick prototyping
- Performance-critical loading

### load_smart() - Production-Ready (80-90% Success Rate)

Intelligent type detection and restoration for production use:

```python
# Complex serialized data
complex_data = {
    "dataframe": pd.DataFrame({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]}),
    "timestamp": datetime.now(),
    "array": np.array([1, 2, 3, 4, 5]),
    "config": {"learning_rate": 0.001}
}

serialized = ds.dump(complex_data)

# Smart loading with type restoration
smart_result = ds.load_smart(serialized)
print(type(smart_result["dataframe"]))  # <class 'pandas.core.frame.DataFrame'>
print(type(smart_result["timestamp"]))  # <class 'datetime.datetime'>
print(type(smart_result["array"]))      # <class 'numpy.ndarray'>
```

**Features:**
- Intelligent type detection
- Good success rate
- Handles complex types
- Production-ready

**When to use:**
- Production applications
- Complex data structures
- General-purpose loading
- When reliability matters

### load_perfect() - 100% Success Rate (Requires Template)

Template-based loading for critical applications requiring 100% reliability:

```python
# Define the expected structure
template = {
    "user_id": int,
    "profile": {
        "name": str,
        "email": str,
        "created": datetime
    },
    "scores": [float],
    "metadata": {
        "version": str,
        "features": [str]
    }
}

json_data = '''
{
    "user_id": 12345,
    "profile": {
        "name": "Alice",
        "email": "alice@example.com",
        "created": "2024-01-01T12:00:00"
    },
    "scores": [95.5, 87.2, 92.1],
    "metadata": {
        "version": "v1.0",
        "features": ["premium", "analytics"]
    }
}
'''

# Perfect restoration using template
perfect_result = ds.load_perfect(json_data, template)
# 100% guaranteed to match template structure
```

**Features:**
- 100% success rate
- Type validation
- Structure enforcement
- Error reporting

**When to use:**
- Critical applications
- Data validation required
- Schema enforcement
- API input validation

### load_typed() - Metadata-Based (95% Success Rate)

Uses embedded type metadata for high-reliability restoration:

```python
# Serialize with type information
original_data = {
    "model": torch.nn.Linear(10, 1),
    "dataframe": pd.DataFrame({"x": [1, 2, 3]}),
    "timestamp": datetime.now()
}

# Serialize with type metadata
serialized_with_types = ds.dump(original_data, include_type_info=True)

# Load using embedded type information
typed_result = ds.load_typed(serialized_with_types)
# Types are restored using embedded metadata
```

**Features:**
- High success rate (95%)
- Uses embedded metadata
- Automatic type restoration
- No template required

**When to use:**
- When original data had type metadata
- High-reliability needs
- Complex type restoration
- Self-describing data

## 🔹 Utility & Discovery Functions

### help_api() - Interactive Guidance

Get personalized recommendations for your use case:

```python
# Get interactive help
ds.help_api()

# Example output:
# 🎯 datason Modern API Guide
#
# SERIALIZATION (Dump Functions):
# • dump() - General purpose with composable options
# • dump_ml() - ML models, tensors, NumPy arrays
# • dump_api() - Web APIs, clean JSON output
# • dump_secure() - Sensitive data with PII redaction
# • dump_fast() - High-throughput scenarios
# • dump_chunked() - Large objects, memory efficiency
#
# DESERIALIZATION (Load Functions):
# • load_basic() - 60-70% success, fastest (exploration)
# • load_smart() - 80-90% success, moderate (production)
# • load_perfect() - 100% success, requires template (critical)
# • load_typed() - 95% success, uses metadata
#
# RECOMMENDATIONS:
# • For ML workflows: dump_ml() + load_smart()
# • For APIs: dump_api() + load_smart()
# • For sensitive data: dump_secure() + load_smart()
# • For exploration: dump() + load_basic()
# • For production: dump() + load_smart() or load_typed()
```

### get_api_info() - API Metadata

Programmatic access to API information:

```python
# Get comprehensive API information
api_info = ds.get_api_info()

print("Dump functions:", api_info['dump_functions'])
print("Load functions:", api_info['load_functions'])
print("Features:", api_info['features'])
print("Recommendations:", api_info['recommendations'])

# Use programmatically
if 'dump_ml' in api_info['dump_functions']:
    # ML optimization available
    result = ds.dump_ml(ml_data)
```

### dumps() / loads() - JSON Compatibility

Drop-in replacement for Python's json module:

```python
import datason as ds

# Instead of:
# import json
# json_str = json.dumps(data)  # Fails with datetime, numpy, etc.
# data = json.loads(json_str)

# Use datason:
data = {
    "timestamp": datetime.now(),
    "array": np.array([1, 2, 3]),
    "dataframe": pd.DataFrame({"x": [1, 2, 3]})
}

# Like json.dumps() but handles complex types
json_str = ds.dumps(data)

# Like json.loads() but restores types
restored = ds.loads(json_str)
print(type(restored["timestamp"]))  # <class 'datetime.datetime'>
print(type(restored["array"]))      # <class 'numpy.ndarray'>
```

## 🎯 Choosing the Right Function

### Quick Decision Tree

```
📊 SERIALIZATION (What are you saving?)
├── 🤖 ML models/tensors/arrays → dump_ml()
├── 🌐 API/web responses → dump_api()
├── 🔒 Sensitive/PII data → dump_secure()
├── ⚡ High-volume/performance → dump_fast()
├── 💾 Very large data → dump_chunked() or stream_dump()
└── 🎯 General purpose → dump()

📥 DESERIALIZATION (How reliable do you need it?)
├── 🔍 Quick exploration (60-70%) → load_basic()
├── 🏭 Production ready (80-90%) → load_smart()
├── 🎯 Critical/validated (100%) → load_perfect()
└── 📋 With metadata (95%) → load_typed()
```

### Usage Patterns by Scenario

#### Data Science Workflow

```python
# Exploratory analysis
raw_data = ds.load_basic(json_file)  # Quick exploration

# Data processing
processed = process_data(raw_data)
result = ds.dump_ml(processed)  # ML-optimized storage

# Production pipeline
validated_data = ds.load_smart(result)  # Reliable loading
```

#### Web API Development

```python
# API endpoint
@app.route('/api/data')
def get_data():
    data = get_database_data()
    return ds.dump_api(data)  # Clean JSON response

# API input validation
@app.route('/api/data', methods=['POST'])
def create_data():
    template = get_validation_template()
    try:
        validated = ds.load_perfect(request.json, template)
        return process_data(validated)
    except TemplateError:
        return {"error": "Invalid data structure"}, 400
```

#### Security-Sensitive Applications

```python
# User data processing
user_input = request.json
secure_data = ds.dump_secure(user_input, redact_pii=True)

# Logging (safe for logs)
logger.info(f"Processed user data: {secure_data}")

# Storage (PII-free)
store_in_database(secure_data)
```

#### High-Performance Systems

```python
# Batch processing
for batch in data_batches:
    processed = ds.dump_fast(batch)  # High-throughput
    queue.put(processed)

# Real-time systems
result = ds.load_basic(incoming_data)  # Fastest loading
process_realtime(result)
```

#### Large Data Systems

```python
# ETL pipeline
with open('large_output.json', 'w') as f:
    ds.stream_dump(massive_dataset, f)  # Direct streaming

# Chunked processing
for chunk in ds.dump_chunked(huge_data, chunk_size=1000):
    process_chunk(chunk)  # Memory-efficient
```

## 🔄 Migration from Traditional API

### Gradual Migration Strategy

```python
# Phase 1: Add modern API alongside traditional
def serialize_data(data, use_modern=False):
    if use_modern:
        return ds.dump_ml(data)  # Modern approach
    else:
        return ds.serialize(data, config=ds.get_ml_config())  # Traditional

# Phase 2: Feature flags for new functionality
def api_response(data):
    if feature_flags.modern_api:
        return ds.dump_api(data)
    else:
        return ds.serialize(data, config=ds.get_api_config())

# Phase 3: Full migration
def process_ml_data(data):
    return ds.dump_ml(data)  # Modern API only
```

### Equivalent Functions

| Traditional API | Modern API | Notes |
|----------------|------------|-------|
| `serialize(data, config=get_ml_config())` | `dump_ml(data)` | Automatic ML optimization |
| `serialize(data, config=get_api_config())` | `dump_api(data)` | Clean JSON output |
| `serialize(data, config=get_performance_config())` | `dump_fast(data)` | Performance optimized |
| `serialize_chunked(data)` | `dump_chunked(data)` | Memory efficient |
| `deserialize(data)` | `load_smart(data)` | General purpose |
| `auto_deserialize(data)` | `load_basic(data)` | Quick exploration |
| `deserialize_with_template(data, template)` | `load_perfect(data, template)` | Template-based |

## 🛠️ Advanced Patterns

### Composable Serialization

```python
# Build up complexity as needed
def adaptive_serialize(data, context):
    options = {}

    if context.has_ml_objects:
        options['ml_mode'] = True

    if context.has_sensitive_data:
        options['secure'] = True

    if context.memory_constrained:
        options['chunked'] = True

    if context.performance_critical:
        options['fast_mode'] = True

    return ds.dump(data, **options)
```

### Progressive Loading

```python
def smart_load(data, reliability_required='medium'):
    """Load data with appropriate reliability level."""

    if reliability_required == 'low':
        return ds.load_basic(data)
    elif reliability_required == 'medium':
        return ds.load_smart(data)
    elif reliability_required == 'high':
        # Requires template
        template = infer_or_get_template(data)
        return ds.load_perfect(data, template)
    else:
        # Use metadata if available
        return ds.load_typed(data)
```

### Error Handling and Fallbacks

```python
def robust_deserialize(data):
    """Try progressively simpler approaches if needed."""

    # Try typed first (highest success rate without template)
    try:
        return ds.load_typed(data)
    except (TypeError, ValueError):
        pass

    # Fall back to smart loading
    try:
        return ds.load_smart(data)
    except (TypeError, ValueError):
        pass

    # Last resort: basic loading
    try:
        return ds.load_basic(data)
    except Exception as e:
        raise DeserializationError(f"All loading methods failed: {e}")
```

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. Deserialization Success Rate Lower Than Expected

```python
# Problem: load_smart() not working well
data = ds.load_smart(json_data)  # Only 60% success

# Solution: Check if you need a template
template = create_template_for_data()
data = ds.load_perfect(json_data, template)  # 100% success
```

#### 2. Performance Issues

```python
# Problem: Slow serialization
result = ds.dump(large_data)  # Too slow

# Solution: Use performance-optimized function
result = ds.dump_fast(large_data)  # Much faster

# Or use chunked processing for memory issues
result = ds.dump_chunked(large_data, chunk_size=1000)
```

#### 3. Security Concerns

```python
# Problem: PII in serialized data
result = ds.dump(user_data)  # Contains PII

# Solution: Use security-focused function
result = ds.dump_secure(user_data, redact_pii=True)  # PII redacted
```

### Debugging Tools

```python
# Get detailed API information
api_info = ds.get_api_info()
print("Available functions:", api_info['dump_functions'])

# Interactive help
ds.help_api()  # Get recommendations

# Test different loading strategies
for loader in [ds.load_basic, ds.load_smart, ds.load_typed]:
    try:
        result = loader(problematic_data)
        print(f"{loader.__name__}: SUCCESS")
        break
    except Exception as e:
        print(f"{loader.__name__}: FAILED - {e}")
```

## 🚀 Best Practices

### 1. Start Simple, Add Complexity

```python
# Start with basic functions
result = ds.dump(data)
loaded = ds.load_smart(result)

# Add features as needed
result = ds.dump(data, ml_mode=True)  # Add ML optimization
result = ds.dump(data, ml_mode=True, secure=True)  # Add security
```

### 2. Choose the Right Tool for the Job

```python
# For each use case, pick the most appropriate function
ml_result = ds.dump_ml(model_data)      # ML objects
api_result = ds.dump_api(response_data) # Web APIs
secure_result = ds.dump_secure(user_data, redact_pii=True)  # PII data
```

### 3. Handle Errors Gracefully

```python
def safe_serialize(data, strategy='smart'):
    try:
        if strategy == 'ml':
            return ds.dump_ml(data)
        elif strategy == 'api':
            return ds.dump_api(data)
        else:
            return ds.dump(data)
    except Exception as e:
        # Fallback to basic serialization
        return ds.dump_fast(data)
```

### 4. Use Discovery Features

```python
# Let the API guide you
ds.help_api()  # Get recommendations

# Check capabilities programmatically
api_info = ds.get_api_info()
if 'dump_ml' in api_info['dump_functions']:
    use_ml_optimization = True
```

## 🎯 Summary

The modern API provides a clear, progressive approach to serialization:

- **🔹 7 Dump Functions**: From general-purpose to highly specialized
- **🔹 4 Load Functions**: Progressive complexity from exploration to production
- **🔹 3 Utility Functions**: Discovery, help, and JSON compatibility
- **🔹 Composable Design**: Mix and match features as needed
- **🔹 Self-Documenting**: Built-in guidance and recommendations

Start with `dump()` and `load_smart()` for general use, then specialize as your needs become clearer. The modern API grows with your application - from quick prototypes to production-critical systems.

Ready to explore real-world examples? Check out our [Examples Gallery](examples/index.md) for comprehensive usage patterns!
