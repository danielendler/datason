# Migration Guide

Comprehensive guide for upgrading from other serialization libraries to datason, with conversion strategies and compatibility information.

## 🎯 Overview

This guide helps you migrate from:

- **Standard JSON**: Add support for complex types
- **Pickle**: Improve security and cross-platform compatibility
- **Other Libraries**: orjson, ujson, joblib, jsonpickle
- **Custom Solutions**: Replace manual serialization code

## 📦 From Standard JSON

### Why Migrate?

Standard JSON has limitations with Python's rich type system:

```python
import json
from datetime import datetime
import uuid

# Standard JSON fails with these
data = {
    "timestamp": datetime.now(),  # ❌ Not serializable
    "user_id": uuid.uuid4(),      # ❌ Not serializable  
    "settings": {1, 2, 3}         # ❌ Sets not supported
}

# json.dumps(data)  # TypeError!
```

### Migration Steps

**Step 1: Install datason**
```bash
pip install datason
```

**Step 2: Simple replacement**
```python
import datason
import json

# Before
# json_string = json.dumps(data)

# After  
json_data = datason.serialize(data)
json_string = json.dumps(json_data)

# Or use datason's convenience function
json_string = datason.serialize_to_json(data)
```

**Step 3: Handle complex types**
```python
from datetime import datetime
import uuid
import datason

# Complex data that standard JSON can't handle
data = {
    "timestamp": datetime.now(),
    "user_id": uuid.uuid4(),
    "tags": {"python", "data", "json"},  # Set
    "config": bytes([1, 2, 3, 4])       # Bytes
}

# datason handles automatically
result = datason.serialize(data)
# Output: All types converted to JSON-compatible format

# Round-trip preservation
restored = datason.deserialize(result)
assert isinstance(restored["timestamp"], datetime)
assert isinstance(restored["user_id"], uuid.UUID)
```

## 🥒 From Pickle

### Migration Benefits

| Feature | Pickle | datason |
|---------|--------|---------|
| **Security** | ❌ Arbitrary code execution | ✅ Safe serialization |
| **Cross-platform** | ❌ Python-only | ✅ JSON-compatible |
| **Human readable** | ❌ Binary format | ✅ Text-based |
| **Performance** | ✅ Very fast | ✅ Good performance |
| **Complex objects** | ✅ Everything | ✅ Most data science objects |

### Direct Migration

```python
import pickle
import datason

# Before: Pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(complex_data, f)

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# After: datason
with open('data.json', 'w') as f:
    json_data = datason.serialize(complex_data)
    json.dump(json_data, f)

with open('data.json', 'r') as f:
    json_data = json.load(f)
    data = datason.deserialize(json_data)
```

### Pickle Bridge Migration

For legacy pickle files, use datason's Pickle Bridge:

```python
import datason

# Convert existing pickle files safely
json_data = datason.from_pickle("legacy_model.pkl")

# Bulk migration
stats = datason.convert_pickle_directory(
    source_dir="old_pickle_files/",
    target_dir="new_json_files/"
)
print(f"Converted {stats['files_converted']} files")
```

### ML Model Migration

```python
# Before: Pickle ML models
import pickle
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# ... train model ...

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# After: datason with metadata preservation
import datason

model_data = {
    'model': model,
    'hyperparameters': {'C': 1.0, 'solver': 'lbfgs'},
    'feature_names': ['feature_1', 'feature_2'],
    'training_metadata': {
        'accuracy': 0.95,
        'training_date': datetime.now(),
        'sklearn_version': sklearn.__version__
    }
}

# Serialize with rich metadata
result = datason.serialize(model_data)

# Save as JSON
with open('model.json', 'w') as f:
    json.dump(result, f, indent=2)
```

## ⚡ From High-Performance JSON Libraries

### From orjson/ujson

```python
# Before: orjson (fast but limited)
import orjson

try:
    json_bytes = orjson.dumps(data)  # Fast but fails on complex types
except TypeError:
    # Manual handling required
    pass

# After: datason with performance config
import datason

config = datason.get_performance_config()
result = datason.serialize(data, config=config)
json_string = json.dumps(result)

# Benefits:
# - Handles complex types automatically  
# - Configurable performance vs features
# - Still good performance (1.53x overhead for compatible data)
```

### Performance Comparison

| Library | Compatible Data | Complex Data | Types Supported |
|---------|----------------|--------------|-----------------|
| **orjson** | ⚡ Fastest | ❌ Fails | Basic JSON only |
| **ujson** | ⚡ Very fast | ❌ Fails | Basic JSON only |
| **datason** | ✅ Good (1.53x) | ✅ **Only option** | 20+ types |
| **jsonpickle** | ❌ Slow | ✅ Works | Many types |

## 📊 From Data Science Libraries

### From joblib (NumPy/Scikit-learn)

```python
# Before: joblib for ML objects
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Save model
model = RandomForestClassifier()
# ... train model ...
joblib.dump(model, 'model.joblib')

# Save arrays
arrays = [np.random.randn(1000, 10) for _ in range(5)]
joblib.dump(arrays, 'arrays.joblib')

# After: datason with better compatibility
import datason

# Complete ML pipeline serialization
ml_pipeline = {
    'model': model,
    'preprocessing': {
        'feature_scaler': StandardScaler().fit(X),
        'selected_features': [0, 1, 3, 7, 9]
    },
    'data': {
        'training_arrays': arrays,
        'feature_names': [f'feature_{i}' for i in range(10)]
    },
    'metadata': {
        'created_date': datetime.now(),
        'model_version': '1.0',
        'performance_metrics': {
            'accuracy': 0.94,
            'f1_score': 0.91
        }
    }
}

# Single serialization call handles everything
result = datason.serialize(ml_pipeline)

# Human-readable JSON output
with open('ml_pipeline.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### From Pandas to_json()

```python
# Before: pandas to_json (limited options)
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=3),
    'categories': pd.Categorical(['A', 'B', 'A']),
    'values': [1.1, 2.2, 3.3]
})

# Limited serialization options
json_str = df.to_json(orient='records')  # Loses categorical info

# After: datason with full type preservation
import datason

# Preserve all pandas metadata
result = datason.serialize(df)
# Categorical information, index, dtypes all preserved

# Round-trip verification
restored_df = datason.deserialize(result)
assert df.equals(restored_df)
assert df.dtypes.equals(restored_df.dtypes)
```

## 🔧 From jsonpickle

### Compatibility Migration

```python
# Before: jsonpickle
import jsonpickle

# jsonpickle handles Python objects but can be unsafe
json_str = jsonpickle.encode(complex_object)
restored = jsonpickle.decode(json_str)

# After: datason with better security
import datason

# Safer, more structured output
result = datason.serialize(complex_object)
restored = datason.deserialize(result)

# Benefits:
# - Better security (no arbitrary code execution)
# - Structured, predictable output format
# - Better performance for data science objects
# - Configurable behavior
```

### Feature Comparison

| Feature | jsonpickle | datason |
|---------|------------|---------|
| **Security** | ⚠️ Can execute code | ✅ Safe by design |
| **Output format** | ⚠️ Complex, nested | ✅ Clean, structured |
| **Performance** | ❌ Slower | ✅ Optimized |
| **Configuration** | ❌ Limited | ✅ Highly configurable |
| **Data science** | ✅ Good | ✅ **Excellent** |

## 🏗️ Migration Strategies

### Gradual Migration

**Phase 1: New code**
```python
# Start using datason for new features
def save_new_data(data):
    return datason.serialize(data)

def load_new_data(json_data):
    return datason.deserialize(json_data)
```

**Phase 2: Replace JSON usage**
```python
# Replace standard json usage
# Before:
# json.dumps(data)

# After:
datason.serialize_to_json(data)
```

**Phase 3: Legacy data conversion**
```python
# Convert existing pickle files
import glob

pickle_files = glob.glob("data/*.pkl")
for pickle_file in pickle_files:
    try:
        json_data = datason.from_pickle(pickle_file)
        json_file = pickle_file.replace('.pkl', '.json')
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Converted {pickle_file} → {json_file}")
    except Exception as e:
        print(f"Failed to convert {pickle_file}: {e}")
```

### Configuration Migration

```python
# Migrate existing configurations
def create_migration_config():
    """Create configuration that matches your current behavior."""

    # For API responses (replacing standard JSON)
    if current_use_case == "api":
        return datason.get_api_config()

    # For ML workflows (replacing pickle/joblib)
    elif current_use_case == "ml":
        return datason.get_ml_config()

    # For performance-critical applications (replacing orjson)
    elif current_use_case == "performance":
        return datason.get_performance_config()

    # For maximum compatibility (replacing jsonpickle)
    else:
        return datason.get_strict_config()
```

## 🧪 Testing Migration

### Validation Framework

```python
def validate_migration(original_data, serialization_func, deserialization_func):
    """Validate that migration preserves data integrity."""

    # Serialize with new method
    serialized = serialization_func(original_data)

    # Deserialize and compare
    restored = deserialization_func(serialized)

    # Type-specific validation
    if isinstance(original_data, pd.DataFrame):
        assert original_data.equals(restored)
        assert original_data.dtypes.equals(restored.dtypes)
    elif isinstance(original_data, np.ndarray):
        assert np.array_equal(original_data, restored)
        assert original_data.dtype == restored.dtype
    elif isinstance(original_data, dict):
        assert original_data == restored
    else:
        assert original_data == restored

    return True

# Test migration
test_data = {
    'dataframe': pd.DataFrame({'A': [1, 2, 3]}),
    'array': np.array([1, 2, 3]),
    'timestamp': datetime.now(),
    'uuid': uuid.uuid4()
}

# Validate datason migration
validate_migration(
    test_data,
    datason.serialize,
    datason.deserialize
)
print("✅ Migration validation passed!")
```

### Performance Testing

```python
def benchmark_migration(data, old_method, new_method, iterations=100):
    """Compare performance between old and new serialization methods."""
    import time

    # Benchmark old method
    old_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        old_method(data)
        old_times.append(time.perf_counter() - start)

    # Benchmark new method
    new_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        new_method(data)
        new_times.append(time.perf_counter() - start)

    old_avg = sum(old_times) / len(old_times)
    new_avg = sum(new_times) / len(new_times)

    print(f"Old method: {old_avg*1000:.2f}ms avg")
    print(f"New method: {new_avg*1000:.2f}ms avg")
    print(f"Performance ratio: {new_avg/old_avg:.2f}x")

# Example: JSON migration benchmark
simple_data = {"users": [{"id": i} for i in range(1000)]}

benchmark_migration(
    simple_data,
    lambda x: json.dumps(x),
    lambda x: datason.serialize_to_json(x)
)
```

## 🔄 Common Migration Patterns

### API Endpoint Migration

```python
# Before: Flask with standard JSON
from flask import Flask, jsonify

@app.route('/api/data')
def get_data():
    data = get_complex_data()  # Contains datetime, UUID, etc.
    # This would fail: return jsonify(data)

    # Manual conversion required
    cleaned_data = manual_clean_for_json(data)
    return jsonify(cleaned_data)

# After: Flask with datason
@app.route('/api/data')
def get_data():
    data = get_complex_data()
    config = datason.get_api_config()
    json_data = datason.serialize(data, config=config)
    return jsonify(json_data)
```

### Data Pipeline Migration

```python
# Before: Mixed serialization
def process_data_pipeline(data):
    # Different serialization for different types
    if isinstance(data, pd.DataFrame):
        return data.to_json()
    elif isinstance(data, np.ndarray):
        return joblib.dump(data, 'temp.joblib')
    elif has_datetime(data):
        return custom_datetime_handler(data)
    else:
        return json.dumps(data)

# After: Unified serialization
def process_data_pipeline(data):
    config = datason.get_ml_config()
    return datason.serialize(data, config=config)
```

### Database Storage Migration

```python
# Before: Multiple formats in database
def save_to_database(data):
    if isinstance(data, dict):
        db.save(json.dumps(data), format='json')
    elif isinstance(data, pd.DataFrame):
        db.save(pickle.dumps(data), format='pickle')
    elif isinstance(data, np.ndarray):
        db.save(joblib.dump(data), format='joblib')

# After: Unified JSON storage
def save_to_database(data):
    config = SerializationConfig(
        date_format=DateFormat.UNIX,  # Database-friendly
        dataframe_orient=DataFrameOrient.SPLIT  # Efficient
    )
    json_data = datason.serialize(data, config=config)
    db.save(json.dumps(json_data), format='json')
```

## 🚀 Best Practices for Migration

### 1. **Start Small**
```python
# Begin with non-critical data
test_data = get_sample_data()
result = datason.serialize(test_data)
```

### 2. **Validate Thoroughly**
```python
# Always test round-trip preservation
original = get_production_data()
serialized = datason.serialize(original)
restored = datason.deserialize(serialized)
assert deep_equals(original, restored)
```

### 3. **Performance Test**
```python
# Benchmark against current solution
current_time = benchmark_current_method()
datason_time = benchmark_datason_method()
print(f"Performance change: {datason_time/current_time:.2f}x")
```

### 4. **Gradual Rollout**
```python
# Use feature flags for gradual migration
if feature_flag('use_datason'):
    result = datason.serialize(data)
else:
    result = legacy_serialize(data)
```

## 🔗 Related Resources

- **[Pickle Bridge](../pickle-bridge/index.md)** - Migrate legacy pickle files
- **[Configuration](../configuration/index.md)** - Match your current behavior
- **[Performance](../performance/index.md)** - Optimize after migration
- **[Security](../../SECURITY.md)** - Security improvements over pickle

## 🚀 Next Steps

- **[Configuration Guide →](../configuration/index.md)** - Customize serialization behavior
- **[Performance Tuning →](../performance/index.md)** - Optimize for your specific needs
