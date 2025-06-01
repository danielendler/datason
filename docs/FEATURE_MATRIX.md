# datason Feature Matrix 🔍

## Library Comparison Matrix

| Feature | datason | json (stdlib) | pickle | joblib | ujson | orjson |
|---------|-----------|---------------|--------|--------|-------|--------|
| **Core Features** |
| Basic JSON Types | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Complex Python Objects | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Human Readable Output | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cross-Language Compatible | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Data Science Support** |
| NumPy Arrays | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Pandas DataFrames | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Pandas Series | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| DateTime Objects | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Timezone Handling | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **ML/AI Support** |
| PyTorch Tensors | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| TensorFlow Tensors | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Scikit-learn Models | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| JAX Arrays | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| HuggingFace Models | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Special Types** |
| UUID Objects | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| NaN/Infinity Handling | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Complex Numbers | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Bytes/ByteArray | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Performance** |
| Small Objects | ⚡ | ⚡⚡ | ⚡ | ⚡ | ⚡⚡⚡ | ⚡⚡⚡ |
| Large Objects | ⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| Memory Efficiency | ⚡⚡ | ⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| **Safety** |
| Type Preservation | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Circular Reference Safe | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Error Handling | ✅ | ⚡ | ⚡ | ⚡ | ⚡ | ⚡ |
| **Ecosystem** |
| Zero Dependencies | ❌* | ✅ | ✅ | ❌ | ❌ | ❌ |
| Optional Dependencies | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Python Version Support | 3.8+ | 3.7+ | 3.7+ | 3.8+ | 3.7+ | 3.8+ |

*Core functionality works without dependencies; ML features require optional packages

## Use Case Recommendations

### 🎯 When to Use datason

**Perfect for:**
- 🤖 **ML/AI Applications**: Model serialization, experiment tracking
- 📊 **Data Science Workflows**: DataFrame processing, scientific computing
- 🌐 **API Development**: Complex response serialization
- 🔬 **Research Projects**: Reproducible experiment data
- 📱 **Cross-platform Data Exchange**: When you need human-readable format

**Example scenarios:**
```python
# ✅ datason excels here
ml_experiment = {
    'model': trained_sklearn_model,
    'predictions': torch_tensor_predictions,
    'data': pandas_dataframe,
    'timestamp': datetime.now(),
    'metrics': {'accuracy': 0.95}
}
```

### 🎯 When to Use Alternatives

**Use `json` (stdlib) when:**
- Simple data types only (dict, list, str, int, float, bool)
- Maximum compatibility needed
- Minimal dependencies required
- Working with web APIs expecting standard JSON

**Use `pickle` when:**
- Python-only environment
- Complex object graphs with references
- Maximum fidelity required (exact object reconstruction)
- Performance is critical and human readability not needed

**Use `joblib` when:**
- Primarily working with NumPy arrays
- Scientific computing focused
- Need compression
- Working with large arrays/matrices

**Use `ujson`/`orjson` when:**
- Maximum performance for basic JSON types
- High-throughput applications
- Simple data structures
- Speed is the primary concern

## Performance Benchmarks

**Real benchmarks (Python 3.13.3, macOS):**

### Simple JSON-Compatible Data (1000 user objects)
```
json:      0.4ms (1.0x baseline)
datason: 0.6ms (1.6x overhead)
```

### Complex Data (500 objects with UUIDs/datetimes)
```
datason: 2.1ms (only option for this data)
pickle:    0.7ms (3.2x faster but binary, Python-only)
```

### High-Throughput Scenarios
```
datason performance:
- Large nested datasets: 272,654 items/second
- NumPy array processing: 5.5M elements/second  
- Pandas DataFrame rows: 195,242 rows/second
- Round-trip (serialize+JSON+deserialize): 1.4ms
```

### Memory Usage (Efficient processing)
```
datason: Optimized for streaming large datasets
joblib:    Good for NumPy arrays with compression
pickle:    High memory usage for complex objects
json:      N/A (can't serialize DataFrames/tensors)
```

*See `benchmark_real_performance.py` for detailed benchmarking methodology

## Feature Deep Dive

### 🤖 ML/AI Object Support

datason provides native support for major ML frameworks:

```python
# PyTorch
tensor = torch.randn(100, 100)
# Result: {"_type": "torch.Tensor", "_shape": [100, 100], "_data": [...]}

# TensorFlow
tf_tensor = tf.constant([[1, 2], [3, 4]])
# Result: {"_type": "tf.Tensor", "_shape": [2, 2], "_data": [[1, 2], [3, 4]]}

# Scikit-learn
model = RandomForestClassifier()
# Result: {"_type": "sklearn.model", "_class": "...", "_params": {...}}
```

### 📊 Data Science Integration

Seamless pandas and NumPy support:

```python
# DataFrame with mixed types
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'created': [datetime.now(), datetime.now(), datetime.now()]
})
# Automatically handles datetime serialization in DataFrame
```

### 🛡️ Safety Features

- **NaN/Infinity Handling**: Converts to `null` for JSON compatibility
- **Circular Reference Detection**: Prevents infinite recursion
- **Graceful Fallbacks**: Falls back to string representation for unknown types
- **Type Preservation**: Maintains type information for accurate reconstruction

### ⚡ Performance Optimizations

- **Early Detection**: Skips processing for already-serialized data
- **Memory Streaming**: Handles large datasets without loading everything in memory
- **Smart Caching**: Reuses serialization results for repeated objects
- **Lazy Loading**: Only imports ML libraries when needed

## Integration Examples

### Flask/FastAPI APIs
```python
@app.route('/model/predict')
def predict():
    result = model.predict(data)
    return jsonify(datason.serialize({
        'predictions': result,  # NumPy array
        'model_info': model,    # sklearn model
        'timestamp': datetime.now()
    }))
```

### Experiment Tracking
```python
experiment = {
    'id': uuid.uuid4(),
    'model': trained_model,
    'data': validation_dataframe,
    'metrics': metrics_dict,
    'artifacts': {
        'weights': model.coef_,
        'predictions': predictions_array
    }
}
experiment_json = datason.serialize(experiment)
```

### Data Pipeline State
```python
pipeline_state = {
    'stage': 'feature_engineering',
    'input_data': raw_dataframe,
    'processed_data': processed_dataframe,
    'transformations': sklearn_preprocessor,
    'statistics': summary_stats,
    'timestamp': datetime.now()
}
```

## Decision Matrix

Use this matrix to choose the right tool:

| Requirement | Recommended Library |
|-------------|-------------------|
| Basic JSON + High Performance | orjson, ujson |
| Standard JSON only | json (stdlib) |
| Python objects + High Performance | pickle |
| NumPy arrays + Compression | joblib |
| **ML/AI data + Human Readable** | **datason** |
| **Complex data + Cross-platform** | **datason** |
| **API responses with mixed types** | **datason** |
| **Data science workflows** | **datason** |

## Migration Guide

### From `json` to datason
```python
# Before
try:
    json_str = json.dumps(data)
except TypeError:
    # Handle unsupported types manually
    pass

# After
json_str = json.dumps(datason.serialize(data))  # Just works!
```

### From `pickle` to datason
```python
# Before (Python-only)
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# After (Cross-platform)
with open('data.json', 'w') as f:
    json.dump(datason.serialize(data), f)
```

### From `joblib` to datason
```python
# Before (Binary format)
joblib.dump(sklearn_model, 'model.joblib')

# After (Human-readable)
with open('model.json', 'w') as f:
    json.dump(datason.serialize(sklearn_model), f)
```

## Key Differentiators Explained

### 🌐 Cross-Language Compatibility

**datason outputs standard JSON** that any programming language can read:

```python
# datason output - works everywhere
{
    "model_results": {
        "_type": "torch.Tensor",
        "_shape": [3, 3],
        "_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "accuracy": 0.95
}
```

**Real-world scenarios where this matters:**

🚀 **Microservices Architecture**
```python
# Python ML service
ml_results = datason.serialize({
    'predictions': torch_predictions,
    'model_version': '2.1.0',
    'confidence': confidence_scores
})
# Send to JavaScript frontend, Java backend, Go service, etc.
```

🌍 **Multi-Language Data Pipelines**
```python
# Python data processing
processed_data = datason.serialize({
    'features': numpy_features,
    'labels': pandas_series,
    'metadata': {'processed_at': datetime.now()}
})
# R analytics, JavaScript visualization, Java storage - all can read this
```

🔗 **API Integrations**
```python
# Your Python API response
@app.route('/api/analysis')
def get_analysis():
    return datason.serialize({
        'dataframe': analysis_df,
        'charts': plot_data,
        'timestamp': datetime.now()
    })
# Any client (React, Vue, mobile apps) can consume this directly
```

**Compare with pickle** (Python-only):
```python
# Pickle output - binary garbage to other languages
b'\x80\x04\x95\x1a\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04name\x94\x8c\x04John\x94s.'
# ❌ JavaScript: "What is this??"
# ❌ Java: "Cannot parse this"
# ❌ R: "Error reading data"
```

### 👁️ Human-Readable Output

**datason produces readable JSON** you can inspect, debug, and version control:

```json
{
  "experiment_id": "exp_20240115_v1",
  "model": {
    "_type": "sklearn.RandomForestClassifier",
    "_params": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    }
  },
  "results": {
    "accuracy": 0.94,
    "precision": 0.91,
    "recall": 0.88
  },
  "training_data": {
    "_type": "pandas.DataFrame",
    "_shape": [1000, 25],
    "_columns": ["feature_1", "feature_2", "..."],
    "_sample": [
      {"feature_1": 1.2, "feature_2": 0.8, "...": "..."}
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Practical benefits:**

🔍 **Easy Debugging**
```bash
# You can actually READ what's happening
cat experiment_results.json | jq '.results.accuracy'
# Output: 0.94

# vs pickle debugging (impossible):
cat experiment_results.pkl
# Output: Binary gibberish
```

📋 **Data Inspection**
```python
# Open in any text editor and see your data structure
# Perfect for:
# - Checking serialization worked correctly
# - Understanding data flow in pipelines  
# - Sharing results with non-Python colleagues
# - Troubleshooting API responses
```

🔄 **Version Control Friendly**
```bash
# Git can show meaningful diffs
git diff experiment_results.json
# Shows actual changes to accuracy, parameters, etc.

# vs pickle in git (useless):
git diff experiment_results.pkl  
# "Binary files differ" - no useful information
```

📊 **Business Stakeholder Sharing**
```python
# Data scientists can share results with business teams
# Marketing can read the JSON and understand conversion rates
# Product managers can see A/B test results directly
# No need for special Python tools to view the data
```

### 🛠️ When Each Tool Excels

| Scenario | Best Tool | Why |
|----------|-----------|-----|
| **Python-only ML pipeline** | pickle | Fastest, perfect object reconstruction |
| **Multi-language microservices** | **datason** | JSON works everywhere |
| **API responses** | **datason** | Human-readable, debuggable |
| **Data science collaboration** | **datason** | Readable by everyone |
| **High-performance NumPy** | joblib | Optimized for arrays |
| **Web APIs (basic JSON)** | orjson/ujson | Maximum speed |
| **Debugging complex data** | **datason** | Inspect results easily |
| **Cross-team data sharing** | **datason** | No language barriers |
| **Scientific reproducibility** | **datason** | Human-readable results |

### 🎯 Concrete Use Case Examples

**❌ Problems with pickle:**
```python
# Data science team (Python)
pickle.dump(ml_results, open('results.pkl', 'wb'))

# Frontend team (JavaScript)
❌ "How do we read this .pkl file?"
❌ "Can you export to CSV for everything?"
❌ "We can't debug API responses"

# Business stakeholders
❌ "We need special tools to see our data"
❌ "Can't inspect experiment results directly"
```

**✅ Solutions with datason:**
```python
# Data science team (Python)
json.dump(datason.serialize(ml_results), open('results.json', 'w'))

# Frontend team (JavaScript)
✅ fetch('/api/results').then(r => r.json()) // Just works!
✅ console.log(results.model.accuracy) // Easy debugging

# Business stakeholders  
✅ Open results.json in any text editor
✅ See experiment results immediately
✅ Share data with external partners easily
```

**🚀 Real-World Success Story:**
```python
# Before: Separate export formats for each team
pickle.dump(model, 'model.pkl')        # For Python team
model.to_csv('model.csv')              # For Excel users  
json.dump(manual_dict, 'api.json')     # For frontend
# 3 different formats, data inconsistency, maintenance nightmare

# After: One format for everyone
datason_result = datason.serialize({
    'model': sklearn_model,
    'predictions': numpy_predictions,
    'metadata': {'timestamp': datetime.now()}
})
# ✅ Python team: Deserialize back to original objects
# ✅ Frontend team: Standard JSON consumption
# ✅ Business team: Human-readable results
# ✅ External partners: Universal JSON format
```
