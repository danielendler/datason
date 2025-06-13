# File Operations Guide

Datason provides comprehensive file operations for saving and loading data in both JSON and JSONL formats, with full integration of all features including ML optimization, security, streaming, and compression.

## Quick Start

```python
import datason
import numpy as np
import pandas as pd

# Save ML data to JSONL
ml_data = {
    "model_weights": np.random.randn(100, 50),
    "training_data": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
    "metrics": {"accuracy": 0.95}
}

# Save with ML optimization
datason.save_ml(ml_data, "model.jsonl.gz")  # Compressed JSONL

# Load back with smart reconstruction
loaded = list(datason.load_smart_file("model.jsonl.gz"))
```

## Formats Supported

### JSON Format (.json)
- **Best for**: Single objects, configuration files, metadata
- **Structure**: Single JSON object or array per file
- **Use when**: You have a complete dataset as one unit

```python
# Save single object
config = {"learning_rate": 0.001, "batch_size": 32}
datason.save_ml(config, "config.json")

# Save array of experiments  
experiments = [{"id": 1, "acc": 0.9}, {"id": 2, "acc": 0.85}]
datason.save_ml(experiments, "experiments.json")
```

### JSONL Format (.jsonl)
- **Best for**: Streaming data, logs, large datasets, line-by-line processing
- **Structure**: One JSON object per line
- **Use when**: You want to stream, append, or process data incrementally

```python
# Save to JSONL (each item on separate line)
training_logs = [
    {"epoch": 1, "loss": 0.8, "weights": np.random.randn(10)},
    {"epoch": 2, "loss": 0.6, "weights": np.random.randn(10)},
    {"epoch": 3, "loss": 0.4, "weights": np.random.randn(10)}
]
datason.save_ml(training_logs, "training.jsonl")
```

## Core Functions

### Save Functions

#### `save_ml(data, filepath, format=None)`
Optimized for ML workflows with automatic type preservation.

```python
# Automatic format detection from extension
datason.save_ml(ml_data, "model.jsonl")     # JSONL format
datason.save_ml(ml_data, "model.json")      # JSON format
datason.save_ml(ml_data, "model.jsonl.gz")  # Compressed JSONL

# Explicit format override
datason.save_ml(ml_data, "model.txt", format="jsonl")
```

#### `save_secure(data, filepath, redact_pii=False, redact_fields=None, format=None)`
Save with security features - PII redaction and data protection.

```python
sensitive_data = {
    "user": {
        "name": "John Doe",
        "email": "john@company.com",
        "ssn": "123-45-6789"
    },
    "api_key": "sk-1234567890"
}

# Automatic PII detection and redaction
datason.save_secure(
    sensitive_data,
    "secure_data.jsonl",
    redact_pii=True,
    redact_fields=["api_key"]
)
```

#### `save_api(data, filepath, format=None)`
Clean API data without ML optimizations.

```python
api_response = {"status": "success", "data": [1, 2, 3]}
datason.save_api(api_response, "api_data.json")
```

#### `save_chunked(data, filepath, chunk_size=1000, format=None)`
Save large datasets in chunks for memory efficiency.

```python
large_dataset = [{"record": i} for i in range(100000)]
datason.save_chunked(large_dataset, "large.jsonl", chunk_size=5000)
```

### Load Functions

#### `load_smart_file(filepath, format=None)`
Intelligent loading with automatic type reconstruction.

```python
# Load with smart type detection
data = list(datason.load_smart_file("model.jsonl"))

# Force specific format
data = list(datason.load_smart_file("data.txt", format="jsonl"))
```

#### `load_perfect_file(filepath, template, format=None)`
Perfect type reconstruction using templates.

```python
# Define template for perfect reconstruction
template = {
    "weights": np.array([[0.0]]),           # NumPy array template
    "dataframe": df.iloc[:1],               # DataFrame template  
    "metadata": {}                          # Regular dict
}

# Load with perfect type reconstruction
data = list(datason.load_perfect_file("model.jsonl", template))
```

#### `load_basic_file(filepath, format=None)`
Basic loading without type reconstruction.

```python
# Load as basic Python types only
data = list(datason.load_basic_file("data.jsonl"))
```

### Streaming Functions

#### `stream_save_ml(filepath, format=None)`
Stream data for memory-efficient processing of large datasets.

```python
# Stream training checkpoints
with datason.stream_save_ml("checkpoints.jsonl") as stream:
    for epoch in range(1000):
        checkpoint = {
            "epoch": epoch,
            "weights": model.get_weights(),
            "metrics": get_epoch_metrics(epoch)
        }
        stream.write(checkpoint)
```

## Advanced Features

### Compression

Automatic compression with `.gz` extension:

```python
# These are automatically compressed
datason.save_ml(large_data, "data.jsonl.gz")
datason.save_ml(large_data, "data.json.gz")

# Load compressed files transparently  
data = list(datason.load_smart_file("data.jsonl.gz"))
```

### Format Conversion

Easy conversion between JSON and JSONL:

```python
# Load JSONL and save as JSON
jsonl_data = list(datason.load_smart_file("experiments.jsonl"))
datason.save_ml(jsonl_data, "experiments.json", format="json")

# Load JSON and save as JSONL  
json_data = list(datason.load_smart_file("config.json"))
datason.save_ml(json_data, "config.jsonl", format="jsonl")
```

### Security and Redaction

Automatic PII detection and redaction:

```python
customer_data = {
    "customers": [
        {
            "name": "Alice Johnson",
            "email": "alice@company.com",
            "ssn": "123-45-6789",
            "credit_card": "4532-1234-5678-9012"
        }
    ],
    "secrets": {
        "api_key": "sk-secret123",
        "database_url": "postgresql://user:pass@db/prod"
    }
}

# Comprehensive redaction
datason.save_secure(
    customer_data,
    "customers.jsonl",
    redact_pii=True,                    # Auto-detect SSN, credit cards, etc.
    redact_fields=["api_key", "database_url"]  # Explicit field redaction
)

# Load back with redaction metadata
secure_data = list(datason.load_smart_file("customers.jsonl"))[0]
print(f"Redacted {secure_data['redaction_summary']['total_redactions']} items")
```

## ML Workflows

### Model Training Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# 1. Prepare training data
X, y = make_classification(n_samples=1000, n_features=20)
feature_info = pd.DataFrame({
    "name": [f"feature_{i}" for i in range(20)],
    "importance": np.random.random(20)
})

# 2. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Package everything
ml_package = {
    "model": model,
    "training_data": {"X": X, "y": y, "features": feature_info},
    "metadata": {"accuracy": model.score(X, y), "timestamp": datetime.now()}
}

# 4. Save with compression
datason.save_ml(ml_package, "trained_model.jsonl.gz")
```

### Streaming Training Logs

```python
# Stream training progress
with datason.stream_save_ml("training_log.jsonl") as stream:
    for epoch in range(100):
        # Training step
        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log progress
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_lr()[0],
            "timestamp": datetime.now(),
            "model_weights_sample": model.layers[0].weight.data[:5].numpy()
        }
        stream.write(log_entry)

# Later: analyze training progress
logs = list(datason.load_smart_file("training_log.jsonl"))
train_losses = [log["train_loss"] for log in logs]
```

### Experiment Tracking

```python
# Track multiple experiments
experiments = []

for lr in [0.01, 0.001, 0.0001]:
    for batch_size in [32, 64, 128]:
        # Run experiment
        config = {"lr": lr, "batch_size": batch_size}
        model = train_model(config)
        results = evaluate_model(model, test_data)

        experiments.append({
            "config": config,
            "results": results,
            "model_weights": model.get_weights(),
            "timestamp": datetime.now()
        })

# Save all experiments
datason.save_ml(experiments, "experiments.jsonl.gz")

# Later: find best experiment
all_experiments = list(datason.load_smart_file("experiments.jsonl.gz"))
best = max(all_experiments, key=lambda x: x["results"]["accuracy"])
```

## Data Types Supported

### NumPy Arrays
Perfect preservation of shape, dtype, and data:

```python
data = {
    "weights": np.random.randn(100, 50).astype(np.float32),
    "labels": np.random.randint(0, 10, 1000).astype(np.int64),
    "mask": np.random.choice([True, False], 1000)
}
datason.save_ml(data, "arrays.jsonl")
loaded = list(datason.load_smart_file("arrays.jsonl"))[0]

assert isinstance(loaded["weights"], np.ndarray)
assert loaded["weights"].dtype == np.float32
assert loaded["weights"].shape == (100, 50)
```

### Pandas DataFrames  
Complete preservation including dtypes and index:

```python
df = pd.DataFrame({
    "id": range(1000),
    "timestamp": pd.date_range("2024-01-01", periods=1000, freq="H"),
    "value": np.random.random(1000),
    "category": pd.Categorical(np.random.choice(["A", "B", "C"], 1000))
})

datason.save_ml({"dataframe": df}, "dataframe.jsonl")
loaded = list(datason.load_smart_file("dataframe.jsonl"))[0]

assert isinstance(loaded["dataframe"], pd.DataFrame)
assert len(loaded["dataframe"]) == 1000
assert "timestamp" in loaded["dataframe"].columns
```

### Complex Nested Structures
Handles arbitrarily nested data:

```python
complex_data = {
    "neural_net": {
        "layers": [
            {"type": "dense", "weights": np.random.randn(784, 128)},
            {"type": "dense", "weights": np.random.randn(128, 10)}
        ],
        "optimizer_state": {
            "momentum": [np.random.randn(784, 128), np.random.randn(128, 10)],
            "learning_rate": 0.001
        }
    },
    "training_history": pd.DataFrame({
        "epoch": range(50),
        "loss": np.random.exponential(1, 50)
    })
}
```

## Performance Tips

### When to Use Each Format

| Use Case | Recommended Format | Reason |
|----------|-------------------|---------|
| Model checkpoints | `.jsonl.gz` | Good compression, line-by-line structure |
| Configuration files | `.json` | Single object, human readable |
| Training logs | `.jsonl` | Streamable, appendable |
| Large datasets | `.jsonl.gz` | Memory efficient, compressed |
| API responses | `.json` | Standard format |
| Experiment results | `.jsonl` | One experiment per line |

### Memory Efficiency

For large datasets, use streaming:

```python
# Instead of loading everything into memory
big_data = [huge_record for huge_record in massive_dataset]  # ❌ Memory intensive
datason.save_ml(big_data, "huge.jsonl")

# Use streaming instead
with datason.stream_save_ml("huge.jsonl") as stream:      # ✅ Memory efficient
    for record in massive_dataset:
        stream.write(record)
```

### Compression Benefits

Compression ratios vary by data type:

- **Text data**: 5-10x compression
- **Repeated values**: 20-50x compression  
- **Numerical arrays**: 2-5x compression
- **Mixed data**: 3-8x compression

```python
# Always use compression for large files
datason.save_ml(large_data, "data.jsonl.gz")  # ✅ Compressed
datason.save_ml(large_data, "data.jsonl")     # ❌ Uncompressed
```

## Best Practices

### File Naming Conventions

```python
# Good naming patterns
"model_v1.2.3.jsonl.gz"           # Versioned model
"training_logs_2024-01-15.jsonl"  # Dated logs  
"experiments_lr_search.jsonl"     # Descriptive purpose
"customer_data_secure.jsonl"      # Security indication

# Use consistent extensions
".jsonl" or ".jsonl.gz"           # For line-by-line data
".json" or ".json.gz"             # For single objects
```

### Directory Structure

```
ml_project/
├── models/
│   ├── checkpoints/
│   │   ├── epoch_001.jsonl.gz
│   │   ├── epoch_002.jsonl.gz
│   │   └── final_model.jsonl.gz
│   └── experiments/
│       ├── hyperparameter_search.jsonl
│       └── architecture_comparison.jsonl
├── data/
│   ├── training_data.jsonl.gz
│   ├── validation_data.jsonl.gz
│   └── features.json
└── logs/
    ├── training_2024-01-15.jsonl
    └── evaluation_2024-01-15.jsonl
```

### Error Handling

```python
try:
    data = list(datason.load_smart_file("model.jsonl"))
except FileNotFoundError:
    print("Model file not found")
except json.JSONDecodeError:
    print("Corrupted file")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## API Reference

### Function Signatures

```python
# Save functions
save_ml(data, filepath, format=None)
save_secure(data, filepath, redact_pii=False, redact_fields=None, format=None)  
save_api(data, filepath, format=None)
save_chunked(data, filepath, chunk_size=1000, format=None)

# Load functions  
load_smart_file(filepath, format=None) -> Iterator
load_perfect_file(filepath, template, format=None) -> Iterator
load_basic_file(filepath, format=None) -> Iterator

# Streaming
stream_save_ml(filepath, format=None) -> ContextManager
```

### Parameters

- **`data`**: Data to save (any serializable object)
- **`filepath`**: Path to save/load (str or Path object)
- **`format`**: Force format ("json" or "jsonl", auto-detected if None)
- **`redact_pii`**: Enable automatic PII detection and redaction
- **`redact_fields`**: List of field names to redact
- **`template`**: Template object for perfect type reconstruction
- **`chunk_size`**: Number of records per chunk for memory efficiency

## Integration with Other Features

File operations integrate seamlessly with all datason features:

- **✅ ML Type Handlers**: Automatic preservation of ML objects
- **✅ Security & Redaction**: PII protection and field redaction  
- **✅ Streaming**: Memory-efficient processing
- **✅ Compression**: Automatic `.gz` detection
- **✅ Templates**: Perfect type reconstruction
- **✅ Progressive Complexity**: basic/smart/perfect loading options

This makes file operations a true first-class citizen in the datason ecosystem!
