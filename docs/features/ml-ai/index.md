# ML/AI Integration

Native support for machine learning and scientific computing objects with intelligent handling of tensors, models, and data arrays.

## 🎯 Overview

datason provides deep integration with popular ML/AI frameworks:

- **PyTorch**: Tensors, models, parameters, optimizers
- **TensorFlow**: Tensors, variables, SavedModel metadata  
- **Scikit-learn**: Fitted models, pipelines, transformers
- **NumPy**: Arrays, scalars, dtypes, structured arrays
- **JAX**: Arrays and computation graphs
- **PIL/Pillow**: Images with format preservation
- **Hugging Face**: Tokenizers and model metadata

## 🧠 Supported Libraries

### PyTorch Integration
```python
import torch
import datason

# Tensors
tensor = torch.randn(3, 4)
result = datason.serialize(tensor)
# Output: {"type": "torch.Tensor", "data": [...], "dtype": "float32", "shape": [3, 4]}

# Models (metadata only for security)
model = torch.nn.Linear(10, 5)
result = datason.serialize(model)
# Output: {"type": "torch.nn.Linear", "class": "Linear", "parameters": {...}}

# CUDA tensors
if torch.cuda.is_available():
    cuda_tensor = torch.randn(2, 3, device='cuda')
    result = datason.serialize(cuda_tensor)
    # Automatically moves to CPU for serialization
```

### TensorFlow Integration
```python
import tensorflow as tf
import datason

# TensorFlow tensors
tensor = tf.constant([[1, 2], [3, 4]])
result = datason.serialize(tensor)
# Output: {"type": "tf.Tensor", "data": [[1, 2], [3, 4]], "dtype": "int32", "shape": [2, 2]}

# Variables
variable = tf.Variable([1.0, 2.0, 3.0])
result = datason.serialize(variable)
# Output: {"type": "tf.Variable", "data": [1.0, 2.0, 3.0], "trainable": true}
```

### Scikit-learn Integration
```python
from sklearn.linear_model import LogisticRegression
import datason

# Fitted models
model = LogisticRegression()
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]
model.fit(X, y)

result = datason.serialize(model)
# Output: {"type": "sklearn.LogisticRegression", "params": {...}, "fitted": true}
```

### NumPy Integration
```python
import numpy as np
import datason

# Arrays
array = np.array([[1, 2, 3], [4, 5, 6]])
result = datason.serialize(array)
# Output: {"type": "numpy.ndarray", "data": [[1, 2, 3], [4, 5, 6]], "dtype": "int64", "shape": [2, 3]}

# Special values
special_array = np.array([np.inf, -np.inf, np.nan])
result = datason.serialize(special_array)
# Handles infinity and NaN values safely

# Structured arrays
dtype = [('name', 'U10'), ('age', 'i4')]
structured = np.array([('Alice', 25), ('Bob', 30)], dtype=dtype)
result = datason.serialize(structured)
# Preserves structure information
```

### PIL/Pillow Integration
```python
from PIL import Image
import datason

# Images
image = Image.new('RGB', (100, 100), color='red')
result = datason.serialize(image)
# Output: {"type": "PIL.Image", "mode": "RGB", "size": [100, 100], "data": "base64..."}
```

## 🔧 Configuration for ML Workflows

### ML-Optimized Configuration
```python
import datason

# Use ML-optimized settings
config = datason.get_ml_config()
result = datason.serialize(ml_data, config=config)

# Features:
# - Optimized for numeric data
# - Efficient handling of large arrays
# - Preservation of ML-specific metadata
```

### Custom ML Configuration
```python
from datason import SerializationConfig, NanHandling

config = SerializationConfig(
    # Handle NaN values appropriately for ML
    nan_handling=NanHandling.PRESERVE,

    # Optimize for numeric data
    preserve_numeric_precision=True,

    # Custom serializers for specific model types
    custom_serializers={
        MyCustomModel: serialize_custom_model
    }
)
```

## 📊 Data Science Workflows

### Complete ML Pipeline Serialization
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datason

# Complete workflow
ml_workflow = {
    "data": {
        "features": np.random.randn(1000, 10),
        "labels": np.random.randint(0, 2, 1000),
        "feature_names": [f"feature_{i}" for i in range(10)]
    },
    "preprocessing": {
        "scaler_params": {"mean": np.array([0.1, 0.2]), "std": np.array([1.0, 1.1])},
        "selected_features": [0, 1, 3, 7, 9]
    },
    "model": LogisticRegression().fit(np.random.randn(100, 5), np.random.randint(0, 2, 100)),
    "evaluation": {
        "accuracy": 0.85,
        "confusion_matrix": np.array([[45, 5], [8, 42]]),
        "feature_importance": np.array([0.2, 0.15, 0.3, 0.1, 0.25])
    },
    "metadata": {
        "created_at": pd.Timestamp.now(),
        "version": "1.0.0",
        "framework_versions": {
            "sklearn": "1.3.0",
            "numpy": "1.24.0"
        }
    }
}

# Serialize entire workflow
result = datason.serialize(ml_workflow)
# Everything is preserved with appropriate type information
```

### Model Comparison and Tracking
```python
# Model comparison data
model_comparison = {
    "experiments": [
        {
            "id": "exp_001",
            "model": LogisticRegression(C=1.0),
            "hyperparams": {"C": 1.0, "solver": "lbfgs"},
            "metrics": {"accuracy": 0.85, "f1": 0.82},
            "predictions": np.array([0, 1, 1, 0, 1]),
            "feature_importance": np.array([0.3, 0.2, 0.5])
        },
        {
            "id": "exp_002",
            "model": LogisticRegression(C=0.1),
            "hyperparams": {"C": 0.1, "solver": "lbfgs"},
            "metrics": {"accuracy": 0.83, "f1": 0.80},
            "predictions": np.array([0, 1, 0, 0, 1]),
            "feature_importance": np.array([0.25, 0.25, 0.5])
        }
    ],
    "best_model_id": "exp_001",
    "dataset_info": {
        "n_samples": 1000,
        "n_features": 10,
        "class_distribution": np.array([600, 400])
    }
}

result = datason.serialize(model_comparison)
```

## 🔒 Security Considerations

### Safe ML Object Handling
```python
# datason handles ML objects safely by default
malicious_model = SomeMLModel()  # Potentially unsafe
result = datason.serialize(malicious_model)
# Only serializes safe metadata, not executable code

# For custom models, use whitelisted serialization
config = datason.SerializationConfig(
    safe_ml_serialization=True,  # Default: True
    allowed_ml_types={
        'sklearn.linear_model.LogisticRegression',
        'torch.nn.Linear',
        'tf.keras.layers.Dense'
    }
)
```

### Tensor Size Limits
```python
# Automatic size validation
large_tensor = torch.randn(10000, 10000)  # Very large
result = datason.serialize(large_tensor)
# Warning: Large tensor detected, consider chunking or compression
```

## ⚡ Performance Optimization

### ML Performance Tips
```python
# For large arrays, use performance config
config = datason.get_performance_config()

# Batch processing for multiple tensors
tensors = [torch.randn(100, 100) for _ in range(10)]
results = [datason.serialize(t, config=config) for t in tensors]

# Memory-efficient streaming for very large data
def serialize_large_dataset(data_generator):
    for batch in data_generator:
        yield datason.serialize(batch, config=config)
```

### Framework-Specific Optimizations
```python
# PyTorch optimization
config = SerializationConfig(
    torch_precision='float32',  # Reduce precision if acceptable
    torch_device='cpu',  # Ensure CPU serialization
    preserve_gradients=False  # Skip gradient information
)

# NumPy optimization
config = SerializationConfig(
    numpy_array_format='list',  # or 'base64' for binary data
    preserve_array_flags=False,  # Skip metadata for performance
    compress_arrays=True  # Enable compression for large arrays
)
```

## 🧪 Testing ML Serialization

### Round-Trip Testing
```python
import torch
import datason

# Original tensor
original = torch.randn(3, 4)

# Serialize and deserialize
serialized = datason.serialize(original)
deserialized = datason.deserialize(serialized)

# Verify preservation
assert torch.allclose(original, deserialized)
assert original.shape == deserialized.shape
assert original.dtype == deserialized.dtype
```

### Model State Preservation
```python
from sklearn.linear_model import LogisticRegression

# Train model
model = LogisticRegression()
X, y = [[1, 2], [3, 4]], [0, 1]
model.fit(X, y)

# Get predictions before serialization
original_pred = model.predict([[2, 3]])

# Serialize and deserialize model
serialized = datason.serialize(model)
restored_model = datason.deserialize(serialized)

# Verify model behavior is preserved
restored_pred = restored_model.predict([[2, 3]])
assert np.array_equal(original_pred, restored_pred)
```

## 🔗 Integration Examples

### Jupyter Notebook Integration
```python
# Save notebook checkpoint with all variables
import datason
import dill

# Serialize entire workspace
workspace = {
    'dataframes': {name: obj for name, obj in locals().items()
                  if isinstance(obj, pd.DataFrame)},
    'models': {name: obj for name, obj in locals().items()
              if hasattr(obj, 'fit') and hasattr(obj, 'predict')},
    'arrays': {name: obj for name, obj in locals().items()
              if isinstance(obj, np.ndarray)}
}

checkpoint = datason.serialize(workspace)
# Save checkpoint to file for later restoration
```

### MLflow Integration
```python
import mlflow
import datason

# Log model with metadata
with mlflow.start_run():
    # Train model
    model = train_model(X, y)

    # Serialize model and metadata
    model_data = datason.serialize({
        'model': model,
        'hyperparams': {'C': 1.0, 'solver': 'lbfgs'},
        'feature_names': feature_names,
        'preprocessing_steps': preprocessing_pipeline
    })

    # Log to MLflow
    mlflow.log_dict(model_data, "model_metadata.json")
```

## 🚀 Best Practices

### 1. Choose Appropriate Serialization Depth
```python
# For model checkpointing (deep serialization)
checkpoint_data = datason.serialize(model, deep=True)

# For model metadata only (shallow serialization)
metadata = datason.serialize(model, deep=False)
```

### 2. Handle Different Tensor Libraries Consistently
```python
def normalize_tensor(tensor):
    """Convert any tensor to a standard format."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return tensor

# Apply normalization before serialization for consistency
```

### 3. Version Compatibility
```python
# Include version information for compatibility tracking
ml_package = {
    'model': model,
    'framework_versions': {
        'torch': torch.__version__,
        'sklearn': sklearn.__version__,
        'numpy': np.__version__
    },
    'datason_version': datason.__version__,
    'serialization_date': pd.Timestamp.now()
}
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - ML-optimized configurations
- **[Pickle Bridge](../pickle-bridge/index.md)** - Migrate legacy ML pickle files
- **[Performance](../performance/index.md)** - Optimize for large datasets
- **[Advanced Types](../advanced-types/index.md)** - Complex data type handling

## 🚀 Next Steps

- **[Configuration →](../configuration/index.md)** - ML-specific configuration options
- **[Pickle Bridge →](../pickle-bridge/index.md)** - Convert legacy ML models
- **[Performance →](../performance/index.md)** - Optimize for production ML workflows
