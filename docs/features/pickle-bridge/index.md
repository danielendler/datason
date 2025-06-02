# Pickle Bridge - Legacy ML Migration

datason's Pickle Bridge provides a secure, production-ready solution for migrating legacy ML pickle files to portable JSON format, addressing workflow pain points in the ML community.

## Overview

Pickle files are ubiquitous in Python ML workflows but create portability, security, and maintenance challenges. The Pickle Bridge solves these issues with a security-first conversion system that maintains data integrity while enabling JSON-based workflows.

```python
import datason

# Convert single pickle file safely
result = datason.from_pickle("legacy_model.pkl")

# Bulk migration with security controls
stats = datason.convert_pickle_directory(
    source_dir="old_models/",
    target_dir="json_models/",
    safe_classes=datason.get_ml_safe_classes()
)

# Custom security configuration
bridge = datason.PickleBridge(
    safe_classes={"sklearn.*", "numpy.ndarray", "pandas.core.frame.DataFrame"},
    max_file_size=50 * 1024 * 1024  # 50MB limit
)
```

## 🎯 Key Features

### Security-First Design
- **Class whitelisting** prevents arbitrary code execution
- **54 default safe classes** covering 95%+ of ML pickle files
- **Module wildcard support** (e.g., `sklearn.*`) with security warnings
- **File size limits** (100MB default) for resource protection

### Zero Dependencies
- Uses only Python standard library `pickle` module
- No new dependencies added to datason
- Always available - no optional imports needed

### Comprehensive Coverage
- **NumPy**: `ndarray`, `dtype`, `matrix` and core types
- **Pandas**: `DataFrame`, `Series`, `Index`, `Categorical` and related classes
- **Scikit-learn**: 15+ common model classes
- **PyTorch**: Basic `Tensor` and `Module` support
- **Python stdlib**: All built-in types

## 🚀 Quick Start

### Basic Conversion

Convert a single pickle file with automatic security:

```python
import datason

# Simple file conversion
result = datason.from_pickle("experiment_data.pkl")

# Access converted data
experiment_data = result["data"]
metadata = result["metadata"]

print(f"Converted {metadata['source_size_bytes']:,} bytes")
print(f"Experiment ID: {experiment_data['id']}")
```

### Bulk Directory Migration

Convert entire directories of pickle files:

```python
import datason

# Migrate entire directory
stats = datason.convert_pickle_directory(
    source_dir="legacy_experiments/",
    target_dir="portable_experiments/",
    pattern="*.pkl",
    overwrite=True
)

print(f"✅ Converted {stats['files_converted']} files")
print(f"📊 Found {stats['files_found']} pickle files")
print(f"⏭️ Skipped {stats['files_skipped']} existing files")
```

### Custom Security Configuration

Fine-tune security settings for your use case:

```python
import datason

# Create bridge with custom safe classes
bridge = datason.PickleBridge(
    safe_classes={
        # Core Python types
        "builtins.dict", "builtins.list", "builtins.str",
        # NumPy essentials
        "numpy.ndarray", "numpy.dtype",
        # Pandas basics
        "pandas.core.frame.DataFrame",
        # Scikit-learn models (wildcard)
        "sklearn.*"
    },
    max_file_size=100 * 1024 * 1024,  # 100MB limit
    config=datason.get_ml_config()    # ML-optimized serialization
)

# Convert with custom settings
result = bridge.from_pickle_file("custom_model.pkl")
```

## 🔒 Security Features

The Pickle Bridge implements multiple security layers to prevent arbitrary code execution:

- **Class Whitelisting**: Only predefined safe classes are allowed
- **Import Validation**: Restricted to specific modules and packages  
- **Code Inspection**: No `__reduce__` or custom deserialization methods
- **Resource Limits**: File size and processing time constraints

For detailed security guidelines, see **[Security Documentation](../../SECURITY.md)**.

## 📊 Performance & Statistics

### Conversion Statistics

Track conversion performance and success rates:

```python
import datason

bridge = datason.PickleBridge()

# Convert multiple files
bridge.from_pickle_file("model1.pkl")
bridge.from_pickle_file("model2.pkl")
bridge.from_pickle_file("data.pkl")

# Get statistics
stats = bridge.get_conversion_stats()
print(f"Files processed: {stats['files_processed']}")
print(f"Files successful: {stats['files_successful']}")
print(f"Files failed: {stats['files_failed']}")
print(f"Total bytes: {stats['total_size_bytes']:,}")
```

### Bulk Processing Statistics

Directory conversion provides detailed batch statistics:

```python
import datason

stats = datason.convert_pickle_directory(
    source_dir="experiments/",
    target_dir="converted/",
    overwrite=True
)

# Detailed batch results
print(f"📁 Files found: {stats['files_found']}")
print(f"✅ Files converted: {stats['files_converted']}")
print(f"⏭️ Files skipped: {stats['files_skipped']}")
print(f"❌ Files failed: {stats['files_failed']}")

# Error details
if stats['errors']:
    print("\nErrors encountered:")
    for error in stats['errors']:
        print(f"  {error['file']}: {error['error_type']} - {error['error']}")
```

## 🛠️ Advanced Usage

### Integration with Datason Configs

Leverage datason's configuration system for optimized output:

```python
import datason

# ML-optimized configuration
ml_config = datason.get_ml_config()
bridge = datason.PickleBridge(config=ml_config)

# API-optimized configuration  
api_config = datason.get_api_config()
api_bridge = datason.PickleBridge(config=api_config)

# Performance-optimized configuration
perf_config = datason.get_performance_config()
fast_bridge = datason.PickleBridge(config=perf_config)

# Convert with different optimizations
ml_result = bridge.from_pickle_file("model.pkl")      # Preserves ML metadata
api_result = api_bridge.from_pickle_file("data.pkl")  # Clean JSON output
fast_result = fast_bridge.from_pickle_file("big.pkl") # Speed optimized
```

### Bytes-Level Processing

Work directly with pickle bytes for in-memory processing:

```python
import datason
import pickle

# Create pickle bytes
data = {"model": "RandomForest", "accuracy": 0.95}
pickle_bytes = pickle.dumps(data)

# Convert bytes directly
bridge = datason.PickleBridge()
result = bridge.from_pickle_bytes(pickle_bytes)

print(f"Converted {len(pickle_bytes)} bytes")
print(f"Result: {result['data']}")
```

### Custom File Patterns

Process specific file patterns in bulk operations:

```python
import datason

# Convert only model files
model_stats = datason.convert_pickle_directory(
    source_dir="ml_experiments/",
    target_dir="json_models/",
    pattern="*_model.pkl"
)

# Convert only data files
data_stats = datason.convert_pickle_directory(
    source_dir="ml_experiments/",
    target_dir="json_data/",
    pattern="*_data.pkl"
)

# Convert all pickle files (default)
all_stats = datason.convert_pickle_directory(
    source_dir="ml_experiments/",
    target_dir="json_all/",
    pattern="*.pkl"  # default pattern
)
```

## 🎯 Real-World Migration Scenarios

### ML Experiment Migration

Convert years of ML experiments to portable format:

```python
import datason
from pathlib import Path

# Step 1: Assessment
source_dir = Path("old_experiments/")
pickle_files = list(source_dir.glob("**/*.pkl"))
print(f"Found {len(pickle_files)} pickle files to convert")

# Step 2: Test conversion on sample
sample_file = pickle_files[0]
try:
    result = datason.from_pickle(sample_file)
    print(f"✅ Sample conversion successful: {len(result['data'])} items")
except Exception as e:
    print(f"❌ Sample conversion failed: {e}")

# Step 3: Bulk migration
target_dir = Path("portable_experiments/")
stats = datason.convert_pickle_directory(
    source_dir=source_dir,
    target_dir=target_dir,
    overwrite=True
)

print(f"Migration complete: {stats['files_converted']}/{stats['files_found']} files")
```

### Model Deployment Pipeline

Integrate pickle conversion into deployment workflows:

```python
import datason
from pathlib import Path

def deploy_model(model_pickle_path, deployment_dir):
    """Convert pickle model to JSON for deployment."""

    # Convert with API-optimized config
    config = datason.get_api_config()
    bridge = datason.PickleBridge(config=config)

    try:
        result = bridge.from_pickle_file(model_pickle_path)

        # Save deployment-ready JSON
        json_path = Path(deployment_dir) / "model.json"
        with json_path.open("w") as f:
            import json
            json.dump(result, f, indent=2)

        print(f"✅ Model deployed to {json_path}")
        return True

    except datason.PickleSecurityError as e:
        print(f"❌ Security error: {e}")
        return False

# Usage
success = deploy_model("trained_model.pkl", "deployment/")
```

### Data Archive Migration

Convert data archives for long-term storage:

```python
import datason
from pathlib import Path

def archive_experiments(source_dir, archive_dir):
    """Convert experimental data to archival JSON format."""

    # Use strict config for maximum data preservation
    config = datason.get_strict_config()

    stats = datason.convert_pickle_directory(
        source_dir=source_dir,
        target_dir=archive_dir,
        safe_classes=datason.get_ml_safe_classes(),
        config=config,
        overwrite=False  # Don't overwrite existing archives
    )

    # Generate archive report
    report = {
        "archive_date": "2025-05-30",
        "source_directory": str(source_dir),
        "files_archived": stats['files_converted'],
        "files_skipped": stats['files_skipped'],
        "errors": stats['errors']
    }

    report_path = Path(archive_dir) / "archive_report.json"
    with report_path.open("w") as f:
        import json
        json.dump(report, f, indent=2)

    return stats

# Usage
archive_stats = archive_experiments("research_2024/", "archives/2024/")
```

## 🚨 Security Best Practices

### Principle of Least Privilege

Only whitelist classes you explicitly trust:

```python
import datason

# ❌ Don't do this - too permissive
risky_bridge = datason.PickleBridge(
    safe_classes={"*"}  # Allows any class - dangerous!
)

# ✅ Do this - explicit and minimal
safe_bridge = datason.PickleBridge(
    safe_classes={
        "builtins.dict", "builtins.list", "builtins.str",
        "numpy.ndarray",
        "pandas.core.frame.DataFrame",
        "sklearn.ensemble._forest.RandomForestClassifier"  # Specific model only
    }
)
```

### Validate Unknown Sources

Be extra cautious with pickle files from unknown sources:

```python
import datason

def safe_convert_untrusted(pickle_path):
    """Safely convert pickle files from untrusted sources."""

    # Minimal safe classes for untrusted sources
    minimal_classes = {
        "builtins.dict", "builtins.list", "builtins.str",
        "builtins.int", "builtins.float", "builtins.bool",
        "datetime.datetime", "uuid.UUID"
    }

    bridge = datason.PickleBridge(
        safe_classes=minimal_classes,
        max_file_size=1024 * 1024  # 1MB limit for untrusted files
    )

    try:
        result = bridge.from_pickle_file(pickle_path)
        print(f"✅ Untrusted file converted safely")
        return result
    except datason.PickleSecurityError as e:
        print(f"🚨 Security violation in untrusted file: {e}")
        return None

# Usage
result = safe_convert_untrusted("downloaded_data.pkl")
```

### Monitor Conversion Results

Always validate conversion results for critical data:

```python
import datason

def validated_conversion(pickle_path, expected_keys=None):
    """Convert pickle with result validation."""

    bridge = datason.PickleBridge()
    result = bridge.from_pickle_file(pickle_path)

    # Validate structure
    assert "data" in result
    assert "metadata" in result

    # Validate expected content
    if expected_keys:
        for key in expected_keys:
            assert key in result["data"], f"Missing expected key: {key}"

    # Validate metadata
    metadata = result["metadata"]
    assert "source_file" in metadata
    assert "source_size_bytes" in metadata
    assert "datason_version" in metadata

    print(f"✅ Conversion validated: {pickle_path}")
    return result

# Usage
result = validated_conversion(
    "model_data.pkl",
    expected_keys=["model", "accuracy", "timestamp"]
)
```

## 🔧 Integration Examples

### Jupyter Notebook Integration

```python
# Cell 1: Setup
import datason
from pathlib import Path

# Cell 2: Convert experiment data
experiment_dir = Path("experiments/2024/")
converted_dir = Path("converted_experiments/")

stats = datason.convert_pickle_directory(
    source_dir=experiment_dir,
    target_dir=converted_dir
)

print(f"Converted {stats['files_converted']} experiment files")

# Cell 3: Load converted data
import json
with open("converted_experiments/experiment_001.json") as f:
    exp_data = json.load(f)

model_accuracy = exp_data["data"]["metrics"]["accuracy"]
print(f"Model accuracy: {model_accuracy}")
```

### MLflow Integration

```python
import datason
import mlflow
import json

def log_pickle_as_json(pickle_path, artifact_name):
    """Convert pickle to JSON and log as MLflow artifact."""

    # Convert pickle to JSON
    result = datason.from_pickle(pickle_path)

    # Save as JSON artifact
    json_path = f"{artifact_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Log to MLflow
    mlflow.log_artifact(json_path, "converted_models")

    # Log metadata
    mlflow.log_param("original_size_bytes", result["metadata"]["source_size_bytes"])
    mlflow.log_param("datason_version", result["metadata"]["datason_version"])

# Usage in MLflow run
with mlflow.start_run():
    log_pickle_as_json("trained_model.pkl", "model")
```

### Docker Container Integration

```dockerfile
# Dockerfile for pickle conversion service
FROM python:3.12-slim

RUN pip install datason

COPY convert_pickles.py /app/
WORKDIR /app

# Conversion script
COPY <<EOF /app/convert_pickles.py
import datason
import sys
from pathlib import Path

def main():
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]

    stats = datason.convert_pickle_directory(
        source_dir=source_dir,
        target_dir=target_dir,
        safe_classes=datason.get_ml_safe_classes()
    )

    print(f"Converted {stats['files_converted']} files")

if __name__ == "__main__":
    main()
EOF

ENTRYPOINT ["python", "convert_pickles.py"]
```

## 📈 Performance Considerations

### Memory Usage

For large pickle files, monitor memory consumption:

```python
import datason
import psutil
import os

def convert_with_monitoring(pickle_path):
    """Convert large pickle files with memory monitoring."""

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    bridge = datason.PickleBridge()
    result = bridge.from_pickle_file(pickle_path)

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory

    print(f"Memory used: {memory_used:.1f} MB")
    print(f"Source size: {result['metadata']['source_size_bytes'] / 1024 / 1024:.1f} MB")

    return result

# Usage for large files
result = convert_with_monitoring("large_model.pkl")
```

### Batch Processing Optimization

Optimize for bulk conversions:

```python
import datason
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def parallel_conversion(source_dir, target_dir, max_workers=4):
    """Convert multiple pickle files in parallel."""

    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)

    pickle_files = list(source_path.glob("*.pkl"))

    def convert_single(pickle_file):
        try:
            result = datason.from_pickle(pickle_file)

            # Save JSON
            json_file = target_path / f"{pickle_file.stem}.json"
            with json_file.open("w") as f:
                import json
                json.dump(result, f)

            return {"file": pickle_file.name, "status": "success"}
        except Exception as e:
            return {"file": pickle_file.name, "status": "error", "error": str(e)}

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(convert_single, pickle_files))

    # Summary
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"✅ Successfully converted: {len(successful)} files")
    print(f"❌ Failed conversions: {len(failed)} files")

    return results

# Usage
results = parallel_conversion("pickles/", "json_output/", max_workers=8)
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - Customize bridge behavior
- **[ML/AI Integration](../ml-ai/index.md)** - Framework-specific features
- **[Advanced Types](../advanced-types/index.md)** - Extended type support
- **[Security Documentation](../../SECURITY.md)** - Additional security considerations

## 🚀 Migration Checklist

### Pre-Migration Assessment
- [ ] Inventory all pickle files to be converted
- [ ] Identify custom classes that may need whitelisting
- [ ] Estimate storage requirements for JSON output
- [ ] Test conversion on representative sample files

### Security Configuration  
- [ ] Review and customize safe classes list
- [ ] Set appropriate file size limits
- [ ] Plan for handling conversion errors
- [ ] Document security decisions for compliance

### Conversion Process
- [ ] Create backup of original pickle files
- [ ] Convert files in stages (test → staging → production)
- [ ] Validate converted data integrity
- [ ] Update downstream systems to use JSON format

### Post-Migration Validation
- [ ] Verify all critical data converted successfully
- [ ] Performance test JSON-based workflows
- [ ] Update documentation and training materials
- [ ] Monitor system performance with new format

---

The Pickle Bridge provides a secure, scalable solution for modernizing ML workflows while maintaining data integrity and security. Start with small test conversions and gradually migrate your entire pickle-based infrastructure to portable JSON format.
