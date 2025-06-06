# 🤖 AI Integration Overview

This guide explains how AI systems and automated workflows can effectively leverage datason for intelligent serialization, data processing, and ML pipeline integration.

## Why datason for AI Systems?

AI systems often deal with complex, heterogeneous data that traditional JSON serialization can't handle. datason solves this by:

- **Automatic Type Detection**: AI systems can serialize any data without manual type handling
- **ML Library Integration**: Native support for PyTorch, TensorFlow, scikit-learn, Hugging Face
- **Schema Inference**: Automatically understand and enforce data structures
- **Streaming & Chunking**: Handle large datasets efficiently in production
- **Privacy & Compliance**: Built-in redaction for sensitive data

## Architecture Integration Patterns

### 1. **Microservices Communication**

```python
import datason as ds

# Service A: ML Model Inference
def ml_inference_service(input_data):
    model_results = {
        "predictions": torch.tensor([0.8, 0.2, 0.7]),
        "features": pd.DataFrame({"feature1": [1, 2, 3]}),
        "metadata": {
            "model_version": "v2.1.0",
            "timestamp": datetime.now(),
            "confidence_scores": np.array([0.95, 0.85, 0.90])
        }
    }

    # Serialize for API response
    config = ds.get_ml_config()
    return ds.serialize(model_results, config=config)

# Service B: Data Processing
def process_ml_results(serialized_data):
    # Deserialize with type preservation
    results = ds.deserialize(serialized_data)

    # Types are automatically restored
    predictions = results["predictions"]  # torch.Tensor
    features = results["features"]        # pd.DataFrame
    metadata = results["metadata"]        # dict with datetime, etc.

    return process_data(predictions, features, metadata)
```

### 2. **ML Pipeline Orchestration**

```python
# Pipeline step configuration
class MLPipelineStep:
    def __init__(self, name: str):
        self.name = name
        self.config = ds.get_ml_config()

    def execute(self, input_data: Any) -> Any:
        # Process data
        result = self.process(input_data)

        # Serialize for next step
        return ds.serialize(result, config=self.config)

    def process(self, data: Any) -> Any:
        # Deserialize input with type preservation
        processed_data = ds.deserialize(data) if isinstance(data, dict) else data

        # Your processing logic here
        return processed_data

# Pipeline orchestrator
class MLPipeline:
    def __init__(self, steps: List[MLPipelineStep]):
        self.steps = steps

    def run(self, initial_data: Any) -> Any:
        current_data = initial_data

        for step in self.steps:
            current_data = step.execute(current_data)

        return ds.deserialize(current_data)  # Final result with proper types
```

### 3. **Real-time Data Streaming**

```python
import asyncio
from datason import StreamingSerializer

async def ml_data_stream_processor():
    """Process ML data in real-time with memory efficiency."""

    # Configure for streaming
    config = ds.get_performance_config()
    streaming_serializer = StreamingSerializer(config=config)

    async for data_batch in data_stream():
        # Handle mixed ML data types
        batch_data = {
            "images": torch.stack(data_batch["images"]),
            "features": pd.concat(data_batch["features"]),
            "metadata": data_batch["metadata"],
            "timestamp": datetime.now()
        }

        # Stream serialize for efficient processing
        async for chunk in streaming_serializer.serialize_async(batch_data):
            await send_to_downstream_service(chunk)
```

## Configuration for AI Systems

### Preset Configurations

datason provides several AI-optimized configurations:

```python
# Machine Learning workflows
ml_config = ds.get_ml_config()
# - Optimized for ML data types
# - Preserves tensor shapes and dtypes
# - Handles large arrays efficiently

# API endpoints
api_config = ds.get_api_config()  
# - Fast serialization
# - Compact output
# - Web-safe formatting

# Research & experimentation
research_config = ds.get_research_config()
# - Preserves maximum information
# - Detailed metadata
# - Reproducibility features

# Production inference
inference_config = ds.get_inference_config()
# - Speed optimized
# - Memory efficient
# - Minimal overhead
```

### Custom AI Configuration

```python
from datason import SerializationConfig, DateFormat, NanHandling

# AI system configuration
ai_config = SerializationConfig(
    # Performance optimizations
    sort_keys=False,           # Preserve order for performance
    ensure_ascii=False,        # Allow Unicode for international data

    # ML-specific settings
    preserve_numpy_types=True, # Keep NumPy dtypes
    include_metadata=True,     # Track data lineage

    # Handle edge cases common in ML
    nan_handling=NanHandling.NULL,  # Convert NaN to null
    date_format=DateFormat.UNIX,    # Unix timestamps for performance

    # Large data handling
    chunk_size=10 * 1024 * 1024,    # 10MB chunks
    enable_streaming=True,           # Memory-efficient processing
)
```

## Automatic Schema Inference

AI systems can automatically understand data structures:

```python
# Infer schema from sample data
sample_ml_data = {
    "model_output": torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
    "features": pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}),
    "metadata": {"model_id": "bert-base", "version": "1.0"}
}

# Create template for consistent structure
template = ds.infer_template_from_data(sample_ml_data)

# Use template to validate and deserialize new data
def process_ml_request(request_data: dict) -> dict:
    try:
        # Enforce schema consistency
        validated_data = ds.deserialize_with_template(request_data, template)

        # Process with guaranteed structure
        return ml_inference(validated_data)

    except ds.TemplateDeserializationError as e:
        return {"error": f"Invalid data structure: {e}"}
```

## Large-Scale Data Processing

### Chunked Processing

```python
import datason as ds

def process_large_dataset(dataset_path: str):
    """Process large ML datasets efficiently."""

    # Configure for large data
    config = ds.get_performance_config()

    # Chunk the data for memory efficiency
    chunked_result = ds.serialize_chunked(
        large_dataset,
        chunk_size=50 * 1024 * 1024,  # 50MB chunks
        config=config
    )

    # Process chunks individually
    processed_chunks = []
    for chunk_info in chunked_result.chunks:
        chunk_data = ds.deserialize(chunk_info["data"])
        processed_chunk = ml_processing_step(chunk_data)
        processed_chunks.append(processed_chunk)

    return combine_results(processed_chunks)
```

### Memory Usage Estimation

```python
# Estimate memory usage before processing
estimated_memory = ds.estimate_memory_usage(ml_data)

if estimated_memory > MAX_MEMORY_THRESHOLD:
    # Use chunked processing
    result = ds.serialize_chunked(ml_data)
else:
    # Process normally
    result = ds.serialize(ml_data)
```

## Privacy & Compliance for AI

AI systems often need to handle sensitive data responsibly:

```python
# Configure privacy-preserving AI pipeline
privacy_config = {
    "redaction_engine": ds.create_healthcare_redaction_engine(),
    "audit_trail": True,
    "data_retention_days": 90
}

def privacy_aware_ml_pipeline(raw_data: dict) -> dict:
    # Redact sensitive information
    redacted_data = privacy_config["redaction_engine"].process_object(raw_data)

    # Process ML inference on redacted data
    ml_results = run_ml_inference(redacted_data)

    # Serialize results with audit trail
    config = ds.get_api_config()
    serialized = ds.serialize(ml_results, config=config)

    # Log for compliance
    audit_trail = privacy_config["redaction_engine"].get_audit_trail()
    log_privacy_action(audit_trail)

    return serialized
```

## Error Handling & Monitoring

### Robust Error Handling

```python
async def robust_ai_serialization(data: Any) -> dict:
    """AI-focused error handling for serialization."""

    try:
        # Attempt ML-optimized serialization
        config = ds.get_ml_config()
        result = ds.serialize(data, config=config)

        return {"status": "success", "data": result}

    except ds.SecurityError as e:
        # Handle security violations
        log_security_event(str(e))
        return {"status": "error", "type": "security", "message": str(e)}

    except MemoryError as e:
        # Handle large data with chunking
        log_performance_event("memory_fallback", str(e))
        chunked_result = ds.serialize_chunked(data)
        return {"status": "success", "data": chunked_result, "chunked": True}

    except Exception as e:
        # Generic fallback
        log_error("serialization_error", str(e))
        return {"status": "error", "type": "serialization", "message": str(e)}
```

### Performance Monitoring

```python
import time
from typing import Dict, Any

class AISerializationMonitor:
    def __init__(self):
        self.metrics = {"total_calls": 0, "total_time": 0, "errors": 0}

    def serialize_with_monitoring(self, data: Any, config: Any = None) -> Dict[str, Any]:
        start_time = time.time()
        self.metrics["total_calls"] += 1

        try:
            result = ds.serialize(data, config=config)

            # Track performance
            duration = time.time() - start_time
            self.metrics["total_time"] += duration

            # Log slow operations
            if duration > 1.0:  # > 1 second
                log_slow_operation("serialization", duration, len(str(data)))

            return {"status": "success", "data": result, "duration": duration}

        except Exception as e:
            self.metrics["errors"] += 1
            log_error("serialization_error", str(e))
            return {"status": "error", "error": str(e)}

    def get_performance_stats(self) -> Dict[str, float]:
        total_calls = self.metrics["total_calls"]
        if total_calls == 0:
            return {"avg_duration": 0, "error_rate": 0}

        return {
            "avg_duration": self.metrics["total_time"] / total_calls,
            "error_rate": self.metrics["errors"] / total_calls,
            "total_calls": total_calls
        }

# Usage
monitor = AISerializationMonitor()
```

## Best Practices for AI Integration

### 1. **Choose the Right Configuration**

```python
# For training pipelines
training_config = ds.get_ml_config()

# For inference APIs
inference_config = ds.get_api_config()

# For research experiments
research_config = ds.get_research_config()
```

### 2. **Handle Large Models Efficiently**

```python
# For large models (>100MB)
def serialize_large_model(model_data: Any) -> dict:
    memory_estimate = ds.estimate_memory_usage(model_data)

    if memory_estimate > 100 * 1024 * 1024:  # 100MB
        return ds.serialize_chunked(model_data, chunk_size=10 * 1024 * 1024)
    else:
        return ds.serialize(model_data)
```

### 3. **Implement Graceful Degradation**

```python
def ai_serialization_with_fallback(data: Any) -> dict:
    """Try optimal serialization, fall back gracefully."""

    try:
        # Try ML-optimized config
        return ds.serialize(data, config=ds.get_ml_config())
    except (MemoryError, ds.SecurityError):
        try:
            # Fall back to basic config
            return ds.serialize(data, config=ds.get_default_config())
        except Exception:
            # Last resort: safe serialization
            return ds.safe_serialize(data)
```

### 4. **Maintain Data Lineage**

```python
def ml_pipeline_with_lineage(data: Any, step_name: str) -> dict:
    """Track data transformations through pipeline."""

    # Add lineage metadata
    enriched_data = {
        "data": data,
        "lineage": {
            "step": step_name,
            "timestamp": datetime.now(),
            "version": "1.0",
            "data_hash": hash(str(data))  # Simple hash for tracking
        }
    }

    config = ds.get_ml_config()
    return ds.serialize(enriched_data, config=config)
```

## Production Deployment

See the [Build & Publish Guide](../BUILD_PUBLISH.md) for:
- Container deployment strategies
- Performance optimization
- Monitoring and alerting
- Scaling considerations

## Next Steps

- [Feature Overview](../features/index.md) - Detailed feature configurations
- [Core Serialization](../features/core/index.md) - What datason detects automatically
- [AI Usage Guide](../AI_USAGE_GUIDE.md) - Extend for domain-specific types
- [Security Policy](../community/security.md) - Security best practices for AI systems
