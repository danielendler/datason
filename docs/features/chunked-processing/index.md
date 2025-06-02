# Chunked Processing & Streaming (v0.4.0)

The chunked processing and streaming capabilities in datason v0.4.0 enable memory-efficient handling of large datasets that exceed available RAM. This feature set provides tools for processing multi-gigabyte datasets without memory overflow.

## Overview

When working with large datasets, traditional serialization can consume excessive memory. Chunked processing breaks large objects into manageable pieces, allowing you to:

- **Process datasets larger than available RAM**
- **Stream data to/from files without loading everything into memory**
- **Optimize memory usage with automatic recommendations**
- **Maintain performance at scale**

## Key Features

### 1. Chunked Serialization

The `serialize_chunked()` function splits large objects into configurable chunks:

```python
import datason

# Large dataset that might not fit in memory
large_data = list(range(1000000))  # 1 million items

# Serialize in chunks of 10,000 items
result = datason.serialize_chunked(large_data, chunk_size=10000)

print(f"Total chunks: {result.metadata['total_chunks']}")
print(f"Chunking strategy: {result.metadata['chunking_strategy']}")

# Process chunks one at a time (memory efficient)
for chunk in result.chunks:
    # Process each chunk without loading all data at once
    process_chunk(chunk)
```

### 2. Streaming Serialization

The `StreamingSerializer` enables continuous data writing without memory accumulation:

```python
import datason
from pathlib import Path

# Stream large amounts of data to file
with datason.stream_serialize("large_dataset.jsonl") as stream:
    for i in range(1000000):
        item = {"id": i, "data": f"item_{i}"}
        stream.write(item)

# Or stream chunked data
large_dataset = create_large_dataset()
with datason.stream_serialize("chunked_data.jsonl") as stream:
    stream.write_chunked(large_dataset, chunk_size=5000)
```

### 3. Memory Estimation

Get optimization recommendations before processing:

```python
import datason

# Analyze your data for memory optimization
stats = datason.estimate_memory_usage(your_large_dataset)

print(f"Object size: {stats['object_size_mb']:.1f} MB")
print(f"Estimated serialized size: {stats['estimated_serialized_mb']:.1f} MB")
print(f"Recommended chunk size: {stats['recommended_chunk_size']:,}")
print(f"Recommended chunks: {stats['recommended_chunks']}")
```

### 4. Chunked File Deserialization

Efficiently read chunked files back:

```python
import datason

# Process chunked file without loading everything
for chunk in datason.deserialize_chunked_file("large_dataset.jsonl"):
    # Process each chunk as it's loaded
    results = analyze_chunk(chunk)
    save_results(results)

# Apply custom processing to each chunk
def normalize_chunk(chunk):
    return [item.upper() if isinstance(item, str) else item for item in chunk]

processed_chunks = list(datason.deserialize_chunked_file(
    "data.jsonl",
    chunk_processor=normalize_chunk
))
```

## Supported Data Types

Chunked processing works with various data structures:

### Lists and Tuples
```python
large_list = list(range(100000))
result = datason.serialize_chunked(large_list, chunk_size=10000)
# Creates 10 chunks of 10,000 items each
```

### DataFrames
```python
import pandas as pd

df = pd.DataFrame({'id': range(50000), 'value': range(50000)})
result = datason.serialize_chunked(df, chunk_size=5000)
# Creates 10 DataFrame chunks of 5,000 rows each
```

### NumPy Arrays
```python
import numpy as np

arr = np.random.random((100000, 10))
result = datason.serialize_chunked(arr, chunk_size=10000)
# Creates 10 array chunks of 10,000 rows each
```

### Dictionaries
```python
large_dict = {f"key_{i}": f"value_{i}" for i in range(100000)}
result = datason.serialize_chunked(large_dict, chunk_size=10000)
# Creates chunks by grouping key-value pairs
```

## File Formats

Chunked processing supports multiple output formats:

### JSONL (JSON Lines)
```python
# One JSON object per line - ideal for streaming
result.save_to_file("data.jsonl", format="jsonl")

# Stream directly to JSONL
with datason.stream_serialize("stream.jsonl", format="jsonl") as stream:
    for item in data:
        stream.write(item)
```

### JSON Array
```python
# Standard JSON array format
result.save_to_file("data.json", format="json")

# Stream to JSON array
with datason.stream_serialize("stream.json", format="json") as stream:
    for item in data:
        stream.write(item)
```

## Memory Management

### Automatic Memory Limits
```python
# Set memory limit to prevent excessive usage
result = datason.serialize_chunked(
    large_data,
    chunk_size=10000,
    memory_limit_mb=500  # Stop if estimated memory exceeds 500MB
)
```

### Memory Monitoring
```python
# Monitor memory usage during processing
import psutil
import os

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss

result = datason.serialize_chunked(large_data, chunk_size=10000)

for i, chunk in enumerate(result.chunks):
    current_memory = process.memory_info().rss
    memory_increase = (current_memory - initial_memory) / (1024 * 1024)
    print(f"Chunk {i}: Memory increase: {memory_increase:.1f} MB")

    # Process chunk
    process_chunk(chunk)
```

## Performance Optimization

### Choosing Chunk Size

The optimal chunk size depends on your data and system:

```python
# Too small: High overhead
serialize_chunked(data, chunk_size=100)

# Too large: Memory pressure
serialize_chunked(data, chunk_size=100000)

# Optimal: Use memory estimation
stats = datason.estimate_memory_usage(data)
optimal_size = stats['recommended_chunk_size']
serialize_chunked(data, chunk_size=optimal_size)
```

### Configuration Integration

Use with domain-specific configurations:

```python
# For ML workloads
ml_config = datason.get_inference_config()
result = datason.serialize_chunked(
    ml_dataset,
    chunk_size=5000,
    config=ml_config
)

# For financial data
financial_config = datason.get_financial_config()
with datason.stream_serialize("trades.jsonl", config=financial_config) as stream:
    for trade in trade_data:
        stream.write(trade)
```

## Real-World Examples

### Processing Large CSV Files
```python
import pandas as pd
import datason

def process_large_csv(file_path, output_path):
    """Process a large CSV file in chunks."""

    # Read and process in chunks
    chunk_size = 10000

    with datason.stream_serialize(output_path) as stream:
        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
            # Process the chunk
            processed_data = chunk_df.to_dict('records')

            # Stream processed data
            for record in processed_data:
                stream.write(record)

# Usage
process_large_csv("massive_dataset.csv", "processed_data.jsonl")
```

### ML Training Data Pipeline
```python
import datason

def create_training_pipeline(raw_data_path, processed_path):
    """Create ML training pipeline with chunked processing."""

    # Memory estimation
    sample_data = load_sample_data(raw_data_path)
    stats = datason.estimate_memory_usage(sample_data)

    print(f"Recommended chunk size: {stats['recommended_chunk_size']:,}")

    # Stream processing
    with datason.stream_serialize(processed_path) as stream:
        for chunk in datason.deserialize_chunked_file(raw_data_path):
            # Feature engineering on chunk
            processed_chunk = apply_feature_engineering(chunk)

            # Write processed chunk
            stream.write_chunked(processed_chunk, chunk_size=1000)

# Usage
create_training_pipeline("raw_features.jsonl", "training_data.jsonl")
```

### Time Series Data Processing
```python
import datason

def process_time_series(data_source, window_size=1000):
    """Process time series data with temporal chunking."""

    config = datason.get_time_series_config()

    with datason.stream_serialize("time_series.jsonl", config=config) as stream:
        # Process data in temporal windows
        for time_window in get_time_windows(data_source, window_size):
            # Apply temporal aggregations
            aggregated_data = aggregate_time_window(time_window)

            # Stream results
            stream.write(aggregated_data)

# Usage
process_time_series(sensor_data_source, window_size=3600)  # 1-hour windows
```

## Best Practices

### 1. Memory Estimation First
Always analyze your data before processing:
```python
stats = datason.estimate_memory_usage(your_data)
chunk_size = stats['recommended_chunk_size']
```

### 2. Use Appropriate Formats
- **JSONL**: For streaming and line-by-line processing
- **JSON**: For structured data with metadata

### 3. Monitor Memory Usage
```python
# Set reasonable memory limits
result = datason.serialize_chunked(
    data,
    chunk_size=optimal_size,
    memory_limit_mb=1000  # Adjust based on available memory
)
```

### 4. Process Chunks Immediately
```python
# Good: Process chunks immediately (low memory)
for chunk in result.chunks:
    process_and_save(chunk)

# Avoid: Collecting all chunks in memory
all_chunks = result.to_list()  # High memory usage
```

### 5. Use Configuration Profiles
```python
# Choose appropriate configuration for your domain
config = datason.get_performance_config()  # For speed
config = datason.get_research_config()     # For reproducibility
```

## Error Handling

### Memory Limit Exceeded
```python
try:
    result = datason.serialize_chunked(
        large_data,
        chunk_size=1000,
        memory_limit_mb=100
    )
except datason.SecurityError as e:
    print(f"Memory limit exceeded: {e}")
    # Reduce chunk size or increase limit
```

### Streaming Errors
```python
try:
    with datason.stream_serialize("output.jsonl") as stream:
        for item in data:
            stream.write(item)
except IOError as e:
    print(f"File writing error: {e}")
except Exception as e:
    print(f"Serialization error: {e}")
```

## API Reference

### Core Functions

- **`serialize_chunked(obj, chunk_size, config=None, memory_limit_mb=None)`**
  - Serialize large objects in chunks
  - Returns `ChunkedSerializationResult`

- **`stream_serialize(file_path, config=None, format="jsonl", buffer_size=8192)`**
  - Create streaming serializer context manager
  - Returns `StreamingSerializer`

- **`deserialize_chunked_file(file_path, format="jsonl", chunk_processor=None)`**
  - Generator for reading chunked files
  - Yields deserialized chunks

- **`estimate_memory_usage(obj, config=None)`**
  - Analyze object for memory optimization
  - Returns estimation dictionary

### Classes

- **`ChunkedSerializationResult`**
  - Container for chunked data
  - Methods: `to_list()`, `save_to_file()`

- **`StreamingSerializer`**
  - Context manager for streaming serialization
  - Methods: `write()`, `write_chunked()`

## See Also

- [Configuration Guide](../configuration/index.md) - Domain-specific configurations
- [Performance Tuning](../performance/index.md) - Optimization strategies
- [Template Deserialization](../template-deserialization/index.md) - Type-safe deserialization
