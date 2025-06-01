# Performance Optimization

Optimizations for speed and memory efficiency in production environments, with benchmarking tools and configuration strategies.

## 🎯 Overview

datason provides multiple optimization strategies:

- **Early Detection**: Skip processing for JSON-compatible data
- **Memory Streaming**: Handle large datasets without full memory loading
- **Configurable Limits**: Prevent resource exhaustion attacks
- **Built-in Benchmarking**: Performance measurement tools
- **Configuration Presets**: Optimized settings for different use cases

## ⚡ Performance Configurations

### Quick Performance Gains

```python
import datason

# Use performance-optimized configuration
config = datason.get_performance_config()
result = datason.serialize(large_dataset, config=config)

# Features:
# - Unix timestamps (fastest date format)
# - Split orientation for large DataFrames
# - Aggressive type coercion
# - Minimal metadata preservation
```

### Configuration Performance Comparison

Based on real benchmarking results:

| Configuration | Performance (Complex Data) | Use Case |
|--------------|---------------------------|----------|
| **Performance Config** | **0.54ms ± 0.03ms** | Speed-critical applications |
| ML Config | 0.56ms ± 0.08ms | ML pipelines |
| Default | 0.58ms ± 0.01ms | General use |
| API Config | 0.59ms ± 0.08ms | API responses |
| Strict Config | 14.04ms ± 1.67ms | Maximum type preservation |

**Performance Config is up to 25x faster than Strict Config!**

## 📊 Real-World Benchmarks

### Simple Data Performance
**Test**: 1000 JSON-compatible user objects
```python
data = {
    "users": [
        {"id": i, "name": f"user_{i}", "active": True, "score": i * 1.5}
        for i in range(1000)
    ]
}
```

| Library | Performance | Relative Speed |
|---------|-------------|----------------|
| Standard JSON | 0.40ms ± 0.03ms | 1.0x (baseline) |
| **datason** | **0.62ms ± 0.02ms** | **1.53x overhead** |

**Analysis**: Only 53% overhead vs standard JSON for compatible data.

### Complex Data Performance
**Test**: 500 session objects with UUIDs and datetimes
```python
data = {
    "sessions": [
        {
            "id": uuid.uuid4(),
            "start_time": datetime.now(),
            "user_data": {"preferences": [...], "last_login": datetime.now()}
        }
        for i in range(500)
    ]
}
```

| Library | Performance | Notes |
|---------|-------------|-------|
| **datason** | **7.04ms ± 0.21ms** | Only option for this data |
| Pickle | 0.98ms ± 0.50ms | Binary format, Python-only |
| Standard JSON | ❌ **Fails** | Cannot serialize UUIDs/datetime |

### High-Throughput Scenarios

**Large Nested Data**: 100 groups × 50 items (5,000 complex objects)
- **Throughput**: 44,624 items/second
- **Performance**: 112.05ms ± 0.37ms total

**NumPy Arrays**: Multiple arrays with ~23K total elements
- **Throughput**: 908,188 elements/second  
- **Performance**: 25.44ms ± 2.56ms total

**Pandas DataFrames**: 5K total rows
- **Throughput**: 1,082,439 rows/second
- **Performance**: 4.71ms ± 0.75ms total

## 🔧 Optimization Strategies

### 1. Choose the Right Configuration

```python
from datason import get_performance_config, get_ml_config, get_api_config

# Speed-critical applications
config = get_performance_config()
# Features: Unix timestamps, aggressive coercion, minimal metadata

# ML/AI workflows
config = get_ml_config()
# Features: Optimized for numeric data, preserved precision

# API responses
config = get_api_config()
# Features: Consistent output, human-readable dates
```

### 2. DataFrame Orientation Optimization

```python
from datason import SerializationConfig, DataFrameOrient

# Small DataFrames (<1K rows) - use values
config = SerializationConfig(dataframe_orient=DataFrameOrient.VALUES)
# Performance: 0.07ms (fastest for small data)

# Large DataFrames (>1K rows) - use split
config = SerializationConfig(dataframe_orient=DataFrameOrient.SPLIT)
# Performance: 1.63ms (scales best for large data)

# API responses - use records
config = SerializationConfig(dataframe_orient=DataFrameOrient.RECORDS)
# Performance: Intuitive structure, moderate speed
```

### 3. Date Format Optimization

```python
from datason import SerializationConfig, DateFormat

# Fastest: Unix timestamps
config = SerializationConfig(date_format=DateFormat.UNIX)
# Performance: 3.11ms ± 0.06ms

# JavaScript compatibility: Unix milliseconds
config = SerializationConfig(date_format=DateFormat.UNIX_MS)
# Performance: 3.26ms ± 0.05ms

# Human readable: ISO format
config = SerializationConfig(date_format=DateFormat.ISO)
# Performance: 3.46ms ± 0.17ms
```

### 4. NaN Handling Optimization

```python
from datason import SerializationConfig, NanHandling

# Fastest: Convert to NULL
config = SerializationConfig(nan_handling=NanHandling.NULL)
# Performance: 2.83ms ± 0.09ms

# Preserve information: Convert to string
config = SerializationConfig(nan_handling=NanHandling.STRING)
# Performance: 2.89ms ± 0.08ms

# Clean data: Drop values
config = SerializationConfig(nan_handling=NanHandling.DROP)
# Performance: 3.00ms ± 0.08ms
```

## 🚀 Memory Optimization

### Large Dataset Handling

```python
# For very large datasets, use chunking
def serialize_large_dataset(data, chunk_size=1000):
    """Process large datasets in chunks to avoid memory issues."""
    if isinstance(data, list) and len(data) > chunk_size:
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            chunks.append(datason.serialize(chunk))
        return {"chunks": chunks, "total_size": len(data)}
    return datason.serialize(data)
```

### Memory-Efficient Configuration

```python
# Minimize memory usage
config = SerializationConfig(
    # Reduce precision when acceptable
    preserve_float_precision=False,

    # Skip unnecessary metadata
    include_type_info=False,
    include_shape_info=False,

    # Use efficient representations
    dataframe_orient=DataFrameOrient.VALUES,
    date_format=DateFormat.UNIX
)
```

### Streaming for Large Files

```python
import json

def stream_serialize_to_file(data, filename, chunk_size=1000):
    """Stream serialization to file for memory efficiency."""
    with open(filename, 'w') as f:
        if isinstance(data, (list, tuple)) and len(data) > chunk_size:
            f.write('[')
            for i, item in enumerate(data):
                if i > 0:
                    f.write(',')
                serialized_item = datason.serialize(item)
                json.dump(serialized_item, f)
            f.write(']')
        else:
            result = datason.serialize(data)
            json.dump(result, f)
```

## 📈 Monitoring Performance

### Built-in Benchmarking

```python
import time
import datason

# Measure serialization performance
def benchmark_serialization(data, iterations=5):
    """Benchmark datason serialization performance."""
    times = []

    # Warm-up
    datason.serialize(data)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        result = datason.serialize(data)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    ops_per_sec = 1.0 / avg_time if avg_time > 0 else 0

    return {
        'average_time_ms': avg_time * 1000,
        'operations_per_second': ops_per_sec,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000
    }

# Example usage
data = {"users": [{"id": i} for i in range(1000)]}
stats = benchmark_serialization(data)
print(f"Average: {stats['average_time_ms']:.2f}ms")
print(f"Throughput: {stats['operations_per_second']:.0f} ops/sec")
```

### Performance Profiling

```python
import cProfile
import datason

def profile_serialization(data):
    """Profile datason serialization to identify bottlenecks."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = datason.serialize(data)

    profiler.disable()
    profiler.print_stats(sort='cumulative')
    return result

# Example: Profile complex data serialization
complex_data = {
    'dataframes': [pd.DataFrame(np.random.randn(100, 10)) for _ in range(5)],
    'arrays': [np.random.randn(1000) for _ in range(10)],
    'timestamps': [datetime.now() for _ in range(100)]
}

profile_serialization(complex_data)
```

## 🔍 Performance Debugging

### Identify Slow Objects

```python
import time
import datason

def find_slow_objects(data_dict, threshold_ms=10):
    """Find objects that take longer than threshold to serialize."""
    slow_objects = []

    for key, value in data_dict.items():
        start = time.perf_counter()
        try:
            datason.serialize(value)
            duration_ms = (time.perf_counter() - start) * 1000

            if duration_ms > threshold_ms:
                slow_objects.append({
                    'key': key,
                    'duration_ms': duration_ms,
                    'type': type(value).__name__,
                    'size': len(str(value)) if hasattr(value, '__len__') else 'unknown'
                })
        except Exception as e:
            slow_objects.append({
                'key': key,
                'duration_ms': float('inf'),
                'type': type(value).__name__,
                'error': str(e)
            })

    return sorted(slow_objects, key=lambda x: x.get('duration_ms', 0), reverse=True)

# Example usage
data = {
    'small_list': list(range(100)),
    'large_dataframe': pd.DataFrame(np.random.randn(10000, 50)),
    'complex_dict': {f'key_{i}': {'nested': list(range(100))} for i in range(100)}
}

slow_items = find_slow_objects(data)
for item in slow_items:
    print(f"{item['key']}: {item['duration_ms']:.2f}ms ({item['type']})")
```

### Memory Usage Monitoring

```python
import psutil
import os
import datason

def monitor_memory_usage(data):
    """Monitor memory usage during serialization."""
    process = psutil.Process(os.getpid())

    # Baseline memory
    baseline_mb = process.memory_info().rss / 1024 / 1024

    # Serialize and measure
    result = datason.serialize(data)
    peak_mb = process.memory_info().rss / 1024 / 1024

    return {
        'baseline_memory_mb': baseline_mb,
        'peak_memory_mb': peak_mb,
        'memory_increase_mb': peak_mb - baseline_mb,
        'serialized_size_chars': len(str(result))
    }

# Example
large_data = {'arrays': [np.random.randn(10000) for _ in range(10)]}
memory_stats = monitor_memory_usage(large_data)
print(f"Memory increase: {memory_stats['memory_increase_mb']:.1f} MB")
```

## 🎯 Production Optimizations

### API Response Optimization

```python
# Optimized for API responses
def serialize_api_response(data):
    """Optimized serialization for API responses."""
    config = datason.get_api_config()

    # Add response metadata efficiently
    response_data = {
        'data': data,
        'timestamp': time.time(),  # Unix timestamp for speed
        'version': '1.0'
    }

    return datason.serialize(response_data, config=config)
```

### Batch Processing Optimization

```python
# Optimized for batch processing
def serialize_batch(data_list, batch_size=100):
    """Process data in optimized batches."""
    config = datason.get_performance_config()
    results = []

    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_result = datason.serialize(batch, config=config)
        results.append(batch_result)

    return results
```

### Database Export Optimization

```python
# Optimized for database export
def serialize_for_database(df):
    """Optimized DataFrame serialization for database export."""
    config = SerializationConfig(
        dataframe_orient=DataFrameOrient.SPLIT,  # Most efficient
        date_format=DateFormat.UNIX,  # Numeric timestamps
        nan_handling=NanHandling.NULL,  # Database-compatible nulls
        preserve_numeric_precision=True  # Maintain data integrity
    )

    return datason.serialize(df, config=config)
```

## 🏆 Performance Best Practices

### 1. **Profile Before Optimizing**
```python
# Always measure first
stats = benchmark_serialization(your_data)
print(f"Baseline: {stats['average_time_ms']:.2f}ms")
```

### 2. **Choose Configuration Wisely**
```python
# For speed-critical applications
config = datason.get_performance_config()

# For human-readable output
config = datason.get_api_config()

# For ML workflows
config = datason.get_ml_config()
```

### 3. **Optimize Data Structure**
```python
# Use efficient pandas dtypes
df = df.astype({
    'category_col': 'category',
    'int_col': 'int32',
    'float_col': 'float32'
})

# Pre-convert problematic types
data = {k: str(v) if isinstance(v, complex_type) else v
        for k, v in data.items()}
```

### 4. **Monitor in Production**
```python
# Add timing to production code
start = time.perf_counter()
result = datason.serialize(data)
duration = time.perf_counter() - start

if duration > 0.1:  # Log slow serializations
    logger.warning(f"Slow serialization: {duration:.3f}s for {type(data)}")
```

## 🔗 Related Features

- **[Configuration System](../configuration/index.md)** - Performance-tuned configurations
- **[Benchmarking](../../BENCHMARKING.md)** - Detailed performance analysis
- **[Core Serialization](../core/index.md)** - Understanding the base layer
- **[Pandas Integration](../pandas/index.md)** - DataFrame-specific optimizations

## 🚀 Next Steps

- **[Configuration →](../configuration/index.md)** - Fine-tune performance settings
- **[Benchmarking →](../../BENCHMARKING.md)** - Run your own performance tests
- **[Production Deployment →](../../CONTRIBUTING.md)** - Production best practices
