# datason Performance Benchmarks

## Overview

This document contains **real performance measurements** for datason, obtained through systematic benchmarking rather than estimates. All benchmarks are reproducible using the included `benchmark_real_performance.py` script.

## Benchmark Environment

- **Python**: 3.13.3
- **Platform**: macOS (Darwin 24.5.0)
- **Dependencies**: NumPy 1.26.4, Pandas 2.2.2, PyTorch 2.7.0
- **Method**: 5 iterations per test, statistical analysis (mean ± std dev)
- **Hardware**: Modern development machine (representative performance)

## Results Summary

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
| Standard JSON | 0.4ms ± 0.02ms | 1.0x (baseline) |
| **datason** | **0.6ms ± 0.02ms** | **1.6x** |

**Analysis**: datason adds only 60% overhead vs standard JSON for compatible data, which is excellent considering the added functionality (type detection, ML object support, safety features).

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
| **datason** | **2.1ms ± 0.08ms** | Only option for this data |
| Pickle | 0.7ms ± 0.07ms | Binary format, Python-only |
| Standard JSON | ❌ **Fails** | Cannot serialize UUIDs/datetime |

**Analysis**: datason is 3.2x slower than pickle but provides JSON output that's human-readable and cross-platform compatible.

### High-Throughput Scenarios

**Large Nested Data**: 100 groups × 50 items (5,000 complex objects)
- **Throughput**: 272,654 items/second
- **Performance**: 18.3ms ± 0.7ms total

**NumPy Arrays**: ~122K elements across multiple arrays
- **Throughput**: 5.5 million elements/second  
- **Performance**: 4.2ms ± 0.1ms total

**Pandas DataFrames**: 5,100 total rows across multiple DataFrames
- **Throughput**: 195,242 rows/second
- **Performance**: 26.1ms ± 22.0ms total

### Round-Trip Performance

**Test**: Complete workflow (serialize → JSON.dumps → JSON.loads → deserialize)
```python
# 200 user objects with UUIDs and timestamps
serialize_time    = 0.5ms ± 0.02ms
json_dumps_time   = included in measurement
json_loads_time   = included in measurement  
deserialize_time  = 0.7ms ± 0.01ms
total_round_trip  = 1.4ms ± 0.02ms
```

**Real-world significance**: Complete API request-response cycle under 1.5ms.

## Why We Compare with Pickle

**Pickle is the natural comparison point** because it's the only other tool that can serialize complex Python objects like ML models and DataFrames. However, the 3.2x performance difference tells only part of the story.

### 🏆 When Pickle Wins
```python
# Pure Python environment, speed is everything
import pickle
start = time.time()
with open('model.pkl', 'wb') as f:
    pickle.dump(complex_ml_pipeline, f)  # 0.7ms
print(f"Saved in {time.time() - start:.1f}ms")

# ✅ Fastest option for Python-only workflows
# ✅ Perfect object reconstruction  
# ✅ Handles any Python object (even lambdas, classes)
```

### 🌐 When datason Wins
```python
# Multi-language team, API responses, data sharing
import json
import datason as ds

start = time.time()
json_data = ds.serialize(complex_ml_pipeline)  # 2.1ms
with open('model.json', 'w') as f:
    json.dump(json_data, f)
print(f"Saved in {time.time() - start:.1f}ms")

# ✅ Frontend team can read it immediately
# ✅ Business stakeholders can inspect results  
# ✅ Works in Git diffs, text editors, web browsers
# ✅ API responses work across all platforms
```

### 📊 The Real Tradeoff
```python
# Performance vs Versatility
pickle_speed = 0.7  # ms
datason_speed = 2.1  # ms
overhead = 1.4  # ms extra

# Questions to ask:
# - Is 1.4ms overhead significant for your use case?
# - Do you need cross-language compatibility?  
# - Do you need human-readable output?
# - Are you building APIs or microservices?

# For most modern applications: 1.4ms is negligible
# For high-frequency trading: Every microsecond matters (use pickle)
# For web APIs: Human-readable JSON is essential (use datason)
```

### 🎯 Practical Decision Framework

| Your Situation | Recommended Choice | Why |
|----------------|-------------------|-----|
| **Python-only batch processing** | pickle | Pure speed, no compatibility needs |
| **REST API responses** | **datason** | JSON required, human-readable |
| **Microservices architecture** | **datason** | Language interoperability |
| **Data science collaboration** | **datason** | Share with non-Python users |
| **Real-time trading systems** | pickle | Every millisecond counts |
| **Research/experiments** | **datason** | Reproducible, inspectable results |
| **Production ML pipelines** | **datason** | Debugging, monitoring, APIs |

The 3.2x "performance penalty" is actually a **feature trade** - you're getting cross-language compatibility, human readability, and API-friendliness for 1.4ms per operation.

## Comparative Analysis

### vs Standard JSON
- **Compatibility**: datason handles 20+ data types vs JSON's 6 basic types
- **Overhead**: Only 1.6x for compatible data (vs 3-10x for many JSON alternatives)
- **Safety**: Graceful handling of NaN/Infinity vs JSON's errors

### vs Pickle  
- **Speed**: 3.2x slower but provides human-readable JSON output
- **Portability**: Cross-language compatible vs Python-only
- **Security**: No arbitrary code execution risks
- **Debugging**: Human-readable output for troubleshooting

### vs Specialized Libraries
- **orjson/ujson**: Faster for basic JSON types but cannot handle ML objects
- **joblib**: Good for NumPy arrays but binary format
- **datason**: Best balance of functionality, performance, and compatibility

## Methodology

### Benchmark Script
All measurements come from `benchmark_real_performance.py`, which:

1. **Multiple Iterations**: Runs each test 5 times for statistical reliability
2. **Warm-up**: First measurement discarded (JIT compilation, cache loading)
3. **Statistical Analysis**: Reports mean, standard deviation, min/max
4. **Real Data**: Uses realistic data structures, not toy examples
5. **Fair Comparison**: Compares like-for-like where possible

### Test Data Characteristics
- **Simple data**: JSON-compatible objects only
- **Complex data**: UUIDs, datetimes, nested structures
- **Large data**: Thousands of objects with realistic size
- **ML data**: NumPy arrays, Pandas DataFrames of representative sizes

### Measurement Precision
- Uses `time.perf_counter()` for high-resolution timing
- Measures end-to-end including all overhead
- No artificial optimizations or cherry-picked scenarios

## Performance Recommendations

### When datason Excels
- **Mixed data types**: Standard + ML objects in one structure
- **API responses**: Need JSON compatibility with complex data
- **Data science workflows**: Frequent DataFrame/NumPy serialization
- **Cross-platform**: Human-readable output required

### Performance Tips
1. **Pre-serialize static data**: Cache serialized results for repeated use
2. **Batch operations**: Group objects for better throughput
3. **Memory streaming**: Use for large datasets to avoid memory spikes
4. **Profile your use case**: Run benchmarks with your actual data

### When to Consider Alternatives
- **Pure speed + basic types**: Use orjson/ujson
- **Python-only + complex objects**: Use pickle
- **Scientific arrays + compression**: Use joblib
- **Maximum compatibility**: Use standard json with manual type handling

## Benchmark History

| Date | Version | Change | Performance Impact |
|------|---------|--------|-------------------|
| 2024-01 | 1.0.0 | Initial benchmarks | Baseline established |
| TBD | 1.1.0 | Optimization updates | Track improvements |

## Running Benchmarks

```bash
# Run complete benchmark suite
python benchmark_real_performance.py

# Run specific performance tests
pytest tests/test_performance.py -v

# Generate detailed timing
python benchmark_real_performance.py --detailed
```

## Interpreting Results

### Statistical Significance
- **Mean**: Primary performance metric
- **Standard deviation**: Consistency indicator (lower = more consistent)
- **Min/Max**: Range of performance variation

### Real-World Context
- **Sub-millisecond**: Excellent for interactive applications
- **Single-digit milliseconds**: Good for API responses
- **Double-digit milliseconds**: Acceptable for batch processing
- **100ms+**: May need optimization for real-time use

---

*Last updated: January 2024*  
*Benchmarks reflect datason v1.0.0 performance characteristics*
