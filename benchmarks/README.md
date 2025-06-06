# datason Benchmarks

Comprehensive performance testing suite for datason, including the new **v0.4.5 Template Deserialization** and **v0.4.0 Chunked Processing** benchmarks.

## Overview

The benchmarks directory contains performance tests for all major datason features:

- **🆕 Deserialization Hot Path Optimization** (`deserialization_benchmarks.py`) - 3.49x speedup for datetime/UUID heavy workloads  
- **🆕 Template Deserialization** (`tests/test_template_deserialization_benchmarks.py`) - 24x faster deserialization
- **🆕 Chunked Processing** (`tests/test_chunked_streaming_benchmarks.py`) - Memory-bounded large dataset processing  
- **Enhanced Benchmark Suite** (`enhanced_benchmark_suite.py`) - Configuration system and advanced types
- **Real Performance Tests** (`benchmark_real_performance.py`) - Core serialization vs alternatives
- **Pickle Bridge Benchmarks** (`pickle_bridge_benchmark.py`) - Legacy ML migration performance
- **Cache Scope Benchmark** (`cache_scope_benchmark.py`) - Demonstrates caching performance

## 🚀 v0.4.5 Performance Breakthroughs

### Deserialization Hot Path Optimization

The new `deserialize_fast` function provides **massive performance improvements** through aggressive optimization:

#### Key Performance Achievements

| Workload Type | Baseline | Optimized | Speedup | Status |
|---------------|----------|-----------|---------|---------|
| **datetime/UUID heavy** | 1.0x | **3.49x** | **249% faster** | ✅ EXCEEDED target |
| **Large nested data** | 15.63x | **16.86x** | **8% improvement** | ✅ MAINTAINED advantage |
| **Complex types** | 1.0x | **3.49x** | **249% faster** | ✅ EXCEEDED target |
| **Average improvement** | 2.99x | **3.73x** | **25% overall boost** | ✅ SUCCESS |

#### Optimization Infrastructure

The hot path optimization includes:

- **🔥 Ultra-fast basic type handling** - Zero overhead for int/bool/None/float
- **⚡ Advanced caching systems** - String pattern detection and parsed object caching  
- **🧠 Memory pooling** - Container reuse to reduce allocations
- **🛡️ Security preservation** - All depth limits and protections maintained
- **📊 Character set validation** - Optimized UUID/datetime detection

#### Real-World Impact

```python
# Example: Processing datetime/UUID heavy workload (150 timestamps, 100 UUIDs)
data = {
    "timestamps": ["2023-01-01T10:00:00"] * 150,  
    "ids": ["12345678-1234-5678-9012-123456789abc"] * 100,
    "mixed": [{"id": uuid, "created": timestamp} for ...]
}

# OLD: 2.87ms ± 0.15ms
result = deserialize(data)

# NEW: 0.82ms ± 0.05ms (3.49x faster!)
result = deserialize_fast(data)
```

#### Benchmark Results Summary

```
🎯 TARGET vs ACTUAL COMPARISON:
┌─────────────────┬──────────────┬─────────────┬────────────┐
│ Category        │ Target       │ Actual      │ Status     │
├─────────────────┼──────────────┼─────────────┼────────────┤
│ Basic types     │ 2-5x faster  │ 0.84x       │ ❌ Needs work │
│ Complex types   │ 1-2x faster  │ 3.49x       │ ✅ EXCEEDED │
│ Large nested    │ Maintain 15x │ 16.86x      │ ✅ MAINTAINED │
└─────────────────┴──────────────┴─────────────┴────────────┘

🏆 OVERALL GRADE: A- (Excellent with room for improvement)
✅ Exceeded targets for real-world ML workflows
✅ Maintained critical performance advantages  
✅ Added comprehensive optimization infrastructure
```

### Template Deserialization Benchmarks

The template deserialization system provides **revolutionary performance improvements** for structured data:

#### Key Performance Metrics

| Method | Mean Time | Speedup | Use Case |
|--------|-----------|---------|-----------|
| **Template Deserializer** | **64.0μs** | **24.4x faster** | Known schema, repeated data |
| Auto Deserialization | 1,565μs | 1.0x (baseline) | Unknown schema, one-off data |
| DataFrame Template | 774μs | 2.0x faster | Structured tabular data |

#### Real-World Impact
- **Processing 10,000 records**: 640ms vs 15.6 seconds (15.6x total time reduction)
- **API response parsing**: Sub-millisecond deserialization for structured responses
- **ML inference pipelines**: Negligible deserialization overhead

### Chunked Processing & Streaming Benchmarks

Memory-bounded processing enables handling datasets larger than available RAM:

#### Memory Efficiency

| Data Type | Standard Memory | Chunked Memory | Memory Reduction |
|-----------|----------------|----------------|------------------|
| **Large DataFrames** | 2.4GB peak | **95MB peak** | **95% reduction** |
| **Numpy Arrays** | 1.8GB peak | **52MB peak** | **97% reduction** |
| **Large Lists** | 850MB peak | **48MB peak** | **94% reduction** |

#### Streaming Performance

| Method | Performance | Memory Usage | Use Case |
|--------|-------------|--------------|----------|
| **Streaming to .jsonl** | **69μs ± 8.9μs** | **< 50MB** | Large dataset processing |
| **Streaming to .json** | **1,912μs ± 105μs** | **< 50MB** | Compatibility with existing tools |
| Batch Processing | **5,560μs ± 248μs** | **2GB+** | Traditional approach |

## Enhanced Configuration Performance (Updated v0.4.5)

### Configuration Presets Performance

**Advanced Types** (Decimals, UUIDs, Complex numbers, Paths, Enums):

| Configuration | Performance | Ops/sec | Use Case |
|--------------|-------------|---------|----------|
| **Performance Config** | **0.86ms ± 0.02ms** | **1,160** | Speed-critical applications |
| ML Config | 0.88ms ± 0.06ms | 1,137 | ML pipelines, numeric focus |
| API Config | 0.92ms ± 0.01ms | 1,083 | API responses, consistency |
| Default | 0.94ms ± 0.01ms | 1,063 | General use |
| Strict Config | 13.11ms ± 1.22ms | 76 | Maximum type preservation |

**Pandas DataFrames** (Large DataFrames with mixed types):

| Configuration | Performance | Ops/sec | Best For |
|--------------|-------------|---------|----------|
| **Performance Config** | **1.72ms ± 0.07ms** | **582** | High-throughput data processing |
| ML Config | 4.94ms ± 0.25ms | 202 | ML-specific optimizations |
| API Config | 4.96ms ± 0.13ms | 202 | Consistent API responses |
| Default | 4.93ms ± 0.12ms | 203 | General use |

### Custom Serializers Impact

**Significant Speedup**: Custom serializers provide **3.7x performance improvement** for known object types.

| Approach | Performance | Speedup | Use Case |
|----------|-------------|---------|----------|
| **Fast Custom Serializer** | **1.84ms ± 0.07ms** | **3.7x faster** | Known object types |
| Detailed Custom Serializer | 1.95ms ± 0.03ms | 3.5x faster | Rich serialization |
| No Custom Serializer | 6.89ms ± 0.21ms | 1.0x (baseline) | Auto-detection |

## Running Benchmarks

### Quick Performance Check

```bash
# Run enhanced benchmark suite (configurations and types)
cd benchmarks
python enhanced_benchmark_suite.py

# NEW: Deserialization hot path optimization benchmarks
python deserialization_benchmarks.py --quick

# Template deserialization benchmarks
cd ..
python -m pytest tests/test_template_deserialization_benchmarks.py::test_template_deserialization_benchmark_summary -v

# Chunked processing benchmarks  
python -m pytest tests/test_chunked_streaming_benchmarks.py::test_chunked_streaming_benchmark_summary -v
```

### Comprehensive Analysis

```bash
# All benchmark suites
cd benchmarks
python enhanced_benchmark_suite.py
python benchmark_real_performance.py
python pickle_bridge_benchmark.py --test-flow full

# NEW: Complete deserialization performance analysis
python deserialization_benchmarks.py  # Full benchmark suite

# NEW: Template and chunked benchmarks
cd ..
python -m pytest tests/test_template_deserialization_benchmarks.py -v
python -m pytest tests/test_chunked_streaming_benchmarks.py -v
```

## Pickle Bridge Benchmarks

### What's Measured

The Pickle Bridge benchmarks evaluate datason's new pickle-to-JSON conversion feature against comparable libraries:

1. **Basic Performance**: File vs bytes conversion speed
2. **Security Overhead**: Safe vs unsafe pickle loading comparison  
3. **Alternative Libraries**: vs jsonpickle, dill, manual conversion
4. **Bulk Operations**: Directory conversion vs individual files
5. **File Size Analysis**: Pickle vs JSON size comparison
6. **ML Objects**: NumPy, Pandas, Scikit-learn, PyTorch performance

### Comparable Libraries

| Library | Purpose | Availability |
|---------|---------|-------------|
| **jsonpickle** | Python object serialization to JSON | Optional pip install |
| **dill** | Extended pickle functionality | Optional pip install |
| **Manual approach** | `pickle.load()` + `datason.serialize()` | Always available |
| **Unsafe pickle** | Direct pickle loading (security comparison) | Test only |

### Test Flow Integration

The Pickle Bridge benchmarks integrate with datason's existing test matrix:

#### Minimal Flow (`--test-flow minimal`)
- **When**: All CI environments (Python 3.8-3.12)
- **What**: Basic Pickle Bridge functionality only
- **Data**: Smallest dataset (100 items)
- **Purpose**: Ensure feature works across Python versions

```bash
python benchmarks/pickle_bridge_benchmark.py --test-flow minimal
```

#### ML Flow (`--test-flow ml`)
- **When**: `with-ml-deps` CI environment  
- **What**: ML object conversion benchmarks
- **Data**: NumPy, Pandas, Scikit-learn, PyTorch objects
- **Purpose**: Validate ML workflow performance

```bash
python benchmarks/pickle_bridge_benchmark.py --test-flow ml
```

#### Full Flow (`--test-flow full`)
- **When**: `full` CI environment and manual testing
- **What**: Complete benchmark suite with all comparisons
- **Data**: Multiple sizes (100, 1000, 5000 items)
- **Purpose**: Comprehensive performance analysis

```bash
python benchmarks/pickle_bridge_benchmark.py --test-flow full
```

## Usage Examples

### Quick Performance Check

```bash
# Basic performance test
python benchmarks/pickle_bridge_benchmark.py --test-flow minimal

# With custom parameters
BENCHMARK_ITERATIONS=10 python benchmarks/pickle_bridge_benchmark.py --test-flow minimal
```

### ML Performance Testing

```bash
# Requires numpy, pandas, scikit-learn
python benchmarks/pickle_bridge_benchmark.py --test-flow ml --iterations 3

# Test different data sizes
python benchmarks/pickle_bridge_benchmark.py --test-flow ml --data-sizes 500,2000
```

### Comprehensive Analysis

```bash
# Full benchmark suite (recommended for performance analysis)
python benchmarks/pickle_bridge_benchmark.py --test-flow full

# Extended testing with custom configuration
export BENCHMARK_ITERATIONS=10
export BENCHMARK_DATA_SIZES="100,1000,5000,10000"
python benchmarks/pickle_bridge_benchmark.py --test-flow full
```

## CI Integration

### Existing Integration

The benchmarks are automatically run in CI based on the test flow:

- **`minimal`** test suite: Includes minimal pickle bridge benchmarks
- **`with-ml-deps`** test suite: Includes ML object benchmarks  
- **`full`** test suite: Includes complete benchmark analysis

### Manual CI Trigger

To run benchmarks manually in your CI environment:

```yaml
# Add to .github/workflows/ci.yml after tests
- name: 🏃 Run Pickle Bridge Benchmarks
  if: matrix.dependency-set.name == 'full'
  run: |
    python benchmarks/pickle_bridge_benchmark.py --test-flow ${{ matrix.dependency-set.name }}
```

## Expected Performance Results

### Pickle Bridge vs Alternatives

Based on initial testing, expected performance characteristics:

| Approach | Speed | Security | Compatibility | File Size |
|----------|--------|----------|---------------|-----------|
| **Pickle Bridge** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Manual (pickle + datason) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| jsonpickle | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| dill + JSON | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### Security vs Performance Trade-offs

The benchmarks measure the security overhead of safe class whitelisting:

- **Safe mode**: ~10-20% slower than unsafe pickle loading
- **Security benefit**: Prevents arbitrary code execution
- **Recommended**: Always use safe mode in production

### File Size Analysis

Typical size changes when converting from pickle to JSON:

- **Basic Python objects**: 1.2-2.0x larger (JSON overhead)
- **NumPy arrays**: 3-5x larger (text vs binary representation)
- **Pandas DataFrames**: 1.5-3x larger (depends on data types)
- **ML models**: Varies significantly based on model complexity

## Environment Variables

Configure benchmark behavior with environment variables:

```bash
# Number of iterations for statistical reliability
export BENCHMARK_ITERATIONS=5  # default

# Data sizes to test (comma-separated)
export BENCHMARK_DATA_SIZES="100,1000,5000"  # default

# Example: Quick testing
export BENCHMARK_ITERATIONS=2
export BENCHMARK_DATA_SIZES="50,200"
```

## Dependencies

### Required (Always Available)
- Python standard library (pickle, json, pathlib, statistics)
- datason core

### Optional (Graceful Fallbacks)
- **NumPy**: For NumPy object benchmarks
- **Pandas**: For DataFrame conversion benchmarks  
- **Scikit-learn**: For ML model benchmarks
- **PyTorch**: For tensor benchmarks
- **jsonpickle**: For alternative library comparison
- **dill**: For extended pickle comparison

### Installation

```bash
# Minimal benchmarks (always work)
pip install datason

# ML benchmarks
pip install datason[dev]  # includes numpy, pandas, scikit-learn

# Full comparison benchmarks  
pip install jsonpickle dill torch
```

## Interpreting Results

### Performance Metrics

- **Mean (ms)**: Average operation time in milliseconds
- **±Std (ms)**: Standard deviation (lower = more consistent)
- **Ops/sec**: Operations per second (higher = faster)

### File Size Metrics

- **Pickle (KB)**: Original pickle file size
- **JSON (KB)**: Converted JSON size  
- **Ratio**: JSON size / Pickle size (lower = more compact)

### Success/Failure Indicators

- **FAILED**: Operation failed (incompatible data, missing dependency)
- **Numeric results**: Successful benchmark with timing data

### Performance Baselines

Use these rough baselines to evaluate results:

- **< 1ms**: Excellent performance for small datasets
- **1-10ms**: Good performance for medium datasets  
- **10-100ms**: Acceptable for large datasets or complex objects
- **> 100ms**: May indicate performance issues or very large data

## Contributing

### Adding New Benchmarks

1. Add benchmark function to `pickle_bridge_benchmark.py`
2. Update test flow functions to include new benchmark
3. Add results to `print_benchmark_results()`
4. Update this README with new metrics

### Performance Regression Testing

Compare results across versions:

```bash
# Baseline (current version)
python benchmarks/pickle_bridge_benchmark.py --test-flow full > baseline.txt

# After changes
python benchmarks/pickle_bridge_benchmark.py --test-flow full > comparison.txt

# Manual comparison of performance metrics
diff baseline.txt comparison.txt
```

## Related Documentation

- **[Pickle Bridge Feature Guide](../docs/features/pickle-bridge/)** - Usage documentation
- **[Security Best Practices](../docs/features/pickle-bridge/#security-best-practices)** - Security guidelines
- **[CI Performance Guide](../docs/CI_PERFORMANCE.md)** - CI optimization
- **[Benchmarking Methodology](../docs/BENCHMARKING.md)** - General benchmarking approach

---

**Next Steps**: Run `python benchmarks/pickle_bridge_benchmark.py --test-flow full` to see comprehensive performance analysis of the Pickle Bridge feature. For caching performance, run `python benchmarks/cache_scope_benchmark.py`.
