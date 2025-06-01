# datason Benchmarks

Comprehensive performance testing suite for datason, including the new **Pickle Bridge** benchmarks.

## Overview

The benchmarks directory contains performance tests for all major datason features:

- **Enhanced Benchmark Suite** (`enhanced_benchmark_suite.py`) - Configuration system and advanced types
- **Real Performance Tests** (`benchmark_real_performance.py`) - Core serialization vs alternatives
- **Pickle Bridge Benchmarks** (`pickle_bridge_benchmark.py`) - Legacy ML migration performance

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

**Next Steps**: Run `python benchmarks/pickle_bridge_benchmark.py --test-flow full` to see comprehensive performance analysis of the Pickle Bridge feature.
