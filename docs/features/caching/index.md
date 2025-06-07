# Configurable Caching System

The datason caching system provides intelligent, configurable caching that adapts to different workflow requirements. Unlike traditional fixed caches, this system offers multiple cache scopes to balance performance with predictability.

## 🎯 Overview

The caching system addresses a critical challenge in ML/AI workflows: **balancing performance with predictability**. Different scenarios require different caching strategies:

- **ML Training**: Maximum performance with process-scoped caches
- **Web APIs**: Request-scoped caches for consistent responses within a request
- **Testing**: Operation-scoped caches for predictable, isolated results
- **Debugging**: Disabled caches for complete predictability

```python
import datason
from datason import CacheScope

# Choose your caching strategy
datason.set_cache_scope(CacheScope.REQUEST)  # For web APIs
datason.set_cache_scope(CacheScope.PROCESS)  # For ML training
datason.set_cache_scope(CacheScope.OPERATION)  # For testing (default)
datason.set_cache_scope(CacheScope.DISABLED)  # For debugging
```

## 🔧 Cache Scopes

### Operation Scope (Default - Safest)

**Best for**: Testing, debugging, applications requiring predictable behavior

```python
# Operation scope clears caches after each deserialize operation
with datason.operation_scope():
    result1 = datason.deserialize_fast(data1)  # Fresh parsing
    result2 = datason.deserialize_fast(data2)  # Fresh parsing
# Cache automatically cleared
```

**Characteristics**:
- ✅ **Predictable**: No cross-operation contamination
- ✅ **Safe**: Prevents test order dependencies
- ✅ **Memory efficient**: Regular cleanup prevents bloat
- ⚠️ **Performance**: Lower cache hit rates

### Request Scope (Balanced)

**Best for**: Web APIs, batch processing, request-response workflows

```python
with datason.request_scope():
    # Multiple operations share cache within this scope
    result1 = datason.deserialize_fast(data1)  # Parse and cache
    result2 = datason.deserialize_fast(data1)  # Cache hit!
    result3 = datason.deserialize_fast(data2)  # Parse and cache
# Cache cleared when scope exits
```

**Characteristics**:
- ✅ **Balanced**: Good performance within requests
- ✅ **Isolated**: Each request has clean cache state
- ✅ **Memory controlled**: Caches cleared between requests
- ✅ **Predictable**: No cross-request contamination

### Process Scope (Maximum Performance)

**Best for**: ML training, data analytics, long-running processes

```python
# Process scope persists caches across all operations
datason.set_cache_scope(CacheScope.PROCESS)

result1 = datason.deserialize_fast(data1)  # Parse and cache
result2 = datason.deserialize_fast(data1)  # Cache hit!
# ... later in the application
result3 = datason.deserialize_fast(data1)  # Still cached!
```

**Characteristics**:
- ✅ **Maximum performance**: Highest cache hit rates
- ✅ **Memory efficient**: Reuses parsed objects
- ⚠️ **Memory growth**: Caches persist until manually cleared
- ⚠️ **Potential contamination**: Cache state affects all operations

### Disabled Scope (Complete Predictability)

**Best for**: Debugging, performance testing baseline, reproducible research

```python
datason.set_cache_scope(CacheScope.DISABLED)

result1 = datason.deserialize_fast(data)  # Parse every time
result2 = datason.deserialize_fast(data)  # Parse again (no cache)
```

**Characteristics**:
- ✅ **Completely predictable**: No caching effects
- ✅ **Debugging friendly**: Pure processing without cache interference
- ✅ **Memory efficient**: No cache storage
- ⚠️ **Slower**: No performance benefits from caching

## 📊 Performance Comparison

Based on real-world benchmarks with datetime and UUID heavy datasets:

| Cache Scope | Performance | Memory Usage | Use Case |
|-------------|-------------|--------------|----------|
| **Disabled** | Baseline (100%) | Minimal | Debugging, Testing |
| **Operation** | 110-120% | Low | Default, Safe Operations |
| **Request** | 130-150% | Medium | Web APIs, Batch Processing |
| **Process** | 150-200% | Higher* | ML Training, Analytics |

*Memory usage grows with cache size but provides better object reuse

### Cache Scope Micro-Benchmark

A simple benchmark script demonstrates real-world gains from caching. Run
`cache_scope_benchmark.py` in the `benchmarks/` directory:

```bash
python benchmarks/cache_scope_benchmark.py
```

Typical results (1000 repeated UUID strings):

| Scope | Time (ms) |
|-------|----------|
| **DISABLED** | ~2.8 |
| **OPERATION** | ~2.3 |
| **REQUEST** | ~1.9 |
| **PROCESS** | ~1.3 |

## 🛠️ Configuration

### Basic Configuration

```python
import datason
from datason.config import SerializationConfig, CacheScope

# Global cache scope
datason.set_cache_scope(CacheScope.REQUEST)

# Configuration with caching options
config = SerializationConfig(
    cache_size_limit=10000,           # Maximum items per cache
    cache_metrics_enabled=True,       # Enable performance monitoring
    cache_warn_on_limit=True,         # Warn when cache limits reached
)

result = datason.deserialize_fast(data, config)
```

### Context Managers

```python
# Temporary scope changes
with datason.operation_scope():
    # Use operation-scoped caching for this block
    result = datason.deserialize_fast(sensitive_data)

with datason.request_scope():
    # Process multiple related items with shared cache
    results = [datason.deserialize_fast(item) for item in batch]
```

### Configuration Presets

```python
# Preset configurations include optimized cache settings

# ML workflows - optimized for performance
ml_config = datason.get_ml_config()
# Includes: process scope, large cache limits, metrics enabled

# API workflows - balanced performance and predictability
api_config = datason.get_api_config()
# Includes: request scope, moderate cache limits

# Development - safe and predictable
dev_config = datason.get_development_config()
# Includes: operation scope, small cache limits, extensive warnings
```

## 📈 Cache Metrics and Monitoring

### Enabling Metrics

```python
from datason import get_cache_metrics, CacheScope

# Enable metrics in configuration
config = datason.SerializationConfig(cache_metrics_enabled=True)

# Use with metrics
result = datason.deserialize_fast(data, config)

# Check performance
metrics = get_cache_metrics()
for scope, stats in metrics.items():
    print(f"{scope}: {stats.hit_rate:.1%} hit rate, {stats.hits} hits")
```

### Sample Metrics Output

```
CacheScope.PROCESS: 78.3% hit rate, 1247 hits, 343 misses, 12 evictions
CacheScope.REQUEST: 45.2% hit rate, 89 hits, 108 misses, 0 evictions
```

### Metrics Interpretation

- **Hit Rate**: Percentage of cache hits vs total accesses
- **Hits**: Number of successful cache retrievals
- **Misses**: Number of cache misses requiring new parsing
- **Evictions**: Number of items removed due to size limits
- **Size Warnings**: Number of times cache size limit was reached

### Performance Monitoring

```python
import datason
from datason import CacheScope, get_cache_metrics, reset_cache_metrics

# Reset metrics before measurement
reset_cache_metrics()

# Your processing code
with datason.request_scope():
    results = process_batch(data_items)

# Analyze performance
metrics = get_cache_metrics(CacheScope.REQUEST)
print(f"Cache efficiency: {metrics.hit_rate:.1%}")

if metrics.hit_rate < 0.3:  # Less than 30% hit rate
    print("Consider using longer-lived cache scope for better performance")

if metrics.evictions > 0:
    print(f"Cache limit reached {metrics.evictions} times - consider increasing cache_size_limit")
```

## 🔍 Object Pooling

The caching system includes intelligent object pooling to reduce memory allocations:

### Dictionary and List Pooling

```python
# Pools are automatically managed but can be controlled via configuration
config = datason.SerializationConfig(
    cache_size_limit=5000,  # Also controls pool sizes
)

# Pools automatically:
# 1. Reuse dict/list objects during deserialization
# 2. Clear objects before reuse (no data contamination)
# 3. Respect cache scope rules (operation/request/process)
# 4. Limit pool size to prevent memory bloat
```

### Pool Efficiency

Object pooling provides:
- **Memory efficiency**: Reduces garbage collection pressure
- **Performance**: Faster object allocation for large datasets
- **Safety**: Objects are cleared before reuse
- **Controlled growth**: Pool size limits prevent memory leaks

## ⚙️ Advanced Usage

### Custom Cache Management

```python
# Manual cache control
datason.clear_all_caches()  # Clear all scopes
datason.clear_caches()      # Clear current scope only

# Check current scope
current_scope = datason.get_cache_scope()
print(f"Currently using: {current_scope}")
```

### ML Pipeline Example

```python
import datason
from datason import CacheScope

def ml_training_pipeline(training_data):
    """Example ML training with optimized caching."""

    # Use process scope for maximum performance during training
    with datason.request_scope():  # or set_cache_scope(CacheScope.PROCESS)

        # Parse training data once, cache for reuse
        parsed_data = []
        for batch in training_data:
            # Repeated datetime/UUID patterns cached automatically
            parsed_batch = datason.deserialize_fast(batch)
            parsed_data.append(parsed_batch)

        # Training loop benefits from cached parsing
        for epoch in range(num_epochs):
            for batch in parsed_data:
                # Fast deserialization with cache hits
                train_step(batch)

    # Cache automatically cleared when scope exits
    # or persists if using process scope

def api_request_handler(request_data):
    """Example API handler with request-scoped caching."""

    with datason.request_scope():
        # Parse request data
        parsed_request = datason.deserialize_fast(request_data)

        # Process multiple related items (cache shared within request)
        results = []
        for item in parsed_request['items']:
            processed = process_item(item)
            results.append(datason.serialize(processed))

        return {'results': results}

    # Cache cleared when request completes
```

### Testing with Reliable Caching

```python
import pytest
import datason
from datason import CacheScope

class TestWithCaching:
    def setup_method(self):
        """Ensure clean cache state for each test."""
        datason.set_cache_scope(CacheScope.OPERATION)  # Safest for testing
        datason.clear_all_caches()

    def test_deserialization_with_cache_isolation(self):
        """Test that demonstrates cache isolation."""
        data = {"timestamp": "2024-01-15T10:30:45", "id": "uuid-string"}

        # Each call uses fresh operation-scoped cache
        result1 = datason.deserialize_fast(data)
        result2 = datason.deserialize_fast(data)

        # Results are correct but cache doesn't persist between operations
        assert result1 == result2
        assert isinstance(result1['timestamp'], datetime.datetime)
```

## 🏆 Best Practices

### 1. Choose the Right Scope

```python
# For web applications
app.before_request(lambda: datason.set_cache_scope(CacheScope.REQUEST))

# For ML training
def train_model():
    datason.set_cache_scope(CacheScope.PROCESS)
    # ... training code

# For testing
@pytest.fixture(autouse=True)
def setup_test_cache():
    datason.set_cache_scope(CacheScope.OPERATION)
    datason.clear_all_caches()
```

### 2. Monitor Performance

```python
# Regular cache monitoring
def monitor_cache_performance():
    metrics = datason.get_cache_metrics()

    for scope, stats in metrics.items():
        if stats.hit_rate < 0.2:  # Less than 20% hit rate
            logger.warning(f"Low cache efficiency for {scope}: {stats.hit_rate:.1%}")

        if stats.evictions > 100:  # Too many evictions
            logger.warning(f"Cache size limit reached frequently for {scope}")
```

### 3. Resource Management

```python
# Clean up in long-running processes
def periodic_cache_cleanup():
    """Call periodically in long-running processes."""
    metrics = datason.get_cache_metrics(CacheScope.PROCESS)

    # Clear cache if it's getting too large with low efficiency
    if metrics.size > 10000 and metrics.hit_rate < 0.1:
        datason.clear_caches()
        logger.info("Cleared inefficient cache")
```

### 4. Configuration Tuning

```python
# Tune cache size based on your data patterns
config = datason.SerializationConfig(
    cache_size_limit=50000,  # Increase for data with many repeated patterns
    cache_warn_on_limit=True,  # Monitor when limits are reached
    cache_metrics_enabled=True,  # Always enable in production for monitoring
)
```

## 🔧 Troubleshooting

### Low Cache Hit Rates

If you're seeing low cache hit rates:

1. **Check data patterns**: Caching works best with repeated datetime/UUID strings
2. **Verify scope**: Use longer-lived scopes (REQUEST/PROCESS) for better hits
3. **Monitor size limits**: Increase `cache_size_limit` if needed
4. **Profile your data**: Use metrics to understand access patterns

### Memory Usage

If memory usage is too high:

1. **Use shorter scopes**: OPERATION scope uses minimal memory
2. **Reduce cache limits**: Lower `cache_size_limit`
3. **Enable cleanup warnings**: Set `cache_warn_on_limit=True`
4. **Manual cleanup**: Call `clear_caches()` periodically

### Test Flakiness

If tests are flaky:

1. **Use OPERATION scope**: Ensures test isolation
2. **Clear caches in setup**: `clear_all_caches()` in test setup
3. **Avoid PROCESS scope**: Can cause test order dependencies
4. **Check metrics**: Ensure predictable cache behavior

## 🌟 Summary

The configurable caching system provides:

- **🔧 Flexible**: Multiple cache scopes for different use cases
- **⚡ Fast**: Significant performance improvements with smart caching
- **🛡️ Safe**: Operation scope prevents contamination by default
- **📊 Observable**: Built-in metrics for performance monitoring
- **🧠 Intelligent**: Object pooling and automatic memory management
- **🧪 Testable**: Predictable behavior for reliable testing

Choose your cache scope based on your needs:
- **OPERATION**: Maximum safety and predictability
- **REQUEST**: Balanced performance for web APIs
- **PROCESS**: Maximum performance for ML/analytics
- **DISABLED**: Complete predictability for debugging
