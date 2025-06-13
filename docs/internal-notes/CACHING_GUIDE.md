# Datason Configurable Caching System

## Overview

Datason now features a sophisticated configurable caching system designed to optimize performance across different ML/data workflows while ensuring predictability and safety. This system addresses the real-world challenges of caching in diverse environments, from simple scripts to multi-tenant production services.

## Cache Scopes

The caching system supports four distinct scopes, each designed for specific use cases:

### 1. Operation-Scoped (Default - Safest)
- **When**: Default behavior, recommended for most use cases
- **Behavior**: Cache cleared after each serialize/deserialize operation
- **Benefits**: Maximum predictability, no cross-contamination
- **Use Cases**: Scripts, one-off analyses, testing, real-time processing

```python
import datason
from datason.config import CacheScope, SerializationConfig

# Default behavior - operation-scoped
config = SerializationConfig()  # cache_scope=CacheScope.OPERATION
result = datason.deserialize(data, config=config)
```

### 2. Request-Scoped (Multi-tenant Safe)
- **When**: Web APIs, multi-tenant applications
- **Behavior**: Cache persists within a single request/context
- **Benefits**: Performance within requests, isolation between requests
- **Use Cases**: Web APIs, microservices, multi-tenant SaaS

```python
from datason.config import get_web_api_config
from datason.cache_manager import request_scope

config = get_web_api_config()  # Uses request-scoped caching

# In your web framework (Flask, FastAPI, etc.)
with request_scope():
    # All datason operations within this block share cache
    result1 = datason.deserialize(data1, config=config)
    result2 = datason.deserialize(data2, config=config)
    # Cache automatically cleared when request ends
```

### 3. Process-Scoped (Maximum Performance)
- **When**: Batch processing, homogeneous workloads
- **Behavior**: Cache persists for the entire process lifetime
- **Benefits**: Maximum performance for repeated operations
- **Use Cases**: Batch processing, ETL pipelines, model training
- **Risks**: Potential cross-contamination, memory growth

```python
from datason.config import get_batch_processing_config
from datason import cache_scope, CacheScope

config = get_batch_processing_config()  # Uses process-scoped caching

# For batch processing
with cache_scope(CacheScope.PROCESS):
    for batch in large_dataset:
        # Benefits from accumulated cache across all batches
        results = [datason.deserialize(item, config=config) for item in batch]
```

### 4. Disabled (Most Predictable)
- **When**: Development, debugging, high-security environments
- **Behavior**: No caching at all
- **Benefits**: Maximum predictability, no memory overhead
- **Use Cases**: Development, testing, security-sensitive applications

```python
from datason.config import get_development_config

config = get_development_config()  # Uses disabled caching
result = datason.deserialize(data, config=config)  # No caching overhead
```

## Preset Configurations

Datason provides preset configurations optimized for common scenarios:

### Batch Processing Configuration
```python
from datason.config import get_batch_processing_config

config = get_batch_processing_config()
# - Process-level caching for maximum performance
# - Large cache size (5000 entries)
# - Metrics enabled for monitoring
# - Aggressive type coercion for compatibility
```

### Web API Configuration
```python
from datason.config import get_web_api_config

config = get_web_api_config()
# - Request-scoped caching for multi-tenant safety
# - Moderate cache size (1000 entries)
# - Metrics disabled for reduced overhead
# - Safe type coercion for reliability
```

### Real-time Configuration
```python
from datason.config import get_realtime_config

config = get_realtime_config()
# - Operation-scoped caching for predictability
# - Small cache size (500 entries) to prevent latency spikes
# - Warnings disabled for real-time contexts
# - Optimized for speed over precision
```

### Development Configuration
```python
from datason.config import get_development_config

config = get_development_config()
# - Caching disabled for maximum predictability
# - Metrics enabled for development insights
# - Preserves all type information for debugging
# - Human-readable output formats
```

## Cache Management

### Context Managers

Use context managers for scoped cache control:

```python
from datason import cache_scope, CacheScope
from datason.cache_manager import operation_scope, request_scope

# Explicit scope control
with cache_scope(CacheScope.PROCESS):
    # All operations use process-level caching
    result = datason.deserialize(data)

# Operation scope with automatic cleanup
with operation_scope():
    # Caches cleared before and after this block
    result = datason.deserialize(data)

# Request scope for web applications
with request_scope():
    # Request-local caching with automatic cleanup
    result = datason.deserialize(data)
```

### Manual Cache Control

```python
from datason import clear_caches, clear_all_caches

# Clear caches for current scope
clear_caches()

# Clear all caches across all scopes (for testing/debugging)
clear_all_caches()

# Legacy compatibility
datason.clear_deserialization_caches()  # Same as clear_caches()
```

### Cache Metrics

Monitor cache performance with built-in metrics:

```python
from datason.cache_manager import get_cache_metrics, reset_cache_metrics
from datason.config import CacheScope

# Enable metrics in configuration
config = SerializationConfig(cache_metrics_enabled=True)

# Get metrics for specific scope
metrics = get_cache_metrics(CacheScope.PROCESS)
print(f"Hit rate: {metrics[CacheScope.PROCESS].hit_rate:.2%}")
print(f"Total hits: {metrics[CacheScope.PROCESS].hits}")
print(f"Total misses: {metrics[CacheScope.PROCESS].misses}")

# Reset metrics
reset_cache_metrics(CacheScope.PROCESS)
```

## Configuration Options

### Cache-Specific Settings

```python
from datason.config import SerializationConfig, CacheScope

config = SerializationConfig(
    cache_scope=CacheScope.REQUEST,           # Cache scope
    cache_size_limit=2000,                    # Maximum cache entries
    cache_warn_on_limit=True,                 # Warn when limit reached
    cache_metrics_enabled=True,               # Enable performance metrics
)
```

### Size Limits and Warnings

The cache system includes built-in protection against memory bloat:

```python
import warnings

config = SerializationConfig(
    cache_size_limit=1000,        # Limit cache to 1000 entries
    cache_warn_on_limit=True,     # Warn when limit is reached
)

# When cache reaches limit, oldest entries are evicted (FIFO)
# and warnings are emitted if enabled
with warnings.catch_warnings():
    warnings.simplefilter("always")
    # Use datason operations that might trigger cache warnings
```

## Best Practices by Use Case

### 1. Long-running Web Applications

```python
from datason.config import get_web_api_config
from datason.cache_manager import request_scope

config = get_web_api_config()

# In your request handler
def handle_request(request_data):
    with request_scope():
        # Safe caching within request boundary
        return datason.deserialize(request_data, config=config)
```

### 2. Batch Processing Pipelines

```python
from datason.config import get_batch_processing_config
from datason import cache_scope, CacheScope

config = get_batch_processing_config()

def process_large_dataset(dataset):
    with cache_scope(CacheScope.PROCESS):
        results = []
        for batch in dataset:
            # Benefits from accumulated cache
            batch_results = [
                datason.deserialize(item, config=config)
                for item in batch
            ]
            results.extend(batch_results)
        return results
```

### 3. Real-time/Streaming Applications

```python
from datason.config import get_realtime_config

config = get_realtime_config()  # Operation-scoped for predictability

def process_stream_event(event_data):
    # Each event processed independently
    return datason.deserialize(event_data, config=config)
```

### 4. Model Training/Research

```python
from datason.config import get_research_config
from datason import cache_scope, CacheScope

config = get_research_config()

# For reproducible research
with cache_scope(CacheScope.OPERATION):
    # Ensures consistent behavior across runs
    training_data = [
        datason.deserialize(sample, config=config)
        for sample in dataset
    ]
```

### 5. Testing and Development

```python
from datason.config import get_development_config

config = get_development_config()  # Caching disabled

def test_deserialization():
    # Predictable behavior for testing
    result = datason.deserialize(test_data, config=config)
    assert expected_result == result
```

## Migration Guide

### From Existing Code

If you're upgrading from a previous version of datason:

```python
# Old code (still works)
result = datason.deserialize(data)
datason.clear_deserialization_caches()

# New code with explicit configuration
from datason.config import get_realtime_config

config = get_realtime_config()
result = datason.deserialize(data, config=config)
datason.clear_caches()  # or clear_all_caches() for thorough cleaning
```

### Gradual Migration

1. **Start with defaults**: The new system defaults to operation-scoped caching, which is safe
2. **Add configuration gradually**: Use preset configurations for common scenarios
3. **Monitor performance**: Enable metrics to understand cache behavior
4. **Optimize for your use case**: Choose appropriate cache scope based on your workflow

## Performance Considerations

### Cache Scope Performance Impact

| Scope | Performance | Memory Usage | Safety | Use Case |
|-------|-------------|--------------|--------|-----------|
| DISABLED | Baseline | Minimal | Highest | Development, Security |
| OPERATION | Low overhead | Minimal | High | Scripts, Real-time |
| REQUEST | Medium gain | Moderate | Medium | Web APIs, Services |
| PROCESS | High gain | Higher | Lower | Batch, ETL |

### Memory Management

- **Automatic eviction**: Caches use FIFO eviction when size limits are reached
- **Size limits**: Configurable per cache type with sensible defaults
- **Pool management**: Object pools are size-limited to prevent memory bloat
- **Context cleanup**: Request and operation scopes automatically clean up

### Monitoring and Debugging

```python
from datason.cache_manager import get_cache_metrics
from datason.config import CacheScope

# Enable metrics for performance monitoring
config = SerializationConfig(
    cache_scope=CacheScope.PROCESS,
    cache_metrics_enabled=True,
    cache_size_limit=5000,
)

# Monitor cache performance
def monitor_cache_performance():
    metrics = get_cache_metrics()
    for scope, metric in metrics.items():
        print(f"{scope.value}: {metric}")
        if metric.hit_rate < 0.5:
            print(f"Low hit rate for {scope.value}: {metric.hit_rate:.2%}")
```

## Troubleshooting

### Common Issues

1. **Memory Growth in Long-running Processes**
   ```python
   # Problem: Using process-scoped caching in long-running service
   # Solution: Switch to request-scoped caching
   config = get_web_api_config()  # Uses request scope
   ```

2. **Unexpected Cross-contamination**
   ```python
   # Problem: Different data sources affecting each other
   # Solution: Use operation or request scoping
   with operation_scope():
       # Isolated processing
       result = datason.deserialize(data)
   ```

3. **Cache Size Warnings**
   ```python
   # Problem: Cache size warnings in logs
   # Solution: Increase size limit or use more restrictive scope
   config = SerializationConfig(
       cache_size_limit=10000,  # Increase limit
       cache_warn_on_limit=False,  # Or disable warnings
   )
   ```

4. **Performance Regression**
   ```python
   # Problem: Slower than expected performance
   # Solution: Enable metrics and choose appropriate scope
   config = SerializationConfig(
       cache_scope=CacheScope.PROCESS,  # More aggressive caching
       cache_metrics_enabled=True,      # Monitor performance
   )
   ```

### Debug Mode

For debugging cache behavior:

```python
from datason.config import get_development_config
from datason.cache_manager import clear_all_caches, get_cache_metrics

# Use development config with caching disabled
config = get_development_config()

# Clear all caches to start fresh
clear_all_caches()

# Enable detailed metrics
config.cache_metrics_enabled = True

# Your code here...

# Check what happened
metrics = get_cache_metrics()
for scope, metric in metrics.items():
    print(f"Scope {scope.value}: {metric}")
```

## Security Considerations

### Multi-tenant Applications

Always use request-scoped or operation-scoped caching in multi-tenant environments:

```python
# SAFE: Request-scoped caching
with request_scope():
    user_data = datason.deserialize(request_data)

# UNSAFE: Process-scoped caching in multi-tenant app
# with cache_scope(CacheScope.PROCESS):  # DON'T DO THIS
#     user_data = datason.deserialize(request_data)
```

### High-security Environments

Consider disabling caching entirely for maximum predictability:

```python
from datason.config import get_development_config

# Disable all caching for security-sensitive applications
config = get_development_config()  # cache_scope=CacheScope.DISABLED
result = datason.deserialize(sensitive_data, config=config)
```

## Future Roadmap

The caching system is designed to be extensible. Future enhancements may include:

- **Custom cache backends**: Redis, memcached integration
- **Cache warming**: Pre-populate caches with common patterns
- **Advanced eviction policies**: LRU, LFU beyond simple FIFO
- **Distributed caching**: Share caches across process boundaries
- **Cache persistence**: Survive process restarts

## Summary

The configurable caching system in datason provides:

✅ **Four cache scopes** for different use cases  
✅ **Preset configurations** for common scenarios  
✅ **Automatic size limits** and memory management  
✅ **Performance metrics** for monitoring  
✅ **Context managers** for easy scope control  
✅ **Backwards compatibility** with existing code  
✅ **Production safety** with multi-tenant considerations  

Choose the right configuration for your use case, monitor performance with metrics, and enjoy the performance benefits while maintaining predictable behavior.
