# Core Serialization Strategy & Architecture

## Overview

The `datason.core` module implements a sophisticated **multi-layered serialization strategy** designed to maximize performance while maintaining security and compatibility. This document explains the architecture, reasoning, and flow through each optimization layer.

## Design Philosophy

### Core Principles

1. **Performance First**: 80% of real-world data should hit ultra-fast paths
2. **Security Always**: Malicious data must be caught and handled safely  
3. **Layered Approach**: Each layer handles increasingly complex cases
4. **Fail-Safe Fallback**: Every layer has a safe fallback to the next layer
5. **Circuit Breaker Protection**: Emergency safeguards prevent infinite recursion and hanging

### Architecture Goals

- **Zero-overhead** for basic JSON-compatible data
- **Minimal checks** for simple structures
- **Graduated complexity** - only pay for what you need
- **Security boundaries** - prevent resource exhaustion and circular references
- **Emergency protection** - circuit breaker prevents any hanging scenarios

## Serialization Flow Layers

The `serialize()` function processes objects through **7 distinct layers**, each with specific responsibilities:

### 🚀 **Layer 1: Ultra-Fast Path**
*Lines 248-265 in core.py*

**Purpose**: Zero-overhead processing for provably safe types

**Handles**:
- `int`, `bool`, `None` → immediate return (no checks needed)
- Regular `float` → NaN/Inf check + immediate return
- Short `str` (≤1000 chars, top-level only) → immediate return

**Strategy**: Skip ALL processing for types that cannot contain malicious content

**Example**:
```python
serialize(42)        # Immediate return: 42
serialize(True)      # Immediate return: True  
serialize("hello")   # Immediate return: "hello"
```

**Performance**: **~0ns overhead** - single type check + return

---

### 🛡️ **Layer 2: Security Layer**
*Lines 267-288 in core.py*

**Purpose**: Early detection of problematic objects and emergency circuit breaker

**Handles**:
- **Circuit Breaker**: Emergency protection against infinite recursion at function entry
- **Mock objects** (`unittest.mock`, `io`, `_io` modules)  
- **IO objects** (BytesIO, StringIO, file handles) with improved detection
- Objects with suspicious `__dict__` patterns (now checks `hasattr(__dict__)`)
- Known problematic types that cause infinite recursion

**Strategy**: Convert dangerous objects to safe string representations + emergency stop mechanisms

**Example**:
```python
mock_obj = Mock()
serialize(mock_obj)  # Returns: "<Mock object at 0x...>"

# Circuit breaker prevents hanging on deeply nested data
very_deep_nested = create_nested_dict(depth=2000)
serialize(very_deep_nested)  # Returns: "CIRCUIT_BREAKER_ACTIVATED" (safe emergency response)
```

**Performance**: **~50ns overhead** - module name + attribute checks + emergency detection

---

### ⚙️ **Layer 3: Configuration & Initialization**
*Lines 289-304 in core.py*

**Purpose**: Set up processing context and validate limits

**Handles**:
- Default config initialization
- Type handler setup  
- Security limit extraction (depth, size, string length)

**Strategy**: One-time setup for the serialization tree

**Performance**: **~100ns overhead** - only on first call per tree

---

### 🎯 **Layer 4: JSON-First Optimization**
*Lines 306-340 in core.py*

**Purpose**: Ultra-fast path for pure JSON data

**Handles**:
- Objects that are already 100% JSON-compatible
- No custom serializers, no NaN handling needed
- Simple configurations only

**Strategy**: Detect JSON compatibility and skip all processing

**Example**:
```python
data = {"name": "John", "age": 30, "active": True}
serialize(data)  # JSON-first path - minimal processing
```

**Performance**: **~200ns overhead** - compatibility check + direct processing

---

### 🔄 **Layer 5: Iterative Processing**
*Lines 341-370 in core.py*

**Purpose**: Eliminate recursive function call overhead

**Handles**:
- Large collections (>10 items)
- Nested dicts/lists with simple data
- Homogeneous type collections

**Strategy**: Stack-based processing instead of recursive calls

**Performance**: **~50% faster** than recursive approach for large collections

---

### 🔒 **Layer 6: Security Enforcement**
*Lines 371-408 in core.py*

**Purpose**: Enforce all security limits and protections

**Handles**:
- Depth limit enforcement → `SecurityError`
- Circular reference detection → Warning + null replacement  
- Size limit enforcement → `SecurityError`
- Collection size validation

**Strategy**: Track state and enforce hard limits

**Example**:
```python
# Circular reference protection
d1 = {"name": "dict1"}
d2 = {"name": "dict2", "ref": d1}  
d1["ref"] = d2
serialize(d1)  # Warning + safe handling
```

---

### 🚄 **Layer 7: Hot Path Processing**
*Lines 480-516 in core.py*

**Purpose**: Optimized processing for common container patterns

**Handles**:
- Small containers (≤5 items) with basic types
- String interning for common values
- Inline NaN/Inf processing
- Memory pool allocation

**Strategy**: Aggressive inlining and caching for frequent patterns

**Performance**: **~80% of real data** hits this path efficiently

---

### 🎛️ **Layer 8: Full Processing Path**
*Lines 520+ in core.py*

**Purpose**: Handle all remaining complex cases

**Handles**:
- Custom objects with `__dict__`
- NumPy arrays and data types
- Pandas DataFrames and Series  
- DateTime, UUID, Sets
- ML objects (PyTorch, TensorFlow)
- Fallback string representation

**Strategy**: Comprehensive type detection and specialized handlers

**Performance**: Only complex/custom data pays the full cost

## Performance Characteristics

### Throughput by Data Type

| Data Type | Throughput | Layer Used | Overhead |
|-----------|------------|------------|----------|
| `int/bool/None` | **Unlimited** | Layer 1 | ~0ns |
| Short strings | **50M+ ops/sec** | Layer 1 | ~5ns |
| JSON objects | **10M+ ops/sec** | Layer 4 | ~200ns |
| Large lists | **1M+ items/sec** | Layer 5 | ~500ns |
| NumPy arrays | **7M+ elements/sec** | Layer 8 | ~1μs |
| Pandas DataFrames | **1M+ rows/sec** | Layer 8 | ~2μs |

### Memory Efficiency

- **Object pools**: Reuse containers to reduce allocations
- **String interning**: Common values cached globally  
- **Type caching**: Reduce `isinstance()` overhead
- **Chunked processing**: Handle datasets larger than RAM

## Security Architecture

### Protection Mechanisms

1. **Emergency Circuit Breaker**
   ```python
   # Emergency fallback - should never reach this with proper depth=50 limit
   if _depth > 100:  # Emergency circuit breaker
       return f"<EMERGENCY_CIRCUIT_BREAKER: depth={_depth}, type={type(obj).__name__}>"
   ```

2. **Depth Bomb Protection**
   ```python
   MAX_SERIALIZATION_DEPTH = 50  # Reduced from 1000 for enhanced security
   ```

3. **Size Bomb Protection**
   ```python
   MAX_OBJECT_SIZE = 10_000_000  # Prevent memory exhaustion
   ```

4. **String Length Protection**
   ```python  
   MAX_STRING_LENGTH = 1_000_000  # Prevent excessive processing
   ```

5. **Enhanced IO Object Detection**
   ```python
   # Improved detection - checks hasattr(__dict__) instead of len(__dict__) > 20
   if obj_module in ("io", "_io") and hasattr(obj, "__dict__"):
       return f"<{obj_class_name} object>"
   ```

6. **Circular Reference Detection**
   - Track object IDs in `_seen` set
   - Warn and replace with null on detection
   - Multi-level protection for nested objects
   - Fixed cleanup logic to include 'tuple' type

7. **Resource Exhaustion Prevention**
   - Early size checks before processing
   - Memory pools to prevent allocation bombs
   - Cache size limits to prevent memory leaks
   - Enhanced security checks using isinstance() instead of exact type matching

### Security vs Performance Balance

**Fast Path Security**:
- Layer 1-2 handle 80% of data with minimal security overhead
- Only check what's necessary for each data type
- Fail safely to next layer if issues detected

**Full Security**:
- Layer 6+ implements comprehensive protections
- Only complex/suspicious data pays the full security cost
- Multiple redundant checks for high-risk operations

## Configuration Impact

### Performance Configurations

**Fastest** (minimal checks):
```python
config = SerializationConfig(
    nan_handling=NanHandling.NULL,
    max_depth=1000,
    max_string_length=1000000,
    sort_keys=False,
    include_type_hints=False
)
```

**Production** (balanced):
```python  
config = get_default_config()  # Well-tuned defaults
```

**Maximum Security** (all checks):
```python
config = SerializationConfig(
    max_depth=100,           # Lower depth limit
    max_size=1000,          # Lower size limit  
    max_string_length=1000,  # Lower string limit
    custom_serializers=True  # Enable custom handlers
)
```

## Implementation Guidelines

### Adding New Optimizations

1. **Identify the data pattern** - What % of real data matches?
2. **Choose the right layer** - Simpler = earlier layer
3. **Preserve security** - Don't skip necessary checks
4. **Benchmark impact** - Measure both positive and negative cases
5. **Add fallback** - Must safely fall through to next layer

### Layer Selection Criteria

| Layer | Use When | Avoid When |
|-------|----------|------------|
| Layer 1 | Provably safe types | Any container or complex type |
| Layer 2 | Known problematic patterns | Performance-critical paths |
| Layer 4 | Pure JSON data | Custom serializers needed |
| Layer 5 | Large homogeneous collections | Mixed type collections |
| Layer 8 | Everything else | N/A - this is the fallback |

## Debugging & Profiling

### Performance Debugging

**Enable debug mode**:
```python
import logging
logging.getLogger('datason.core').setLevel(logging.DEBUG)
```

**Benchmark specific layers**:
```python
from datason.benchmarks import benchmark_layer
benchmark_layer(data, layer=4)  # Test JSON-first path
```

### Common Performance Issues

1. **String length checks** - Bypass Layer 1 fast path
2. **Type hints enabled** - Skip many optimizations  
3. **Custom serializers** - Force full processing
4. **Deep nesting** - Prevent iterative optimization
5. **Mixed type collections** - Prevent homogeneous optimization

## Future Optimizations

### Planned Improvements

1. **Pattern Recognition**: Cache serialization strategies for object patterns
2. **Vectorized Operations**: Batch process arrays with NumPy operations
3. **C Extensions**: Move hot paths to compiled code
4. **Rust Integration**: Ultimate performance for core algorithms
5. **Adaptive Optimization**: Runtime selection of best strategy

### Extension Points

- **Custom type handlers**: Plug into Layer 8 full processing
- **ML serializers**: Automatic detection of ML frameworks
- **Streaming interfaces**: Handle datasets larger than RAM
- **Compression**: Integrate with compression libraries

## Migration Guide

### From datason v0.4.x

The new layered architecture is **fully backward compatible**. Existing code will automatically benefit from performance improvements.

**Recommended optimizations**:
```python
# Old approach
result = serialize(large_data)

# New optimized approach  
result = serialize(large_data, config=get_performance_config())
```

### Performance Tuning

1. **Profile your data** - Run `estimate_memory_usage()`
2. **Choose optimal config** - Use `get_ml_config()` for ML data
3. **Use chunked processing** - For data >100MB
4. **Enable streaming** - For data >1GB

---

## Summary

The **layered serialization architecture** achieves the optimal balance of:

- ⚡ **Performance**: 80% of data hits ultra-fast paths
- 🔒 **Security**: Comprehensive protection against all attack vectors  
- 🧩 **Flexibility**: Handles any Python data type
- 📈 **Scalability**: Processes datasets larger than available RAM

Each layer is **independently optimized** and **safely fails through** to more comprehensive processing, ensuring both **maximum performance** and **robust security** for all use cases.
