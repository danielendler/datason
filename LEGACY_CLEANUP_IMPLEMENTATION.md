# Legacy Cleanup Implementation Plan

## 🎯 Phase 1: Immediate Safe Cleanup (v0.7.6)

### 1. Remove Legacy Cache Function Alias

**Files to Modify**:
- `datason/__init__.py` - Remove alias export
- `30+ test files` - Replace function calls
- `deserialization_audit.py` - Update function call
- `datason/deserializers.py` - Remove old function

**Implementation Steps**:

#### Step 1: Update datason/__init__.py
```python
# REMOVE these lines:
clear_caches as clear_deserialization_caches,  # Legacy alias
"clear_deserialization_caches",  # From __all__

# KEEP only:
"clear_caches",
```

#### Step 2: Update all test files (30+ files)
```python
# REPLACE all instances of:
from datason.deserializers import _clear_deserialization_caches
_clear_deserialization_caches()

# WITH:
import datason
datason.clear_caches()
```

#### Step 3: Update datason/deserializers.py
```python
# REMOVE entire function:
def _clear_deserialization_caches() -> None:
    """Clear all deserialization caches.

    This function clears pattern recognition caches and object pools
    to ensure clean state between operations. Primarily used for testing.
    """
    # [entire function body - remove]
```

#### Step 4: Simplify clear_caches function
```python
# CURRENT:
def clear_caches() -> None:
    """Clear all caches - new name for _clear_deserialization_caches."""
    _clear_deserialization_caches()

# SIMPLIFIED:
def clear_caches() -> None:
    """Clear all deserialization caches and object pools."""
    # Move the actual implementation here directly
    global _PARSED_OBJECT_CACHE, _STRING_PATTERN_CACHE, _TYPE_CACHE
    global _RESULT_DICT_POOL, _RESULT_LIST_POOL

    # Clear pattern caches
    _PARSED_OBJECT_CACHE.clear()
    _STRING_PATTERN_CACHE.clear()
    _TYPE_CACHE.clear()

    # Clear object pools
    _RESULT_DICT_POOL.clear()
    _RESULT_LIST_POOL.clear()

    # Clear cache manager if available
    try:
        from .cache_manager import clear_all_caches_internal
        clear_all_caches_internal()
    except ImportError:
        pass
```

### 2. Remove Redundant Configuration Presets

**Analysis of Current Presets**:
```python
# CORE (keep these):
get_ml_config()          # ML workflows - high usage
get_api_config()         # API responses - high usage  
get_strict_config()      # Type preservation - useful
get_performance_config() # Speed optimization - useful

# NICHE (remove these):
get_financial_config()   # Too specific, can be custom
get_time_series_config() # Too specific, can be custom
get_inference_config()   # Redundant with ml_config
get_research_config()    # Redundant with strict_config
get_logging_config()     # Too specific, can be custom
get_batch_processing_config()  # Redundant with performance_config
get_web_api_config()     # Redundant with api_config
get_realtime_config()    # Too specific, can be custom
get_development_config() # Useful for debugging (keep?)
```

**Cleanup Implementation**:
```python
# In datason/config.py - REMOVE these functions:
def get_financial_config() -> SerializationConfig: ...      # REMOVE
def get_time_series_config() -> SerializationConfig: ...    # REMOVE  
def get_inference_config() -> SerializationConfig: ...      # REMOVE
def get_research_config() -> SerializationConfig: ...       # REMOVE
def get_logging_config() -> SerializationConfig: ...        # REMOVE
def get_batch_processing_config() -> SerializationConfig: ...  # REMOVE
def get_web_api_config() -> SerializationConfig: ...        # REMOVE
def get_realtime_config() -> SerializationConfig: ...       # REMOVE

# KEEP ONLY core presets:
get_ml_config()          # Essential for ML workflows
get_api_config()         # Essential for APIs
get_strict_config()      # Essential for type preservation
get_performance_config() # Essential for optimization
get_development_config() # Useful for debugging
```

### 3. Add Deprecation Warnings to Legacy ML Formats

**Implementation in datason/deserializers.py**:
```python
# In _deserialize_with_type_metadata function:
def _deserialize_with_type_metadata(obj: Dict[str, Any]) -> Any:
    """Handle objects with type metadata for reconstruction."""

    # Check for legacy _type format
    if "_type" in obj and "__datason_type__" not in obj:
        warnings.warn(
            "Legacy '_type' format is deprecated and will be removed in v0.8.0. "
            "Use '__datason_type__' format instead.",
            DeprecationWarning,
            stacklevel=3
        )
        # Continue processing but log the warning

    # [rest of function unchanged]
```

---

## 🎯 Phase 2: Major Cleanup (v0.8.0)

### 1. Remove Legacy ML Format Support Entirely

**Files to Modify**:
- `datason/deserializers.py` - Remove all legacy format handlers
- `datason/core.py` - Remove legacy format conversions

**Implementation**:
```python
# REMOVE entire section from deserializers.py:
# ENHANCED LEGACY TYPE FORMATS (priority 2) - Handle older serialization formats
# Lines 527-643 (approximately 116 lines of legacy support)

# REMOVE from core.py:
# Convert legacy ML format to new type metadata format
# Lines 795-808 (legacy type conversion)
```

### 2. Simplify API Based on Usage Patterns

**Add High-Level Convenience Functions**:
```python
# Add to datason/__init__.py:
def serialize_with_types(obj: Any, **kwargs: Any) -> Dict[str, Any]:
    """Serialize with automatic type metadata for perfect round-trips.

    Equivalent to: serialize(obj, config=SerializationConfig(include_type_hints=True))
    """
    from .config import SerializationConfig
    config = SerializationConfig(include_type_hints=True, **kwargs)
    return serialize(obj, config=config)

def deserialize_with_types(data: Any) -> Any:
    """Deserialize with automatic type reconstruction.

    Handles both new format (__datason_type__) and automatic type detection.
    """
    from .deserializers import deserialize_fast
    return deserialize_fast(data, auto_detect_types=True)

# Add to __all__:
"serialize_with_types",
"deserialize_with_types",
```

### 3. Remove Unused Configuration Options

**Audit and Remove**:
```python
# In SerializationConfig, remove rarely used options:
auto_detect_types: bool = False  # REMOVE - replace with function parameter
check_if_serialized: bool = False  # REMOVE - always do this optimization

# Simplify to essential options only:
@dataclass
class SerializationConfig:
    # Core formatting
    date_format: DateFormat = DateFormat.ISO
    custom_date_format: Optional[str] = None
    dataframe_orient: DataFrameOrient = DataFrameOrient.RECORDS

    # Value handling  
    nan_handling: NanHandling = NanHandling.NULL
    type_coercion: TypeCoercion = TypeCoercion.SAFE

    # Type preservation
    preserve_decimals: bool = True
    preserve_complex: bool = True
    include_type_hints: bool = False

    # Security limits
    max_depth: int = 50
    max_size: int = 100_000
    max_string_length: int = 1_000_000

    # Caching
    cache_scope: CacheScope = CacheScope.OPERATION
    cache_size_limit: int = 1000
    cache_metrics_enabled: bool = False

    # Extensibility
    custom_serializers: Optional[Dict[type, Callable[[Any], Any]]] = None

    # Output formatting
    sort_keys: bool = False
    ensure_ascii: bool = False
```

---

## 🎯 Phase 3: API Modernization (v0.8.5)

### 1. Introduce Modern High-Level API

**New Primary Interface**:
```python
# datason/modern_api.py (new file)
"""Modern high-level API for datason."""

from typing import Any, Dict, Optional
from .config import SerializationConfig
from .core import serialize as _serialize_core
from .deserializers import deserialize_fast

def dump(obj: Any, *, with_types: bool = False, **config_options) -> Dict[str, Any]:
    """Modern serialize function with clean interface.

    Args:
        obj: Object to serialize
        with_types: Include type metadata for round-trip support
        **config_options: Configuration options (date_format, nan_handling, etc.)

    Returns:
        JSON-compatible dictionary

    Example:
        >>> data = dump(my_model, with_types=True, date_format='unix')
    """
    config = SerializationConfig(include_type_hints=with_types, **config_options)
    return _serialize_core(obj, config=config)

def load(data: Any, *, auto_detect: bool = True) -> Any:
    """Modern deserialize function with smart type detection.

    Args:
        data: JSON-compatible data to deserialize
        auto_detect: Enable automatic type detection

    Returns:
        Reconstructed Python object

    Example:
        >>> obj = load(json_data, auto_detect=True)
    """
    return deserialize_fast(data, auto_detect_types=auto_detect)

# Add convenience functions
def dump_json(obj: Any, **kwargs) -> str:
    """Serialize to JSON string."""
    import json
    return json.dumps(dump(obj, **kwargs))

def load_json(json_str: str, **kwargs) -> Any:
    """Load from JSON string."""
    import json
    return load(json.loads(json_str), **kwargs)
```

### 2. Deprecate Complex Low-Level API

**Add Deprecation Warnings**:
```python
# In datason/__init__.py, add warnings to complex functions:
def serialize_with_config(obj: Any, **kwargs: Any) -> Any:
    """Serialize with quick configuration options.

    DEPRECATED: Use dump() instead for cleaner API.
    """
    warnings.warn(
        "serialize_with_config is deprecated. Use datason.dump() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # [existing implementation]
```

### 3. Provide Migration Guide

**Add to documentation**:
```markdown
# Migration Guide v0.7 → v0.8

## API Changes

### Old API (still works but deprecated):
```python
from datason.config import SerializationConfig
config = SerializationConfig(include_type_hints=True, date_format='unix')
result = datason.serialize(data, config=config)
reconstructed = datason.deserialize_fast(result)
```

### New API (recommended):
```python
# Simple serialization
result = datason.dump(data, with_types=True, date_format='unix')
reconstructed = datason.load(result)

# JSON string handling
json_str = datason.dump_json(data, with_types=True)
reconstructed = datason.load_json(json_str)
```

## Removed Features
- Legacy cache function `clear_deserialization_caches()` → use `clear_caches()`
- Legacy ML type formats (`_type`) → use `__datason_type__`
- Redundant config presets → use core presets or custom configs
```

---

## 🧪 Testing Strategy for Cleanup

### Phase 1 Testing:
```bash
# Test cache function cleanup
python -m pytest tests/ -v -k "cache"

# Test configuration preset usage
python -m pytest tests/ -v -k "config"

# Ensure no breaking changes to core functionality
python -m pytest tests/ --tb=short
```

### Phase 2 Testing:
```bash
# Test legacy format removal doesn't break new formats
python -m pytest tests/integration/ -v

# Performance regression testing
python -m pytest tests/performance/ -v

# Round-trip testing with type metadata
python -m pytest tests/integration/test_round_trip_serialization.py -v
```

### Phase 3 Testing:
```bash
# Test new modern API
python -m pytest tests/test_modern_api.py -v

# Backward compatibility testing
python -m pytest tests/test_backward_compatibility.py -v
```

---

## 📊 Impact Assessment

### Benefits of Cleanup:
1. **Reduced Complexity**: 40% fewer configuration options
2. **Cleaner API**: Modern high-level functions for common use cases
3. **Better Performance**: Remove legacy code paths
4. **Easier Maintenance**: Less code to maintain and test
5. **Clearer Documentation**: Simpler API surface to document

### Risks:
1. **Breaking Changes**: Some existing code may need updates
2. **Migration Effort**: Need to update test files and scripts
3. **Lost Functionality**: Some niche presets removed

### Mitigation:
1. **Personal Use**: Breaking changes acceptable
2. **Phased Approach**: Gradual cleanup over 3 phases
3. **Testing**: Comprehensive test suite ensures core functionality preserved
4. **Documentation**: Clear migration guide for any breaking changes

---

## ✅ Implementation Checklist

### Phase 1 (Immediate - v0.7.6):
- [ ] Remove `clear_deserialization_caches` alias from `__init__.py`
- [ ] Update all test files to use `clear_caches()`
- [ ] Remove old cache function from `deserializers.py`
- [ ] Remove redundant configuration presets (8 functions)
- [ ] Add deprecation warnings to legacy ML formats
- [ ] Run full test suite to ensure no regressions

### Phase 2 (Major - v0.8.0):
- [ ] Remove all legacy ML format support (116 lines)
- [ ] Add high-level convenience functions
- [ ] Simplify SerializationConfig class
- [ ] Update documentation for API changes
- [ ] Run comprehensive testing including performance tests

### Phase 3 (Modernization - v0.8.5):
- [ ] Create modern API module
- [ ] Add deprecation warnings to complex low-level functions
- [ ] Create migration guide documentation
- [ ] Add backward compatibility test suite
- [ ] Final testing and validation

**Estimated Effort**: 2-3 days for Phase 1, 1 week each for Phases 2-3

**Success Criteria**: All tests pass, API simplified, performance maintained, documentation updated
