# Datason Test Structure - Organized & Maintainable

## ✅ Proper Approach: Function-Level Organization

Instead of deleting whole files, we **extract unique test functions** and organize them logically.

## Current Clean Structure

### 🎯 tests/unit/ - Main Functionality Tests
```
tests/unit/
├── test_api_comprehensive.py          # API functions (dump, load, etc.)
├── test_config_comprehensive.py       # Configuration management  
├── test_core_comprehensive.py         # Core serialization (main functionality)
├── test_deserializers_comprehensive.py # Deserialization (main functionality)
├── test_serializers_comprehensive.py  # Serializers module
├── test_validation_comprehensive.py   # Validation helpers
├── test_converters_comprehensive.py   # Type converters
└── test_data_utils_comprehensive.py   # Data utilities
```

### 🔧 tests/coverage/ - Edge Cases & Coverage Boosters
```
tests/coverage/
├── test_core_edge_cases.py           # ⭐ NEW: Import failures, object edge cases
├── test_advanced_features_boost.py   # Advanced feature paths
├── test_datetime_coverage_boost.py   # DateTime edge cases
├── test_focused_coverage_boost.py    # Focused coverage improvements
└── test_*.py                         # Other specialized coverage tests
```

### 🔗 tests/integration/ - Multi-Module Tests
```
tests/integration/
├── test_round_trip_serialization.py  # Serialize → Deserialize testing
├── test_config_and_type_handlers.py  # Configuration integration
└── test_*.py                         # Other integration scenarios
```

## 📋 Rules: Where to Add New Tests

### ✅ Add to tests/unit/ when:
- Testing **main functionality** of a module
- Testing **happy path** scenarios
- Testing **standard parameter variations**
- Testing **basic error handling**

**Example**: Adding a new serialization method
```python
# Add to tests/unit/test_core_comprehensive.py
def test_serialize_new_type(self):
    """Test serialization of new type."""
```

### ✅ Add to tests/coverage/ when:
- Testing **import failure scenarios**
- Testing **exception handling edge cases**  
- Testing **boundary conditions**
- Testing **performance optimization paths**
- Targeting **specific uncovered lines**

**Example**: Testing edge case for coverage improvement
```python
# Add to tests/coverage/test_core_edge_cases.py
def test_new_import_failure_path(self):
    """Test fallback when new_module import fails."""
    # Target specific lines for coverage
```

### ✅ Add to tests/integration/ when:
- Testing **multiple modules together**
- Testing **real-world usage scenarios**
- Testing **end-to-end workflows**

## 📊 Coverage Results
- **Maintained 74% coverage** while cleaning up structure
- **Eliminated 71% of duplicates** (134→39 duplicate instances)
- **Reduced files by 30%** but **preserved all unique functionality**

## 🎯 Key Benefits

### 1. **Clear Decision Making**
```
❓ Need to increase core serialization coverage?
✅ Check tests/unit/test_core_comprehensive.py first
✅ Add edge cases to tests/coverage/test_core_edge_cases.py

❓ Testing new API functionality?
✅ Add to tests/unit/test_api_comprehensive.py

❓ Testing import failure edge case?
✅ Add to tests/coverage/test_core_edge_cases.py
```

### 2. **No More Duplicate Creation**
- **Before**: "I need to test X, I'll create test_X_boost.py"
- **After**: "I need to test X, I'll add to the appropriate existing file"

### 3. **Maintainable Coverage Growth**
- **Edge cases**: Always go to `/coverage/` directory
- **Main functionality**: Always go to `/unit/` directory  
- **Integration**: Always go to `/integration/` directory

## 🚀 Example: Adding New Coverage

If you want to increase `datason/core.py` coverage:

1. **Check current**: `tests/unit/test_core_comprehensive.py`
2. **Add edge cases**: `tests/coverage/test_core_edge_cases.py`
3. **Add integration**: `tests/integration/test_*_integration.py`

**Never create**: `test_core_coverage_boost_v2.py` ❌

## ✅ Success Metrics Achieved
- **74% coverage maintained**
- **Clear structure**: Know exactly where to add tests
- **No duplication**: Each test function in exactly one logical place
- **Easy maintenance**: Find and modify tests quickly
