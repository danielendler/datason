# Enhanced Type Support Implementation Plan

## 🎯 Current Status
- **Overall Success**: 69.1% (47/68 tests) ⬆️ +1.5% from cache fix
- **Target v0.7.5**: 85%+ success rate  
- **Gap to close**: 16% (need 11+ more passing tests)

## 📊 Prioritized Fix Strategy

### Phase 1: Quick Wins (Target: +10% success rate)
**Priority: HIGH - Fix auto-detection gaps**

#### 1.1 Pandas DataFrame Auto-Detection (4 failing tests → potential +6%)
- `dataframe_simple`: Add smart list-of-dicts → DataFrame detection
- `dataframe_orient_*`: Fix orientation-specific reconstruction
- **Impact**: 4 tests × 1.5% = 6% improvement

#### 1.2 NumPy Array Auto-Detection (4 failing tests → potential +6%)
- `array_1d`, `array_2d`, `array_float32`, `array_int64`: Add smart list → ndarray detection
- **Impact**: 4 tests × 1.5% = 6% improvement

#### 1.3 PyTorch Tensor Comparison Fix (2 failing tests → potential +3%)
- Fix tensor comparison logic (currently: "Boolean value of Tensor with more than one value is ambiguous")
- **Impact**: 2 tests × 1.5% = 3% improvement

#### 1.4 Nested Structure Verification (1 failing test → potential +1.5%)
- Fix audit script's verification logic for set → list conversions
- **Impact**: 1 test × 1.5% = 1.5% improvement

**Phase 1 Total**: +16.5% potential → **Target: 85%+ achieved** ✅

### Phase 2: ML Integration (Target: +5% success rate)
**Priority: MEDIUM - Complete ML round-trip support**

#### 2.1 Sklearn Model Reconstruction (4 failing tests)
- Fix metadata deserialization for unfitted/fitted models
- Enhance `_deserialize_with_type_metadata()` for sklearn objects

#### 2.2 Advanced PyTorch Support
- Enhance tensor attribute preservation (device, dtype, etc.)
- Add support for complex tensor operations

### Phase 3: Advanced Features (Target: +5% success rate)
**Priority: LOW - Edge cases and optimizations**

#### 3.1 DataFrame Orientation Mastery
- Complete support for all pandas DataFrame orientations
- Optimize for different use cases

#### 3.2 Advanced NumPy Support
- Enhanced dtype preservation
- Complex array structures

## 🛠️ Implementation Strategy

### Strategy 1: Enhanced Auto-Detection
**Add intelligence to `deserialize_fast()` without breaking hot path**

```python
# In _process_dict_optimized():
# Add checks for:
# 1. List-of-dicts → DataFrame pattern
# 2. Nested lists → ndarray pattern  
# 3. Specific ML object patterns
```

### Strategy 2: Improved Type Metadata Handling
**Enhance `_deserialize_with_type_metadata()` for better ML support**

```python
# Fix issues with:
# 1. Sklearn model reconstruction
# 2. PyTorch tensor attributes
# 3. Complex nested type preservation
```

### Strategy 3: Smart Verification
**Update audit script verification logic**

```python
# Handle expected type conversions:
# 1. set → list (acceptable without type hints)
# 2. tuple → list (acceptable without type hints)
# 3. Complex nested structures
```

## 📁 File Organization

### New Test Structure
```
tests/enhanced_types/
├── test_basic_type_enhancements.py      ✅ Created
├── test_pandas_auto_detection.py        🔄 Next
├── test_numpy_auto_detection.py         🔄 Next
├── test_ml_integration.py               🔄 Next
└── test_verification_improvements.py    🔄 Next
```

### Core Implementation Files
```
datason/
├── deserializers.py                     🔄 Main enhancements
├── core.py                             🔄 Type metadata improvements  
└── ml_serializers.py                   🔄 ML-specific fixes
```

## 🎯 Success Metrics

### v0.7.5 Targets
- **Overall Success**: 85%+ (58+ tests passing)
- **Basic Types**: 95%+ (maintain excellence)
- **Complex Types**: 95%+ (maintain improvement)
- **NumPy Types**: 90%+ (major improvement)
- **Pandas Types**: 70%+ (significant improvement)
- **ML Types**: 50%+ (basic functionality)

### v0.8.0 Targets  
- **Overall Success**: 95%+ (65+ tests passing)
- **All categories**: 90%+ success rates
- **Production-ready ML workflows**

### v0.8.5 Targets
- **Overall Success**: 99%+ (67+ tests passing)
- **Perfect round-trip support for all major ML libraries**
- **Complete edge case coverage**

## 🚀 Implementation Order

1. **Phase 1.1**: Pandas DataFrame auto-detection
2. **Phase 1.2**: NumPy array auto-detection  
3. **Phase 1.3**: PyTorch tensor comparison fix
4. **Phase 1.4**: Nested structure verification
5. **Phase 2**: ML integration improvements
6. **Phase 3**: Advanced features and edge cases

## 🔄 Testing Strategy

- **Continuous audit monitoring**: Run `deserialization_audit.py` after each fix
- **Regression prevention**: Maintain 1060+ passing integration tests
- **Performance validation**: Ensure no hot path degradation
- **Security verification**: Maintain 28/28 security tests passing
