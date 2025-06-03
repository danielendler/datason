# Plugin Architecture Testing Strategy

This document explains how datason implements and tests its plugin-style architecture where the core package has zero required dependencies but gains functionality when optional dependencies are available.

## 🎯 Architecture Goals

### Core Principle: Zero Dependencies
```bash
pip install datason  # ← Works with no additional dependencies!
```

### Enhanced Functionality
```bash
pip install datason[numpy]     # ← Adds NumPy support
pip install datason[pandas]    # ← Adds Pandas support  
pip install datason[ml]        # ← Adds ML library support
pip install datason[all]       # ← All optional features
```

## 🧪 Testing Strategy

### CI Matrix Testing
Our CI tests multiple dependency scenarios:

1. **`minimal`** - Core functionality only (no optional dependencies)
2. **`with-numpy`** - Core + NumPy support
3. **`with-pandas`** - Core + Pandas support
4. **`with-ml-deps`** - Core + ML dependencies (sklearn, etc.)
5. **`full`** - All dependencies (complete test suite)

### Test Categories

#### 🟢 Core Tests (Always Run)
- Basic JSON serialization
- Datetime handling
- UUID handling
- Error handling
- Security features

**Files**: `tests/core/test_core.py`, `tests/core/test_deserializers.py`, `tests/core/test_converters.py`, `tests/core/test_security.py`, `tests/core/test_edge_cases.py`, `tests/core/test_circular_references.py`, `tests/core/test_dataframe_orientation_regression.py`

#### 🟡 Feature Tests (Conditional)
- NumPy array serialization
- Pandas DataFrame handling
- ML model serialization
- Chunked streaming
- Template deserialization
- Auto-detection and metadata

**Files**: `tests/features/test_ml_serializers.py`, `tests/features/test_chunked_streaming.py`, `tests/features/test_auto_detection_and_metadata.py`, `tests/features/test_template_deserialization.py`

#### 🔵 Integration Tests (Multi-component)
- Configuration system integration
- Optional dependency integration
- Pickle bridge functionality

**Files**: `tests/integration/test_config_and_type_handlers.py`, `tests/integration/test_optional_dependencies.py`, `tests/integration/test_pickle_bridge.py`

#### ⚪ Coverage Tests (Edge cases)
- Coverage boost for specific modules
- Edge case testing
- Error condition testing

**Files**: `tests/coverage/test_*_coverage_boost.py`

#### 🔴 Performance Tests (Separate Pipeline)
- Benchmark tests
- Performance regression detection
- Memory usage analysis

**Files**: `tests/benchmarks/test_*_benchmarks.py` (runs in separate performance pipeline)

## 📝 Test Markers

Use pytest markers to categorize tests:

```python
import pytest
from conftest import requires_numpy, requires_pandas, requires_sklearn

@pytest.mark.core
def test_basic_serialization():
    """Core functionality - always runs."""
    pass

@requires_numpy
@pytest.mark.numpy
def test_numpy_arrays():
    """Only runs when numpy is available."""
    pass

@requires_pandas  
@pytest.mark.pandas
def test_pandas_dataframes():
    """Only runs when pandas is available."""
    pass

@pytest.mark.fallback
def test_numpy_fallback():
    """Tests fallback when numpy isn't available."""
    with patch("datason.core.np", None):
        # Test fallback behavior
        pass
```

## 🚀 Running Tests Locally

### Test Minimal Install
```bash
# Create clean environment
python -m venv test_minimal
source test_minimal/bin/activate
pip install -e .
pip install pytest pytest-cov

# Run core tests only
pytest tests/core/ -v
```

### Test With Specific Dependencies
```bash
# Test with numpy
pip install numpy
pytest tests/core/ tests/features/test_ml_serializers.py -v

# Test with pandas
pip install pandas  
pytest tests/core/ tests/features/test_auto_detection_and_metadata.py tests/features/test_chunked_streaming.py tests/features/test_template_deserialization.py -v

# Test ML features
pip install numpy pandas scikit-learn
pytest tests/core/ tests/features/ tests/integration/test_optional_dependencies.py -v
```

### Test Full Suite
```bash
pip install -e ".[dev]"
pytest tests/core/ tests/features/ tests/integration/ tests/coverage/ -v
```

### Test Performance (Separate)
```bash
# Performance tests run in separate pipeline
pytest tests/benchmarks/ --benchmark-only -v
```

## 📋 CI Test Matrix

Each CI job tests a specific scenario:

| Job | Dependencies | Tests Run | Purpose |
|-----|-------------|-----------|---------|
| `minimal` | None | tests/core/ | Verify zero-dependency functionality |
| `with-numpy` | numpy | tests/core/ + ML features | Basic array support |
| `with-pandas` | pandas | tests/core/ + data features | DataFrame support |
| `with-ml-deps` | numpy, pandas, sklearn | tests/core/ + tests/features/ + integration | ML model serialization |
| `full` | All deps | tests/core/ + tests/features/ + tests/integration/ + tests/coverage/ | Complete functional testing |
| `performance` | All deps | tests/benchmarks/ (separate pipeline) | Performance regression tracking |

## 🔧 Adding New Optional Features

When adding support for a new optional dependency:

1. **Add to `pyproject.toml`**:
   ```toml
   [project.optional-dependencies]
   newfeature = ["newlibrary>=1.0.0"]
   ```

2. **Add conditional import**:
   ```python
   try:
       import newlibrary
       HAS_NEWLIBRARY = True
   except ImportError:
       HAS_NEWLIBRARY = False
   ```

3. **Add to CI matrix**:
   ```yaml
   - name: "with-newfeature"
     install: "pip install -e . && pip install newlibrary"
     description: "Core + NewLibrary support"
   ```

4. **Add tests with proper markers**:
   ```python
   @pytest.mark.skipif(not HAS_NEWLIBRARY, reason="newlibrary not available")
   @pytest.mark.newlibrary
   def test_newlibrary_feature():
       pass
   ```

## ✅ Benefits

- **Lightweight**: Users only install what they need
- **Robust**: Core functionality always works
- **Flexible**: Easy to add new optional features
- **Well-tested**: All scenarios covered in CI
- **User-friendly**: Clear error messages when deps missing

This architecture ensures datason works for everyone while providing enhanced functionality for users who need it.
