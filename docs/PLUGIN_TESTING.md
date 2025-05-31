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

**Files**: `test_core.py`, `test_deserializers.py`, `test_converters.py`, `test_data_utils.py`

#### 🟡 Optional Dependency Tests (Conditional)
- NumPy array serialization
- Pandas DataFrame handling
- ML model serialization
- Image processing

**Files**: `test_ml_serializers.py`, `test_optional_dependencies.py`

#### 🔴 Fallback Behavior Tests (Mock-based)
- Test what happens when optional deps aren't available
- Verify graceful degradation
- Test import error handling

**Files**: `test_targeted_coverage_boost.py`, `test_fallback_behavior.py`

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
pytest tests/test_core.py tests/test_deserializers.py -v
```

### Test With Specific Dependencies
```bash
# Test with numpy
pip install numpy
pytest -m "core or numpy" -v

# Test with pandas
pip install pandas  
pytest -m "core or pandas" -v

# Test ML features
pip install numpy pandas scikit-learn
pytest -m "core or ml" -v
```

### Test Full Suite
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 📋 CI Test Matrix

Each CI job tests a specific scenario:

| Job | Dependencies | Tests Run | Purpose |
|-----|-------------|-----------|---------|
| `minimal` | None | Core only | Verify zero-dependency functionality |
| `with-numpy` | numpy | Core + NumPy | Basic array support |
| `with-pandas` | pandas | Core + Pandas | DataFrame support |
| `with-ml-deps` | numpy, pandas, sklearn | Core + ML | ML model serialization |
| `full` | All deps | Complete suite | Integration testing |

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
