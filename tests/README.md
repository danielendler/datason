# Datason Test Suite Organization

This document describes the reorganized test structure for optimal development workflow and CI performance.

> **📊 Performance Benchmarking**: Major performance benchmarking is now handled by the external [datason-benchmarks](https://github.com/danielendler/datason-benchmarks) repository, which runs automatically on every PR via `.github/workflows/pr-performance-check.yml`.

## 🏗️ Test Structure

The test suite is organized into logical categories for better performance and maintainability:

```
tests/
├── unit/           # Unit tests for individual modules (~30-45 seconds)
│   ├── test_api_*.py                   # API module tests
│   ├── test_core_*.py                  # Core functionality tests
│   ├── test_datetime_*.py              # DateTime utilities tests
│   ├── test_deserializers_*.py         # Deserialization tests
│   ├── test_utils_*.py                 # Utility function tests
│   └── test_*_comprehensive.py         # Comprehensive module tests
│
├── integration/    # Integration tests (~15-25 seconds)
│   ├── test_auto_detection_and_metadata.py  # Auto-detection
│   ├── test_chunked_streaming.py       # Streaming/chunking features
│   ├── test_ml_serializers.py          # ML library integrations
│   ├── test_modern_api.py              # Modern API integration
│   ├── test_pickle_bridge.py           # Pickle bridge functionality
│   └── test_template_deserializer.py   # Template deserialization
│
├── edge_cases/     # Edge case tests (~10-15 seconds)
│   ├── test_core_edge_cases.py         # Core edge cases
│   ├── test_datetime_coverage_boost.py # DateTime edge cases
│   ├── test_integrity_and_security.py  # Security edge cases
│   └── test_uuid_edge_cases.py         # UUID edge cases
│
├── performance/    # Specific performance tests (kept locally)
│   └── test_idempotency_performance.py # Idempotency performance tests
│
├── conftest.py     # Shared test configuration
└── README.md       # This documentation
```

## 🚀 Quick Start

### Fast Development Testing (Recommended)
```bash
# Run fast core tests only (~7-10 seconds)
./scripts/run_tests.sh fast

# Or using pytest directly:
python -m pytest tests/core -m "core and not slow" --maxfail=5 --tb=short
```

### Full Functionality Testing  
```bash
# Run all tests except benchmarks (~30-60 seconds)
./scripts/run_tests.sh full

# Or using pytest directly:
python -m pytest tests/core tests/features tests/integration -m "not benchmark"
```

### Coverage Testing
```bash
# Run coverage boost tests to improve coverage metrics
./scripts/run_tests.sh coverage

# Or using pytest directly:
python -m pytest tests/coverage
```

### Performance Testing
```bash
# Run local performance tests for specific functionality
python -m pytest tests/performance/

# Note: Main performance benchmarking is now handled by the external
# datason-benchmarks repository via automated PR performance checks
```

### Complete Testing
```bash
# Run all local tests (unit, integration, edge cases, performance)
./scripts/run_tests.sh all

# Or using pytest directly:
python -m pytest tests/
```

## 📊 Performance Improvements

The reorganized structure provides significant performance improvements:

| Test Category | Test Count | Execution Time | Use Case |
|---------------|------------|----------------|----------|
| **Fast Core** | 137 tests | ~7-10 seconds | Development, quick validation |
| **Full Suite** | ~400 tests | ~30-60 seconds | Pre-commit, CI main |
| **Coverage** | ~200 tests | ~10-30 seconds | Coverage improvement |
| **Performance** | ~5 tests | ~5-10 seconds | Local performance validation |
| **Complete** | ~450 tests | ~60-90 seconds | Release validation |

**Previous:** All tests took ~103 seconds  
**Now:** Fast tests take ~7 seconds (93% faster!)

## 🏷️ Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.core` - Core functionality (no optional dependencies)
- `@pytest.mark.features` - Feature-specific tests  
- `@pytest.mark.integration` - Integration scenarios
- `@pytest.mark.coverage` - Coverage boost tests
- `@pytest.mark.performance` - Local performance tests
- `@pytest.mark.slow` - Long-running tests (excluded from fast runs)

### Dependency Markers
- `@pytest.mark.numpy` - Requires numpy
- `@pytest.mark.pandas` - Requires pandas  
- `@pytest.mark.sklearn` - Requires scikit-learn
- `@pytest.mark.ml` - Requires ML libraries

## 🔧 Configuration

The test configuration is optimized in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Optimized for faster testing
addopts = ["-v", "--tb=short"]

# Test search paths
testpaths = ["tests/unit", "tests/integration", "tests/edge_cases", "tests/performance"]
```

## 🚦 CI/CD Integration

### GitHub Actions Workflow

```yaml
# Fast tests for every commit
- name: Run Fast Tests  
  run: python -m pytest tests/core -m "core and not slow"

# Full tests for pull requests
- name: Run Full Tests
  run: python -m pytest tests/unit tests/integration tests/edge_cases

# Performance benchmarks now handled by external datason-benchmarks repo
# Triggered automatically on PRs via .github/workflows/pr-performance-check.yml
```

### Local Development Workflow

1. **During Development:** `./scripts/run_tests.sh fast`
2. **Before Commit:** `./scripts/run_tests.sh full`  
3. **Performance Check:** Handled automatically by external datason-benchmarks
4. **Coverage Check:** `./scripts/run_tests.sh coverage`

## 🎯 Best Practices

### Writing New Tests

1. **Core functionality** → `tests/unit/`
2. **New features** → `tests/features/`
3. **Integration scenarios** → `tests/integration/`
4. **Coverage gaps** → `tests/coverage/`
5. **Performance tests** → `tests/performance/` (minimal local tests only)

### Test Performance Guidelines

- Core tests should complete in < 10 seconds total
- Individual test functions should be < 100ms
- Use `@pytest.mark.slow` for tests > 500ms
- Major performance benchmarking is handled by external datason-benchmarks repo

### Markers Usage

```python
import pytest

@pytest.mark.core
def test_basic_serialization():
    """Fast core functionality test."""
    pass

@pytest.mark.features  
@pytest.mark.pandas
def test_dataframe_feature():
    """Feature test requiring pandas."""
    pass

@pytest.mark.performance
def test_local_performance():
    """Local performance validation test."""
    pass
```

## 🔍 Troubleshooting

### Common Issues

**Import errors after reorganization:**
```bash
# Clear pytest cache
rm -rf .pytest_cache
python -m pytest --collect-only
```

**Missing tests:**
```bash
# Verify all test files are found
find tests/ -name "test_*.py" | wc -l
```

**Performance benchmarking:**
```bash
# Major benchmarks now run automatically via external datason-benchmarks repo
# Check PR comments for performance analysis results
```

### Performance Issues

If tests are still slow:
1. Check for slow tests in non-performance directories
2. Look for tests with expensive setup/teardown  
3. Use `--durations=10` to identify slow tests
4. Major performance analysis is now handled by external datason-benchmarks
4. Consider moving slow tests to appropriate directories

## 📈 Coverage Reporting

Coverage is collected across all test directories:

```bash
# Run with coverage
python -m pytest tests/core tests/features --cov=datason --cov-report=html

# View coverage report
open htmlcov/index.html
```

The coverage boost tests in `tests/coverage/` are specifically designed to improve coverage metrics for edge cases and error conditions.

## 🤝 Contributing

When adding new tests:

1. Choose the appropriate directory based on test purpose
2. Add relevant pytest markers  
3. Ensure tests run quickly (< 100ms per test for core/)
4. Update this documentation if adding new categories

For questions or suggestions about the test structure, please open an issue or discussion.
