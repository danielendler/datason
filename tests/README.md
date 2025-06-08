# Datason Test Suite Organization

This document describes the reorganized test structure for optimal development workflow and CI performance.

## 🏗️ Test Structure

The test suite is organized into logical categories for better performance and maintainability:

```
tests/
├── core/           # Fast core functionality tests (~7-10 seconds)
│   ├── test_core.py                    # Basic serialization
│   ├── test_security.py                # Security features  
│   ├── test_circular_references.py     # Circular reference handling
│   ├── test_edge_cases.py              # Edge cases and error handling
│   ├── test_converters.py              # Type converters
│   ├── test_deserializers.py           # Deserialization functionality
│   └── test_dataframe_orientation_regression.py
│
├── features/       # Feature-specific tests (~10-20 seconds)
│   ├── test_ml_serializers.py          # ML library integrations
│   ├── test_chunked_streaming.py       # Streaming/chunking features
│   ├── test_auto_detection_and_metadata.py  # Auto-detection
│   └── test_template_deserialization.py     # Template deserialization
│
├── integration/    # Integration tests (~5-15 seconds)
│   ├── test_config_and_type_handlers.py     # Configuration integration
│   ├── test_optional_dependencies.py        # Dependency integrations
│   └── test_pickle_bridge.py                # Pickle bridge functionality
│
├── coverage/       # Coverage boost tests (~10-30 seconds)
│   ├── test_coverage_boost.py               # General coverage boosting
│   ├── test_core_coverage_boost.py          # Core module coverage
│   ├── test_datetime_coverage_boost.py      # DateTime utils coverage
│   ├── test_focused_coverage_boost.py       # Targeted coverage
│   ├── test_init_coverage_boost.py          # __init__.py coverage
│   ├── test_ml_serializers_coverage_boost.py # ML serializers coverage
│   └── test_targeted_coverage_boost.py      # Additional targeted tests
│
├── benchmarks/     # Performance benchmark tests (~60-120 seconds)
│   ├── test_benchmarks.py                   # Core performance benchmarks
│   ├── test_chunked_streaming_benchmarks.py # Streaming performance
│   ├── test_template_deserialization_benchmarks.py # Template performance
│   └── test_performance.py                  # General performance tests
│
├── conftest.py     # Shared test configuration
└── __init__.py     # Test package marker
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
# Run benchmark tests for performance analysis (~60-120 seconds)
./scripts/run_tests.sh benchmarks

# Or using pytest directly:
python -m pytest tests/benchmarks --benchmark-only
```

### Complete Testing
```bash
# Run everything including benchmarks
./scripts/run_tests.sh all

# Or using pytest directly:
python -m pytest tests/ tests/benchmarks --benchmark-skip
```

## 📊 Performance Improvements

The reorganized structure provides significant performance improvements:

| Test Category | Test Count | Execution Time | Use Case |
|---------------|------------|----------------|----------|
| **Fast Core** | 137 tests | ~7-10 seconds | Development, quick validation |
| **Full Suite** | ~400 tests | ~30-60 seconds | Pre-commit, CI main |
| **Coverage** | ~200 tests | ~10-30 seconds | Coverage improvement |
| **Benchmarks** | ~90 tests | ~60-120 seconds | Performance analysis |
| **Complete** | ~540 tests | ~90-150 seconds | Release validation |

**Previous:** All tests took ~103 seconds  
**Now:** Fast tests take ~7 seconds (93% faster!)

## 🏷️ Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.core` - Core functionality (no optional dependencies)
- `@pytest.mark.features` - Feature-specific tests  
- `@pytest.mark.integration` - Integration scenarios
- `@pytest.mark.coverage` - Coverage boost tests
- `@pytest.mark.benchmark` - Performance benchmarks
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
# Exclude benchmarks by default for faster testing
addopts = ["-m", "not benchmark"]

# Test search paths
testpaths = ["tests/core", "tests/features", "tests/integration", "tests/coverage"]
```

## 🚦 CI/CD Integration

### GitHub Actions Workflow

```yaml
# Fast tests for every commit
- name: Run Fast Tests  
  run: python -m pytest tests/core -m "core and not slow"

# Full tests for pull requests
- name: Run Full Tests
  run: python -m pytest tests/core tests/features tests/integration

# Benchmarks for performance monitoring (optional)
- name: Run Benchmarks
  run: python -m pytest tests/benchmarks --benchmark-only
```

### Local Development Workflow

1. **During Development:** `./scripts/run_tests.sh fast`
2. **Before Commit:** `./scripts/run_tests.sh full`  
3. **Performance Check:** `./scripts/run_tests.sh benchmarks`
4. **Coverage Check:** `./scripts/run_tests.sh coverage`

## 🎯 Best Practices

### Writing New Tests

1. **Core functionality** → `tests/unit/`
2. **New features** → `tests/features/`
3. **Integration scenarios** → `tests/integration/`
4. **Coverage gaps** → `tests/coverage/`
5. **Performance tests** → `tests/benchmarks/`

### Test Performance Guidelines

- Core tests should complete in < 10 seconds total
- Individual test functions should be < 100ms
- Use `@pytest.mark.slow` for tests > 500ms
- Benchmark tests can be longer but should be in `tests/benchmarks/`

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

@pytest.mark.benchmark
def test_performance(benchmark):
    """Performance benchmark test."""
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

**Benchmark tests not running:**
```bash
# Install benchmark plugin
pip install pytest-benchmark
```

### Performance Issues

If tests are still slow:
1. Check for benchmark tests in non-benchmark directories
2. Look for tests with expensive setup/teardown  
3. Use `--durations=10` to identify slow tests
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
