# Contributing to datason 🤝

Thank you for your interest in contributing to datason! This guide will help you understand our development process, coding standards, and how to make meaningful contributions.

## 🎯 Core Development Principles

datason follows three fundamental principles that guide all development decisions:

### 1. 🪶 **Minimal Dependencies**

**Philosophy**: The core functionality should work without any external dependencies, with optional enhancements when libraries are available.

**Guidelines**:
- ✅ Core serialization must work with just Python standard library
- ✅ Import optional dependencies only when needed with graceful fallbacks
- ✅ Use feature detection rather than forcing installations
- ✅ Document which features require which dependencies

**Example**:
```python
# Good: Graceful fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

def serialize_pandas_object(obj):
    if not HAS_PANDAS:
        return str(obj)  # Fallback to string representation
    # Full pandas logic here
```

### 2. ✅ **Comprehensive Test Coverage**

**Target**: Maintain 80%+ test coverage across all modules

**Testing Strategy**:
- **Edge cases first**: Test error conditions, malformed data, boundary cases
- **Optional dependency matrix**: Test with and without optional packages
- **Performance regression**: Benchmark critical paths
- **Cross-platform**: Test on multiple Python versions and platforms

**Coverage Requirements**:
- 🎯 **Overall target**: 80-85% coverage
- 🔴 **Core modules**: 85%+ required (`core.py`, `converters.py`)  
- 🟡 **Feature modules**: 75%+ required (`ml_serializers.py`, `datetime_utils.py`)
- 🟢 **Utility modules**: 70%+ acceptable (`data_utils.py`)

**Testing Commands**:
```bash
# Run all tests with coverage
pytest --cov=datason --cov-report=term-missing --cov-report=html

# Test core functionality only
pytest tests/test_core.py tests/test_converters.py tests/test_deserializers.py

# Test with optional dependencies
pytest tests/test_optional_dependencies.py tests/test_ml_serializers.py

# Performance benchmarks
pytest tests/test_performance.py
```

### 3. ⚡ **Performance First**

**Performance Standards** (based on real benchmarks):
- 🚀 **Simple data**: < 1ms for JSON-compatible objects (achieved: 0.6ms)
- 🚀 **Complex data**: < 5ms for 500 objects with UUIDs/datetimes (achieved: 2.1ms)
- 🚀 **Large datasets**: > 250K items/second throughput (achieved: 272K/sec)
- 💾 **Memory**: Linear memory usage, no memory leaks
- 🔄 **Round-trip**: Complete cycle < 2ms (achieved: 1.4ms)
- 📊 **NumPy**: > 5M elements/second (achieved: 5.5M/sec)
- 🐼 **Pandas**: > 150K rows/second (achieved: 195K/sec)

**Optimization Guidelines**:
- Early exits for already-serialized data
- Minimize object creation in hot paths
- Use efficient algorithms for type detection
- Benchmark before and after changes with `benchmark_real_performance.py`

**Performance Testing**:
```python
# Always benchmark performance-critical changes
@pytest.mark.benchmark
def test_large_dataset_performance():
    large_data = create_large_test_dataset()

    start_time = time.time()
    result = serialize(large_data)
    elapsed = time.time() - start_time

    assert elapsed < 1.0  # Must complete in under 1 second
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+ (we support 3.8-3.12)
- Git
- Virtual environment (recommended)

### Local Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/danielendler/datason.git
cd datason

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev,all,ml]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run initial tests to verify setup
pytest -v
```

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature-name`
2. **Make changes** following the coding standards below
3. **Add tests** for new functionality
4. **Run tests**: `pytest --cov=datason`
5. **Check coverage**: Ensure coverage stays above 80%
6. **Run linting and formatting**: `ruff check --fix . && ruff format .`
7. **Create pull request** with detailed description

## 📋 Coding Standards

### Code Style
- **Linter & Formatter**: Ruff (unified tool for linting and formatting)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google-style for all public functions

### File Structure
```
datason/
├── core.py              # Core serialization logic
├── converters.py        # Safe type converters
├── deserializers.py     # Deserialization and parsing
├── datetime_utils.py    # Date/time handling
├── data_utils.py        # Data processing utilities
├── serializers.py       # Specialized serializers
├── ml_serializers.py    # ML/AI library support
└── __init__.py          # Public API exports
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `serialize_object`)
- **Classes**: `PascalCase` (e.g., `JSONSerializer`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private**: `_leading_underscore` (e.g., `_internal_function`)

### Documentation Standards

```python
def serialize_complex_object(obj: Any, **kwargs: Any) -> Dict[str, Any]:
    """Serialize a complex object to JSON-compatible format.

    Args:
        obj: The object to serialize (any Python type)
        **kwargs: Additional serialization options

    Returns:
        JSON-compatible dictionary representation

    Raises:
        SerializationError: If object cannot be serialized

    Examples:
        >>> data = {'date': datetime.now(), 'array': np.array([1, 2, 3])}
        >>> result = serialize_complex_object(data)
        >>> isinstance(result, dict)
        True
    """
```

## 🧪 Testing Guidelines

### Test Organization
```
tests/
├── test_core.py                    # Core functionality tests
├── test_converters.py              # Type converter tests
├── test_deserializers.py           # Deserialization tests
├── test_optional_dependencies.py   # Pandas/numpy integration
├── test_ml_serializers.py          # ML library tests
├── test_edge_cases.py              # Edge cases and error handling
├── test_performance.py             # Performance benchmarks
└── test_coverage_boost.py          # Coverage improvement tests
```

### Test Types

**Unit Tests**: Test individual functions in isolation
```python
def test_serialize_datetime():
    """Test datetime serialization preserves information."""
    dt = datetime(2023, 1, 1, 12, 0, 0)
    result = serialize(dt)
    assert result.endswith('T12:00:00')
```

**Integration Tests**: Test component interaction
```python
def test_round_trip_serialization():
    """Test complete serialize → JSON → deserialize cycle."""
    data = {'uuid': uuid.uuid4(), 'date': datetime.now()}
    json_str = json.dumps(serialize(data))
    restored = deserialize(json.loads(json_str))
    assert isinstance(restored['uuid'], uuid.UUID)
```

**Edge Case Tests**: Test error conditions and boundaries
```python
def test_circular_reference_handling():
    """Test graceful handling of circular references."""
    circular = {}
    circular['self'] = circular
    result = serialize(circular)  # Should not raise
    assert isinstance(result, (dict, str))
```

**Performance Tests**: Benchmark critical paths
```python
@pytest.mark.performance
def test_large_dataset_serialization():
    """Benchmark serialization of large datasets."""
    large_data = create_large_test_data(10000)

    start = time.time()
    result = serialize(large_data)
    elapsed = time.time() - start

    assert elapsed < 2.0  # Performance requirement
```

### Coverage Requirements

Each new feature must include tests that maintain or improve coverage:

```bash
# Check coverage before changes
pytest --cov=datason --cov-report=term-missing

# Identify uncovered lines
# Lines marked in red in HTML report need tests

# Add tests targeting specific lines
pytest tests/test_your_new_feature.py --cov=datason --cov-report=term-missing
```

## 🔄 Pull Request Process

### Before Submitting
- [ ] All tests pass: `pytest`
- [ ] Coverage maintained: `pytest --cov=datason`
- [ ] Code quality: `ruff check --fix .`
- [ ] Code formatting: `ruff format .`
- [ ] Type checking: `mypy datason/`
- [ ] Security scan: `bandit -r datason/`
- [ ] Documentation updated if needed

### CI/CD Pipeline
datason uses a modern multi-pipeline CI/CD architecture. See our **[CI/CD Pipeline Guide](CI_PIPELINE_GUIDE.md)** for complete details on:
- 🔍 **Quality Pipeline** - Ruff linting, formatting, security scanning (~30s)
- 🧪 **Main CI** - Testing, coverage, package building (~2-3min)
- 📚 **Docs Pipeline** - Documentation generation and deployment
- 📦 **Publish Pipeline** - Automated PyPI releases

All pipelines use intelligent caching for 2-5x speedup on repeat runs.

### PR Template
```markdown
## 🎯 What does this PR do?
Brief description of the changes

## ✅ Testing
- [ ] Added unit tests for new functionality
- [ ] Added integration tests if applicable
- [ ] Performance benchmarks if performance-related
- [ ] Coverage maintained/improved

## 📊 Coverage Impact
Current coverage: X%
After changes: Y%

## 🚀 Performance Impact  
[Include benchmarks if performance-related]

## 📚 Documentation
- [ ] Updated README if user-facing changes
- [ ] Added docstrings for new public APIs
- [ ] Updated examples if needed
```

## 🎁 Types of Contributions

### 🐛 Bug Fixes
- Always include a test that reproduces the bug
- Ensure fix doesn't break existing functionality
- Update documentation if behavior changes

### ✨ New Features
- Discuss major features in an issue first
- Follow the minimal dependencies principle
- Include comprehensive tests
- Update documentation and examples

### 📈 Performance Improvements
- Include before/after benchmarks
- Ensure no functionality regressions
- Add performance tests to prevent future regressions

### 📚 Documentation
- Fix typos, improve clarity
- Add examples for complex use cases
- Update API documentation

### 🧪 Testing
- Improve test coverage
- Add edge case tests
- Performance benchmarks

## 🏆 Recognition

Contributors are recognized in:
- `CHANGELOG.md` for each release
- GitHub contributors page
- Special recognition for significant contributions

## ❓ Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## 📜 Code of Conduct

We follow the Contributor Covenant Code of Conduct. Please be respectful and inclusive in all interactions.

---

**Thank you for contributing to datason!** 🙏 Your contributions help make Python data serialization better for everyone.
