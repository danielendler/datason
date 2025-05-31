# Modern Python Tooling Guide

## Overview

datason uses **best-in-class modern Python tooling** for development, testing, security, and documentation. This guide explains our choices and how to use them effectively.

## 🚀 **Tool Stack Summary**

| Category | Tool | Replaces | Why Better |
|----------|------|----------|-------------|
| **Linting & Formatting** | Ruff | Black + isort + flake8 + pyupgrade | 10-100x faster, single tool |
| **Type Checking** | mypy | - | Industry standard, strict typing |
| **Testing** | pytest + coverage | unittest | Better DX, plugins, fixtures |
| **Security** | Bandit + Safety + pip-audit | Manual reviews | Automated vulnerability detection |
| **Documentation** | MkDocs Material | Sphinx | Modern, beautiful, fast |
| **Dependency Management** | pip-tools + Dependabot | Manual updates | Automated, deterministic |
| **Git Hooks** | pre-commit | Manual checks | Automatic quality gates |
| **Performance** | pytest-benchmark | Manual timing | Statistical benchmarking |

## 🔧 **Core Tooling Decisions**

### ⚡ Ruff (Linting & Formatting)

**Why Ruff over Black + isort + flake8?**

```bash
# Old approach (slow, multiple tools)
black datason/ tests/     # ~2 seconds
isort datason/ tests/     # ~1 second  
flake8 datason/ tests/    # ~3 seconds
pyupgrade **/*.py           # ~2 seconds
# Total: ~8 seconds

# New approach (fast, single tool)
ruff check --fix datason/ tests/    # ~0.1 seconds
ruff format datason/ tests/          # ~0.1 seconds  
# Total: ~0.2 seconds (40x faster!)
```

**Ruff Configuration:**
```toml
[tool.ruff]
target-version = "py38"
line-length = 88

[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes  
    "I",         # isort
    "N",         # pep8-naming
    "D",         # pydocstyle
    "UP",        # pyupgrade
    "S",         # bandit security
    "B",         # bugbear
    "C4",        # comprehensions
    "SIM",       # simplify
    "RUF",       # ruff-specific
]
```

**Usage:**
```bash
# Check and fix issues
ruff check --fix .

# Format code  
ruff format .

# Check specific rule
ruff check --select E501 .  # Line too long
```

### 🔒 Security Tools

**Multi-layered security approach:**

1. **Bandit** - Static code analysis for security issues
2. **Safety** - Known vulnerability database checking
3. **pip-audit** - Dependency vulnerability scanning
4. **Semgrep** - Advanced pattern-based security scanning

```bash
# Run all security checks
bandit -c pyproject.toml -r datason/
safety check --json
pip-audit --format=json
semgrep --config=auto datason/
```

**Common security issues caught:**
- Hardcoded passwords/secrets
- SQL injection patterns
- Use of `eval()` or `exec()`
- Weak random number generation
- Insecure hash functions

### 📚 Documentation with MkDocs Material

**Why MkDocs over Sphinx?**

```bash
# Modern, beautiful, fast
mkdocs serve    # Live reload dev server
mkdocs build    # Static site generation
mkdocs gh-deploy # Deploy to GitHub Pages
```

**Features:**
- 🎨 **Beautiful UI** - Material Design
- 🔍 **Search** - Full-text search built-in
- 📱 **Mobile** - Responsive design
- 🌙 **Dark mode** - Automatic theme switching
- 📊 **Diagrams** - Mermaid support
- 🔗 **Cross-refs** - Automatic API linking

### 🧪 Testing Excellence

**Comprehensive testing setup:**

```bash
# Run all tests with coverage
pytest --cov=datason --cov-report=html

# Parallel testing (faster)
pytest -n auto

# Performance benchmarks
pytest tests/test_performance.py --benchmark-only

# Specific test markers
pytest -m "not slow"        # Skip slow tests
pytest -m "ml"              # Only ML-related tests
pytest -m "integration"     # Integration tests only
```

**Test markers configured:**
```python
# In pyproject.toml
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
    "ml: marks tests that require ML libraries",
]
```

### 🔄 Pre-commit Hooks

**Automatic quality gates:**

```yaml
# .pre-commit-config.yaml highlights
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff           # Linting
      - id: ruff-format    # Formatting

  - repo: https://github.com/PyCQA/bandit
    hooks:
      - id: bandit         # Security scanning
```

**Benefits:**
- ✅ **Prevents bad commits** - Catches issues before they reach CI
- ✅ **Consistent formatting** - Everyone uses same style
- ✅ **Security checks** - Vulnerabilities caught early
- ✅ **Fast feedback** - Issues fixed immediately

## 🛠️ **Development Workflow**

### Initial Setup
```bash
# 1. Install development dependencies
pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install

# 3. Run initial checks
pre-commit run --all-files
```

### Daily Development
```bash
# Before committing (automatic via pre-commit)
ruff check --fix .          # Fix linting issues
ruff format .               # Format code
pytest                      # Run tests
mypy datason/            # Type checking
```

### Release Process
```bash
# 1. Security audit
bandit -c pyproject.toml -r datason/
safety check
pip-audit

# 2. Full test suite
pytest --cov=datason --cov-report=term-missing

# 3. Documentation
mkdocs build

# 4. Build and publish
python -m build
twine upload dist/*
```

## 📊 **Performance Optimization**

### Benchmarking with pytest-benchmark
```python
def test_serialization_performance(benchmark):
    """Benchmark serialization performance."""
    data = create_test_data()

    result = benchmark(datason.serialize, data)

    # Assertions on performance
    assert benchmark.stats.median < 0.01  # < 10ms
```

### Profiling
```bash
# Profile specific functions
python -m cProfile -o profile.stats benchmark_real_performance.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

## 🔄 **Dependency Management**

### pip-tools for Reproducible Builds
```bash
# Create requirements.txt from pyproject.toml
pip-compile pyproject.toml

# Update dependencies
pip-compile --upgrade pyproject.toml

# Install exact versions
pip-sync requirements.txt
```

### Dependabot Configuration
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

## 📈 **Monitoring & Metrics**

### Coverage Tracking
```bash
# Generate coverage report
pytest --cov=datason --cov-report=html

# Coverage thresholds in pyproject.toml
[tool.coverage.report]
fail_under = 80
show_missing = true
```

### Performance Monitoring
```python
# Benchmark regression testing
@pytest.mark.benchmark
def test_performance_regression(benchmark):
    """Ensure performance doesn't regress."""
    result = benchmark(serialize_large_dataset)
    # CI will fail if significantly slower than baseline
```

## 🎯 **Best Practices Summary**

### Code Quality
- ✅ **Ruff** for all linting/formatting (replaces 4 tools)
- ✅ **mypy** with strict mode for type safety
- ✅ **Pre-commit hooks** for automatic quality gates
- ✅ **100% test coverage** for critical paths

### Security
- ✅ **Multi-tool scanning** (Bandit, Safety, pip-audit)
- ✅ **Regular dependency updates** via Dependabot  
- ✅ **Secret detection** in pre-commit
- ✅ **Security policy** with responsible disclosure

### Documentation
- ✅ **MkDocs Material** for beautiful docs
- ✅ **API reference** auto-generated from docstrings
- ✅ **Examples** for all major features
- ✅ **Cross-references** and search

### Performance
- ✅ **Benchmark suite** with statistical analysis
- ✅ **Performance regression testing** in CI
- ✅ **Profiling tools** for optimization
- ✅ **Real-world performance data** in docs

## 🚀 **Upgrading from Legacy Tools**

### Migration Guide

**From Black + isort + flake8 → Ruff:**
```bash
# Remove old tools
pip uninstall black isort flake8

# Install Ruff
pip install ruff

# Update pre-commit
# Replace separate hooks with ruff hooks

# Run once
ruff check --fix .
ruff format .
```

**Benefits:**
- 🚀 **40x faster** linting and formatting
- 🧹 **Single tool** instead of multiple
- 🔧 **More rules** - catches more issues
- 📦 **Smaller dependency** tree

## 🎉 **Results**

This modern tooling setup delivers:

- ⚡ **10x faster** development workflow
- 🛡️ **Better security** with automated scanning  
- 📚 **Professional docs** with zero maintenance
- 🧪 **Reliable testing** with performance tracking
- 🤖 **Automated quality** via pre-commit hooks

**Developer Experience Score: 10/10** 🎯

The investment in modern tooling pays off immediately in developer productivity and code quality!
