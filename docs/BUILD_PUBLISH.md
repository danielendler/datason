# Building & Publishing Guide

This guide covers the complete **build and publishing workflow** for datason using modern Python packaging standards.

## 🏗️ **Modern Python Packaging Overview**

datason uses the **latest Python packaging standards** for a clean, maintainable build process:

```toml
# pyproject.toml - Single source of truth
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Why this setup is excellent**:
- ✅ **PEP 517/518 compliant** - Standard modern build system
- ✅ **Zero config** - Hatchling works out of the box
- ✅ **Fast builds** - Optimized for performance
- ✅ **Reproducible** - Same build everywhere

## 📦 **Build Process**

### **1. Install Build Tools**

```bash
# Install modern Python build tools
pip install build twine hatch

# Or install all dev dependencies
pip install -e ".[dev]"
```

### **2. Build Distribution Packages**

```bash
# Clean any previous builds
rm -rf dist/ build/ *.egg-info/

# Build both wheel and source distribution
python -m build

# This creates:
# dist/datason-0.1.0-py3-none-any.whl  (wheel - fast install)
# dist/datason-0.1.0.tar.gz             (source - full source)
```

**What gets built**:
- 📦 **Wheel** (`.whl`) - Optimized binary distribution for fast installs
- 📄 **Source Distribution** (`.tar.gz`) - Complete source code package

### **3. Verify the Build**

```bash
# Check package contents
python -m zipfile -l dist/datason-*.whl

# Test install in clean environment
pip install dist/datason-*.whl

# Verify it works
python -c "import datason; print('✅ Package works!')"
```

## 🚀 **Publishing Workflow**

### **Test Publishing (TestPyPI)**

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ datason
```

### **Production Publishing (PyPI)**

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Verify successful upload
pip install datason
python -c "import datason; print(f'✅ Version {datason.__version__} published!')"
```

## 🔐 **Authentication & Security**

### **Secure Publishing with API Tokens**

**Never use passwords** - use API tokens for security:

```bash
# Configure PyPI API token (recommended)
# 1. Create token at https://pypi.org/manage/account/
# 2. Store in ~/.pypirc:

[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Your API token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwI...  # Your TestPyPI token
```

### **Environment Variables (CI/CD)**

For automated publishing:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcC...
python -m twine upload dist/*
```

## 🔄 **Version Management**

### **Semantic Versioning**

datason follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
  0.1.0 ← Current version
```

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)  
- **PATCH**: Bug fixes (backward compatible)

### **Updating Version**

Update version in `pyproject.toml`:

```toml
[project]
name = "datason"
version = "0.2.0"  # ← Update this
```

### **Automated Version Bumping**

```bash
# Using hatch (recommended)
hatch version minor  # 0.1.0 → 0.2.0
hatch version patch  # 0.2.0 → 0.2.1
hatch version major  # 0.2.1 → 1.0.0

# Or manually edit pyproject.toml
```

## 🤖 **Automated Publishing (GitHub Actions)**

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # For trusted publishing

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install build tools
      run: pip install build

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      # Uses OIDC trusted publishing - no tokens needed!
```

## 📋 **Release Checklist**

### **Pre-Release**

- [ ] All tests passing (`pytest`)
- [ ] Code quality checks (`ruff check`)
- [ ] Security scan clean (`bandit -r datason/`)
- [ ] Documentation updated
- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated

### **Release Process**

```bash
# 1. Final testing
pytest tests/ -v
ruff check datason/
bandit -r datason/

# 2. Build packages
rm -rf dist/
python -m build

# 3. Test on TestPyPI
python -m twine upload --repository testpypi dist/*

# 4. Test install
pip install --index-url https://test.pypi.org/simple/ datason

# 5. Publish to PyPI
python -m twine upload dist/*

# 6. Create GitHub release
git tag v0.2.0
git push origin v0.2.0
# Create release on GitHub UI
```

### **Post-Release**

- [ ] Verify PyPI upload successful
- [ ] Test install from PyPI
- [ ] Update documentation if needed
- [ ] Announce release (if applicable)

## 🛠️ **Development Builds**

### **Editable Install**

For development work:

```bash
# Install in development mode
pip install -e .

# With optional dependencies
pip install -e ".[dev,ml,pandas]"

# Changes to source code immediately available
```

### **Local Testing**

```bash
# Build and test locally
python -m build
pip install dist/datason-*.whl --force-reinstall

# Test specific functionality
python -c "
import datason
import pandas as pd
result = datason.serialize({'df': pd.DataFrame({'A': [1,2,3]})})
print('✅ Local build works!')
"
```

## 📊 **Build Verification**

### **Package Contents**

```bash
# Check wheel contents
python -m zipfile -l dist/datason-*.whl

# Expected contents:
# datason/__init__.py
# datason/core.py
# datason/ml_serializers.py
# datason-0.1.0.dist-info/METADATA
# datason-0.1.0.dist-info/WHEEL
```

### **Metadata Verification**

```bash
# Check package metadata
python -m pip show datason

# Verify dependencies are correct
python -c "
import pkg_resources
dist = pkg_resources.get_distribution('datason')
print(f'Name: {dist.project_name}')
print(f'Version: {dist.version}')
print(f'Dependencies: {[str(req) for req in dist.requires()]}')
"
```

## 🔍 **Troubleshooting**

### **Common Build Issues**

**Problem**: `ModuleNotFoundError` during build
```bash
# Solution: Install build dependencies
pip install build hatchling
```

**Problem**: Permission denied during upload
```bash
# Solution: Check API token
python -m twine check dist/*
python -m twine upload --verbose dist/*
```

**Problem**: Package already exists
```bash
# Solution: Bump version in pyproject.toml
hatch version patch
python -m build
```

### **Build Performance**

```bash
# Fast development builds (skip tests)
python -m build --no-isolation

# Clean builds (recommended for releases)
rm -rf build/ dist/ *.egg-info/
python -m build
```

## 📈 **Advanced Features**

### **Conditional Dependencies**

Already configured in `pyproject.toml`:

```toml
[project.optional-dependencies]
# Users can install specific features
# pip install datason[ml]      # Just ML dependencies
# pip install datason[pandas]  # Just pandas
# pip install datason[all]     # Everything
```

### **Platform-Specific Builds**

```bash
# Build for specific platforms (if needed)
python -m build --wheel

# Check wheel compatibility
python -m pip debug --verbose
```

---

## 🎯 **Summary**

datason uses **modern Python packaging** for:

✅ **Simple builds** with `python -m build`  
✅ **Secure publishing** with API tokens  
✅ **Automated releases** via GitHub Actions  
✅ **Professional distribution** on PyPI  

The entire process from development to PyPI takes **< 5 minutes** and is **fully automated**! 🚀
