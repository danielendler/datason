# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-01-09

### 🏗️ DevOps & Infrastructure Overhaul

#### CI/CD Pipeline Architecture
- **🔄 Multi-Pipeline System**: Established sophisticated 3-pipeline CI/CD architecture
  - **🧪 Main CI**: Testing, coverage, package building (~2-3 min, excludes docs changes)
  - **🔍 Quality**: Ruff linting, formatting, security scanning (~30-60s, all changes)
  - **📚 Docs**: Documentation generation (only docs changes)
  - **📦 Publish**: PyPI releases (GitHub releases only)
- **⚡ Performance Optimizations**: Added comprehensive dependency caching (pip, pre-commit, MkDocs) for 2-5x speedup
- **🎯 Smart Triggering**: Intelligent path-based pipeline triggering to avoid unnecessary runs
- **📊 Parallel Execution**: Optimized job dependencies for maximum parallel efficiency

#### Code Coverage & Quality Assurance
- **📈 Codecov Integration**: Comprehensive coverage reporting with GitHub integration
  - 80% project coverage target, 75% patch coverage requirement
  - Proper exclusions for tests, benchmarks, scripts, and examples
  - Automated coverage comments on pull requests
  - Flag-based coverage tracking (core, ml, utils modules)
- **🔧 Pre-commit Modernization**: Updated all hooks to latest versions
  - Ruff v0.11.12 (was v0.1.6) - resolved local/CI formatting inconsistencies
  - Enhanced security scanning with Bandit 1.8.3
  - Lightweight coverage check for changed files only
- **⚙️ Configuration Refinement**: Fixed ruff formatter conflicts by adding `ISC001` to ignore list

#### Documentation & Project Structure
- **📖 Comprehensive Documentation**: Created detailed CI/CD pipeline guide (`docs/CI_PIPELINE_GUIDE.md`)
  - Mermaid diagrams explaining pipeline architecture
  - Performance metrics and optimization strategies
  - Troubleshooting guides for common CI issues
- **🤖 Dependabot Integration**: Automated dependency management with comprehensive documentation (`docs/DEPENDABOT_GUIDE.md`)
  - Weekly dev dependency updates for latest tooling
  - Monthly core/optional dependency updates for stability
  - Conservative ML library update strategy with major version protection
  - Integrated with CI/CD pipelines for automated validation
- **📁 Project Organization**: Moved `benchmark_real_performance.py` to `benchmarks/` directory
- **📝 Contributing Guidelines**: Updated `CONTRIBUTING.md` from outdated tools (black/flake8) to current ruff-based workflow
- **🔗 Navigation Fixes**: Fixed broken documentation links and mkdocs navigation structure

#### Bug Fixes & Stability
- **🐛 Serialization Bug Fix**: Resolved critical optimization issue in `serialpy/core.py`
  - Objects were being added to `_seen` set before optimization checks
  - Fixed identity preservation for complex nested structures
  - Maintained performance while ensuring correctness
- **🔑 GPG Signing Resolution**: Configured proper commit signing with `GPG_TTY=$(tty)`
- **🧹 Environment Alignment**: Resolved local vs CI Python/ruff version mismatches

### 🔧 Technical Improvements

#### Development Workflow
- **✅ Streamlined Testing**: Removed pre-commit from main CI (kept local-only) since separate quality pipeline exists
- **📋 Coverage Configuration**: Updated local coverage settings to exclude scripts and examples
- **🚀 Performance Monitoring**: Enhanced benchmark organization and execution
- **🔒 Security Scanning**: Integrated bandit security checks across all pipelines

#### Code Quality
- **📏 Linting Standardization**: Unified ruff configuration across local and CI environments
- **🎨 Formatting Consistency**: Resolved pre-commit vs CI formatter conflicts
- **📊 Metrics Tracking**: Comprehensive coverage reporting with proper exclusions
- **🧪 Test Organization**: Improved test structure and execution efficiency

### 📦 Configuration Updates

#### Build & Packaging
- **⚙️ pyproject.toml**: Enhanced coverage configuration with proper omit patterns
- **🔧 codecov.yml**: Complete Codecov configuration with targets, thresholds, and GitHub integration
- **🪝 .pre-commit-config.yaml**: Updated all hooks to latest stable versions
- **🤖 .github/dependabot.yml**: Automated dependency management with smart update scheduling

#### CI/CD Configuration Files
- **📋 GitHub Workflows**: Four specialized workflow files for different concerns
- **🚀 Dependency Caching**: Intelligent caching strategies across all pipelines
- **📊 Performance Optimization**: Path-based triggering and parallel job execution

### 🎯 Developer Experience

#### Faster Feedback Loops
- **⚡ Quick Quality Checks**: 30-60 second quality pipeline for immediate feedback
- **🔍 Targeted Testing**: Lightweight coverage checks for changed files only
- **📝 Clear Documentation**: Comprehensive guides for CI/CD pipeline usage

#### Better Debugging
- **📊 Coverage Reports**: Detailed HTML and XML coverage reports
- **🔍 Pipeline Visualization**: Mermaid diagrams showing pipeline flow and dependencies
- **📋 Troubleshooting Guides**: Common issues and solutions documented

### 🚧 Breaking Changes
- **📁 File Structure**: Moved benchmarks from root to `benchmarks/` directory
- **⚙️ Coverage Configuration**: Updated omit patterns (may affect local coverage reports)

### 🐛 Bug Fixes
- Fixed serialization identity preservation bug in core optimization logic
- Resolved pre-commit ruff version conflicts between local and CI
- Fixed documentation link breakages and navigation issues
- Corrected GPG signing configuration for authenticated commits

### 📈 Performance Improvements
- 2-5x faster CI pipeline execution through comprehensive dependency caching
- Eliminated unnecessary pipeline runs through intelligent path-based triggering
- Streamlined test execution with parallel job optimization
- Reduced pre-commit overhead with lightweight coverage checks

### 🔒 Security Enhancements
- Enhanced security scanning with updated Bandit configuration
- Proper dependency vulnerability checking in quality pipeline
- Secure GPG commit signing configuration
- Updated all dependencies to latest secure versions

---

## [0.1.0] - 2025-05-30

### 🚀 Major Features Added

#### Bidirectional Serialization
- **Complete Python ↔ JSON round-trip support** with intelligent type restoration
- `deserialize()` function for converting JSON back to Python objects
- `deserialize_to_pandas()` for pandas-optimized deserialization
- `safe_deserialize()` for graceful error handling during JSON parsing
- Smart parsing controls with `parse_dates` and `parse_uuids` parameters

#### ML/AI Ecosystem Support
- **PyTorch integration**: Tensor serialization with metadata preservation
- **TensorFlow support**: Tensor data and shape information
- **Scikit-learn compatibility**: Model parameters and pipeline metadata
- **JAX arrays**: Full array serialization with dtype preservation
- **Scipy integration**: Sparse matrix serialization (CSR, CSC, COO formats)
- **PIL/Pillow support**: Image metadata and base64 encoding
- **HuggingFace transformers**: Tokenizer and model metadata
- Automatic ML object detection with `detect_and_serialize_ml_object()`

#### Advanced Data Science Features
- **Enhanced pandas support**: DataFrames, Series, Index, and Timestamps
- **Comprehensive numpy integration**: All data types, arrays, and edge cases
- **Smart datetime handling**: ISO format conversion with timezone support
- **UUID preservation**: Round-trip UUID serialization and restoration
- **NaN/Inf handling**: Graceful conversion of special float values

#### Performance & Production Features
- **Optimization engine**: Early detection of already-serialized data
- **Memory efficiency**: Streaming support for large datasets
- **Error resilience**: Comprehensive edge case handling
- **Type safety**: Full type hints and validation
- **Circular reference protection**: Safe handling of recursive structures

### 🔧 Core Improvements

#### Serialization Engine
- Enhanced `serialize()` function with ML/AI object detection
- Performance optimizations for JSON-compatible data
- Better handling of nested structures and mixed types
- Improved fallback mechanisms for unknown objects

#### Deserialization Engine
- Intelligent type restoration from JSON strings
- Configurable parsing options for selective type conversion
- Pandas-specific optimizations for DataFrame reconstruction
- Robust error handling for malformed JSON

#### Utility Functions
- `parse_datetime_string()` for smart datetime parsing
- `parse_uuid_string()` for UUID validation and conversion
- `get_ml_library_info()` for checking available ML libraries
- Enhanced datetime utilities with timezone support

### 📊 Test Coverage & Quality

#### Comprehensive Testing
- **82% test coverage** across all modules
- **128 passing tests** with extensive edge case coverage
- **Performance benchmarks** for large dataset handling
- **Round-trip testing** for data integrity verification
- **Optional dependency testing** for graceful degradation

#### CI/CD Pipeline
- Multi-version Python testing (3.8-3.12)
- Performance regression testing
- Security scanning with Bandit and Safety
- Automated publishing to PyPI
- Code quality checks with flake8, black, mypy

### 🎯 Use Case Coverage

#### Machine Learning Workflows
- Experiment tracking and metadata serialization
- Model pipeline persistence and restoration
- Hyperparameter optimization data handling
- Training metrics and performance data

#### Data Science Applications
- Time series analysis with pandas integration
- Large dataset serialization and streaming
- Statistical computation results preservation
- Data preprocessing pipeline persistence

#### API Development
- REST API request/response serialization
- Database ORM integration
- Microservices data exchange
- Real-time data streaming

### 📦 Package Structure

#### New Modules
- `serialpy/ml_serializers.py` - ML/AI library integration
- `serialpy/deserializers.py` - Bidirectional deserialization
- `tests/test_deserializers.py` - Deserialization test suite
- `tests/test_performance.py` - Performance benchmarks
- `tests/test_optional_dependencies.py` - Optional library tests
- `examples/advanced_ml_examples.py` - ML/AI workflow examples
- `examples/bidirectional_example.py` - Round-trip demonstrations

#### Enhanced Documentation
- Comprehensive README with ML/AI examples
- Advanced use case demonstrations
- Performance benchmarking results
- Contributing guidelines and development setup

### 🔍 Supported Libraries

#### Core Data Science (✅ Full Support)
- **pandas** >= 1.0.0: DataFrames, Series, Timestamps, NaT
- **numpy** >= 1.20.0: All data types, arrays, NaN/Inf handling

#### Machine Learning (✅ Metadata & Tensors)
- **PyTorch** >= 1.9.0: Tensors, device info, gradients
- **TensorFlow** >= 2.8.0: Tensors, shapes, dtypes
- **scikit-learn** >= 1.0.0: Models, pipelines, parameters
- **JAX** >= 0.3.0: Arrays, transformations

#### Scientific Computing (✅ Arrays & Matrices)
- **scipy** >= 1.7.0: Sparse matrices, special functions
- **PIL/Pillow** >= 8.0.0: Images, metadata

#### NLP & Transformers (✅ Models & Tokenizers)
- **HuggingFace transformers** >= 4.0.0: Tokenizers, model configs

#### Core Python (✅ Complete)
- **datetime**, **UUID**, **Decimal**: Full round-trip support
- **pathlib**, **collections**: Smart object handling

### 🚧 Installation Options

```bash
# Basic installation (core + pandas/numpy)
pip install serialpy[all]

# ML/AI ecosystem support
pip install serialpy[ml]

# Development environment
pip install serialpy[dev]

# Minimal core-only installation
pip install serialpy
```

### 🎭 Breaking Changes
- None (initial release)

### 🐛 Bug Fixes
- N/A (initial release)

### 📈 Performance Improvements
- Optimized serialization paths for already-JSON-compatible data
- Early return optimization for basic types
- Efficient numpy array conversion
- Smart pandas DataFrame handling

### 🔒 Security Enhancements
- Safe object serialization without code execution
- Input validation for all parsing functions
- Graceful error handling for malicious inputs
- No eval() or exec() usage in parsing

### 📝 Documentation
- Comprehensive API documentation
- Real-world usage examples
- Performance benchmarking results
- Contributing guidelines
- Advanced ML/AI workflow demonstrations

---

## Development Roadmap

### Planned for v0.2.0
- TensorFlow SavedModel serialization
- XGBoost/LightGBM model support
- Arrow/Polars DataFrame integration
- Enhanced image serialization options
- Streaming serialization for very large datasets

### Future Considerations
- OpenCV integration for computer vision workflows
- Matplotlib figure serialization
- Distributed computing framework support (Dask, Ray)
- Custom serialization plugins architecture
- WebAssembly compatibility

---

## Contributors

- SerialPy Contributors

## License

MIT License - see LICENSE file for details.
