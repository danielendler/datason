# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

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
