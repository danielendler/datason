# datason Product Roadmap

> **Mission**: Make ML/data workflows reliably portable, readable, and structurally type-safe using human-friendly JSON.

---

## 🎯 Core Principles (Non-Negotiable)

### ✅ **Minimal Dependencies**
- **Zero required dependencies** for core functionality
- Optional dependencies only for specific integrations (pandas, torch, etc.)
- Never add dependencies that duplicate Python stdlib functionality

### ✅ **Performance First**
- Maintain <3x stdlib JSON overhead for simple types
- Benchmark-driven development with regression prevention
- Memory efficiency through configurable limits and smart defaults

### ✅ **Comprehensive Test Coverage**
- Maintain >90% test coverage across all features
- Test all edge cases and failure modes
- Performance regression testing for every release

---

## 🎯 Current State (v0.1.4)

### ✅ **Foundation Complete**
- **Core Serialization**: 20+ data types, circular reference detection, security limits
- **Configuration System**: 4 preset configs + 13+ configurable options
- **Advanced Type Handling**: Complex numbers, decimals, UUIDs, paths, enums, collections
- **ML/AI Integration**: PyTorch, TensorFlow, scikit-learn, NumPy, JAX, PIL
- **Pandas Deep Integration**: 6 DataFrame orientations, Series, Categorical, NaN handling
- **Performance Optimizations**: Early detection, memory streaming, configurable limits
- **Comprehensive Testing**: 83% coverage, 300+ tests, benchmark suite

### 📊 **Performance Baseline**
- Simple JSON: 1.6x overhead vs stdlib (excellent for added functionality)
- Complex types: Only option for UUIDs/datetime/ML objects in pure JSON
- Advanced configs: 15-40% performance improvement over default

---

## 🚀 Focused Roadmap

> **Philosophy**: Deepen what datason uniquely does well rather than expanding scope

### **v0.3.0 - Pickle Bridge**
> *"Convert legacy ML pickle files to portable JSON - solve a real workflow pain point"*

#### 🎯 **Unique Value Proposition**
No other JSON serializer handles the ML community's massive pickle legacy. This bridges the gap safely.

```python
# Convert pickle to datason JSON (unique capability)
import datason

# Safe conversion with class whitelisting  
json_data = datason.from_pickle("model.pkl",
                                safe_classes=["sklearn", "numpy", "pandas"])

# Bulk migration tools for ML teams
datason.convert_pickle_directory("old_models/", "json_models/")
```

#### 🔧 **Implementation Goals**
- **Zero new dependencies** - use stdlib `pickle` module only
- **Security-first** - whitelist approach for safe class loading
- **Leverage existing type handlers** - reuse 100% of current ML object support
- **Maintain performance** - streaming conversion for large pickle files

#### 📈 **Success Metrics**
- Support 95%+ of sklearn/torch/pandas pickle files
- Zero security vulnerabilities in pickle processing
- <5% performance overhead vs direct pickle loading
- No new dependencies added

---

### **v0.3.5 - Advanced ML Types**
> *"Handle more ML framework objects that competitors can't serialize"*

#### 🎯 **Unique Value Proposition**
Extend datason's unique strength - serializing ML objects that other JSON libraries simply can't handle.

```python
# New ML object support (competitors can't do this)
import xarray as xr
import dask.dataframe as dd

data = {
    "xarray_dataset": xr.Dataset({"temp": (["x", "y"], np.random.random((3, 4)))}),
    "dask_dataframe": dd.from_pandas(large_df, npartitions=4),
    "pytorch_dataset": torchvision.datasets.MNIST(root="./data"),
    "huggingface_tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased")
}

# Only datason can serialize this to readable JSON
result = datason.serialize(data, config=get_ml_config())
```

#### 🔧 **Implementation Goals**
- **Extend type handler system** - no architectural changes needed
- **Optional dependencies only** - graceful fallbacks when libs unavailable
- **Consistent with existing patterns** - reuse configuration system
- **Maintain performance** - efficient handling of large scientific objects

#### 📈 **Success Metrics**
- Support 10+ additional ML/scientific libraries
- Zero new required dependencies
- Maintain existing performance characteristics
- 100% backward compatibility

---

### **v0.4.0 - Performance & Memory Optimization**
> *"Make datason the fastest option for ML object serialization"*

#### 🎯 **Unique Value Proposition**
Other JSON libraries either can't handle ML objects or are slow. Make datason both capable AND fast.

```python
# Optimized for large ML workflows
import datason

# Memory-efficient streaming for large objects
with datason.stream_serialize("large_experiment.json") as stream:
    stream.write({"model": huge_model})
    stream.write({"data": massive_dataset})
    # Memory usage stays bounded

# Parallel serialization for multi-object workflows
results = datason.serialize_parallel([model1, model2, model3])
```

#### 🔧 **Implementation Goals**
- **Zero new dependencies** - optimize existing algorithms
- **Memory streaming** - handle objects larger than RAM
- **Parallel processing** - utilize multiple cores efficiently  
- **Profile-driven optimization** - target real-world ML bottlenecks

#### 📈 **Success Metrics**
- 50%+ performance improvement for large ML objects
- Handle 10GB+ objects with <2GB RAM usage
- Maintain <2x stdlib overhead for simple JSON
- Zero new dependencies

---

### **v0.4.5 - Typed Deserialization & Round-Trip Support**
> *"Complete the portability story with safe data reconstruction"*

#### 🎯 **Unique Value Proposition**
Enable truly portable ML workflows by safely reconstructing Python objects from datason JSON.

```python
# Template-based deserialization with type safety
import datason

# Infer template from existing objects
template = datason.infer_template({"features": np.array([[1,2,3]]), "timestamp": datetime.now()})
# Result: {"features": Array(dtype="float64", shape=(-1, 3)), "timestamp": DateTime()}

# Type-safe reconstruction
json_data = '{"features": [[1.0, 2.0, 3.0]], "timestamp": "2024-01-15T10:30:00"}'
reconstructed = datason.cast_to_template(json_data, template)
assert reconstructed["features"].dtype == np.float64  # Type guaranteed
```

#### 🔧 **Implementation Goals**
- **Leverage existing type handlers** - reuse serialization logic in reverse
- **Zero new dependencies** - use stdlib and existing optional deps
- **Template inference** - automatically generate cast templates from examples
- **Type safety** - prevent runtime type errors in ML pipelines

#### 📈 **Success Metrics**
- 99%+ fidelity for numpy array round-trips (dtype, shape, values)
- Support for 15+ cast types (arrays, datetime, ML objects)
- <20% overhead vs naive JSON parsing
- Zero runtime type errors with proper templates

---

### **v0.5.0 - Configuration Refinement**
> *"Perfect the configuration system based on real-world usage"*

#### 🎯 **Unique Value Proposition**
No other JSON serializer offers ML-specific configuration presets. Refine this unique advantage.

```python
# Enhanced presets based on user feedback
import datason
from datason.config import get_inference_config, get_research_config

# New specialized configurations
inference_config = get_inference_config()  # Optimized for model serving
research_config = get_research_config()    # Preserve maximum information
logging_config = get_logging_config()      # Safe for production logs
training_config = get_training_config()    # Balance speed and fidelity

# Environment-based auto-configuration
datason.auto_configure()  # Detects ML environment and optimizes
```

#### 🔧 **Implementation Goals**
- **Refine existing system** - no architectural changes
- **User feedback driven** - address real pain points
- **Maintain simplicity** - keep API surface small
- **Performance tuning** - optimize configuration combinations

#### 📈 **Success Metrics**
- 90%+ user satisfaction with default configurations
- 25%+ performance improvement for common workflows
- Zero breaking changes to existing API
- Maintain zero required dependencies

---

### **v0.5.5 - Production Safety & Redaction**
> *"Make datason safe for production ML logging and compliance"*

#### 🎯 **Unique Value Proposition**
ML workflows often contain sensitive data. Provide built-in redaction without breaking serialization fidelity.

```python
# Safe logging configuration for production ML
import datason
from datason.config import SerializationConfig

# Field-level redaction with smart patterns
config = SerializationConfig(
    redact_fields=["password", "api_key", "*.secret", "user.email"],
    redact_large_objects=True,  # Auto-redact >10MB objects
    redact_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"],  # Credit cards
    redaction_replacement="<REDACTED>",
    include_redaction_summary=True
)

# Safe for production logs
result = datason.serialize(ml_experiment_data, config=config)
```

#### 🔧 **Implementation Goals**
- **Extend configuration system** - no new dependencies
- **Pattern-based redaction** - regex and field path matching  
- **Audit trails** - track what was redacted and why
- **Performance conscious** - minimal overhead when not redacting

#### 📈 **Success Metrics**
- 99.9%+ sensitive data detection for common patterns
- <5% false positive rate for redaction
- <10% performance overhead for redaction processing
- Zero new dependencies

---

### **v0.6.0 - Snapshot Testing & ML DevX**
> *"Turn datason's readable JSON into powerful ML testing infrastructure"*

#### 🎯 **Unique Value Proposition**
Leverage datason's human-readable JSON to create the best ML testing experience available.

```python
# Snapshot testing for ML workflows
import datason

# Generate readable test snapshots
@datason.snapshot_test("test_model_prediction")
def test_model_output():
    model = load_trained_model()
    prediction = model.predict(test_data)

    # Auto-generates human-readable JSON snapshot
    datason.assert_snapshot(prediction, normalize_floats=True)

# Update snapshots when behavior intentionally changes
datason.update_snapshots("test_model_*", reason="Added new features")

# Compare model outputs semantically
datason.assert_equivalent(old_predictions, new_predictions,
                         tolerance=1e-6, ignore_fields=["timestamp"])
```

#### 🔧 **Implementation Goals**
- **Leverage existing serialization** - build on proven JSON generation
- **Git-friendly diffs** - human-readable changes in model outputs
- **ML-specific normalization** - handle float precision, timestamps, etc.
- **Integration friendly** - work with pytest, unittest, CI/CD

#### 📈 **Success Metrics**
- 50%+ reduction in ML test maintenance overhead
- Support for 95%+ of ML output types
- <10s snapshot update time for large test suites
- Zero false positive failures from irrelevant changes

---

### **v0.6.5 - Test Infrastructure & Quality**
> *"Achieve industry-leading reliability for production ML workflows"*

#### 🎯 **Unique Value Proposition**
ML workflows need bulletproof reliability. Make datason the most tested serialization library.

```python
# Enhanced testing and validation tools
import datason

# Built-in validation for ML workflows
result = datason.serialize(model, validate=True)  # Catches issues early

# Property-based testing for edge cases
datason.test_roundtrip(my_custom_object)  # Ensures perfect fidelity

# Performance monitoring
with datason.monitor() as m:
    result = datason.serialize(large_data)
    print(f"Serialized {m.size_mb}MB in {m.time_ms}ms")
```

#### 🔧 **Implementation Goals**
- **Expand test coverage to 95%+** - test all edge cases
- **Property-based testing** - use hypothesis for edge case discovery
- **Performance regression tests** - prevent performance degradation
- **ML-specific test utilities** - help users test their integrations

#### 📈 **Success Metrics**
- 95%+ test coverage across all modules
- Zero critical bugs in production deployments
- Comprehensive property-based test suite
- Built-in performance monitoring tools

---

### **v0.7.0 - Delta Serialization & Efficiency**
> *"Optimize storage and improve traceability for evolving ML objects"*

#### 🎯 **Unique Value Proposition**
Make ML experiment tracking and model versioning storage-efficient with structural diffs.

```python
# Delta-aware serialization for efficient storage
import datason

# Only serialize what changed
baseline_model = load_model("v1.0")
updated_model = train_incremental(baseline_model, new_data)

delta = datason.serialize_delta(updated_model, baseline=baseline_model)
# Result: {"changed_params": {"layer_3.weight": [...]}, "metadata": {...}}

# Reconstruct from baseline + delta
reconstructed = datason.apply_delta(baseline_model, delta)

# Great for Git-friendly experiment tracking
datason.save_experiment_delta("experiment_v2.json", new_experiment,
                             baseline="experiment_v1.json")
```

#### 🔧 **Implementation Goals**
- **Structural diff algorithms** - efficient comparison of deep objects
- **Configurable sensitivity** - control what constitutes a "change"
- **Storage optimization** - significant space savings for incremental updates
- **Git integration** - meaningful diffs for version control

#### 📈 **Success Metrics**
- 80%+ storage reduction for incremental model updates
- <100ms delta computation for typical ML models
- Support delta chains of 50+ steps without degradation
- Human-readable diffs in version control

---

### **v0.8.0 - Documentation & Ecosystem**
> *"Make datason the easiest ML serialization library to adopt"*

#### 🎯 **Unique Value Proposition**
Complex ML serialization made simple through excellent documentation and examples.

```python
# Comprehensive examples for every ML use case
import datason

# Interactive documentation with runnable examples
datason.examples.pytorch_model_serialization()
datason.examples.pandas_workflow()
datason.examples.sklearn_pipeline()

# Migration guides from other solutions
datason.migrate_from_pickle(existing_workflow)
datason.migrate_from_joblib(sklearn_artifacts)
```

#### 🔧 **Implementation Goals**
- **Comprehensive documentation** - cover every use case
- **Interactive examples** - runnable code for all features
- **Migration guides** - help users switch from pickle/joblib
- **Performance guides** - help users optimize for their workloads

#### 📈 **Success Metrics**
- 100% API documentation coverage
- Migration guides for 5+ popular alternatives
- Interactive examples for all major ML frameworks
- <5 minute time-to-first-success for new users

---

## 🚫 What We Won't Build

### **Schema Validation**
- **Why Not**: Covered excellently by Pydantic, marshmallow, cerberus
- **Instead**: Focus on serializing whatever users already validate

### **Cloud Storage Integration**
- **Why Not**: Adds dependencies, covered by cloud SDKs
- **Instead**: Focus on generating JSON that works with any storage

### **Cross-Format Support (Arrow, Protobuf)**
- **Why Not**: Violates "human-friendly JSON" mission
- **Instead**: Perfect JSON serialization, let users convert if needed

### **Enterprise Features (Auth, Monitoring, etc.)**
- **Why Not**: Adds complexity and dependencies far beyond core mission
- **Instead**: Integrate well with existing enterprise tools

### **Plugin System**
- **Why Not**: Adds architectural complexity, can revisit post-v1.0
- **Decision**: Document for potential v1.0+ exploration, focus on core excellence first

---

## 🎯 Success Metrics

### **Technical Excellence**
- **Performance**: Always <3x stdlib JSON for simple types
- **Reliability**: >99.9% uptime for critical ML workflows
- **Coverage**: Support 95%+ of common ML objects without dependencies
- **Quality**: 95%+ test coverage with zero critical production bugs

### **Adoption Goals**
- **v0.3.0**: 5,000+ monthly active users in ML community
- **v0.5.0**: Standard tool in 3+ major ML frameworks' documentation
- **v0.7.0**: 50,000+ downloads, referenced in ML courses/tutorials

### **Community Impact**
- **Unique Value**: Only JSON serializer that "just works" for ML objects
- **Reliability**: Teams trust datason for production ML pipelines
- **Simplicity**: Stays true to zero-dependency, minimal-complexity philosophy

---

## 🤝 Community & Feedback

### **Current Users**
- Share your ML serialization pain points
- Report performance bottlenecks with real workloads
- Suggest ML frameworks we should prioritize

### **ML Framework Authors**
- Partner on official integration examples
- Provide feedback on type handler implementations
- Help test edge cases in your frameworks

### **Enterprise Teams**
- Share production use cases and requirements
- Provide feedback on configuration presets
- Help validate reliability improvements

---

*Roadmap Principles: Stay focused, stay fast, stay simple, solve real problems*

*Last updated: JUne 2025 | Next review: Q3 2025*
