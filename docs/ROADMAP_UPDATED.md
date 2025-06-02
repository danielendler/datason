# datason Product Roadmap (Updated with Integration Feedback)

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

## 🎯 Current State (v0.2.0)

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

### ⚠️ **Known Issues from Real-World Usage**
- DataFrame orientation configuration not working as documented
- Limited output type flexibility (always returns JSON-safe primitives)
- Missing round-trip capabilities for production workflows

---

## 🚀 Updated Focused Roadmap

> **Philosophy**: Deepen what datason uniquely does well, prioritizing real user pain points

### **v0.2.5 - Critical Configuration Fixes** (URGENT)
> *"Fix core functionality blocking production adoption"*

#### 🎯 **Unique Value Proposition**
Make existing configuration system work reliably before adding new features.

```python
# Fix DataFrame orientation (currently broken)
config = SerializationConfig(dataframe_orient="split")
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
result = datason.serialize(df, config=config)
# Must actually return split format: {"index": [0, 1], "columns": ["a", "b"], "data": [[1, 3], [2, 4]]}

# Add basic output type control
config = SerializationConfig(
    datetime_output="object",     # Return datetime objects instead of ISO strings
    series_output="list"          # Return lists instead of dicts
)
```

#### 🔧 **Implementation Goals**
- **Fix existing bugs** - DataFrame orientation configuration
- **Add output type flexibility** - datetime_output, series_output options
- **Performance skip optimization** - check_if_serialized parameter
- **Zero new dependencies** - work within existing architecture

#### 📈 **Success Metrics**
- 100% of documented configuration options work correctly
- Support for 3+ output type options (datetime, series, basic types)
- No breaking changes to existing API
- <48 hour release cycle for critical fixes

---

### **v0.3.0 - Enhanced Configuration & Pickle Bridge**
> *"Convert legacy ML pickle files while providing flexible output options"*

#### 🎯 **Unique Value Proposition**
Bridge the pickle legacy gap while solving the configuration flexibility problem.

```python
# Complete output type control
config = SerializationConfig(
    datetime_output: Literal["iso_string", "timestamp", "object"] = "iso_string",
    series_output: Literal["dict", "list", "object"] = "dict",  
    dataframe_output: Literal["records", "split", "values", "object"] = "records",
    numpy_output: Literal["python_types", "arrays", "objects"] = "python_types"
)

# Enhanced pickle conversion with flexible output
json_data = datason.from_pickle("model.pkl",
                                safe_classes=["sklearn", "numpy", "pandas"],
                                config=config)

# Type hints for round-trip support  
serialized = datason.serialize(data, include_type_hints=True)
# → {"timestamp": {"__value__": "2023-01-01T00:00:00", "__type__": "datetime"}}
```

#### 🔧 **Implementation Goals**
- **Complete configuration flexibility** - all output type options
- **Pickle bridge with security** - whitelist approach for safe class loading
- **Type metadata support** - include_type_hints option
- **Maintain performance** - <5% overhead for new features

#### 📈 **Success Metrics**
- Support 95%+ of sklearn/torch/pandas pickle files
- 100% configurable output types (datetime, series, dataframe, numpy)
- Type hints enable 90%+ round-trip accuracy
- Zero new dependencies added

---

### **v0.3.5 - Smart Deserialization & Advanced ML Types**
> *"Auto-detect types while expanding ML framework support"*

#### 🎯 **Unique Value Proposition**
Intelligent type reconstruction combined with broader ML ecosystem support.

```python
# Smart auto-detection deserialization
reconstructed = datason.safe_deserialize(json_data)  
# Uses heuristics: "2023-01-01T00:00:00" → datetime, [1,2,3] → list/array

# Template-based reconstruction  
template = datason.infer_template(example_data)
typed_data = datason.cast_to_template(json_data, template)

# New ML frameworks
data = {
    "xarray_dataset": xr.Dataset({"temp": (["x", "y"], np.random.random((3, 4)))}),
    "dask_dataframe": dd.from_pandas(large_df, npartitions=4),
    "huggingface_tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased")
}
result = datason.serialize(data, config=get_ml_config())
```

#### 🔧 **Implementation Goals**
- **Heuristic type detection** - safe_deserialize with smart guessing
- **Template inference** - automatic template generation from examples  
- **Extended ML support** - xarray, dask, huggingface, more scientific libs
- **Performance skip checks** - check_if_serialized optimization

#### 📈 **Success Metrics**
- 85%+ accuracy in auto-type detection for common patterns
- Support 10+ additional ML/scientific libraries
- Template inference for 95%+ of common object structures
- <20% performance overhead for smart features

---

### **v0.4.0 - Performance & Memory Optimization with Chunking**
> *"Make datason the fastest option for large-scale ML serialization"*

#### 🎯 **Unique Value Proposition**
Handle massive ML objects efficiently with streaming and chunked processing.

```python
# Memory-efficient streaming (existing plan)
with datason.stream_serialize("large_experiment.json") as stream:
    stream.write({"model": huge_model})
    stream.write({"data": massive_dataset})

# NEW: Chunked processing for very large DataFrames
chunks = datason.serialize_chunked(massive_df, chunk_size=10000)
for chunk in chunks:
    # Process each chunk separately, bounded memory usage
    store_chunk(chunk)

# Parallel processing with chunking
results = datason.serialize_parallel_chunked([df1, df2, df3], chunk_size=5000)
```

#### 🔧 **Implementation Goals**
- **Existing streaming optimization** - handle objects larger than RAM
- **NEW: Chunked processing** - break large objects into manageable pieces
- **Parallel processing** - utilize multiple cores efficiently
- **Memory bounds** - strict limits on memory usage regardless of input size

#### 📈 **Success Metrics**
- Handle 50GB+ DataFrames with <4GB RAM usage
- 50%+ performance improvement for large ML objects
- Chunked processing for objects 10x larger than available memory
- Maintain <2x stdlib overhead for simple JSON

---

### **v0.4.5 - Complete Round-Trip & Type Safety**
> *"Perfect bidirectional type preservation for production workflows"*

#### 🎯 **Unique Value Proposition**
Industry-leading type fidelity for ML pipeline round-trips.

```python
# Enhanced template-based approach (from original plan)
template = datason.infer_template({"features": np.array([[1,2,3]]), "timestamp": datetime.now()})
reconstructed = datason.cast_to_template(json_data, template)

# NEW: Metadata-based round-trips (from user feedback)
serialized = datason.serialize(data, preserve_types="metadata")
# → {"data": [...], "__datason_types__": {"path.to.field": "numpy.float64"}}
reconstructed = datason.deserialize_with_types(serialized)

# Hybrid approach for maximum compatibility
config = SerializationConfig(
    round_trip_method="auto",  # Choose best method per object type
    type_safety_level="strict"  # Fail fast on type mismatches
)
```

#### 🔧 **Implementation Goals**
- **Template AND metadata approaches** - give users choice
- **Automatic method selection** - optimize for each object type
- **Type safety guarantees** - prevent runtime type errors
- **Production reliability** - extensive testing for edge cases

#### 📈 **Success Metrics**
- 99.9%+ fidelity for numpy array round-trips (dtype, shape, values)
- Support for 20+ round-trip types (arrays, datetime, ML objects)
- <15% overhead vs naive JSON parsing
- Zero runtime type errors with proper configuration

---

### **v0.5.0 - Enhanced Configuration & Domain Presets**
> *"Perfect configuration system with real-world domain expertise"*

#### 🎯 **Unique Value Proposition**
Best-in-class configuration system tailored for specific ML/data domains.

```python
# Enhanced presets (existing + new from feedback)
inference_config = get_inference_config()      # Optimized for model serving  
research_config = get_research_config()        # Preserve maximum information
financial_config = get_financial_config()      # Financial ML workflows
time_series_config = get_time_series_config()  # Temporal data analysis
api_config = get_api_config()                  # REST API responses
logging_config = get_logging_config()          # Safe for production logs

# Smart environment detection
datason.auto_configure()  # Detects context and optimizes automatically

# Custom preset creation
my_config = datason.create_preset(
    base=financial_config,
    datetime_output="timestamp",
    enable_chunking=True,
    name="trading_pipeline"
)
```

#### 🔧 **Implementation Goals**
- **Domain expertise encoding** - capture best practices for each field
- **User feedback integration** - address real pain points from production use
- **Configuration composition** - build complex configs from simple parts
- **Environment awareness** - auto-detect and optimize for context

#### 📈 **Success Metrics**
- 95%+ user satisfaction with domain-specific presets
- 8+ specialized configurations covering major use cases
- 30%+ performance improvement with optimized presets
- Auto-configuration success rate >80% for common environments

---

### **v0.5.5 - Production Safety & Redaction**
> *"Make datason safe for production ML logging and compliance"*

#### 🎯 **Unique Value Proposition**
Enterprise-ready safety features for sensitive ML data.

```python
# Enhanced safety features (existing plan + chunking integration)
config = SerializationConfig(
    redact_fields=["password", "api_key", "*.secret", "user.email"],
    redact_large_objects=True,  # Auto-redact >10MB objects
    redact_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"],  # Credit cards
    redaction_replacement="<REDACTED>",
    include_redaction_summary=True,

    # NEW: Integration with chunking
    redact_chunk_boundaries=True,  # Apply redaction per chunk
    audit_trail=True              # Track all redaction operations
)

# Safe chunked processing
safe_chunks = datason.serialize_chunked(sensitive_data,
                                       config=config,
                                       chunk_size=1000)
```

#### 🔧 **Implementation Goals**
- **Enhanced pattern matching** - more sophisticated redaction rules
- **Chunk-aware redaction** - apply safety rules to chunked processing
- **Comprehensive audit trails** - track all redaction operations
- **Performance optimization** - minimal overhead when not redacting

#### 📈 **Success Metrics**
- 99.95%+ sensitive data detection for financial patterns
- <2% false positive rate for redaction
- Audit trails for 100% of redaction operations
- <8% performance overhead for redaction processing

---

### **v0.6.0 - Snapshot Testing & ML DevX**
> *"Transform datason's readable JSON into ML testing infrastructure"*

#### 🎯 **Unique Value Proposition**
Best-in-class testing experience for ML workflows using human-readable diffs.

```python
# Snapshot testing with chunking support (existing + enhanced)
@datason.snapshot_test("test_model_prediction")
def test_model_output():
    model = load_trained_model()
    prediction = model.predict(test_data)

    # Handle large outputs with chunking
    datason.assert_snapshot(prediction,
                           normalize_floats=True,
                           chunk_large_outputs=True)

# NEW: Integration testing across serialization configurations
@datason.config_test([financial_config, api_config, inference_config])
def test_cross_config_compatibility(config):
    result = datason.serialize(test_data, config=config)
    # Ensure all configs produce compatible outputs
```

#### 🔧 **Implementation Goals**
- **Chunk-aware testing** - handle large test outputs efficiently
- **Configuration testing** - verify compatibility across presets
- **ML-specific normalization** - better handling of floating-point precision
- **CI/CD integration** - seamless workflow integration

#### 📈 **Success Metrics**
- 60%+ reduction in ML test maintenance overhead
- Support for chunked outputs in snapshot testing
- Cross-configuration compatibility testing for all presets
- <5s snapshot update time for large test suites

---

## 🚨 Critical Changes from Original Roadmap

### **Added Based on Real-World Feedback**

1. **v0.2.5 - Critical Fixes** (NEW)
   - Fix DataFrame orientation bug (production blocker)
   - Add basic output type control (adoption blocker)
   - Performance skip optimization

2. **Enhanced Configuration Throughout**
   - Complete output type flexibility (v0.3.0)
   - Smart auto-detection (v0.3.5)
   - Domain-specific presets expansion (v0.5.0)

3. **Chunked Processing Integration**
   - Memory-bounded chunked serialization (v0.4.0)
   - Chunk-aware redaction (v0.5.5)
   - Chunked snapshot testing (v0.6.0)

4. **Dual Round-Trip Approaches**
   - Template-based (original plan)
   - Metadata-based (user request)
   - Hybrid auto-selection (best of both)

### **Maintained from Original Roadmap**

- **Core principles** - zero dependencies, performance first, comprehensive testing
- **Pickle bridge** - critical for ML community adoption
- **Advanced ML types** - unique competitive advantage
- **Performance focus** - streaming, parallel processing
- **Production safety** - redaction and audit trails
- **Testing infrastructure** - snapshot testing and ML DevX

### **Timeline Adjustments**

- **Accelerated v0.2.5** - Address critical bugs immediately
- **Enhanced v0.3.0** - Add configuration flexibility alongside pickle bridge  
- **Maintained v0.4.x-v0.6.x** - Keep planned timeline with enhancements

---

## 🎯 Success Metrics (Updated)

### **Technical Excellence**
- **Reliability**: All documented configuration options work correctly
- **Flexibility**: 5+ output type options for all major data types
- **Performance**: <3x stdlib JSON for simple types, bounded memory for large objects
- **Quality**: 95%+ test coverage with zero critical production bugs

### **Real-World Adoption**
- **Production Use**: Users can replace 70%+ of custom serialization utilities
- **Configuration Satisfaction**: 90%+ user satisfaction with domain presets
- **Round-Trip Fidelity**: 99%+ accuracy for ML object reconstruction
- **Large Object Handling**: Support objects 20x larger than available memory

### **Community Impact**
- **v0.3.0**: 8,000+ monthly active users (increased from real-world validation)
- **v0.5.0**: Standard tool in 5+ major ML frameworks' documentation
- **v0.7.0**: 75,000+ downloads, referenced in production ML tutorials

---

## 🤝 Integration Feedback Validation

This updated roadmap directly addresses:

✅ **Critical Blocking Issues**
- DataFrame orientation bug fix (v0.2.5)
- Output type inflexibility (v0.3.0)
- Round-trip deserialization (v0.4.5)

✅ **High-Value Enhancements**  
- Auto-detection deserialization (v0.3.5)
- Chunked processing (v0.4.0)
- Domain-specific presets (v0.5.0)

✅ **Production Requirements**
- Type metadata support (v0.3.0)
- Performance skip optimizations (v0.3.5)
- Enhanced safety features (v0.5.5)

---

*Roadmap Principles: Stay focused, stay fast, stay simple, solve real problems*

*Updated: June 1, 2025 based on financial ML pipeline integration feedback*  
*Next review: Q2 2025 after v0.3.0 release*
