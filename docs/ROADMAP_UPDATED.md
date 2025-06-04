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

### ⚠️ **Critical Issues from Real-World Usage**
- **DataFrame orientation configuration not working as documented** (URGENT)
- **Limited output type flexibility** (always returns JSON-safe primitives) (HIGH)
- **Incomplete round-trip capabilities** - Serialization works, but deserialization gaps exist (CRITICAL)
- **Missing performance monitoring** for production optimization (MEDIUM)

### 🔍 **Validated by Financial ML Team Integration**
**Results from Production Use**:
- ✅ **+20% serialization performance improvement**
- ✅ **+15% API response time improvement**
- ✅ **+25% memory efficiency for DataFrame operations**
- ✅ **-40% manual type conversion code eliminated**
- ✅ **100% type preservation accuracy for serialization**
- ❌ **Round-trip deserialization gaps identified**

---

## 🚀 Updated Focused Roadmap

> **Philosophy**: Perfect bidirectional ML serialization before expanding scope

### **v0.2.5 - Critical Configuration Fixes** (URGENT - 1-2 weeks)
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

### **v0.3.0 - Complete Round-Trip Support & Pickle Bridge** (HIGH PRIORITY - 4-6 weeks)
> *"Perfect bidirectional type preservation - the foundation of ML portability"*

#### 🎯 **Unique Value Proposition**
**PRIORITY SHIFT**: Reliable deserialization is equally important as serialization for production ML workflows.

```python
# Complete bidirectional support with type metadata
data = {
    "model": sklearn_model,
    "features": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    "timestamp": datetime.now(),
    "config": {"learning_rate": Decimal("0.001")}
}

# Serialize with type hints for perfect reconstruction
serialized = datason.serialize(data, include_type_hints=True)
# → {"model": {...}, "__datason_types__": {"features": "numpy.float32[1,3]"}}

# Perfect reconstruction - this MUST work reliably
reconstructed = datason.deserialize_with_types(serialized)
assert type(reconstructed["features"]) == np.ndarray
assert reconstructed["features"].dtype == np.float32
assert reconstructed["features"].shape == (1, 3)

# Legacy pickle bridge with perfect round-trips
json_data = datason.from_pickle("model.pkl", include_type_hints=True)
original_model = datason.deserialize_with_types(json_data)
```

#### 🔧 **Implementation Goals**
- **CRITICAL**: Audit all 20+ supported types for round-trip completeness
- **Type metadata system** - include_type_hints for perfect reconstruction
- **Pickle bridge with round-trips** - convert legacy files with full fidelity
- **Complete output type control** - datetime, series, dataframe, numpy options
- **Zero new dependencies** - extend existing type handler system

#### 📈 **Success Metrics**
- **99.9% round-trip fidelity** for all supported types (dtype, shape, values)
- **100% of ML objects** can be serialized AND reconstructed perfectly
- Support 95%+ of sklearn/torch/pandas pickle files with full round-trips
- Zero new dependencies added
- Complete type reconstruction test suite

---

### **v0.3.5 - Smart Deserialization & Enhanced ML Types** (6-8 weeks)
> *"Intelligent type reconstruction + domain-specific type handlers"*

#### 🎯 **Unique Value Proposition**
Combine auto-detection with custom domain types (financial ML team request).

```python
# Smart auto-detection deserialization
reconstructed = datason.safe_deserialize(json_data)  
# Uses heuristics: "2023-01-01T00:00:00" → datetime, [1,2,3] → list/array

# Custom domain type handlers (financial ML team validated need)
@datason.register_type_handler
class MonetaryAmount:
    def serialize(self, value):
        return {"amount": str(value.amount), "currency": value.currency}

    def deserialize(self, data):
        return MonetaryAmount(data["amount"], data["currency"])

# Extended ML framework support
data = {
    "xarray_dataset": xr.Dataset({"temp": (["x", "y"], np.random.random((3, 4)))}),
    "dask_dataframe": dd.from_pandas(large_df, npartitions=4),
    "financial_instrument": MonetaryAmount("100.50", "USD")
}
result = datason.serialize(data, config=get_ml_config())
reconstructed = datason.deserialize_with_types(result)  # Perfect round-trip
```

#### 🔧 **Implementation Goals**
- **Custom type handler registration** - extensible type system for domain types
- **Auto-detection deserialization** - safe_deserialize with smart guessing
- **Extended ML support** - xarray, dask, huggingface, scientific libs
- **Migration utilities** - help teams convert from other formats

#### 📈 **Success Metrics**
- 85%+ accuracy in auto-type detection for common patterns
- Support 10+ additional ML/scientific libraries
- Custom type handler system for domain-specific types
- Migration utilities for common serialization libraries

---

### **v0.4.0 - Performance Optimization & Monitoring** (8-10 weeks)
> *"Make datason the fastest option with full visibility into performance"*

#### 🎯 **Unique Value Proposition**
**ACCELERATED FROM v0.6.5**: Financial ML team validated need for performance monitoring.

```python
# Performance monitoring (financial team high-priority request)
with datason.profile() as prof:
    result = datason.serialize(large_financial_dataset)

print(prof.report())
# Output:
# Serialization Time: 1.2s, Memory Peak: 45MB
# Type Conversions: 1,247 (UUID: 89, DateTime: 445, DataFrame: 1)  
# Bottlenecks: DataFrame orientation conversion (0.8s)

# Memory-efficient streaming for large objects  
with datason.stream_serialize("large_experiment.json") as stream:
    stream.write({"model": huge_model})
    stream.write({"data": massive_dataset})

# Enhanced chunked processing (conditional based on user demand)
chunks = datason.serialize_chunked(massive_df, chunk_size=10000)
for chunk in chunks:
    store_chunk(chunk)  # Bounded memory usage
```

#### 🔧 **Implementation Goals**
- **Performance profiling tools** - detailed bottleneck identification
- **Memory streaming optimization** - handle objects larger than RAM
- **Chunked processing** - conditional feature based on additional user validation
- **Zero new dependencies** - use stdlib profiling tools

#### 📈 **Success Metrics**
- Built-in performance monitoring with zero dependencies
- 50%+ performance improvement for large ML objects
- Handle 10GB+ objects with <2GB RAM usage
- Maintain <2x stdlib overhead for simple JSON

---

### **v0.4.5 - Advanced ML Types & Domain Configurations** (10-12 weeks)  
> *"Domain expertise and comprehensive ML ecosystem support"*

#### 🎯 **Unique Value Proposition**
**MOVED UP FROM v0.5.0**: Financial team explicitly requested domain-specific configurations.

```python
# Domain-specific configurations (financial team validated)
financial_config = get_financial_config()     # Optimized for financial ML
time_series_config = get_time_series_config() # Temporal data analysis  
inference_config = get_inference_config()     # Model serving optimized
research_config = get_research_config()       # Maximum information preservation

# Auto-environment detection
datason.auto_configure()  # Detects context and optimizes automatically

# Custom preset creation
trading_config = datason.create_preset(
    base=financial_config,
    datetime_output="timestamp",
    decimal_precision=8,
    name="trading_pipeline"
)

# Enhanced ML ecosystem support
advanced_ml_data = {
    "huggingface_model": AutoModel.from_pretrained("bert-base"),
    "ray_dataset": ray.data.from_pandas(large_df),
    "optuna_study": study.best_params
}
```

#### 🔧 **Implementation Goals**
- **Domain-specific presets** - financial, time-series, inference, research configs
- **Advanced ML framework support** - huggingface, ray, optuna, MLflow
- **Auto-configuration** - environment detection and optimization
- **Zero new dependencies** - extend existing configuration system

#### 📈 **Success Metrics**
- 95%+ user satisfaction with domain-specific presets  
- 8+ specialized configurations covering major use cases
- 30%+ performance improvement with optimized presets
- Support for 15+ additional ML/AI frameworks

---

### **v0.5.0 - Production Safety & Enterprise Readiness** (12-14 weeks)
> *"Enterprise-grade safety without compromising simplicity"*

#### 🎯 **Unique Value Proposition**
Production-ready features for sensitive ML data without enterprise bloat.

```python
# Enhanced safety features for production ML
config = SerializationConfig(
    redact_fields=["password", "api_key", "*.secret", "user.email"],
    redact_large_objects=True,  # Auto-redact >10MB objects  
    redact_patterns=[r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"],  # Credit cards
    redaction_replacement="<REDACTED>",
    include_redaction_summary=True,
    audit_trail=True  # Track all redaction operations
)

# Safe processing with audit trails
result = datason.serialize(sensitive_ml_data, config=config)
```

#### 🔧 **Implementation Goals**
- **Enhanced redaction** - pattern matching and field-level safety
- **Audit trails** - track all redaction operations  
- **Production logging safety** - prevent sensitive data leaks
- **Zero new dependencies** - extend existing configuration system

#### 📈 **Success Metrics**
- 99.95%+ sensitive data detection for financial patterns
- <2% false positive rate for redaction
- Comprehensive audit trails for compliance
- <8% performance overhead for redaction processing

---

### **v0.5.5 - Snapshot Testing & ML DevX** (14-16 weeks)
> *"Transform readable JSON into powerful ML testing infrastructure"*

#### 🎯 **Unique Value Proposition**
Best-in-class testing experience for ML workflows using human-readable diffs.

```python
# Snapshot testing for ML workflows
@datason.snapshot_test("test_model_prediction")
def test_model_output():
    model = load_trained_model()
    prediction = model.predict(test_data)

    # Auto-generates human-readable JSON snapshot
    datason.assert_snapshot(prediction, normalize_floats=True)

# Cross-configuration compatibility testing
@datason.config_test([financial_config, api_config, inference_config])
def test_cross_config_compatibility(config):
    result = datason.serialize(test_data, config=config)
    # Ensure all configs produce compatible outputs
```

#### 🔧 **Implementation Goals**
- **ML-specific snapshot testing** - handle large outputs efficiently
- **Configuration compatibility testing** - verify cross-preset compatibility
- **CI/CD integration** - seamless workflow integration
- **Zero new dependencies** - build on existing serialization

#### 📈 **Success Metrics**
- 60%+ reduction in ML test maintenance overhead
- Support for chunked outputs in snapshot testing
- Cross-configuration compatibility testing for all presets
- <5s snapshot update time for large test suites

---

## 🚨 Critical Changes from Original Roadmap

### **PRIORITY SHIFTS Based on Real-World Feedback**

1. **Round-Trip Support Moved to v0.3.0** (was v0.4.5)
   - **Rationale**: Deserialization gaps are blocking production adoption
   - **Impact**: Perfect bidirectional support is foundation for ML workflows
   - **Risk**: Low - extends existing architecture

2. **Performance Monitoring Accelerated** (was v0.6.5 → v0.4.0)  
   - **Rationale**: Financial team explicitly validated need
   - **Impact**: Production teams need visibility into optimization
   - **Risk**: Low - uses stdlib profiling tools

3. **Domain Configurations Prioritized** (accelerated timeline)
   - **Rationale**: Financial team requested specific domain presets
   - **Impact**: Reduces onboarding friction for domain experts
   - **Risk**: Low - extends existing configuration system

### **FEATURES REJECTED (Avoiding Feature Creep)**

❌ **Schema Generation & Validation**
- **Reason**: Covered by Pydantic/marshmallow, would require massive dependencies
- **Decision**: Focus on serialization, not validation

❌ **Database ORM Integration**  
- **Reason**: Violates minimal dependencies principle, not ML-specific
- **Decision**: Provide examples, but no direct integration

❌ **Multi-Format Support** (Parquet, Arrow, etc.)
- **Reason**: Violates "human-friendly JSON" mission  
- **Decision**: Stay focused on JSON excellence

❌ **Enterprise Platform Features** (Caching, monitoring, auth)
- **Reason**: Conflicts with simplicity principle
- **Decision**: Integrate with existing enterprise tools, don't build them

### **SELECTIVE ADOPTION Strategy**

**✅ Adopted (~30% of suggestions)**:
- Performance monitoring and profiling
- Custom domain type handlers  
- Enhanced configuration profiles
- Migration utilities
- Streaming optimizations

**❌ Rejected (~70% of suggestions)**:
- Schema validation system
- ORM integration
- Multi-format conversion
- Intelligent caching layer
- Enterprise observability features

---

## 🎯 Success Metrics (Updated)

### **Technical Excellence**  
- **Round-Trip Fidelity**: 99.9%+ accuracy for all supported ML objects
- **Performance**: <2x stdlib JSON for simple types, competitive for complex types
- **Reliability**: All documented features work correctly in production
- **Quality**: 95%+ test coverage with comprehensive round-trip testing

### **Real-World Adoption**
- **Production Use**: Teams can replace 90%+ of custom serialization utilities
- **Configuration Satisfaction**: 95%+ user satisfaction with domain presets  
- **Performance Monitoring**: Built-in profiling adopted by 80%+ of teams
- **Migration Success**: 95%+ successful conversions from pickle/other formats

### **Community Impact**
- **v0.3.0**: 10,000+ monthly active users (round-trip support drives adoption)
- **v0.4.5**: Standard tool in 5+ major ML frameworks' documentation
- **v0.5.5**: 100,000+ downloads, referenced in production ML tutorials

---

## 🔍 **Validation from Financial ML Integration**

This updated roadmap directly addresses production-validated requirements:

✅ **CRITICAL**: Perfect round-trip support (moved to v0.3.0)
✅ **HIGH VALUE**: Performance monitoring (accelerated to v0.4.0)  
✅ **PROVEN NEED**: Domain-specific configurations (financial team request)
✅ **PRACTICAL**: Migration utilities (help teams adopt datason)

**Key Insight**: The financial team achieved impressive results (+20% performance, -40% code reduction) but identified deserialization gaps as the primary blocker for complete adoption.

---

*Roadmap Principles: Perfect bidirectional ML serialization, stay focused, stay fast, stay simple*

*Updated: December 2024 based on financial ML pipeline integration feedback and deserialization audit*  
*Next review: Q1 2025 after v0.3.0 round-trip completion*
