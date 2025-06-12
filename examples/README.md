# 📚 Datason Examples Gallery

Comprehensive examples demonstrating datason's **simple & direct API** alongside traditional features, with special focus on **UUID/Pydantic compatibility** and web framework integration.

## 🎯 **Simple & Direct API - No Configuration Needed!**

**New in v0.9.0:** Just pick the right function for your use case!

```python
import datason as ds

# Web APIs - automatic UUID handling, clean JSON
api_data = ds.dump_api(response_data)  # UUIDs become strings automatically

# ML models - automatic framework detection  
ml_data = ds.dump_ml(model_data)      # Optimized for ML objects

# Security - automatic PII redaction
safe_data = ds.dump_secure(user_data) # Redacts emails, SSNs, etc.

# Choose your loading success rate
basic_data = ds.load_basic(json_string)    # 60-70% success, fastest
smart_data = ds.load_smart(json_string)    # 80-90% success, balanced
perfect_data = ds.load_perfect(json_string, template)  # 100% success
```

## 🚀 **Quick Start: UUID + Pydantic Problem Solved**

**Start here if you're using FastAPI, Django, or Pydantic:**

### [`uuid_pydantic_quickstart.py`](uuid_pydantic_quickstart.py) ⭐ **Most Important**

**THE solution for UUID compatibility issues with Pydantic models.**

```bash
# Run the interactive demo
python examples/uuid_pydantic_quickstart.py
```

**What you'll see:**
- ❌ The problem: UUIDs become objects, breaking Pydantic validation
- ✅ The solution: Use `get_api_config()` to keep UUIDs as strings  
- 🎉 Result: Perfect FastAPI/Pydantic/Django compatibility

**Perfect for:** FastAPI APIs, Django REST Framework, Flask JSON endpoints

---

## 🌐 **Web Framework Integration Examples**

### [`fastapi_integration_guide.py`](fastapi_integration_guide.py)

**Complete FastAPI application with datason integration.**

```bash
# Run the FastAPI server
python examples/fastapi_integration_guide.py
# Visit: http://localhost:8000/docs
```

**Features:**
- Real-world FastAPI application patterns
- Pydantic model integration with string UUIDs
- Database simulation with proper UUID handling
- Performance testing endpoints
- Middleware integration examples

**Key endpoints:**
- `GET /users/` - List users with UUID processing
- `POST /users/` - Create user with Pydantic validation
- `GET /demo/complex-data` - Complex nested UUID handling
- `GET /demo/performance` - Performance benchmarking

### [`django_integration_guide.py`](django_integration_guide.py)

**Django models, views, and admin integration.**

```bash
# Setup and run Django example
python examples/django_integration_guide.py
```

**Features:**
- Django model serialization with datason mixins
- Admin interface with datason processing
- Django REST Framework integration (if available)
- JSON field processing with datason
- Form data handling patterns

### [`uuid_api_compatibility_demo.py`](uuid_api_compatibility_demo.py)

**Interactive demonstration of the UUID solution in action.**

```bash
python examples/uuid_api_compatibility_demo.py
```

**Shows:**
- Before/after performance comparison
- Different configuration strategies
- Real-world data patterns
- Integration with various frameworks

---

## 🧠 **Core Functionality Examples**

### [`basic_usage.py`](basic_usage.py) ⭐ **Simple API First**

**Start here! Showcases the simple & direct API alongside traditional operations.**

```bash
python examples/basic_usage.py
```

**Learn:**
- ✨ **Simple API:** `dump_api()`, `dump_ml()`, `dump_secure()`, `load_basic()`, `load_smart()`
- 🔧 **Traditional API:** `serialize()`, `deserialize()` with configuration
- 🆔 **UUID handling:** Automatic string conversion for APIs
- 🔒 **Security:** Automatic PII redaction patterns
- 📊 **Progressive loading:** Choose your success rate (60% → 100%)

### [`modern_api_demo.py`](modern_api_demo.py)

**Complete tour of the modern intention-revealing API.**

```bash
python examples/modern_api_demo.py
```

**Features:**
- 🎯 **Intention-revealing names:** Clear function purposes
- 📈 **Progressive complexity:** Choose your level of sophistication
- 🔍 **API discovery:** Built-in help and recommendations

### [`advanced_serialization_demo.py`](advanced_serialization_demo.py)

**Advanced serialization patterns and configurations.**

```bash
python examples/advanced_serialization_demo.py
```

**Features:**
- Custom configuration creation
- Security and size limits
- Performance optimization
- Edge case handling

---

## 🤖 **Machine Learning Examples**

### [`ai_ml_examples.py`](ai_ml_examples.py) ⭐ **Simple ML API**

**ML/AI workflows using the simple & direct API.**

```bash
python examples/ai_ml_examples.py
```

**Features:**
- ✨ **Simple API:** Just use `dump_ml()` and `dump_api()` - no configuration needed!
- 🧠 **Auto-detection:** Automatically handles ML frameworks
- 🌐 **API responses:** Clean JSON for FastAPI/Flask with `dump_api()`
- 📊 **15+ frameworks:** PyTorch, TensorFlow, Scikit-learn, NumPy, Pandas, and more

### [`advanced_ml_examples.py`](advanced_ml_examples.py)

**Comprehensive ML workflow examples across multiple frameworks.**

```bash
python examples/advanced_ml_examples.py
```

**Advanced ML Scenarios:**
- 🔥 **PyTorch:** Tensor serialization and training workflows
- 🤖 **Scikit-learn:** Pipeline metadata and model comparison
- 👁️ **Computer Vision:** Image processing and CNN workflows
- 📊 **Time Series:** Forecasting and temporal analysis
- 📝 **NLP:** Text processing and transformer models
- 🔬 **Experiment Tracking:** Hyperparameter optimization workflows

## 🔗 **Framework Integration Examples**

**Framework-specific examples are in the [`framework_integrations/`](framework_integrations/) directory:**

```bash
ls examples/framework_integrations/
```

**Available integrations:**
- 🤖 **ML Serving:** BentoML, Ray Serve, MLflow, Seldon/KServe
- 🌐 **Web Frameworks:** FastAPI, Django with UUID/Pydantic compatibility
- 🎨 **UI Frameworks:** Streamlit, Gradio dashboards

See [`framework_integrations/README.md`](framework_integrations/README.md) for details.

---

## 🔧 **Advanced Features**

### [`auto_detection_and_metadata_demo.py`](auto_detection_and_metadata_demo.py)

**Advanced type detection and perfect round-trip serialization.**

```bash
python examples/auto_detection_and_metadata_demo.py
```

**Learn:**
- ✨ **Simple API:** Progressive loading (`load_basic()` → `load_smart()` → `load_perfect()`)
- 🔍 **Auto-detection:** Intelligent type recognition from JSON
- 🎯 **Aggressive mode:** Pandas DataFrame/Series detection
- 🔄 **Type metadata:** Perfect round-trip serialization
- ⚡ **Performance:** Comparison of different detection modes

### [`chunked_and_template_demo.py`](chunked_and_template_demo.py)

**Large data processing and template-based deserialization.**

```bash
python examples/chunked_and_template_demo.py
```

**Features:**
- Memory-efficient chunked processing
- Template-based validation
- Streaming for large datasets
- Schema enforcement

### [`pickle_bridge_demo.py`](pickle_bridge_demo.py)

**Migration from pickle to datason.**

```bash
python examples/pickle_bridge_demo.py
```

**Learn:**
- Pickle compatibility layer
- Migration strategies
- Security improvements over pickle
- Performance comparisons

---

## 🛡️ **Security & Privacy Examples**

### [`security_patterns_demo.py`](security_patterns_demo.py)

**Security features and data protection.**

```bash
python examples/security_patterns_demo.py
```

**Features:**
- PII redaction patterns
- Size and depth limits
- Safe deserialization
- Audit logging

---

## 🎯 **Domain-Specific Examples**

### [`domain_config_demo.py`](domain_config_demo.py)

**Specialized configurations for different domains.**

```bash
python examples/domain_config_demo.py
```

**Configurations:**
- Financial data processing
- Healthcare data handling
- Research data management
- API response formatting

### [`modern_api_demo.py`](modern_api_demo.py)

**Modern intention-revealing API examples.**

```bash
python examples/modern_api_demo.py
```

**Features:**
- Progressive complexity patterns
- Domain-specific functions
- Composable utilities
- Self-documenting code

---

## 🔧 **Utility Examples**

### [`enhanced_utils_example.py`](enhanced_utils_example.py)

**Utility functions and helper patterns.**

```bash
python examples/enhanced_utils_example.py
```

### [`bidirectional_example.py`](bidirectional_example.py)

**Bidirectional serialization patterns.**

```bash
python examples/bidirectional_example.py
```

---

## 🎯 **Quick Navigation by Use Case**

### 🌐 **Web Development**
```bash
# FastAPI + Pydantic compatibility
python examples/uuid_pydantic_quickstart.py

# Complete FastAPI integration
python examples/fastapi_integration_guide.py

# Django patterns
python examples/django_integration_guide.py
```

### 🤖 **Machine Learning**
```bash
# Basic ML integration
python examples/ai_ml_examples.py

# Advanced ML workflows
python examples/advanced_ml_examples.py

# Model serving examples
python examples/bentoml_integration_guide.py
python examples/ray_serve_integration_guide.py
python examples/streamlit_gradio_integration.py
python examples/mlflow_artifact_tracking.py
python examples/seldon_kserve_integration.py
```

### 🔒 **Security & Production**
```bash
# Security patterns
python examples/security_patterns_demo.py

# Domain-specific configs
python examples/domain_config_demo.py
```

### ⚡ **Performance & Scale**
```bash
# Large data handling
python examples/chunked_and_template_demo.py

# Performance optimization
python examples/advanced_serialization_demo.py
```

---

## 🚀 **Integration Checklist**

### For FastAPI Projects
- [ ] Run `uuid_pydantic_quickstart.py` to understand the solution
- [ ] Review `fastapi_integration_guide.py` for patterns
- [ ] Set `API_CONFIG = get_api_config()` in your app
- [ ] Use `datason.auto_deserialize(data, config=API_CONFIG)` consistently

### For Django Projects  
- [ ] Run `django_integration_guide.py` for model patterns
- [ ] Add datason mixins to your models
- [ ] Configure JSON field processing
- [ ] Set up API views with proper UUID handling

### For ML/AI Projects
- [ ] Run `ai_ml_examples.py` for library support
- [ ] Review `advanced_ml_examples.py` for workflows
- [ ] Choose between `get_ml_config()` and `get_api_config()` based on needs
- [ ] Try `bentoml_integration_guide.py` for BentoML services
- [ ] Use `ray_serve_integration_guide.py` for Ray Serve deployments
- [ ] Explore `streamlit_gradio_integration.py` for UI demos
- [ ] Log experiments with `mlflow_artifact_tracking.py`
- [ ] Wrap models using `seldon_kserve_integration.py`

---

## 💡 **Tips for Success**

1. **Start with UUID compatibility**: If you use web frameworks, run `uuid_pydantic_quickstart.py` first
2. **Use consistent configuration**: Pick one config (usually `get_api_config()`) and use it throughout your app
3. **Test with real data**: Try your actual data patterns with the examples
4. **Performance matters**: Use `chunked_and_template_demo.py` for large datasets
5. **Security first**: Review `security_patterns_demo.py` for production deployments

---

## 🔗 **Related Documentation**

- [📖 **API Integration Guide**](../docs/features/api-integration.md) - Complete web framework integration
- [⚙️ **Configuration System**](../docs/api/configuration.md) - All configuration options
- [🚀 **Quick Start Guide**](../docs/user-guide/quick-start.md) - Getting started
- [🏠 **Main Documentation**](../docs/index.md) - Complete documentation

---

**🎉 Happy coding with datason!**

*Most developers start with `uuid_pydantic_quickstart.py` - it solves the #1 integration issue!*
