# 📚 DataSON Examples Gallery

**Complete examples demonstrating DataSON's modern API - organized from simple to complex**

Welcome to the comprehensive DataSON examples! This gallery showcases the full power of DataSON's modern, intention-revealing API alongside traditional features. Examples are organized by complexity level to provide a smooth learning path.

---

## 🎯 **Quick Start: Choose Your Learning Path**

### 🟢 **Beginner (Just Getting Started)**
- [`basic_usage.py`](#-1-basic-usage---your-first-steps) - Start here! Modern API basics
- [`modern_api_demo.py`](#-2-modern-api-showcase) - Complete modern API tour

### 🟡 **Intermediate (Building Applications)**  
- [`uuid_pydantic_quickstart.py`](#-3-uuidpydantic-integration) - Web API compatibility
- [`enhanced_utils_example.py`](#-4-enhanced-utilities) - Advanced data operations
- [`file_operations_guide.py`](#-5-file-operations) - File I/O patterns

### 🔴 **Advanced (Production & ML)**
- [`ai_ml_examples.py`](#-6-ml--ai-integration) - Machine learning workflows
- [`advanced_serialization_demo.py`](#-7-advanced-serialization) - Complex configurations
- [`framework_integrations/`](#-8-framework-integrations) - Production deployments

---

## 🚀 **Modern API Overview**

DataSON v0.11+ provides a **clean, intention-revealing API** that makes your code more readable and maintainable:

```python
import datason as ds

# 🌐 Web APIs - Clean JSON with UUID string conversion
api_response = ds.dump_api(user_data)  # Perfect for FastAPI/Django

# 🤖 ML Models - Optimized for ML frameworks  
ml_data = ds.dump_ml(model_artifacts)  # Handles PyTorch, NumPy, pandas

# 🔒 Security - Automatic PII redaction
safe_data = ds.dump_secure(sensitive_data)  # Redacts emails, SSNs

# 🧠 Smart Loading - Progressive intelligence
basic = ds.load_basic(json_str)     # 70% success, fastest
smart = ds.load_smart(json_str)     # 90% success, balanced  
perfect = ds.load_perfect(json_str, template)  # 100% success
```

---

## 📈 **Learning Progression: Simple → Complex**

### 🟢 **Level 1: Basic Operations**

#### 📘 1. Basic Usage - Your First Steps
**File:** [`basic_usage.py`](basic_usage.py) ⭐ **START HERE**

**What you'll learn:**
- ✨ Modern API functions (`dump_api`, `dump_ml`, `load_smart`)
- 🔧 Traditional API for existing codebases
- 🆔 UUID handling for web APIs
- 📊 Progressive loading strategies

```bash
python examples/basic_usage.py
```

**Perfect for:** Complete beginners, quick API overview, first integration

---

#### 📗 2. Modern API Showcase  
**File:** [`modern_api_demo.py`](modern_api_demo.py)

**What you'll learn:**
- 🎯 Intention-revealing function names
- 📈 Progressive complexity options
- 🔍 API discovery and help system
- 🛠️ Configuration best practices

```bash
python examples/modern_api_demo.py
```

**Perfect for:** Understanding the full modern API scope

---

### 🟡 **Level 2: Real-World Applications**

#### 📙 3. UUID/Pydantic Integration
**File:** [`uuid_pydantic_quickstart.py`](uuid_pydantic_quickstart.py) ⭐ **WEB DEVELOPERS**

**What you'll learn:**
- ❌ The UUID compatibility problem
- ✅ The `get_api_config()` solution  
- 🌐 FastAPI/Django/Flask integration
- 🎉 Perfect Pydantic model validation

```bash
python examples/uuid_pydantic_quickstart.py
```

**Perfect for:** Web API developers, FastAPI users, Django REST framework

---

#### 📒 4. Enhanced Utilities
**File:** [`enhanced_utils_example.py`](enhanced_utils_example.py)

**What you'll learn:**
- 🔧 Data transformation utilities
- 📊 Advanced type handling
- 🔍 Data validation patterns
- ⚡ Performance optimization techniques

```bash
python examples/enhanced_utils_example.py
```

**Perfect for:** Data processing applications, utility functions

---

#### 📕 5. File Operations
**File:** [`file_operations_guide.py`](file_operations_guide.py)

**What you'll learn:**
- 📁 Multi-format file handling (JSON, JSONL, CSV)
- 🔄 Streaming and chunked processing
- 💾 Save/load patterns with configurations
- 📦 Batch processing workflows

```bash
python examples/file_operations_guide.py
```

**Perfect for:** File processing applications, data pipelines

---

### 🔴 **Level 3: Advanced & Production**

#### 📚 6. ML & AI Integration
**File:** [`ai_ml_examples.py`](ai_ml_examples.py) ⭐ **ML ENGINEERS**

**What you'll learn:**
- 🤖 15+ ML framework integrations
- 🧠 Automatic framework detection
- 📊 Model artifact management
- 🌐 API response generation

```bash
python examples/ai_ml_examples.py
```

**Perfect for:** ML engineers, data scientists, model serving

---

#### 📘 7. Advanced Serialization  
**File:** [`advanced_serialization_demo.py`](advanced_serialization_demo.py)

**What you'll learn:**
- 🔧 Custom configuration creation
- 🛡️ Security and size limits
- ⚡ Performance optimization
- 🚫 Edge case handling

```bash
python examples/advanced_serialization_demo.py
```

**Perfect for:** Complex applications, custom requirements

---

## 🌐 **Framework Integrations**

**Directory:** [`framework_integrations/`](framework_integrations/)

Production-ready examples for popular frameworks:

### **Web Frameworks**
- 🚀 **[FastAPI](framework_integrations/fastapi_integration_guide.py)** - Complete REST API with UUID handling
- 🌐 **[Django](framework_integrations/django_integration_guide.py)** - Models, views, and admin integration

### **ML Serving Platforms**
- 🤖 **[BentoML](framework_integrations/bentoml_integration_guide.py)** - Model serving with DataSON APIs
- ☁️ **[Ray Serve](framework_integrations/ray_serve_integration_guide.py)** - Scalable ML deployments
- 📈 **[MLflow](framework_integrations/mlflow_artifact_tracking.py)** - Experiment tracking and artifacts
- ⚙️ **[Seldon/KServe](framework_integrations/seldon_kserve_integration.py)** - Kubernetes-native serving

### **Interactive Applications**
- 📊 **[Streamlit](framework_integrations/streamlit_gradio_integration.py)** - Data apps and dashboards
- 🎨 **[Gradio](framework_integrations/streamlit_gradio_integration.py)** - ML interfaces and demos

Each framework integration includes:
- ✅ Complete working examples
- 🔧 Production deployment patterns  
- 📖 Step-by-step setup instructions
- 🐳 Docker configurations
- 🧪 Testing and validation

---

## 🔧 **Advanced Examples**

### **Data Processing & Analytics**
- [`auto_detection_and_metadata_demo.py`](auto_detection_and_metadata_demo.py) - Type detection and metadata
- [`chunked_and_template_demo.py`](chunked_and_template_demo.py) - Large data processing
- [`domain_config_demo.py`](domain_config_demo.py) - Domain-specific configurations

### **Enterprise Features**
- [`integrity_verification_demo.py`](integrity_verification_demo.py) - Data integrity and validation
- [`uuid_api_compatibility_demo.py`](uuid_api_compatibility_demo.py) - UUID compatibility patterns

### **Complete Applications**
- [`advanced_ml_examples.py`](advanced_ml_examples.py) - Multi-framework ML workflows

---

## 🎯 **Use Case Quick Finder**

### 🌐 **Building Web APIs?**
1. Start with [`uuid_pydantic_quickstart.py`](uuid_pydantic_quickstart.py)
2. Explore [`framework_integrations/fastapi_integration_guide.py`](framework_integrations/fastapi_integration_guide.py)
3. Review [`uuid_api_compatibility_demo.py`](uuid_api_compatibility_demo.py)

### 🤖 **Working with ML Models?**
1. Begin with [`ai_ml_examples.py`](ai_ml_examples.py)
2. Check [`framework_integrations/mlflow_artifact_tracking.py`](framework_integrations/mlflow_artifact_tracking.py)
3. Explore serving with [`framework_integrations/bentoml_integration_guide.py`](framework_integrations/bentoml_integration_guide.py)

### 📊 **Processing Large Datasets?**
1. Start with [`file_operations_guide.py`](file_operations_guide.py)
2. Learn chunked processing in [`chunked_and_template_demo.py`](chunked_and_template_demo.py)
3. Review [`auto_detection_and_metadata_demo.py`](auto_detection_and_metadata_demo.py)

### 🎨 **Building Interactive Apps?**
1. Explore [`framework_integrations/streamlit_gradio_integration.py`](framework_integrations/streamlit_gradio_integration.py)
2. Check basic patterns in [`enhanced_utils_example.py`](enhanced_utils_example.py)

---

## 🚀 **Getting Started Commands**

```bash
# Clone and setup
git clone <datason-repo>
cd datason/examples

# Install dependencies
pip install datason[all]  # Includes optional ML dependencies

# Run examples by complexity level
python basic_usage.py                    # Level 1: Basics
python uuid_pydantic_quickstart.py       # Level 2: Web APIs  
python ai_ml_examples.py                 # Level 3: ML/AI

# Launch interactive demos
streamlit run framework_integrations/streamlit_gradio_integration.py
python framework_integrations/streamlit_gradio_integration.py --gradio

# Explore specific frameworks
python framework_integrations/fastapi_integration_guide.py
python framework_integrations/mlflow_artifact_tracking.py
```

---

## 📚 **Key Features Demonstrated**

### ✨ **Modern API (v0.11+)**
- **`dump_api()`** - Clean JSON for web APIs
- **`dump_ml()`** - ML-optimized serialization
- **`dump_secure()`** - Security-focused with PII redaction
- **`load_smart()`** - Intelligent parsing (90% success rate)
- **`load_basic()`** - Fast parsing (70% success rate)
- **`load_perfect()`** - Template-based (100% success rate)

### 🔧 **Configuration System**
- **`get_api_config()`** - Web API optimized
- **`get_ml_config()`** - ML framework optimized
- **`get_performance_config()`** - Speed optimized
- **`get_strict_config()`** - Validation focused

### 🌟 **Advanced Features**
- 🆔 **UUID String Conversion** - Perfect FastAPI/Pydantic compatibility
- 🧠 **Smart Type Detection** - Automatic datetime, UUID, complex type handling
- 🔒 **Built-in Security** - PII redaction, data sanitization
- ⚡ **High Performance** - Optimized for large datasets
- 🔄 **Progressive Loading** - Choose your success rate vs speed
- 📊 **ML Integration** - 15+ frameworks supported

---

## 💡 **Tips for Success**

1. **Start Simple**: Begin with `basic_usage.py` even if you're experienced
2. **Use Modern API**: Prefer `dump_api()` over legacy `serialize()` for new code
3. **Choose Right Function**: `dump_api()` for web, `dump_ml()` for ML, `dump_secure()` for sensitive data
4. **Progressive Loading**: Start with `load_smart()`, use `load_perfect()` for critical data
5. **Framework Integration**: Check `framework_integrations/` for your specific stack
6. **Read Comments**: Examples include detailed explanations and best practices

---

## 🤝 **Contributing**

Found an issue or want to add an example?

1. **Bug Reports**: Include the example file and expected vs actual behavior
2. **New Examples**: Follow the complexity progression and include comprehensive comments
3. **Framework Integrations**: Include setup instructions, production patterns, and testing

---

## 📞 **Need Help?**

- 📖 **Documentation**: Check `/docs` directory
- 💬 **Issues**: Report problems on GitHub
- 🚀 **Feature Requests**: Suggest new examples or integrations

---

**🎉 Happy serializing with DataSON! Start with [`basic_usage.py`](basic_usage.py) and work your way up.**
