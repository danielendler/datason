# 🌐 Framework Integrations with DataSON

**Production-ready examples for popular frameworks - enhanced with comprehensive modern API usage**

This directory contains full-featured examples showing how to integrate DataSON with popular web frameworks, ML serving platforms, and interactive applications. Each example has been significantly enhanced to demonstrate modern DataSON API patterns and real-world deployment scenarios.

---

## 🚀 **Recently Enhanced Examples**

All framework integration examples have been **completely rewritten** to showcase:

- ✨ **Modern DataSON API** (`dump_api`, `load_smart`, `dump_ml`, `dump_secure`)
- 🔧 **Production deployment patterns** with Docker and Kubernetes
- 📊 **Comprehensive error handling** and monitoring
- 🧪 **Complete testing and validation** workflows
- 📖 **Step-by-step setup instructions** for real deployments

---

## 🌐 **Web Frameworks**

### 🚀 FastAPI Integration
**File:** [`fastapi_integration_guide.py`](fastapi_integration_guide.py)

**What's New:**
- Complete FastAPI application with realistic endpoints
- Perfect UUID/Pydantic compatibility using `get_api_config()`
- Database simulation with proper DataSON serialization
- Performance benchmarking and monitoring endpoints
- Production-ready middleware and error handling

**Key Features:**
```python
# Perfect UUID handling for Pydantic models
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    processed_data = ds.load_smart(user.dict(), config=API_CONFIG)
    return ds.dump_api(created_user)  # UUIDs become strings automatically
```

**Run it:**
```bash
python fastapi_integration_guide.py
# Visit: http://localhost:8000/docs
```

---

### 🌐 Django Integration  
**File:** [`django_integration_guide.py`](django_integration_guide.py)

**What's New:**
- Django model mixins with DataSON serialization
- Admin interface integration with enhanced data display
- Django REST Framework compatibility (when available)
- JSON field processing with smart type detection
- Form handling with automatic validation

**Key Features:**
```python
class DataSONModelMixin:
    def to_datason(self):
        return ds.dump_api(model_to_dict(self))

    @classmethod
    def from_datason(cls, data):
        processed = ds.load_smart(data, config=API_CONFIG)
        return cls(**processed)
```

---

## 🤖 **ML Serving Platforms**

### 🤖 BentoML Integration
**File:** [`bentoml_integration_guide.py`](bentoml_integration_guide.py)

**What's New:**
- Complete BentoML service with multiple input/output types
- JSON, NumPy array, and text endpoint examples
- DataSON serialization for enhanced API responses
- Health checks and model information endpoints
- Production deployment configurations

**Key Features:**
```python
@svc.api(input=JSON(), output=JSON())
def predict_json(input_data: dict) -> dict:
    parsed_input = ds.load_smart(input_data, config=API_CONFIG)
    result = model.predict(parsed_input["features"])
    return ds.dump_api({"prediction": result, "metadata": {...}})
```

**Run it:**
```bash
bentoml serve bentoml_integration_guide:svc --reload
```

---

### ☁️ Ray Serve Integration
**File:** [`ray_serve_integration_guide.py`](ray_serve_integration_guide.py)

**What's New:**
- Scalable ML model serving with Ray Serve
- Batch and single prediction endpoints
- DataSON-enhanced request/response handling
- Health monitoring and service metrics
- Production deployment with auto-scaling

**Key Features:**
```python
@serve.deployment(num_replicas=2, max_concurrent_queries=10)
class DataSONModelServing:
    async def __call__(self, request):
        parsed_data = ds.load_smart(await request.json(), config=API_CONFIG)
        result = self.model.predict(parsed_data["data"])
        return ds.dump_api({"prediction": result, "processing_info": {...}})
```

---

### 📈 MLflow Integration
**File:** [`mlflow_artifact_tracking.py`](mlflow_artifact_tracking.py)

**What's New:**
- Complete ML experiment tracking with DataSON artifacts
- Model metadata serialization and versioning
- Cross-experiment comparison with enhanced data structures
- Artifact loading with smart parsing
- Integration with scikit-learn for real ML workflows

**Key Features:**
```python
# Enhanced experiment metadata with DataSON
experiment_data = {
    "model_config": {...},
    "performance": {...},
    "feature_importance": model.feature_importances_
}
serialized_data = ds.dump_ml(experiment_data)
mlflow.log_artifact(temp_path, "experiment_metadata")
```

---

### ⚙️ Kubernetes ML Serving (Seldon/KServe)
**File:** [`seldon_kserve_integration.py`](seldon_kserve_integration.py)

**What's New:**
- Complete Kubernetes deployment configurations
- Both Seldon Core and KServe examples
- Production-ready YAML configurations
- Docker build instructions and Dockerfiles
- Comprehensive health checks and monitoring

**Key Features:**
```python
class DataSONSeldonModel(SeldonComponent):
    def predict(self, features, **kwargs):
        processed_features = ds.load_smart(features, config=API_CONFIG)
        result = self.model.predict(processed_features)
        return ds.dump_api({"prediction": result, "seldon_info": {...}})
```

**Includes:**
- Kubernetes YAML configurations
- Docker build files
- Production deployment guides

---

## 🎨 **Interactive Applications**

### 📊 Streamlit & Gradio Integration
**File:** [`streamlit_gradio_integration.py`](streamlit_gradio_integration.py)

**What's New:**
- **Comprehensive Streamlit app** with multi-tab interface
- **Full Gradio interface** with interactive processing
- Real-time data validation and transformation
- File upload/download with DataSON serialization
- ML prediction interfaces with visualization

**Streamlit Features:**
- 🧪 Interactive data processing with multiple DataSON modes
- 🤖 ML predictions with feature analysis
- 📁 File operations with drag-and-drop support
- 📊 Analytics dashboard with processing history
- 🔍 API explorer with live examples

**Gradio Features:**
- 🔄 JSON data processing with real-time feedback
- 🤖 ML inference with comma-separated input
- 📄 Batch file processing
- 📚 Built-in API documentation

**Run them:**
```bash
# Streamlit app
streamlit run streamlit_gradio_integration.py

# Gradio interface  
python streamlit_gradio_integration.py --gradio
```

---

## 🔧 **Installation & Setup**

### **Base Requirements**
```bash
pip install datason  # Core DataSON functionality
```

### **Framework-Specific Requirements**

```bash
# Web frameworks
pip install fastapi uvicorn pydantic
pip install django djangorestframework

# ML serving platforms
pip install bentoml
pip install ray[serve]
pip install mlflow scikit-learn
pip install seldon-core kserve

# Interactive applications
pip install streamlit gradio pandas numpy
```

### **Complete Installation**
```bash
# Install everything for full examples
pip install datason[all] fastapi uvicorn django bentoml ray[serve] mlflow seldon-core kserve streamlit gradio pandas numpy scikit-learn
```

---

## 🚀 **Quick Start Guide**

### 1. **Web API Development** (Most Common)
```bash
# Start with UUID/Pydantic compatibility
python ../uuid_pydantic_quickstart.py

# Then explore FastAPI integration
python fastapi_integration_guide.py
```

### 2. **ML Model Serving**
```bash
# Begin with basic ML integration
python ../ai_ml_examples.py

# Choose your serving platform
python bentoml_integration_guide.py    # Local serving
python ray_serve_integration_guide.py  # Scalable serving
python seldon_kserve_integration.py    # Kubernetes serving
```

### 3. **Interactive Applications**
```bash
# Launch Streamlit dashboard
streamlit run streamlit_gradio_integration.py

# Or try Gradio interface
python streamlit_gradio_integration.py --gradio
```

---

## 🎯 **Use Case Matrix**

| Framework | Best For | Complexity | Production Ready |
|-----------|----------|------------|------------------|
| **FastAPI** | REST APIs, Web services | 🟡 Medium | ✅ Yes |
| **Django** | Web applications, Admin | 🟡 Medium | ✅ Yes |
| **BentoML** | ML model serving | 🟡 Medium | ✅ Yes |
| **Ray Serve** | Scalable ML serving | 🔴 Advanced | ✅ Yes |
| **MLflow** | Experiment tracking | 🟢 Beginner | ✅ Yes |
| **Seldon/KServe** | Kubernetes ML | 🔴 Advanced | ✅ Yes |
| **Streamlit** | Data apps, dashboards | 🟢 Beginner | 🟡 Prototyping |
| **Gradio** | ML demos, interfaces | 🟢 Beginner | 🟡 Prototyping |

---

## 🔍 **Key Patterns Demonstrated**

### **UUID String Conversion** (Critical for Web APIs)
```python
from datason.config import get_api_config
API_CONFIG = get_api_config()  # Ensures UUIDs become strings

# In your endpoints
processed_data = ds.load_smart(input_data, config=API_CONFIG)
response = ds.dump_api(result)  # UUIDs automatically stringified
```

### **ML-Optimized Serialization**
```python
# For ML artifacts and model serving
model_data = ds.dump_ml({"model": model, "metrics": metrics})
experiment_results = ds.load_smart(data, config=get_ml_config())
```

### **Security-Focused Processing**
```python
# For sensitive data
safe_data = ds.dump_secure(user_data, redact_pii=True)
```

### **Progressive Loading**
```python
# Choose your success rate vs performance
basic_result = ds.load_basic(json_data)      # 70% success, fastest
smart_result = ds.load_smart(json_data)      # 90% success, balanced
perfect_result = ds.load_perfect(json_data, template)  # 100% success
```

---

## 🐳 **Production Deployment**

Several examples include production deployment configurations:

### **Docker Examples**
- **BentoML**: Container-ready service definitions
- **Ray Serve**: Multi-container scaling configurations  
- **Seldon/KServe**: Kubernetes-native containers

### **Kubernetes Configurations**
- **Seldon Core**: Complete deployment YAML
- **KServe**: InferenceService configurations
- **Health checks**: Liveness and readiness probes

### **Monitoring & Observability**
- **Health endpoints**: Built into all serving examples
- **Metrics collection**: Request/response monitoring
- **Error handling**: Comprehensive error responses

---

## 📚 **Documentation Links**

- [🏠 Main Examples](../README.md) - Complete examples overview
- [📖 DataSON Documentation](../../docs/) - Full API documentation
- [🚀 Quick Start](../../docs/user-guide/quick-start.md) - Getting started guide
- [⚙️ Configuration](../../docs/api/configuration.md) - Configuration options

---

## 🤝 **Contributing New Integrations**

Want to add a new framework integration? Follow this pattern:

1. **Comprehensive example** with real-world usage patterns
2. **Modern DataSON API** usage throughout
3. **Production deployment** configurations
4. **Complete setup instructions** with dependencies
5. **Testing and validation** examples
6. **Error handling** and monitoring

**Template structure:**
```python
#!/usr/bin/env python3
"""
[Framework] Integration with DataSON

A comprehensive example showing how to integrate DataSON with [Framework]
for [use case].

Features:
- Modern DataSON API (dump_api, load_smart, dump_ml)
- [Framework-specific features]
- Production deployment patterns
"""

# Framework availability check
try:
    import framework
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

import datason as ds
from datason.config import get_api_config

# Implementation with fallback for missing dependencies
```

---

**🎉 Ready to integrate DataSON with your favorite framework? Pick an example and start building!**
