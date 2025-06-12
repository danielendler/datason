# 🔗 Framework Integration Examples

This directory contains examples showing how datason integrates with specific frameworks and platforms. These are specialized use cases that demonstrate production deployment patterns.

## 📋 Available Integrations

### Machine Learning Frameworks
- **[`advanced_bentoml_integration.py`](advanced_bentoml_integration.py)** - BentoML model serving integration
- **[`production_ml_serving_guide.py`](production_ml_serving_guide.py)** - Production ML serving patterns
- **[`bentoml_integration_guide.py`](bentoml_integration_guide.py)** - BentoML integration guide
- **[`mlflow_artifact_tracking.py`](mlflow_artifact_tracking.py)** - MLflow experiment tracking
- **[`ray_serve_integration_guide.py`](ray_serve_integration_guide.py)** - Ray Serve deployment
- **[`seldon_kserve_integration.py`](seldon_kserve_integration.py)** - Seldon/KServe Kubernetes deployment

### Web Frameworks  
- **[`django_integration_guide.py`](django_integration_guide.py)** - Django REST API integration
- **[`fastapi_integration_guide.py`](fastapi_integration_guide.py)** - FastAPI integration patterns

### UI Frameworks
- **[`streamlit_gradio_integration.py`](streamlit_gradio_integration.py)** - Streamlit/Gradio dashboard integration

## 🎯 How Framework Examples Differ

| **Main Examples** | **Framework Examples** |
|-------------------|-------------------------|
| Core datason functionality | Framework-specific patterns |
| Simple & direct API | Production deployment |
| Universal patterns | Platform integration |
| Quick start guides | Enterprise use cases |

## 🚀 Quick Start

For general datason usage, start with the main examples:
- [`../basic_usage.py`](../basic_usage.py) - Simple & direct API
- [`../ai_ml_examples.py`](../ai_ml_examples.py) - ML workflows
- [`../modern_api_demo.py`](../modern_api_demo.py) - Modern API features

Then explore framework-specific examples based on your deployment needs.

## 🛠️ Prerequisites

Framework examples may require additional dependencies:

```bash
# ML serving frameworks
pip install bentoml ray[serve] mlflow

# Web frameworks  
pip install fastapi django uvicorn

# UI frameworks
pip install streamlit gradio

# Kubernetes deployment
pip install seldon-core kserve
```

## 📚 Integration Patterns

All framework examples demonstrate:
- ✅ **Simple API usage** with `dump_api()` and `dump_ml()`
- ✅ **UUID/Pydantic compatibility** for web frameworks
- ✅ **Production configuration** and error handling
- ✅ **Monitoring and logging** integration
- ✅ **Performance optimization** for production workloads

## 🎯 Usage

Each example is self-contained and can be run independently:

```bash
# Run a specific framework example
python framework_integrations/fastapi_integration_guide.py

# Or explore the production ML serving guide
python framework_integrations/production_ml_serving_guide.py
```

Most examples include both demonstration code and production-ready templates you can adapt for your use case.
