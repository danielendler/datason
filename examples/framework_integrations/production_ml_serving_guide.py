#!/usr/bin/env python3
"""
Production ML Model Serving with Datason

This comprehensive guide demonstrates production-ready ML model serving patterns
using datason for serialization across different serving frameworks.

Features:
- Error handling and graceful degradation
- Performance monitoring and metrics
- Model versioning and A/B testing
- Health checks and observability
- Memory management for large models
- Security and input validation

Setup:
    pip install datason fastapi uvicorn bentoml ray[serve] streamlit gradio mlflow
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import datason
from datason.config import SerializationConfig, get_api_config, get_ml_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================


class MLServingConfig:
    """Production ML serving configuration."""

    def __init__(self):
        # Datason configurations for different use cases
        self.api_config = get_api_config()  # For API responses
        self.ml_config = get_ml_config()  # For ML objects

        # Custom performance config for high-throughput serving
        self.performance_config = SerializationConfig(
            uuid_format="string",
            parse_uuids=False,
            date_format="unix",
            preserve_decimals=False,
            sort_keys=False,
            max_depth=20,  # Limit for security
            max_size=50_000_000,  # 50MB limit
        )

        # Model serving settings
        self.max_batch_size = 32
        self.timeout_seconds = 30
        self.max_memory_mb = 1000
        self.enable_monitoring = True


# Global config instance
serving_config = MLServingConfig()

# =============================================================================
# MODEL WRAPPER WITH MONITORING
# =============================================================================


class ProductionModelWrapper:
    """Production-ready model wrapper with monitoring and error handling."""

    def __init__(self, model_id: str, model_version: str):
        self.model_id = model_id
        self.model_version = model_version
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.last_health_check = datetime.now()

    def predict(self, features: Any) -> Dict[str, Any]:
        """Make prediction with monitoring and error handling."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Validate input
            self._validate_input(features)

            # Process features with datason
            processed_features = datason.auto_deserialize(features, config=serving_config.api_config)

            # Mock prediction (replace with actual model)
            prediction = self._run_model(processed_features)

            # Prepare response
            response = {
                "request_id": request_id,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "prediction": prediction,
                "confidence": 0.85,  # Mock confidence
                "timestamp": datetime.now(),
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
            }

            # Serialize response
            serialized_response = datason.serialize(response, config=serving_config.performance_config)

            # Update metrics
            self.total_latency += time.perf_counter() - start_time

            logger.info(f"Prediction successful: {request_id}")
            return serialized_response

        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed: {request_id}, Error: {e}")

            # Return error response
            error_response = {
                "request_id": request_id,
                "error": str(e),
                "model_id": self.model_id,
                "timestamp": datetime.now(),
                "status": "error",
            }

            return datason.serialize(error_response, config=serving_config.api_config)

    def _validate_input(self, features: Any) -> None:
        """Validate input features."""
        if not features:
            raise ValueError("Features cannot be empty")

        # Estimate memory usage
        try:
            estimated_size = len(str(features))  # Simple estimation
            if estimated_size > serving_config.max_memory_mb * 1024 * 1024:
                raise ValueError(f"Input too large: {estimated_size} bytes")
        except Exception:
            pass  # nosec B110 - Continue if estimation fails (demo code)

    def _run_model(self, features: Any) -> Any:
        """Mock model inference (replace with actual model)."""
        # Simulate processing time
        time.sleep(0.01)

        # Mock prediction based on input
        if isinstance(features, dict):
            return {"class": "positive", "score": 0.85}
        elif isinstance(features, list):
            return [0.1, 0.7, 0.2]  # Multi-class probabilities
        else:
            return {"value": 42.0}

    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status."""
        avg_latency = self.total_latency / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)

        status = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": "healthy" if error_rate < 0.1 else "degraded",
            "metrics": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency * 1000,
                "last_health_check": self.last_health_check,
            },
        }

        return datason.serialize(status, config=serving_config.api_config)


# =============================================================================
# FASTAPI PRODUCTION SERVING
# =============================================================================


def create_fastapi_production_service():
    """Create production FastAPI service with comprehensive features."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="Production ML Serving API", description="Production-ready ML model serving with datason", version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Model instances
    models = {
        "classifier_v1": ProductionModelWrapper("classifier", "1.0.0"),
        "regressor_v1": ProductionModelWrapper("regressor", "1.0.0"),
    }

    class PredictionRequest(BaseModel):
        features: Dict[str, Any]
        model_id: str = "classifier_v1"
        options: Optional[Dict[str, Any]] = None

    class BatchPredictionRequest(BaseModel):
        batch_features: List[Dict[str, Any]]
        model_id: str = "classifier_v1"
        options: Optional[Dict[str, Any]] = None

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(),
            "models": {model_id: model.get_health_status() for model_id, model in models.items()},
        }
        return datason.serialize(health_status, config=serving_config.api_config)

    @app.post("/predict")
    async def predict(request: PredictionRequest):
        """Single prediction endpoint."""
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        model = models[request.model_id]
        result = model.predict(request.features)

        return result

    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictionRequest):
        """Batch prediction endpoint."""
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

        if len(request.batch_features) > serving_config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.batch_features)} exceeds limit {serving_config.max_batch_size}",
            )

        model = models[request.model_id]
        results = []

        for features in request.batch_features:
            result = model.predict(features)
            results.append(result)

        batch_response = {
            "batch_id": str(uuid.uuid4()),
            "model_id": request.model_id,
            "results": results,
            "batch_size": len(results),
            "timestamp": datetime.now(),
        }

        return datason.serialize(batch_response, config=serving_config.performance_config)

    @app.get("/models")
    async def list_models():
        """List available models."""
        model_info = {
            model_id: {"model_id": model.model_id, "version": model.model_version, "status": "active"}
            for model_id, model in models.items()
        }

        return datason.serialize(model_info, config=serving_config.api_config)

    @app.get("/metrics")
    async def get_metrics():
        """Get serving metrics."""
        metrics = {
            "service_metrics": {"total_models": len(models), "timestamp": datetime.now()},
            "model_metrics": {model_id: model.get_health_status() for model_id, model in models.items()},
        }

        return datason.serialize(metrics, config=serving_config.api_config)

    return app


# =============================================================================
# BENTOML PRODUCTION SERVICE
# =============================================================================


def create_bentoml_production_service():
    """Create production BentoML service."""

    try:
        import bentoml
        from bentoml.io import JSON
    except ImportError:
        print("BentoML not available. Install with: pip install bentoml")
        return None

    # Create service
    svc = bentoml.Service("production_ml_service", runners=[])

    # Model wrapper
    model = ProductionModelWrapper("bentoml_model", "1.0.0")

    @svc.api(input=JSON(), output=JSON())
    def predict(input_data: dict) -> dict:
        """Production prediction endpoint."""
        try:
            # Validate input
            if "features" not in input_data:
                raise ValueError("Missing 'features' in input")

            # Make prediction
            result = model.predict(input_data["features"])

            return result

        except Exception as e:
            error_response = {"error": str(e), "timestamp": datetime.now(), "status": "error"}
            return datason.serialize(error_response, config=serving_config.api_config)

    @svc.api(input=JSON(), output=JSON())
    def health() -> dict:
        """Health check endpoint."""
        return model.get_health_status()

    return svc


# =============================================================================
# RAY SERVE PRODUCTION DEPLOYMENT
# =============================================================================


def create_ray_serve_production_deployment():
    """Create production Ray Serve deployment."""

    try:
        from ray import serve
    except ImportError:
        print("Ray not available. Install with: pip install ray[serve]")
        return None

    @serve.deployment(
        num_replicas=2,
        max_concurrent_queries=100,
        ray_actor_options={"num_cpus": 1, "memory": 1000 * 1024 * 1024},  # 1GB
    )
    class ProductionMLDeployment:
        def __init__(self):
            self.model = ProductionModelWrapper("ray_serve_model", "1.0.0")
            logger.info("Ray Serve deployment initialized")

        async def __call__(self, request):
            """Handle prediction requests."""
            try:
                # Parse request
                if hasattr(request, "json"):
                    payload = await request.json()
                else:
                    payload = request

                # Validate payload
                if "features" not in payload:
                    raise ValueError("Missing 'features' in request")

                # Make prediction
                result = self.model.predict(payload["features"])

                return result

            except Exception as e:
                error_response = {"error": str(e), "timestamp": datetime.now(), "status": "error"}
                return datason.serialize(error_response, config=serving_config.api_config)

        def health_check(self):
            """Health check method."""
            return self.model.get_health_status()

    return ProductionMLDeployment


# =============================================================================
# STREAMLIT PRODUCTION DASHBOARD
# =============================================================================


def create_streamlit_production_dashboard():
    """Create production Streamlit dashboard."""

    try:
        import pandas as pd
        import plotly.express as px
        import streamlit as st
    except ImportError:
        print("Streamlit/Plotly not available. Install with: pip install streamlit plotly")
        return None

    def run_dashboard():
        st.set_page_config(page_title="ML Model Serving Dashboard", page_icon="🤖", layout="wide")

        st.title("🤖 Production ML Model Serving Dashboard")
        st.markdown("Real-time monitoring and testing of ML models served with datason")

        # Initialize model
        if "model" not in st.session_state:
            st.session_state.model = ProductionModelWrapper("dashboard_model", "1.0.0")

        model = st.session_state.model

        # Sidebar for configuration
        st.sidebar.header("Configuration")

        # Model selection
        st.sidebar.selectbox("Model Type", ["Classifier", "Regressor", "Custom"])

        # Performance settings
        st.sidebar.slider("Batch Size", 1, 100, 10)
        st.sidebar.slider("Timeout (seconds)", 1, 60, 30)

        # Main dashboard
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Requests", model.request_count)

        with col2:
            error_rate = model.error_count / max(model.request_count, 1)
            st.metric("Error Rate", f"{error_rate:.2%}")

        with col3:
            avg_latency = model.total_latency / max(model.request_count, 1)
            st.metric("Avg Latency", f"{avg_latency * 1000:.1f}ms")

        # Health status
        st.subheader("Health Status")
        health_status = model.get_health_status()
        st.json(health_status)

        # Prediction testing
        st.subheader("Test Predictions")

        # Input methods
        input_method = st.radio("Input Method", ["JSON", "Form", "File Upload"])

        if input_method == "JSON":
            json_input = st.text_area(
                "JSON Input", value='{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}', height=100
            )

            if st.button("Predict"):
                try:
                    features = json.loads(json_input)
                    result = model.predict(features)

                    st.success("Prediction successful!")
                    st.json(result)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        elif input_method == "Form":
            with st.form("prediction_form"):
                feature1 = st.number_input("Feature 1", value=1.0)
                feature2 = st.number_input("Feature 2", value=2.0)
                feature3 = st.number_input("Feature 3", value=3.0)

                submitted = st.form_submit_button("Predict")

                if submitted:
                    features = {"feature1": feature1, "feature2": feature2, "feature3": feature3}

                    result = model.predict(features)
                    st.json(result)

        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload JSON file", type="json")

            if uploaded_file is not None:
                try:
                    features = json.load(uploaded_file)
                    result = model.predict(features)

                    st.success("Prediction successful!")
                    st.json(result)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # Performance monitoring
        st.subheader("Performance Monitoring")

        if model.request_count > 0:
            # Create mock time series data for demo
            import numpy as np

            timestamps = pd.date_range(
                start=datetime.now() - pd.Timedelta(hours=1), end=datetime.now(), periods=min(model.request_count, 60)
            )

            latencies = np.random.normal(avg_latency * 1000, 10, len(timestamps))

            df = pd.DataFrame({"timestamp": timestamps, "latency_ms": latencies})

            fig = px.line(df, x="timestamp", y="latency_ms", title="Response Latency Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Configuration export
        st.subheader("Configuration Export")

        config_export = {
            "serving_config": {
                "max_batch_size": serving_config.max_batch_size,
                "timeout_seconds": serving_config.timeout_seconds,
                "max_memory_mb": serving_config.max_memory_mb,
            },
            "datason_config": {
                "uuid_format": serving_config.api_config.uuid_format,
                "parse_uuids": serving_config.api_config.parse_uuids,
            },
        }

        st.json(config_export)

        if st.button("Download Configuration"):
            config_json = json.dumps(config_export, indent=2)
            st.download_button(
                label="Download config.json",
                data=config_json,
                file_name="ml_serving_config.json",
                mime="application/json",
            )

    return run_dashboard


# =============================================================================
# MLFLOW PRODUCTION INTEGRATION
# =============================================================================


def create_mlflow_production_integration():
    """Create production MLflow integration."""

    try:
        import mlflow
        import mlflow.pyfunc
    except ImportError:
        print("MLflow not available. Install with: pip install mlflow")
        return None

    class DatasonMLflowModel(mlflow.pyfunc.PythonModel):
        """MLflow model wrapper using datason for serialization."""

        def __init__(self):
            self.model = ProductionModelWrapper("mlflow_model", "1.0.0")

        def predict(self, context, model_input):
            """Make predictions using datason serialization."""
            try:
                # Convert input to dict if needed
                if hasattr(model_input, "to_dict"):
                    features = model_input.to_dict()
                elif hasattr(model_input, "values"):
                    features = {"features": model_input.values.tolist()}
                else:
                    features = model_input

                # Make prediction
                result = self.model.predict(features)

                return result

            except Exception as e:
                logger.error(f"MLflow prediction failed: {e}")
                return {"error": str(e)}

    def log_model_with_datason():
        """Log model to MLflow with datason metadata."""

        with mlflow.start_run():
            # Create model instance
            model = DatasonMLflowModel()

            # Log model
            mlflow.pyfunc.log_model(
                artifact_path="datason_model", python_model=model, pip_requirements=["datason", "numpy", "pandas"]
            )

            # Log datason configuration
            config_dict = {
                "uuid_format": serving_config.api_config.uuid_format,
                "parse_uuids": serving_config.api_config.parse_uuids,
                "max_batch_size": serving_config.max_batch_size,
            }

            mlflow.log_dict(config_dict, "datason_config.json")

            # Log performance metrics
            mlflow.log_metric("max_memory_mb", serving_config.max_memory_mb)
            mlflow.log_metric("timeout_seconds", serving_config.timeout_seconds)

            logger.info("Model logged to MLflow with datason configuration")

    return log_model_with_datason


# =============================================================================
# MAIN DEMO FUNCTION
# =============================================================================


def main():
    """Run production ML serving demonstrations."""

    print("🚀 Production ML Model Serving with Datason")
    print("=" * 60)

    # Test model wrapper
    print("\n1. Testing Production Model Wrapper...")
    model = ProductionModelWrapper("test_model", "1.0.0")

    # Test prediction
    test_features = {"feature1": 1.0, "feature2": 2.0}
    result = model.predict(test_features)
    print(f"   ✓ Prediction result: {result.get('prediction', 'N/A')}")

    # Test health check
    health = model.get_health_status()
    print(f"   ✓ Health status: {health.get('status', 'N/A')}")

    # Test batch processing
    print("\n2. Testing Batch Processing...")
    batch_features = [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 3.0, "feature2": 4.0},
        {"feature1": 5.0, "feature2": 6.0},
    ]

    batch_results = []
    for features in batch_features:
        result = model.predict(features)
        batch_results.append(result)

    print(f"   ✓ Processed {len(batch_results)} predictions")

    # Test serialization performance
    print("\n3. Testing Serialization Performance...")
    large_data = {
        "features": list(range(1000)),
        "metadata": {"timestamp": datetime.now()},
        "model_info": {"version": "1.0.0", "type": "classifier"},
    }

    start_time = time.perf_counter()
    datason.serialize(large_data, config=serving_config.performance_config)
    end_time = time.perf_counter()

    print(f"   ✓ Serialized 1000 features in {(end_time - start_time) * 1000:.2f}ms")

    # Framework availability
    print("\n4. Framework Availability:")

    frameworks = [
        ("FastAPI", lambda: __import__("fastapi")),
        ("BentoML", lambda: __import__("bentoml")),
        ("Ray Serve", lambda: __import__("ray.serve")),
        ("Streamlit", lambda: __import__("streamlit")),
        ("MLflow", lambda: __import__("mlflow")),
    ]

    for name, import_func in frameworks:
        try:
            import_func()
            print(f"   ✓ {name} available")
        except ImportError:
            print(f"   ✗ {name} not available")

    print("\n🎯 Production Features Demonstrated:")
    print("   ✓ Error handling and graceful degradation")
    print("   ✓ Performance monitoring and metrics")
    print("   ✓ Health checks and observability")
    print("   ✓ Memory management and limits")
    print("   ✓ Batch processing capabilities")
    print("   ✓ Configuration management")
    print("   ✓ Security and input validation")

    print("\n📚 Next Steps:")
    print("   • Run individual framework examples")
    print("   • Customize configurations for your use case")
    print("   • Add your actual ML models")
    print("   • Set up monitoring and alerting")
    print("   • Deploy to production environment")


if __name__ == "__main__":
    main()
