# Model Serving Integration

Comprehensive guides for using **datason** with popular model serving and experiment tracking frameworks.

These recipes show how to plug datason into each framework so your ML objects round-trip cleanly between training code, servers, and web APIs.

## 🎯 Quick Start

### Basic Integration Pattern

```python
import datason
from datason.config import get_api_config

# Universal pattern for all frameworks
API_CONFIG = get_api_config()

def process_ml_request(input_data: dict) -> dict:
    """Universal ML request processing with datason."""

    # 1. Deserialize input (handles UUIDs, dates, etc.)
    processed_input = datason.auto_deserialize(input_data, config=API_CONFIG)

    # 2. Run your model
    prediction = your_model.predict(processed_input)

    # 3. Serialize response (preserves types for APIs)
    return datason.serialize(prediction, config=API_CONFIG)
```

## 🚀 Framework Integrations

### BentoML

**Basic Service:**
```python
import bentoml
from bentoml.io import JSON
import datason
from datason.config import get_api_config

svc = bentoml.Service("my_model_service")
API_CONFIG = get_api_config()

@svc.api(input=JSON(), output=JSON())
def predict(parsed_json: dict) -> dict:
    # Parse incoming JSON with datason
    data = datason.auto_deserialize(parsed_json, config=API_CONFIG)
    prediction = run_model(data)
    # Serialize response for BentoML
    return datason.serialize(prediction, config=API_CONFIG)
```

**Production Service with Monitoring:**
```python
import bentoml
from bentoml.io import JSON
from bentoml import metrics
import datason
from datason.config import get_api_config
import time
import uuid
from datetime import datetime

# Metrics
request_counter = metrics.Counter("prediction_requests_total")
error_counter = metrics.Counter("prediction_errors_total")
latency_histogram = metrics.Histogram("prediction_latency_seconds")

svc = bentoml.Service("production_ml_service")
API_CONFIG = get_api_config()

@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    """Production prediction with comprehensive monitoring."""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        request_counter.inc()

        # Input validation
        if "features" not in input_data:
            raise ValueError("Missing 'features' in input")

        # Process with datason
        data = datason.auto_deserialize(input_data["features"], config=API_CONFIG)

        # Model inference
        prediction = run_model(data)

        # Build response
        response = {
            "request_id": request_id,
            "prediction": prediction,
            "model_version": "1.0.0",
            "timestamp": datetime.now(),
            "processing_time_ms": (time.perf_counter() - start_time) * 1000
        }

        # Record latency
        latency_histogram.observe(time.perf_counter() - start_time)

        return datason.serialize(response, config=API_CONFIG)

    except Exception as e:
        error_counter.inc()

        error_response = {
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.now(),
            "status": "error"
        }

        return datason.serialize(error_response, config=API_CONFIG)

@svc.api(input=JSON(), output=JSON())
def health() -> dict:
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_version": "1.0.0"
    }
    return datason.serialize(health_status, config=API_CONFIG)

# Deploy with: bentoml serve production_service:svc --production
```

### Ray Serve

**Basic Deployment:**
```python
from ray import serve
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

@serve.deployment
class ModelDeployment:
    async def __call__(self, request):
        payload = await request.json()
        data = datason.auto_deserialize(payload, config=API_CONFIG)
        result = run_model(data)
        return datason.serialize(result, config=API_CONFIG)
```

**Production Deployment with Autoscaling:**
```python
from ray import serve
import ray
import datason
from datason.config import get_api_config
import logging
import time
import uuid
from datetime import datetime

API_CONFIG = get_api_config()

@serve.deployment(
    num_replicas=2,
    max_concurrent_queries=100,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5
    },
    ray_actor_options={
        "num_cpus": 1,
        "memory": 2000 * 1024 * 1024  # 2GB
    }
)
class ProductionMLDeployment:
    def __init__(self):
        self.logger = logging.getLogger("ray_serve_deployment")
        self.request_count = 0
        self.error_count = 0
        self.logger.info("Ray Serve deployment initialized")

    async def __call__(self, request):
        """Handle requests with comprehensive error handling."""
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            self.request_count += 1

            # Parse request
            if hasattr(request, 'json'):
                payload = await request.json()
            else:
                payload = request

            # Validate
            if "features" not in payload:
                raise ValueError("Missing 'features' in request")

            # Process with datason
            data = datason.auto_deserialize(payload["features"], config=API_CONFIG)

            # Model inference
            prediction = run_model(data)

            # Build response
            response = {
                "request_id": request_id,
                "prediction": prediction,
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "timestamp": datetime.now()
            }

            return datason.serialize(response, config=API_CONFIG)

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Request {request_id} failed: {e}")

            error_response = {
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.now(),
                "status": "error"
            }

            return datason.serialize(error_response, config=API_CONFIG)

    def health_check(self):
        """Health check for Ray Serve."""
        error_rate = self.error_count / max(self.request_count, 1)

        return {
            "status": "healthy" if error_rate < 0.1 else "degraded",
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate
        }

app = ProductionMLDeployment.bind()

# Deploy with: serve run production_deployment:app
```

### Streamlit Production Dashboard

**Basic Demo:**
```python
import streamlit as st
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

uploaded = st.file_uploader("Upload JSON")
if uploaded:
    raw = uploaded.read().decode()
    data = datason.loads(raw, config=API_CONFIG)
    prediction = run_model(data)
    st.json(datason.dumps(prediction, config=API_CONFIG))
```

**Production Dashboard:**
```python
import streamlit as st
import datason
from datason.config import get_api_config
import pandas as pd
import plotly.express as px
import json
import time
from datetime import datetime

API_CONFIG = get_api_config()

st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production ML Model Dashboard")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

# Sidebar configuration
st.sidebar.header("Model Configuration")
model_version = st.sidebar.selectbox("Model Version", ["v1.0.0", "v1.1.0", "v2.0.0"])
batch_size = st.sidebar.slider("Batch Size", 1, 100, 10)

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Requests", st.session_state.request_count)

with col2:
    error_rate = st.session_state.error_count / max(st.session_state.request_count, 1)
    st.metric("Error Rate", f"{error_rate:.2%}")

with col3:
    if st.session_state.predictions:
        avg_latency = sum(p.get('processing_time_ms', 0) for p in st.session_state.predictions) / len(st.session_state.predictions)
        st.metric("Avg Latency", f"{avg_latency:.1f}ms")
    else:
        st.metric("Avg Latency", "0ms")

# Prediction interface
st.subheader("Make Predictions")

input_method = st.radio("Input Method", ["JSON", "Form", "File Upload"])

if input_method == "JSON":
    json_input = st.text_area(
        "JSON Input",
        value='{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}',
        height=100
    )

    if st.button("Predict"):
        try:
            start_time = time.perf_counter()
            features = json.loads(json_input)

            # Process with datason
            processed_features = datason.auto_deserialize(features, config=API_CONFIG)
            prediction = run_model(processed_features)

            processing_time = (time.perf_counter() - start_time) * 1000

            # Build response
            response = {
                "prediction": prediction,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now(),
                "model_version": model_version
            }

            # Serialize response
            serialized_response = datason.serialize(response, config=API_CONFIG)

            # Update session state
            st.session_state.predictions.append(serialized_response)
            st.session_state.request_count += 1

            st.success("Prediction successful!")
            st.json(serialized_response)

        except Exception as e:
            st.session_state.error_count += 1
            st.session_state.request_count += 1
            st.error(f"Prediction failed: {e}")

elif input_method == "Form":
    with st.form("prediction_form"):
        feature1 = st.number_input("Feature 1", value=1.0)
        feature2 = st.number_input("Feature 2", value=2.0)
        feature3 = st.number_input("Feature 3", value=3.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            features = {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3
            }

            try:
                start_time = time.perf_counter()
                processed_features = datason.auto_deserialize(features, config=API_CONFIG)
                prediction = run_model(processed_features)
                processing_time = (time.perf_counter() - start_time) * 1000

                response = {
                    "prediction": prediction,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now(),
                    "model_version": model_version
                }

                serialized_response = datason.serialize(response, config=API_CONFIG)
                st.session_state.predictions.append(serialized_response)
                st.session_state.request_count += 1

                st.json(serialized_response)

            except Exception as e:
                st.session_state.error_count += 1
                st.session_state.request_count += 1
                st.error(f"Prediction failed: {e}")

# Performance monitoring
if st.session_state.predictions:
    st.subheader("Performance Monitoring")

    # Create performance DataFrame
    df_data = []
    for i, pred in enumerate(st.session_state.predictions[-50:]):  # Last 50 predictions
        df_data.append({
            'request_id': i,
            'latency_ms': pred.get('processing_time_ms', 0),
            'timestamp': pred.get('timestamp', datetime.now())
        })

    df = pd.DataFrame(df_data)

    # Latency chart
    fig = px.line(df, x='request_id', y='latency_ms', title='Response Latency Over Time')
    st.plotly_chart(fig, use_container_width=True)

    # Recent predictions
    st.subheader("Recent Predictions")
    st.dataframe(df.tail(10))

def run_model(data):
    """Mock model function - replace with your actual model."""
    time.sleep(0.01)  # Simulate processing time
    return {"class": "positive", "confidence": 0.85}

# Run with: streamlit run dashboard.py
```

### MLflow Production Integration

**Basic Artifact Logging:**
```python
import mlflow
import datason

with mlflow.start_run():
    result = train_model()
    mlflow.log_dict(datason.serialize(result), "results.json")
```

**Production Model Registry:**
```python
import mlflow
import mlflow.pyfunc
import datason
from datason.config import get_api_config
import pandas as pd

class DatasonMLflowModel(mlflow.pyfunc.PythonModel):
    """Production MLflow model with datason serialization."""

    def __init__(self):
        self.api_config = get_api_config()
        self.request_count = 0
        self.error_count = 0

    def predict(self, context, model_input):
        """Make predictions with datason processing."""
        try:
            self.request_count += 1

            # Convert input to appropriate format
            if isinstance(model_input, pd.DataFrame):
                features = model_input.to_dict('records')[0]
            elif hasattr(model_input, 'to_dict'):
                features = model_input.to_dict()
            else:
                features = model_input

            # Process with datason
            processed_features = datason.auto_deserialize(features, config=self.api_config)

            # Model inference (replace with your model)
            prediction = self._run_model(processed_features)

            # Serialize response
            response = {
                "prediction": prediction,
                "model_version": context.artifacts.get("model_version", "unknown"),
                "request_count": self.request_count
            }

            return datason.serialize(response, config=self.api_config)

        except Exception as e:
            self.error_count += 1
            return {"error": str(e), "request_count": self.request_count}

    def _run_model(self, features):
        """Replace with your actual model inference."""
        return {"class": "positive", "score": 0.85}

def log_production_model():
    """Log model to MLflow with comprehensive metadata."""

    with mlflow.start_run():
        # Create model
        model = DatasonMLflowModel()

        # Log model
        mlflow.pyfunc.log_model(
            artifact_path="datason_model",
            python_model=model,
            pip_requirements=["datason>=1.0.0", "numpy", "pandas"],
            signature=mlflow.models.infer_signature(
                {"feature1": 1.0, "feature2": 2.0},
                {"prediction": {"class": "positive", "score": 0.85}}
            )
        )

        # Log datason configuration
        config_dict = {
            "uuid_format": model.api_config.uuid_format,
            "parse_uuids": model.api_config.parse_uuids,
            "date_format": str(model.api_config.date_format)
        }
        mlflow.log_dict(config_dict, "datason_config.json")

        # Log model metadata
        mlflow.log_param("serialization_framework", "datason")
        mlflow.log_param("model_type", "classification")
        mlflow.log_metric("initial_accuracy", 0.95)

        print("Model logged to MLflow with datason configuration")

# Register model
def register_production_model():
    """Register model for production use."""

    # Log model
    log_production_model()

    # Register in model registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/datason_model"

    mlflow.register_model(
        model_uri=model_uri,
        name="production_datason_model",
        tags={"framework": "datason", "stage": "production"}
    )

# Usage
register_production_model()

# Load and use model
model = mlflow.pyfunc.load_model("models:/production_datason_model/latest")
prediction = model.predict({"feature1": 1.0, "feature2": 2.0})
print(prediction)
```

### Seldon Core / KServe Production

**Basic Model:**
```python
from seldon_core.user_model import SeldonComponent
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

class Model(SeldonComponent):
    def predict(self, features, **kwargs):
        data = datason.auto_deserialize(features, config=API_CONFIG)
        output = run_model(data)
        return datason.serialize(output, config=API_CONFIG)
```

**Production Model with Monitoring:**
```python
from seldon_core.user_model import SeldonComponent
import datason
from datason.config import get_api_config
import logging
import time
import uuid
from datetime import datetime

API_CONFIG = get_api_config()

class ProductionModel(SeldonComponent):
    """Production Seldon model with comprehensive monitoring."""

    def __init__(self):
        self.logger = logging.getLogger("seldon_model")
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.logger.info("Production Seldon model initialized")

    def predict(self, features, **kwargs):
        """Make prediction with monitoring and error handling."""
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            self.request_count += 1
            self.logger.info(f"Processing request {request_id}")

            # Input validation
            if features is None or len(features) == 0:
                raise ValueError("Features cannot be empty")

            # Process with datason
            data = datason.auto_deserialize(features, config=API_CONFIG)

            # Model inference
            prediction = self._run_model(data)

            # Build response
            response = {
                "request_id": request_id,
                "prediction": prediction,
                "model_version": "1.0.0",
                "timestamp": datetime.now(),
                "processing_time_ms": (time.perf_counter() - start_time) * 1000
            }

            # Update metrics
            self.total_latency += time.perf_counter() - start_time

            return datason.serialize(response, config=API_CONFIG)

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Request {request_id} failed: {e}")

            error_response = {
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.now(),
                "status": "error"
            }

            return datason.serialize(error_response, config=API_CONFIG)

    def health_status(self):
        """Health check endpoint."""
        avg_latency = self.total_latency / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)

        status = {
            "status": "healthy" if error_rate < 0.1 else "degraded",
            "metrics": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency * 1000
            }
        }

        return datason.serialize(status, config=API_CONFIG)

    def _run_model(self, data):
        """Model inference - replace with your actual model."""
        return {"class": "positive", "confidence": 0.85}

# Deployment configuration
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: datason-model
spec:
  predictors:
  - name: default
    graph:
      name: classifier
      implementation: PYTHON_MODEL
      modelUri: gs://your-bucket/datason-model
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: your-registry/datason-model:latest
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
```

## 🔒 Security and Production Considerations

### Input Validation

```python
from pydantic import BaseModel, validator
import jsonschema

class SecureModelWrapper:
    """Model wrapper with security validation."""

    def __init__(self):
        self.api_config = get_api_config()

        # Define input schema
        self.input_schema = {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                }
            },
            "required": ["features"],
            "additionalProperties": False
        }

        # Security limits
        self.max_input_size = 1024 * 1024  # 1MB
        self.max_features = 1000

    def validate_input(self, data):
        """Comprehensive input validation."""

        # Schema validation
        try:
            jsonschema.validate(data, self.input_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Schema validation failed: {e.message}")

        # Size validation
        import sys
        if sys.getsizeof(data) > self.max_input_size:
            raise ValueError("Input size exceeds limit")

        # Feature count validation
        features = data.get("features", {})
        if len(features) > self.max_features:
            raise ValueError(f"Too many features: {len(features)}")

    def predict(self, input_data):
        """Secure prediction with validation."""

        # Validate input
        self.validate_input(input_data)

        # Process with datason
        processed_data = datason.auto_deserialize(input_data, config=self.api_config)

        # Run model
        prediction = run_model(processed_data)

        return datason.serialize(prediction, config=self.api_config)
```

### Rate Limiting

```python
from collections import defaultdict
import time
from threading import Lock

class RateLimitedModel:
    """Model with rate limiting."""

    def __init__(self):
        self.requests_per_minute = 100
        self.request_times = defaultdict(list)
        self.lock = Lock()

    def check_rate_limit(self, client_id="default"):
        """Check rate limits."""
        current_time = time.time()

        with self.lock:
            # Clean old requests
            self.request_times[client_id] = [
                req_time for req_time in self.request_times[client_id]
                if current_time - req_time < 60  # Last minute
            ]

            # Check limit
            if len(self.request_times[client_id]) >= self.requests_per_minute:
                raise ValueError("Rate limit exceeded")

            # Record request
            self.request_times[client_id].append(current_time)

    def predict(self, input_data, client_id="default"):
        """Prediction with rate limiting."""
        self.check_rate_limit(client_id)

        # Process normally
        return run_model(input_data)
```

## 📊 Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
PREDICTION_REQUESTS = Counter(
    'ml_prediction_requests_total',
    'Total prediction requests',
    ['model_id', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency',
    ['model_id']
)

ACTIVE_MODELS = Gauge(
    'ml_active_models',
    'Number of active models'
)

class MonitoredModel:
    """Model with Prometheus metrics."""

    def __init__(self, model_id):
        self.model_id = model_id
        ACTIVE_MODELS.inc()

    def predict(self, input_data):
        """Prediction with metrics."""
        start_time = time.perf_counter()

        try:
            result = run_model(input_data)

            # Record success
            PREDICTION_REQUESTS.labels(
                model_id=self.model_id,
                status='success'
            ).inc()

            return result

        except Exception as e:
            # Record error
            PREDICTION_REQUESTS.labels(
                model_id=self.model_id,
                status='error'
            ).inc()

            raise

        finally:
            # Record latency
            PREDICTION_LATENCY.labels(
                model_id=self.model_id
            ).observe(time.perf_counter() - start_time)

# Start metrics server
start_http_server(8080)
```

## 🎯 Best Practices

### ✅ Production Checklist

- [ ] **Input validation** with schema enforcement
- [ ] **Error handling** with graceful degradation
- [ ] **Monitoring** with metrics and health checks
- [ ] **Security** with rate limiting and access controls
- [ ] **Performance** optimization with caching and async processing
- [ ] **Observability** with structured logging and tracing
- [ ] **Deployment** strategies (blue-green, canary)
- [ ] **Configuration** management with environment variables

### 🚀 Performance Tips

1. **Use `get_api_config()`** for web APIs (preserves UUIDs as strings)
2. **Enable caching** for repeated predictions
3. **Implement batching** for high-throughput scenarios
4. **Use async processing** for I/O-bound operations
5. **Set resource limits** to prevent memory issues
6. **Monitor latency** and optimize bottlenecks

### 🔒 Security Guidelines

1. **Validate all inputs** with schemas
2. **Implement rate limiting** per client
3. **Set size limits** on requests
4. **Use HTTPS** in production
5. **Log security events** for monitoring
6. **Sanitize error messages** to prevent information leakage

These integrations ensure consistent JSON handling across your ML stack—BentoML or Ray Serve for serving, Streamlit/Gradio for demos, and MLflow for experiment tracking—all powered by datason's ML-friendly serialization.

## 🏗️ Architecture Overview

For a comprehensive view of how datason integrates across your entire ML serving pipeline, see the [Architecture Overview](architecture-overview.md). This includes:

- **High-level system architecture** with Mermaid diagrams
- **Data flow sequences** showing request/response patterns
- **Framework integration patterns** across all major platforms
- **Production deployment strategies** with monitoring and observability
- **End-to-end data flow** from clients to storage systems

The architecture overview provides visual diagrams and detailed explanations of how datason serves as the universal serialization layer across your ML infrastructure.

## 📚 Next Steps

- [Architecture Overview](architecture-overview.md) - Complete system architecture with diagrams
- [Production Patterns](production-patterns.md) - Advanced deployment strategies
- [Production Patterns](production-patterns.md) - Production deployment patterns
- [Architecture Overview](architecture-overview.md) - System architecture diagrams
