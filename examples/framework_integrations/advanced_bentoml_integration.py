#!/usr/bin/env python3
"""
Advanced BentoML Integration with Datason

This comprehensive example demonstrates production-ready BentoML integration
with datason, including:

- Model versioning and A/B testing
- Comprehensive monitoring and metrics
- Health checks and observability
- Batch processing optimization
- Error handling and graceful degradation
- Custom model runners
- Performance optimization

Setup:
    pip install bentoml datason prometheus-client

Usage:
    bentoml serve advanced_bentoml_integration:svc --production
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import bentoml
import numpy as np
from bentoml import metrics
from bentoml.io import JSON, NumpyNdarray

import datason
from datason.config import get_api_config, get_ml_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

# BentoML metrics
request_counter = metrics.Counter(
    name="prediction_requests_total",
    documentation="Total prediction requests",
    labelnames=["model_version", "endpoint", "status"],
)

error_counter = metrics.Counter(
    name="prediction_errors_total", documentation="Total prediction errors", labelnames=["model_version", "error_type"]
)

latency_histogram = metrics.Histogram(
    name="prediction_latency_seconds",
    documentation="Prediction latency in seconds",
    labelnames=["model_version", "endpoint"],
)

batch_size_histogram = metrics.Histogram(
    name="batch_size_distribution", documentation="Distribution of batch sizes", buckets=[1, 5, 10, 20, 50, 100, 200]
)

model_health_gauge = metrics.Gauge(
    name="model_health_score", documentation="Model health score (0-1)", labelnames=["model_version"]
)

# =============================================================================
# DATASON CONFIGURATIONS
# =============================================================================

# API configuration for web responses
API_CONFIG = get_api_config()

# ML configuration for model artifacts
ML_CONFIG = get_ml_config()

# Custom performance configuration
PERFORMANCE_CONFIG = datason.SerializationConfig(
    uuid_format="string",
    parse_uuids=False,
    date_format="unix",
    preserve_decimals=False,
    sort_keys=False,
    max_depth=20,
    max_size=10_000_000,  # 10MB limit
)

# =============================================================================
# MODEL WRAPPER WITH MONITORING
# =============================================================================


class ProductionModelWrapper:
    """Production model wrapper with comprehensive monitoring."""

    def __init__(self, model_id: str, model_version: str):
        self.model_id = model_id
        self.model_version = model_version
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.last_health_check = datetime.now()

        # Model-specific metrics
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Initialized model wrapper: {model_id} v{model_version}")

    def predict(self, features: Any, use_cache: bool = True) -> Dict[str, Any]:
        """Make prediction with monitoring and caching."""
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            self.request_count += 1

            # Input validation
            self._validate_input(features)

            # Check cache if enabled
            if use_cache:
                cache_key = self._generate_cache_key(features)
                if cache_key in self.prediction_cache:
                    self.cache_hits += 1
                    cached_result = self.prediction_cache[cache_key]
                    cached_result["request_id"] = request_id
                    cached_result["cached"] = True
                    return cached_result

            self.cache_misses += 1

            # Process with datason
            processed_features = datason.auto_deserialize(features, config=API_CONFIG)

            # Model inference
            prediction = self._run_inference(processed_features)

            # Build response
            response = {
                "request_id": request_id,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "prediction": prediction,
                "confidence": self._calculate_confidence(prediction),
                "timestamp": datetime.now(),
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "cached": False,
            }

            # Cache result
            if use_cache:
                self.prediction_cache[cache_key] = response.copy()
                # Limit cache size
                if len(self.prediction_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self.prediction_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.prediction_cache[key]

            # Update metrics
            self.total_latency += time.perf_counter() - start_time

            # Serialize response
            return datason.serialize(response, config=PERFORMANCE_CONFIG)

        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed for request {request_id}: {e}")

            error_response = {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "timestamp": datetime.now(),
                "status": "error",
            }

            return datason.serialize(error_response, config=API_CONFIG)

    def predict_batch(self, batch_features: List[Any]) -> List[Dict[str, Any]]:
        """Batch prediction with optimization."""
        batch_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        logger.info(f"Processing batch {batch_id} with {len(batch_features)} items")

        try:
            # Validate batch size
            if len(batch_features) > 100:
                raise ValueError(f"Batch size {len(batch_features)} exceeds limit of 100")

            # Process each item
            results = []
            for i, features in enumerate(batch_features):
                try:
                    result = self.predict(features, use_cache=True)
                    result["batch_id"] = batch_id
                    result["batch_index"] = i
                    results.append(result)
                except Exception as e:
                    error_result = {"batch_id": batch_id, "batch_index": i, "error": str(e), "status": "error"}
                    results.append(datason.serialize(error_result, config=API_CONFIG))

            # Batch metadata
            batch_response = {
                "batch_id": batch_id,
                "batch_size": len(batch_features),
                "successful_predictions": len([r for r in results if r.get("status") != "error"]),
                "failed_predictions": len([r for r in results if r.get("status") == "error"]),
                "total_processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "results": results,
                "timestamp": datetime.now(),
            }

            return datason.serialize(batch_response, config=PERFORMANCE_CONFIG)

        except Exception as e:
            logger.error(f"Batch processing failed for batch {batch_id}: {e}")

            error_response = {
                "batch_id": batch_id,
                "error": str(e),
                "timestamp": datetime.now(),
                "status": "batch_error",
            }

            return datason.serialize(error_response, config=API_CONFIG)

    def _validate_input(self, features: Any) -> None:
        """Validate input features."""
        if not features:
            raise ValueError("Features cannot be empty")

        # Size estimation
        try:
            estimated_size = len(str(features))
            if estimated_size > 1024 * 1024:  # 1MB limit
                raise ValueError(f"Input too large: {estimated_size} bytes")
        except Exception:
            pass  # nosec B110 - Size estimation failure is non-critical

    def _generate_cache_key(self, features: Any) -> str:
        """Generate cache key from features."""
        import hashlib
        import json

        try:
            # Convert to JSON string for hashing
            if isinstance(features, np.ndarray):
                features_str = str(features.tolist())
            else:
                features_str = json.dumps(features, sort_keys=True)

            return hashlib.md5(features_str.encode(), usedforsecurity=False).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.md5(str(features).encode(), usedforsecurity=False).hexdigest()

    def _run_inference(self, features: Any) -> Any:
        """Run model inference (replace with actual model)."""
        # Simulate processing time
        time.sleep(0.01)

        # Mock prediction based on input type
        if isinstance(features, dict):
            return {
                "class": "positive" if sum(features.values()) > 0 else "negative",
                "score": min(abs(sum(features.values())) / 10, 1.0),
            }
        elif isinstance(features, list):
            return [0.1, 0.7, 0.2]  # Multi-class probabilities
        elif isinstance(features, np.ndarray):
            return {"prediction": float(np.mean(features)), "shape": features.shape}
        else:
            return {"value": 42.0}

    def _calculate_confidence(self, prediction: Any) -> float:
        """Calculate prediction confidence."""
        if isinstance(prediction, dict):
            if "score" in prediction:
                return float(prediction["score"])
            elif "confidence" in prediction:
                return float(prediction["confidence"])

        # Default confidence
        return 0.85

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics."""
        avg_latency = self.total_latency / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        health_score = max(0, 1 - error_rate - (avg_latency / 10))  # Simple health score

        metrics = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": "healthy" if error_rate < 0.1 and avg_latency < 1.0 else "degraded",
            "health_score": health_score,
            "metrics": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency * 1000,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.prediction_cache),
                "last_health_check": self.last_health_check,
            },
            "timestamp": datetime.now(),
        }

        # Update health gauge
        model_health_gauge.labels(model_version=self.model_version).set(health_score)

        return datason.serialize(metrics, config=API_CONFIG)


# =============================================================================
# A/B TESTING MODEL WRAPPER
# =============================================================================


class ABTestingModelWrapper:
    """A/B testing wrapper for comparing model versions."""

    def __init__(self, model_a: ProductionModelWrapper, model_b: ProductionModelWrapper, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.requests_to_a = 0
        self.requests_to_b = 0

        logger.info(f"A/B testing setup: {model_a.model_version} vs {model_b.model_version}, split: {traffic_split}")

    def predict(self, features: Any) -> Dict[str, Any]:
        """Route prediction to A or B model based on traffic split."""
        import random

        if random.random() < self.traffic_split:  # nosec B311 - Demo A/B testing, not cryptographic
            # Route to model A
            self.requests_to_a += 1
            result = self.model_a.predict(features)
            result["ab_test_model"] = "A"
            result["ab_test_version"] = self.model_a.model_version
        else:
            # Route to model B
            self.requests_to_b += 1
            result = self.model_b.predict(features)
            result["ab_test_model"] = "B"
            result["ab_test_version"] = self.model_b.model_version

        return result

    def get_ab_test_stats(self) -> Dict[str, Any]:
        """Get A/B testing statistics."""
        total_requests = self.requests_to_a + self.requests_to_b

        stats = {
            "total_requests": total_requests,
            "model_a": {
                "version": self.model_a.model_version,
                "requests": self.requests_to_a,
                "percentage": self.requests_to_a / max(total_requests, 1),
                "error_rate": self.model_a.error_count / max(self.model_a.request_count, 1),
                "avg_latency_ms": self.model_a.total_latency / max(self.model_a.request_count, 1) * 1000,
            },
            "model_b": {
                "version": self.model_b.model_version,
                "requests": self.requests_to_b,
                "percentage": self.requests_to_b / max(total_requests, 1),
                "error_rate": self.model_b.error_count / max(self.model_b.request_count, 1),
                "avg_latency_ms": self.model_b.total_latency / max(self.model_b.request_count, 1) * 1000,
            },
            "traffic_split_target": self.traffic_split,
            "timestamp": datetime.now(),
        }

        return datason.serialize(stats, config=API_CONFIG)


# =============================================================================
# BENTOML SERVICE DEFINITION
# =============================================================================

# Initialize models
model_v1 = ProductionModelWrapper("classifier", "1.0.0")
model_v2 = ProductionModelWrapper("classifier", "2.0.0")
ab_test_wrapper = ABTestingModelWrapper(model_v1, model_v2, traffic_split=0.8)

# Create BentoML service
svc = bentoml.Service("advanced_datason_ml_service")

# =============================================================================
# API ENDPOINTS
# =============================================================================


@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    """Single prediction endpoint with comprehensive monitoring."""

    try:
        # Validate input
        if "features" not in input_data:
            raise ValueError("Missing 'features' in input")

        # Extract options
        model_version = input_data.get("model_version", "auto")
        use_ab_testing = input_data.get("use_ab_testing", True)

        # Route to appropriate model
        if use_ab_testing and model_version == "auto":
            result = ab_test_wrapper.predict(input_data["features"])
        elif model_version == "1.0.0":
            result = model_v1.predict(input_data["features"])
        elif model_version == "2.0.0":
            result = model_v2.predict(input_data["features"])
        else:
            # Default to A/B testing
            result = ab_test_wrapper.predict(input_data["features"])

        # Update metrics
        request_counter.labels(
            model_version=result.get("model_version", "unknown"), endpoint="predict", status="success"
        ).inc()

        return result

    except Exception as e:
        # Update error metrics
        error_counter.labels(model_version="unknown", error_type=type(e).__name__).inc()

        request_counter.labels(model_version="unknown", endpoint="predict", status="error").inc()

        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(),
            "status": "error",
        }

        return datason.serialize(error_response, config=API_CONFIG)


@svc.api(input=JSON(), output=JSON())
def predict_batch(input_data: dict) -> dict:
    """Batch prediction endpoint."""

    try:
        # Validate input
        if "batch_features" not in input_data:
            raise ValueError("Missing 'batch_features' in input")

        batch_features = input_data["batch_features"]
        model_version = input_data.get("model_version", "1.0.0")

        # Record batch size
        batch_size_histogram.observe(len(batch_features))

        # Route to appropriate model
        if model_version == "1.0.0":
            result = model_v1.predict_batch(batch_features)
        elif model_version == "2.0.0":
            result = model_v2.predict_batch(batch_features)
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        # Update metrics
        request_counter.labels(model_version=model_version, endpoint="predict_batch", status="success").inc()

        return result

    except Exception as e:
        # Update error metrics
        error_counter.labels(
            model_version=input_data.get("model_version", "unknown"), error_type=type(e).__name__
        ).inc()

        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(),
            "status": "batch_error",
        }

        return datason.serialize(error_response, config=API_CONFIG)


@svc.api(input=JSON(), output=JSON())
def health(input_data: dict = None) -> dict:
    """Comprehensive health check endpoint."""

    try:
        # Get health metrics from all models
        health_data = {
            "service_status": "healthy",
            "timestamp": datetime.now(),
            "models": {"v1.0.0": model_v1.get_health_metrics(), "v2.0.0": model_v2.get_health_metrics()},
            "ab_testing": ab_test_wrapper.get_ab_test_stats(),
            "system_info": {
                "total_requests": model_v1.request_count + model_v2.request_count,
                "total_errors": model_v1.error_count + model_v2.error_count,
                "uptime_seconds": time.time() - start_time,
            },
        }

        # Determine overall health
        v1_health = model_v1.get_health_metrics()
        v2_health = model_v2.get_health_metrics()

        if v1_health.get("status") == "degraded" or v2_health.get("status") == "degraded":
            health_data["service_status"] = "degraded"

        return datason.serialize(health_data, config=API_CONFIG)

    except Exception as e:
        error_response = {"service_status": "error", "error": str(e), "timestamp": datetime.now()}

        return datason.serialize(error_response, config=API_CONFIG)


@svc.api(input=JSON(), output=JSON())
def metrics_endpoint(input_data: dict = None) -> dict:
    """Custom metrics endpoint with detailed statistics."""

    try:
        metrics_data = {
            "timestamp": datetime.now(),
            "models": {
                "v1.0.0": {
                    "health": model_v1.get_health_metrics(),
                    "cache_stats": {
                        "cache_hits": model_v1.cache_hits,
                        "cache_misses": model_v1.cache_misses,
                        "cache_size": len(model_v1.prediction_cache),
                        "hit_rate": model_v1.cache_hits / max(model_v1.cache_hits + model_v1.cache_misses, 1),
                    },
                },
                "v2.0.0": {
                    "health": model_v2.get_health_metrics(),
                    "cache_stats": {
                        "cache_hits": model_v2.cache_hits,
                        "cache_misses": model_v2.cache_misses,
                        "cache_size": len(model_v2.prediction_cache),
                        "hit_rate": model_v2.cache_hits / max(model_v2.cache_hits + model_v2.cache_misses, 1),
                    },
                },
            },
            "ab_testing": ab_test_wrapper.get_ab_test_stats(),
            "datason_config": {
                "api_config": {
                    "uuid_format": API_CONFIG.uuid_format,
                    "parse_uuids": API_CONFIG.parse_uuids,
                    "date_format": str(API_CONFIG.date_format),
                },
                "performance_config": {
                    "max_size": PERFORMANCE_CONFIG.max_size,
                    "max_depth": PERFORMANCE_CONFIG.max_depth,
                    "preserve_decimals": PERFORMANCE_CONFIG.preserve_decimals,
                },
            },
        }

        return datason.serialize(metrics_data, config=API_CONFIG)

    except Exception as e:
        error_response = {"error": str(e), "timestamp": datetime.now(), "status": "metrics_error"}

        return datason.serialize(error_response, config=API_CONFIG)


@svc.api(input=JSON(), output=JSON())
def configure_ab_test(input_data: dict) -> dict:
    """Configure A/B testing parameters."""

    try:
        new_traffic_split = input_data.get("traffic_split")

        if new_traffic_split is not None:
            if not 0 <= new_traffic_split <= 1:
                raise ValueError("Traffic split must be between 0 and 1")

            old_split = ab_test_wrapper.traffic_split
            ab_test_wrapper.traffic_split = new_traffic_split

            response = {
                "message": "A/B test configuration updated",
                "old_traffic_split": old_split,
                "new_traffic_split": new_traffic_split,
                "timestamp": datetime.now(),
            }
        else:
            response = {
                "current_traffic_split": ab_test_wrapper.traffic_split,
                "stats": ab_test_wrapper.get_ab_test_stats(),
                "timestamp": datetime.now(),
            }

        return datason.serialize(response, config=API_CONFIG)

    except Exception as e:
        error_response = {"error": str(e), "timestamp": datetime.now(), "status": "config_error"}

        return datason.serialize(error_response, config=API_CONFIG)


# =============================================================================
# NUMPY ARRAY ENDPOINT
# =============================================================================


@svc.api(input=NumpyNdarray(), output=JSON())
def predict_numpy(input_array: np.ndarray) -> dict:
    """Prediction endpoint for NumPy arrays."""

    try:
        # Process with datason
        result = model_v1.predict(input_array)

        # Update metrics
        request_counter.labels(model_version="1.0.0", endpoint="predict_numpy", status="success").inc()

        return result

    except Exception as e:
        error_counter.labels(model_version="1.0.0", error_type=type(e).__name__).inc()

        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "input_shape": input_array.shape if hasattr(input_array, "shape") else None,
            "timestamp": datetime.now(),
            "status": "error",
        }

        return datason.serialize(error_response, config=API_CONFIG)


# =============================================================================
# SERVICE STARTUP
# =============================================================================

# Record service start time
start_time = time.time()

# Log service initialization
logger.info("Advanced BentoML service with datason initialized")
logger.info(f"Model v1.0.0: {model_v1.model_id}")
logger.info(f"Model v2.0.0: {model_v2.model_id}")
logger.info(f"A/B testing enabled with {ab_test_wrapper.traffic_split} traffic split")

if __name__ == "__main__":
    print("🚀 Advanced BentoML Integration with Datason")
    print("=" * 60)
    print()
    print("Features:")
    print("  ✓ Production model wrapper with monitoring")
    print("  ✓ A/B testing between model versions")
    print("  ✓ Comprehensive health checks")
    print("  ✓ Batch processing optimization")
    print("  ✓ Prediction caching")
    print("  ✓ Prometheus metrics integration")
    print("  ✓ Error handling and graceful degradation")
    print("  ✓ NumPy array support")
    print()
    print("Endpoints:")
    print("  • POST /predict - Single prediction with A/B testing")
    print("  • POST /predict_batch - Batch predictions")
    print("  • POST /predict_numpy - NumPy array predictions")
    print("  • GET /health - Comprehensive health check")
    print("  • GET /metrics_endpoint - Detailed metrics")
    print("  • POST /configure_ab_test - A/B test configuration")
    print()
    print("Usage:")
    print("  bentoml serve advanced_bentoml_integration:svc --production")
    print("  bentoml serve advanced_bentoml_integration:svc --reload  # Development")
    print()
    print("Example requests:")
    print('  curl -X POST "http://localhost:3000/predict" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"features": {"feature1": 1.0, "feature2": 2.0}}\'')
    print()
    print('  curl -X POST "http://localhost:3000/predict_batch" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"batch_features": [{"f1": 1}, {"f1": 2}], "model_version": "1.0.0"}\'')
