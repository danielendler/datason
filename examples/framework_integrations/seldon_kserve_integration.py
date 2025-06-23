#!/usr/bin/env python3
"""
Seldon Core / KServe Integration with DataSON

A comprehensive example showing how to integrate DataSON with Seldon Core
and KServe for Kubernetes-native ML model serving and deployment.

Features:
- Modern DataSON API (dump_api, load_smart, dump_ml)
- Seldon Core model serving with DataSON serialization
- KServe inference service integration
- Kubernetes deployment patterns
- Model monitoring and health checks
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

try:
    from seldon_core.user_model import SeldonComponent

    SELDON_AVAILABLE = True
except ImportError:
    SELDON_AVAILABLE = False
    SeldonComponent = None

try:
    import kserve

    KSERVE_AVAILABLE = True
except ImportError:
    KSERVE_AVAILABLE = False
    kserve = None

import numpy as np

import datason as ds
from datason.config import get_api_config

API_CONFIG = get_api_config()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSONMLModel:
    """ML Model with DataSON integration for Seldon/KServe."""

    def __init__(self, model_name: str = "datason_k8s_model"):
        self.model_name = model_name
        self.version = "2.0.0"
        self.loaded_at = time.time()
        self.model_config = {
            "framework": "sklearn",
            "model_type": "ensemble_classifier",
            "input_features": 20,
            "output_classes": 5,
            "performance": {"accuracy": 0.96, "latency_ms": 12},
        }
        self.prediction_count = 0

        logger.info(f"🚀 DataSON ML Model initialized: {model_name}")

    def predict(self, features: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Make predictions with comprehensive metadata."""
        self.prediction_count += 1

        # Handle different input formats
        if isinstance(features, np.ndarray):
            features = features.tolist()

        # Simulate model prediction
        np.random.seed(42 + self.prediction_count)

        if isinstance(features[0], (list, np.ndarray)):
            # Batch prediction
            batch_size = len(features)
            predictions = np.random.randint(0, 5, batch_size).tolist()
            probabilities = np.random.dirichlet(np.ones(5), batch_size).tolist()

            result = {
                "predictions": predictions,
                "probabilities": probabilities,
                "batch_info": {
                    "batch_size": batch_size,
                    "input_shape": [len(features), len(features[0])],
                    "prediction_id": f"batch_{self.prediction_count}",
                },
            }
        else:
            # Single prediction
            prediction = int(np.random.randint(0, 5))
            probabilities = np.random.dirichlet(np.ones(5)).tolist()

            result = {
                "prediction": prediction,
                "probabilities": probabilities,
                "single_info": {"input_shape": len(features), "prediction_id": f"single_{self.prediction_count}"},
            }

        # Add model metadata
        result.update(
            {
                "model_metadata": {
                    "model_name": self.model_name,
                    "version": self.version,
                    "prediction_count": self.prediction_count,
                    "timestamp": time.time(),
                },
                "processing_info": {"datason_processed": True, "latency_estimate_ms": 12 + np.random.randint(-3, 4)},
            }
        )

        return result

    def health_check(self) -> Dict[str, Any]:
        """Model health check with DataSON formatting."""
        return {
            "status": "healthy",
            "model_info": {
                "name": self.model_name,
                "version": self.version,
                "loaded_at": self.loaded_at,
                "predictions_served": self.prediction_count,
            },
            "system_info": {
                "uptime_seconds": time.time() - self.loaded_at,
                "memory_status": "normal",
                "last_prediction": time.time(),
            },
            "datason_info": {
                "version": getattr(ds, "__version__", "unknown"),
                "api_config_active": API_CONFIG is not None,
            },
        }


if SELDON_AVAILABLE:

    class DataSONSeldonModel(SeldonComponent):
        """Seldon Core model with DataSON integration."""

        def __init__(self, model_name: str = "datason_seldon_model"):
            super().__init__()
            self.model = DataSONMLModel(model_name)
            logger.info("🎯 DataSON Seldon Model Component initialized")

        def predict(
            self, features: Union[List, np.ndarray], names: Optional[List[str]] = None, **kwargs
        ) -> Union[np.ndarray, List, Dict]:
            """Seldon predict method with DataSON processing."""
            try:
                logger.info(
                    f"📊 Received prediction request: {type(features)}, shape: {np.array(features).shape if isinstance(features, (list, np.ndarray)) else 'unknown'}"
                )

                # Process input with DataSON smart loading if it's a dict
                if isinstance(features, dict):
                    processed_features = ds.load_smart(features, config=API_CONFIG)
                    actual_features = processed_features.get("data", processed_features)
                else:
                    actual_features = features

                # Make prediction
                prediction_result = self.model.predict(actual_features)

                # Enhanced response with Seldon metadata
                enhanced_result = {
                    "prediction_result": prediction_result,
                    "seldon_info": {
                        "component_type": "DataSONSeldonModel",
                        "names": names,
                        "kwargs": kwargs,
                        "processed_at": time.time(),
                    },
                }

                # Use DataSON API serialization for clean output
                serialized_result = ds.dump_api(enhanced_result)

                logger.info(f"✅ Prediction completed: {prediction_result.get('prediction', 'batch')}")
                return serialized_result

            except Exception as e:
                logger.error(f"❌ Prediction failed: {e}")
                error_response = {
                    "error": str(e),
                    "status": "seldon_prediction_failed",
                    "model_info": self.model.model_name,
                    "timestamp": time.time(),
                }
                return ds.dump_api(error_response)

        def health_status(self) -> Dict[str, Any]:
            """Seldon health status with DataSON formatting."""
            health_info = self.model.health_check()
            health_info["seldon_component"] = {
                "type": "DataSONSeldonModel",
                "status": "ready",
                "last_health_check": time.time(),
            }
            return ds.dump_api(health_info)


if KSERVE_AVAILABLE:

    class DataSONKServeModel(kserve.Model):
        """KServe model with DataSON integration."""

        def __init__(self, name: str = "datason-kserve-model"):
            super().__init__(name)
            self.model = DataSONMLModel(name)
            self.ready = True
            logger.info("🎯 DataSON KServe Model initialized")

        def load(self) -> bool:
            """Load the model (KServe lifecycle method)."""
            try:
                # Model is already loaded in __init__, but we can add loading logic here
                logger.info("📦 Loading DataSON KServe model...")
                self.ready = True
                return True
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                self.ready = False
                return False

        def predict(self, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
            """KServe predict method with DataSON processing."""
            try:
                logger.info("📊 KServe prediction request received")

                # Process KServe payload with DataSON
                if "instances" in payload:
                    # Standard KServe format
                    instances = payload["instances"]
                    processed_instances = ds.load_smart(instances, config=API_CONFIG)
                elif "inputs" in payload:
                    # Alternative KServe format
                    processed_instances = ds.load_smart(payload["inputs"], config=API_CONFIG)
                else:
                    # Direct data format
                    processed_instances = ds.load_smart(payload, config=API_CONFIG)

                # Handle single vs batch
                if isinstance(processed_instances, list) and len(processed_instances) == 1:
                    # Single instance
                    prediction_result = self.model.predict(processed_instances[0])
                else:
                    # Batch or direct features
                    prediction_result = self.model.predict(processed_instances)

                # Format response for KServe
                kserve_response = {
                    "predictions": prediction_result,
                    "kserve_info": {
                        "model_name": self.name,
                        "headers": headers or {},
                        "processed_at": time.time(),
                        "datason_enhanced": True,
                    },
                }

                serialized_response = ds.dump_api(kserve_response)
                logger.info("✅ KServe prediction completed")

                return serialized_response

            except Exception as e:
                logger.error(f"❌ KServe prediction failed: {e}")
                error_response = {
                    "error": str(e),
                    "status": "kserve_prediction_failed",
                    "model_name": self.name,
                    "timestamp": time.time(),
                }
                return ds.dump_api(error_response)

        def is_ready(self) -> bool:
            """KServe readiness check."""
            return self.ready

        def is_alive(self) -> bool:
            """KServe liveness check."""
            return self.ready


class K8sMLServingDemo:
    """Demonstration of Kubernetes ML serving with DataSON."""

    def __init__(self):
        self.model = DataSONMLModel("demo_k8s_model")

    def create_seldon_deployment_yaml(self) -> str:
        """Generate Seldon Core deployment YAML with DataSON integration."""
        seldon_config = {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {
                "name": "datason-seldon-deployment",
                "labels": {"datason-integration": "true", "version": "v1"},
            },
            "spec": {
                "protocol": "seldon",
                "transport": "rest",
                "replicas": 2,
                "predictors": [
                    {
                        "name": "datason-predictor",
                        "replicas": 1,
                        "componentSpecs": [
                            {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "datason-model",
                                            "image": "your-registry/datason-seldon-model:latest",
                                            "env": [{"name": "DATASON_CONFIG", "value": "api_mode"}],
                                            "resources": {
                                                "requests": {"memory": "1Gi", "cpu": "500m"},
                                                "limits": {"memory": "2Gi", "cpu": "1000m"},
                                            },
                                        }
                                    ]
                                }
                            }
                        ],
                        "graph": {"name": "datason-model", "type": "MODEL", "endpoint": {"type": "REST"}},
                    }
                ],
            },
        }

        return ds.dumps_json(seldon_config, indent=2)

    def create_kserve_service_yaml(self) -> str:
        """Generate KServe InferenceService YAML with DataSON integration."""
        kserve_config = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": "datason-kserve-service",
                "annotations": {"datason.io/integration": "enabled", "datason.io/api-mode": "true"},
            },
            "spec": {
                "predictor": {
                    "serviceAccountName": "datason-kserve-sa",
                    "minReplicas": 1,
                    "maxReplicas": 3,
                    "containers": [
                        {
                            "name": "datason-predictor",
                            "image": "your-registry/datason-kserve-model:latest",
                            "env": [{"name": "DATASON_MODE", "value": "kserve_api"}],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1000m"},
                                "limits": {"memory": "4Gi", "cpu": "2000m"},
                            },
                            "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                        }
                    ],
                }
            },
        }

        return ds.dumps_json(kserve_config, indent=2)

    def create_sample_requests(self) -> Dict[str, Any]:
        """Create sample requests for testing Kubernetes deployments."""

        # Seldon Core request format
        seldon_request = {
            "data": {
                "ndarray": [
                    [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                        13.0,
                        14.0,
                        15.0,
                        16.0,
                        17.0,
                        18.0,
                        19.0,
                        20.0,
                    ]
                ]
            },
            "meta": {"client_id": "k8s_client", "request_type": "seldon_prediction", "timestamp": time.time()},
        }

        # KServe request format
        kserve_request = {
            "instances": [
                [
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                    21.0,
                ]
            ],
            "parameters": {"client_id": "kserve_client", "prediction_mode": "standard"},
        }

        return {"seldon_request": seldon_request, "kserve_request": kserve_request}

    def generate_docker_files(self) -> Dict[str, str]:
        """Generate Dockerfiles for Seldon and KServe deployments."""

        seldon_dockerfile = """
FROM python:3.9-slim

# Install dependencies
RUN pip install seldon-core datason numpy

# Copy model code
COPY seldon_kserve_integration.py /app/
WORKDIR /app

# Expose Seldon port
EXPOSE 9000

# Run Seldon model
CMD ["python", "-c", "from seldon_kserve_integration import DataSONSeldonModel; import seldon_core.microservice as microservice; microservice.main(DataSONSeldonModel())"]
"""

        kserve_dockerfile = """
FROM python:3.9-slim

# Install dependencies
RUN pip install kserve datason numpy

# Copy model code
COPY seldon_kserve_integration.py /app/
WORKDIR /app

# Expose KServe port
EXPOSE 8080

# Run KServe model
CMD ["python", "-c", "from seldon_kserve_integration import DataSONKServeModel; import kserve; model = DataSONKServeModel(); kserve.ModelServer().start([model])"]
"""

        return {"seldon_dockerfile": seldon_dockerfile, "kserve_dockerfile": kserve_dockerfile}


def run_k8s_ml_serving_demo():
    """Run a comprehensive Kubernetes ML serving demonstration."""
    print("🚀 Kubernetes ML Serving + DataSON Integration Demo")
    print("=" * 60)

    demo = K8sMLServingDemo()

    # Check availability
    seldon_status = "✅ Available" if SELDON_AVAILABLE else "❌ Not installed"
    kserve_status = "✅ Available" if KSERVE_AVAILABLE else "❌ Not installed"

    print("\n📦 Framework Status:")
    print(f"  - Seldon Core: {seldon_status}")
    print(f"  - KServe: {kserve_status}")

    if not (SELDON_AVAILABLE or KSERVE_AVAILABLE):
        print("\n💡 Install frameworks:")
        print("  pip install seldon-core kserve")

    print("\n1️⃣ DataSON Features for K8s ML Serving:")
    print("  - 🌐 dump_api() for standardized responses")
    print("  - 🧠 load_smart() for flexible input processing")
    print("  - 📊 Enhanced metadata and monitoring")
    print("  - 🔄 Automatic serialization/deserialization")
    print("  - ⚡ High-performance processing")

    print("\n2️⃣ Deployment Configurations:")

    # Generate Seldon deployment
    print("\n📋 Seldon Core Deployment YAML:")
    seldon_yaml = demo.create_seldon_deployment_yaml()
    if len(seldon_yaml) > 500:
        print(seldon_yaml[:500] + "...")
    else:
        print(seldon_yaml)

    # Generate KServe service
    print("\n📋 KServe InferenceService YAML:")
    kserve_yaml = demo.create_kserve_service_yaml()
    if len(kserve_yaml) > 500:
        print(kserve_yaml[:500] + "...")
    else:
        print(kserve_yaml)

    print("\n3️⃣ Sample API Requests:")
    sample_requests = demo.create_sample_requests()

    for platform, request_data in sample_requests.items():
        print(f"\n{platform.upper()}:")
        serialized_request = ds.dumps_json(ds.dump_api(request_data), indent=2)
        print(f"curl -X POST http://your-k8s-cluster/{platform.split('_')[0]}/predict \\")
        print("  -H 'Content-Type: application/json' \\")
        print(f"  -d '{serialized_request[:100]}...'")

    print("\n4️⃣ Docker Build Instructions:")
    docker_files = demo.generate_docker_files()

    for dockerfile_name, dockerfile_content in docker_files.items():
        print(f"\n{dockerfile_name.upper()}:")
        print(f"# Save as {dockerfile_name.replace('_', '.')}")
        print(
            f"docker build -t your-registry/datason-{dockerfile_name.split('_')[0]}-model:latest -f {dockerfile_name.replace('_', '.')} ."
        )

    # Demonstrate local processing
    print("\n5️⃣ Local DataSON Processing Demo:")
    for platform, request_data in sample_requests.items():
        print(f"\n🔄 Processing {platform}:")

        # Parse request with DataSON
        json_string = ds.dumps_json(request_data)
        parsed_request = ds.load_smart(json_string, config=API_CONFIG)

        # Extract features
        if "data" in parsed_request and "ndarray" in parsed_request["data"]:
            features = parsed_request["data"]["ndarray"][0]
        elif "instances" in parsed_request:
            features = parsed_request["instances"][0]
        else:
            features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # Make prediction
        result = demo.model.predict(features)

        # Format response
        response = ds.dump_api({"prediction_result": result, "platform": platform, "processed_with_datason": True})

        print(f"  ✅ Prediction: {result.get('prediction', 'batch')}")
        print(f"  📊 Response size: {len(ds.dumps_json(response))} bytes")

    print("\n✅ Kubernetes ML Serving Demo Completed!")
    print("\n🚀 Next Steps:")
    print("1. Build Docker images with the provided Dockerfiles")
    print("2. Apply the Kubernetes YAML configurations")
    print("3. Test the deployments with the sample requests")
    print("4. Monitor with kubectl and DataSON's built-in metrics")

    return {
        "seldon_available": SELDON_AVAILABLE,
        "kserve_available": KSERVE_AVAILABLE,
        "demo_completed": True,
        "configurations_generated": 2,
        "sample_requests": len(sample_requests),
    }


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_k8s_ml_serving_demo()

    print(f"\n🎯 Demo Results: {results}")

    # Additional information
    print("\n📚 Additional Resources:")
    print("- Seldon Core Docs: https://docs.seldon.io/")
    print("- KServe Docs: https://kserve.github.io/website/")
    print("- DataSON K8s Integration: Enhanced serialization for cloud-native ML")

    if SELDON_AVAILABLE:
        print("\n🔧 Seldon Component Available:")
        print("  from seldon_kserve_integration import DataSONSeldonModel")

    if KSERVE_AVAILABLE:
        print("\n🔧 KServe Model Available:")
        print("  from seldon_kserve_integration import DataSONKServeModel")
