#!/usr/bin/env python3
"""
🚀 BentoML + DataSON Integration Guide
=====================================

Comprehensive example showcasing BentoML model serving with DataSON for:
- JSON endpoint with intelligent parsing
- NumPy array processing with ML serialization
- Text processing with smart data handling
- Health checks and monitoring
- Production deployment patterns

Key DataSON features:
- dump_api() for clean JSON responses
- load_smart() for intelligent request parsing
- dump_ml() for ML-optimized serialization
- dumps_secure() for PII-safe logging

Installation:
    pip install bentoml datason scikit-learn numpy

Usage:
    python bentoml_integration_guide.py
    bentoml serve datason_service:svc --reload
"""

import time
from typing import Any, Dict, List, Optional

try:
    import bentoml
    import numpy as np
    from pydantic import BaseModel

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None

import os
import sys

# Add parent directory to path to import datason
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import datason as ds
except ImportError:
    print("❌ DataSON not available. Please install from the parent directory.")
    sys.exit(1)

import json
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    data: List[List[float]]
    metadata: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predictions: List[float]
    model_info: Dict[str, Any]
    processing_time: float
    timestamp: str


class TextRequest(BaseModel):
    text: str
    options: Optional[Dict[str, Any]] = None


class TextResponse(BaseModel):
    processed_text: str
    analysis: Dict[str, Any]
    timestamp: str


# Mock ML model for demonstration
class DataSONMLModel:
    """Mock ML model with DataSON integration."""

    def __init__(self, model_name: str = "datason_demo_model"):
        self.model_name = model_name
        self.version = "1.2.0"
        self.trained_at = time.time()
        self.model_metadata = {"architecture": "ensemble", "features": 10, "classes": 3, "accuracy": 0.94}

    def predict_single(self, features: List[float]) -> Dict[str, Any]:
        """Single prediction with metadata."""
        # Simulate prediction
        np.random.seed(42)
        prediction = np.random.choice([0, 1, 2])
        confidence = 0.7 + 0.3 * np.random.random()

        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "feature_count": len(features),
            "model_version": self.version,
        }

    def predict_batch(self, batch_features: List[List[float]]) -> Dict[str, Any]:
        """Batch prediction with comprehensive metadata."""
        predictions = []
        confidences = []

        for features in batch_features:
            result = self.predict_single(features)
            predictions.append(result["prediction"])
            confidences.append(result["confidence"])

        return {
            "predictions": predictions,
            "confidences": confidences,
            "batch_size": len(batch_features),
            "avg_confidence": float(np.mean(confidences)),
            "model_metadata": self.model_metadata,
            "processed_at": time.time(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "trained_at": self.trained_at,
            "metadata": self.model_metadata,
            "datason_integration": True,
            "status": "ready",
        }


if BENTOML_AVAILABLE:
    # Initialize model for BentoML service
    model = DataSONMLModel("production_classifier")

    # Create BentoML service with DataSON integration
    svc = bentoml.Service("datason_ml_service")

    @bentoml.api
    def predict_json(request: PredictionRequest) -> PredictionResponse:
        """JSON endpoint with DataSON intelligent parsing."""
        start_time = datetime.now()

        try:
            print(f"📥 Received prediction request with {len(request.data)} samples")

            # Use DataSON for enhanced data processing
            enhanced_data = ds.load_smart(json.dumps(request.data))

            # Mock ML prediction (replace with your model)
            predictions = model.predict_batch(enhanced_data)

            # Create response with DataSON API serialization
            model_info = {
                "model_type": "mock_regressor",
                "version": "1.0.0",
                "input_shape": np.array(request.data).shape,
                "datason_features": ["smart_parsing", "type_inference"],
            }

            # Use dump_api for clean response data
            clean_model_info = ds.dump_api(model_info)

            processing_time = (datetime.now() - start_time).total_seconds()

            response = PredictionResponse(
                predictions=predictions["predictions"],
                model_info=clean_model_info,
                processing_time=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            print(f"✅ Prediction completed in {processing_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            # Return error response
            return PredictionResponse(
                predictions=[],
                model_info={"error": str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    @bentoml.api
    def predict_numpy(json_input: Dict[str, Any]) -> Dict[str, Any]:
        """NumPy endpoint with ML-optimized serialization."""
        start_time = datetime.now()

        try:
            print("📊 Processing NumPy array prediction...")

            # Extract NumPy array from request
            if "array" not in json_input:
                raise ValueError("Missing 'array' field in request")

            # Use DataSON for intelligent array parsing
            array_data = ds.load_smart(json.dumps(json_input["array"]))
            np_array = np.array(array_data)

            print(f"📐 Array shape: {np_array.shape}")

            # Mock ML processing
            processed_array = np_array * 2 + 1  # Simple transformation
            predictions = np.sum(processed_array, axis=-1)

            # Prepare ML response with DataSON
            ml_response = {
                "predictions": predictions.tolist(),
                "input_stats": {
                    "shape": np_array.shape,
                    "mean": float(np.mean(np_array)),
                    "std": float(np.std(np_array)),
                    "min": float(np.min(np_array)),
                    "max": float(np.max(np_array)),
                },
                "processing_info": {
                    "operation": "linear_transformation",
                    "formula": "2x + 1",
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                },
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "datason_version": getattr(ds, "__version__", "unknown"),
                },
            }

            # Use dump_ml for ML-optimized response
            return ds.dump_ml(ml_response)

        except Exception as e:
            logger.error(f"❌ NumPy processing error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }

    @bentoml.api
    def process_text(request: TextRequest) -> TextResponse:
        """Text processing endpoint with smart data handling."""
        start_time = datetime.now()

        try:
            print(f"📝 Processing text: {request.text[:50]}...")

            # Use DataSON for enhanced text processing
            {
                "original_text": request.text,
                "options": request.options or {},
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Enhanced text processing
            processed_text = request.text.upper()  # Simple transformation

            # Text analysis
            analysis = {
                "character_count": len(request.text),
                "word_count": len(request.text.split()),
                "contains_numbers": any(char.isdigit() for char in request.text),
                "contains_special_chars": not request.text.isalnum(),
                "processing_options": request.options or {},
            }

            # Use DataSON for comprehensive analysis
            enhanced_analysis = ds.dump_api(analysis)

            response = TextResponse(
                processed_text=processed_text,
                analysis=enhanced_analysis,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Text processing completed in {processing_time:.3f}s")

            return response

        except Exception as e:
            logger.error(f"❌ Text processing error: {e}")
            return TextResponse(
                processed_text="", analysis={"error": str(e)}, timestamp=datetime.now(timezone.utc).isoformat()
            )

    @bentoml.api
    def health_check(json_input: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint with comprehensive system status."""
        print("🏥 Health check requested...")

        # System health information
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_info": {"name": "datason_ml_service", "version": "1.0.0", "datason_integration": True},
            "system_stats": {
                "uptime": "unknown",  # Would be calculated in real deployment
                "memory_usage": "unknown",  # Would use psutil in real deployment
                "active_requests": 0,
            },
            "datason_features": {
                "smart_parsing": True,
                "ml_optimization": True,
                "api_serialization": True,
                "secure_redaction": True,
            },
        }

        # Use DataSON for comprehensive health response
        return ds.dump_api(health_data)


class BentoMLDataSONDemo:
    """Demonstration of BentoML + DataSON integration."""

    def __init__(self):
        self.model = DataSONMLModel("demo_model")

    def create_sample_requests(self) -> Dict[str, Any]:
        """Create sample requests for testing different endpoints."""

        samples = {
            "json_single": {
                "features": [1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3, 4.4, 5.5],
                "metadata": {"client_id": "demo_client", "request_type": "single_prediction", "timestamp": time.time()},
            },
            "json_batch": {
                "features": [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ],
                "metadata": {"client_id": "batch_client", "request_type": "batch_prediction", "batch_id": "batch_001"},
            },
            "numpy_array": np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
            "text_csv": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0",
            "text_json": ds.dumps_json(
                {
                    "features": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                    "metadata": {"source": "text_endpoint"},
                }
            ),
        }

        return samples

    def demonstrate_serialization(self):
        """Demonstrate DataSON serialization features."""
        print("🔄 DataSON Serialization Demonstration")
        print("-" * 40)

        # Create complex data
        complex_data = {
            "model_results": {
                "predictions": [0, 1, 2, 1, 0],
                "confidences": [0.95, 0.87, 0.92, 0.78, 0.89],
                "processed_at": time.time(),
            },
            "metadata": {
                "model_version": "1.2.0",
                "feature_engineering": ["normalize", "scale", "transform"],
                "performance_metrics": {"accuracy": 0.94, "precision": 0.91},
            },
        }

        # Demonstrate different serialization modes
        serialization_modes = [
            ("dump_api", "API-friendly serialization"),
            ("dump_ml", "ML-optimized serialization"),
            ("dump_secure", "Security-focused serialization"),
        ]

        for mode_name, description in serialization_modes:
            print(f"\n📊 {description}:")

            if mode_name == "dump_api":
                serialized = ds.dump_api(complex_data)
            elif mode_name == "dump_ml":
                serialized = ds.dump_ml(complex_data)
            elif mode_name == "dump_secure":
                serialized = ds.dump_secure(complex_data)

            # Show size and parsing
            json_str = ds.dumps_json(serialized, indent=2)
            parsed_back = serialized  # Already parsed

            print(f"  Size: {len(json_str)} bytes")
            print(f"  Parsed successfully: {parsed_back['model_results']['predictions'][:3]}...")


def run_bentoml_demo():
    """Run a comprehensive BentoML + DataSON demonstration."""
    print("🚀 BentoML + DataSON Integration Demo")
    print("=" * 50)

    if not BENTOML_AVAILABLE:
        print("❌ BentoML is not available.")
        print("💡 Install with: pip install bentoml")
        print("\nFallback: Showing DataSON features for ML serving...")

        # Show DataSON features without BentoML
        demo = BentoMLDataSONDemo()

        print("\n1️⃣ DataSON Serialization for ML Services:")
        demo.demonstrate_serialization()

        print("\n2️⃣ Sample Request/Response Processing:")
        samples = demo.create_sample_requests()

        for request_type, request_data in samples.items():
            if request_type.startswith("json"):
                print(f"\n📋 {request_type.upper()} Request:")

                # Serialize request
                serialized_request = ds.dump_api(request_data)
                print(f"  Request size: {len(ds.dumps_json(serialized_request))} bytes")

                # Parse and process
                parsed_request = serialized_request  # Already parsed
                features = parsed_request.get("features", [])

                # Mock prediction
                if features and isinstance(features[0], list):
                    result = demo.model.predict_batch(features)
                else:
                    result = demo.model.predict_single(features)

                # Serialize response
                response = {"prediction_result": result, "processing_info": {"datason_processed": True}}
                serialized_response = ds.dump_api(response)

                print(f"  Response size: {len(ds.dumps_json(serialized_response))} bytes")
                print(f"  Prediction: {result.get('prediction', result.get('predictions', 'N/A'))}")

        print("\n✅ DataSON demonstration completed!")
        print("💡 Install BentoML for full service demo: pip install bentoml")
        return {"status": "demo_completed_without_bentoml"}

    # Full BentoML demo
    print("\n1️⃣ BentoML Service Available!")
    print("📋 Service Endpoints:")
    print("  - POST /predict_json - JSON input/output")
    print("  - POST /predict_numpy - NumPy array input")
    print("  - POST /predict_text - Text input (CSV or JSON)")
    print("  - POST /model_info - Model information")
    print("  - POST /health_check - Health diagnostics")

    print("\n2️⃣ DataSON Features Integrated:")
    print("  - 🧠 load_smart() for intelligent input parsing")
    print("  - 🌐 dump_api() for clean JSON responses")
    print("  - 🔄 Automatic type handling and validation")
    print("  - 📊 Enhanced error responses with metadata")

    # Create demo
    demo = BentoMLDataSONDemo()

    print("\n3️⃣ Sample Requests:")
    samples = demo.create_sample_requests()

    for endpoint, sample_data in samples.items():
        if isinstance(sample_data, dict):
            serialized = ds.dumps_json(ds.dump_api(sample_data), indent=2)
            print(f"\n{endpoint.upper()}:")
            print(f"curl -X POST http://localhost:3000/{endpoint.replace('_', '/')} \\")
            print("  -H 'Content-Type: application/json' \\")
            print(f"  -d '{serialized[:100]}...'")

    print("\n4️⃣ Start the service:")
    print(f"bentoml serve {__file__.replace('.py', '')}:svc --reload")

    print("\n✅ BentoML + DataSON integration ready!")
    return {"status": "service_ready", "endpoints": 5}


def demonstrate_client_usage():
    """Demonstrate how to interact with the BentoML + DataSON service."""
    print("\n🧪 Client Usage Examples")
    print("=" * 30)

    # Example requests
    examples = {
        "json_prediction": {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "metadata": {"source": "demo", "version": "1.0"},
        },
        "numpy_prediction": {"array": [[1, 2], [3, 4], [5, 6]]},
        "text_processing": {
            "text": "Hello DataSON + BentoML Integration!",
            "options": {"uppercase": True, "analyze": True},
        },
    }

    print("📋 Example requests:")
    for endpoint, data in examples.items():
        print(f"\n{endpoint}:")
        # Use DataSON for clean example formatting
        api_data = ds.dump_api(data)
        formatted = ds.dumps(api_data)
        print(formatted)

    print("\n🌐 cURL Examples:")
    print("curl -X POST http://localhost:3000/predict_json \\")
    print("  -H 'Content-Type: application/json' \\")
    print('  -d \'{"data": [[1.0, 2.0]], "metadata": {"source": "curl"}}\'')

    print("\ncurl -X POST http://localhost:3000/health_check \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{}'")


def create_deployment_config():
    """Generate deployment configuration files."""
    print("\n🚀 Creating deployment configurations...")

    # BentoML configuration
    bentofile_config = {
        "service": "datason_service:svc",
        "labels": {"owner": "datason-team", "stage": "demo"},
        "include": ["*.py", "requirements.txt"],
        "python": {
            "packages": ["datason", "bentoml>=1.4.0", "numpy>=1.21.0", "scikit-learn>=1.0.0", "pydantic>=2.0.0"]
        },
        "docker": {"distro": "debian", "python_version": "3.11"},
    }

    # Use DataSON for clean YAML-like output
    config_json = ds.dumps(ds.dump_api(bentofile_config))

    print("📄 bentofile.yaml content:")
    print(config_json)

    # Production deployment example
    deployment_info = {
        "deployment_commands": [
            "bentoml build",
            "bentoml containerize datason_ml_service:latest",
            "docker run -p 3000:3000 datason_ml_service:latest",
        ],
        "kubernetes_deployment": {
            "replicas": 3,
            "resources": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"cpu": "2000m", "memory": "4Gi"}},
        },
        "monitoring": {"metrics": ["request_count", "response_time", "error_rate"], "health_check": "/health_check"},
    }

    print("\n🏗️  Deployment Configuration:")
    print(ds.dumps(ds.dump_api(deployment_info)))


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_bentoml_demo()

    print(f"\n🎯 Demo Results: {results}")

    if BENTOML_AVAILABLE:
        print("\n💡 Next Steps:")
        print("1. Save this file as 'bentoml_datason_service.py'")
        print("2. Run: bentoml serve datason_service:svc")
        print("3. Visit: http://localhost:3000 for the Swagger UI")
        print("4. Test the endpoints with the sample requests above")
    else:
        print("\n💡 To run the full demo:")
        print("pip install bentoml numpy")
        print("python bentoml_integration_guide.py")

    # Demonstrate client usage
    demonstrate_client_usage()

    # Create deployment configurations
    create_deployment_config()
