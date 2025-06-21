#!/usr/bin/env python3
"""
BentoML Integration with DataSON

A comprehensive example showing how to integrate DataSON with BentoML for
ML model serving, API creation, and production deployment.

Features:
- Modern DataSON API (dump_api, load_smart, dump_ml)
- BentoML service creation with DataSON serialization
- Input/output validation and transformation
- Model serving patterns
- Production deployment examples
"""

import time
from typing import Any, Dict, List

try:
    import bentoml
    from bentoml.io import JSON, NumpyNdarray, Text

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None

import numpy as np

import datason as ds
from datason.config import get_api_config

API_CONFIG = get_api_config()


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
    svc = bentoml.Service("datason_ml_service", runners=[])

    @svc.api(input=JSON(), output=JSON())
    def predict_json(input_data: dict) -> dict:
        """JSON prediction endpoint with DataSON processing."""
        try:
            # Use DataSON's smart loading for input validation and parsing
            parsed_input = ds.load_smart(input_data, config=API_CONFIG)

            # Extract features
            if "features" in parsed_input:
                features = parsed_input["features"]
                metadata = parsed_input.get("metadata", {})
            else:
                features = parsed_input
                metadata = {}

            # Make prediction
            result = model.predict_batch(features) if isinstance(features[0], list) else model.predict_single(features)

            # Enhance result with request metadata
            enhanced_result = {
                "prediction_result": result,
                "request_metadata": metadata,
                "processing_info": {"endpoint": "predict_json", "datason_processed": True, "timestamp": time.time()},
            }

            # Use DataSON's API serialization for clean output
            return ds.dump_api(enhanced_result)

        except Exception as e:
            error_response = {"error": str(e), "status": "prediction_failed", "timestamp": time.time()}
            return ds.dump_api(error_response)

    @svc.api(input=NumpyNdarray(), output=JSON())
    def predict_numpy(input_array: np.ndarray) -> dict:
        """NumPy array prediction endpoint with DataSON output."""
        try:
            # Convert numpy array to list for processing
            if input_array.ndim == 1:
                features = input_array.tolist()
                result = model.predict_single(features)
            else:
                features = input_array.tolist()
                result = model.predict_batch(features)

            # Enhanced response with numpy metadata
            response = {
                "prediction_result": result,
                "input_info": {"shape": input_array.shape, "dtype": str(input_array.dtype), "size": input_array.size},
                "processing_info": {"endpoint": "predict_numpy", "input_type": "numpy_array", "timestamp": time.time()},
            }

            return ds.dump_api(response)

        except Exception as e:
            return ds.dump_api({"error": str(e), "status": "numpy_prediction_failed"})

    @svc.api(input=Text(), output=JSON())
    def predict_text(input_text: str) -> dict:
        """Text-based prediction endpoint with DataSON parsing."""
        try:
            # Parse text input using DataSON (could be JSON string)
            try:
                parsed_data = ds.loads(input_text)
            except Exception:
                # If not JSON, treat as comma-separated values
                features = [float(x.strip()) for x in input_text.split(",")]
                parsed_data = {"features": features}

            # Process with smart loading
            processed_input = ds.load_smart(parsed_data, config=API_CONFIG)

            # Make prediction
            features = processed_input.get("features", processed_input)
            result = model.predict_single(features)

            response = {
                "prediction_result": result,
                "input_info": {
                    "original_text": input_text,
                    "parsed_features": features,
                    "input_length": len(input_text),
                },
                "processing_info": {
                    "endpoint": "predict_text",
                    "parsing_method": "datason_smart",
                    "timestamp": time.time(),
                },
            }

            return ds.dump_api(response)

        except Exception as e:
            return ds.dump_api({"error": str(e), "status": "text_prediction_failed", "input_text": input_text})

    @svc.api(input=JSON(), output=JSON())
    def model_info(input_data: dict) -> dict:
        """Model information endpoint with DataSON formatting."""
        try:
            # Get comprehensive model info
            info = model.get_model_info()

            # Add service information
            service_info = {
                "service_name": "datason_ml_service",
                "available_endpoints": ["predict_json", "predict_numpy", "predict_text", "model_info", "health_check"],
                "datason_features": ["smart_loading", "api_serialization", "type_handling", "error_recovery"],
                "created_at": time.time(),
            }

            response = {
                "model_info": info,
                "service_info": service_info,
                "request_metadata": ds.load_smart(input_data, config=API_CONFIG),
            }

            return ds.dump_api(response)

        except Exception as e:
            return ds.dump_api({"error": str(e), "status": "info_request_failed"})

    @svc.api(input=JSON(), output=JSON())
    def health_check(input_data: dict = None) -> dict:
        """Health check endpoint with DataSON diagnostics."""
        health_info = {
            "status": "healthy",
            "model_status": "ready",
            "model_info": model.get_model_info(),
            "service_metrics": {
                "uptime": time.time() - model.trained_at,
                "memory_usage": "normal",  # Could integrate actual metrics
                "last_prediction": "recent",
            },
            "datason_info": {
                "version": getattr(ds, "__version__", "unknown"),
                "api_config_loaded": API_CONFIG is not None,
                "features_enabled": ["smart_loading", "api_serialization", "ml_optimization"],
            },
            "timestamp": time.time(),
        }

        return ds.dump_api(health_info)


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
            parsed_back = ds.load_smart(serialized, config=API_CONFIG)

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
                parsed_request = ds.load_smart(serialized_request, config=API_CONFIG)
                features = parsed_request.get("features", [])

                # Mock prediction
                if isinstance(features[0], list):
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


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_bentoml_demo()

    print(f"\n🎯 Demo Results: {results}")

    if BENTOML_AVAILABLE:
        print("\n💡 Next Steps:")
        print("1. Save this file as 'bentoml_datason_service.py'")
        print("2. Run: bentoml serve bentoml_datason_service:svc")
        print("3. Visit: http://localhost:3000 for the Swagger UI")
        print("4. Test the endpoints with the sample requests above")
    else:
        print("\n💡 To run the full demo:")
        print("pip install bentoml numpy")
        print("python bentoml_integration_guide.py")
