#!/usr/bin/env python3
"""
Ray Serve Integration with DataSON

A comprehensive example showing how to integrate DataSON with Ray Serve for
ML model serving, request/response handling, and scalable deployments.

Features:
- Modern DataSON API (dump_api, load_smart)
- Request/response serialization
- Batch processing with Ray Serve
- Model serving patterns
- Health checks and monitoring
"""

import asyncio
import time
from typing import Any, Dict, List

try:
    import ray
    from ray import serve

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    serve = None
    ray = None

import numpy as np

import datason as ds
from datason.config import get_api_config

API_CONFIG = get_api_config()


# Mock ML model for demonstration
class MockMLModel:
    """Mock ML model for demonstration purposes."""

    def __init__(self, model_type: str = "classifier"):
        self.model_type = model_type
        self.version = "1.0.0"
        self.loaded_at = time.time()

    def predict(self, data: Any) -> Dict[str, Any]:
        """Make predictions on input data."""
        # Simulate processing time
        time.sleep(0.01)

        if isinstance(data, (list, np.ndarray)):
            # Batch prediction
            predictions = [0.8 + 0.2 * np.random.random() for _ in range(len(data))]
            return {"predictions": predictions, "model_version": self.version, "batch_size": len(data)}
        else:
            # Single prediction
            return {"prediction": 0.8 + 0.2 * np.random.random(), "model_version": self.version, "confidence": 0.95}

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {"model_type": self.model_type, "version": self.version, "loaded_at": self.loaded_at, "status": "ready"}


if RAY_AVAILABLE:

    @serve.deployment(num_replicas=2, max_ongoing_requests=10)
    class DataSONModelServing:
        """Ray Serve deployment with DataSON integration."""

        def __init__(self, model_type: str = "classifier"):
            """Initialize the model serving deployment."""
            self.model = MockMLModel(model_type)
            print(f"🚀 DataSON Model Serving initialized: {model_type}")

        async def __call__(self, request) -> Dict[str, Any]:
            """Handle incoming requests with DataSON serialization."""
            try:
                # Parse request - already returns parsed dict
                raw_data = await request.json()

                # Since raw_data is already parsed, we don't need load_smart for parsing
                # We can use it for enhanced processing if we need type detection
                if isinstance(raw_data, dict) and "data" in raw_data:
                    input_data = raw_data["data"]
                    metadata = raw_data.get("metadata", {})
                else:
                    input_data = raw_data
                    metadata = {}

                # Make prediction
                prediction_result = self.model.predict(input_data)

                # Enhance result with request metadata
                response = {
                    "prediction_result": prediction_result,
                    "request_metadata": metadata,
                    "processing_info": {
                        "timestamp": time.time(),
                        "model_info": self.model.get_info(),
                        "datason_processing": True,
                    },
                }

                # Serialize response using DataSON's API mode for clean JSON
                serialized_response = ds.dump_api(response)

                return serialized_response

            except Exception as e:
                # Error handling with DataSON
                error_response = {"error": str(e), "status": "failed", "timestamp": time.time()}
                return ds.dump_api(error_response)

        async def health_check(self) -> Dict[str, Any]:
            """Health check endpoint."""
            health_info = {
                "status": "healthy",
                "model_status": self.model.get_info(),
                "timestamp": time.time(),
                "datason_version": getattr(ds, "__version__", "unknown"),
            }
            return ds.dump_api(health_info)

        async def batch_predict(self, request) -> Dict[str, Any]:
            """Handle batch predictions."""
            try:
                raw_data = await request.json()

                # Extract batch data directly since raw_data is already parsed
                batch_data = raw_data.get("batch_data", [])
                batch_metadata = raw_data.get("metadata", {})

                if not batch_data:
                    return ds.dump_api({"error": "No batch data provided"})

                # Process batch
                batch_results = []
                for i, item in enumerate(batch_data):
                    result = self.model.predict(item)
                    result["batch_index"] = i
                    batch_results.append(result)

                response = {
                    "batch_results": batch_results,
                    "batch_size": len(batch_data),
                    "batch_metadata": batch_metadata,
                    "processing_info": {"timestamp": time.time(), "model_info": self.model.get_info()},
                }

                return ds.dump_api(response)

            except Exception as e:
                error_response = {"error": str(e), "status": "batch_failed", "timestamp": time.time()}
                return ds.dump_api(error_response)

    @serve.deployment
    class DataSONProxyService:
        """Proxy service demonstrating DataSON data transformation."""

        def __init__(self):
            self.request_count = 0

        async def __call__(self, request) -> Dict[str, Any]:
            """Transform requests using DataSON before forwarding."""
            self.request_count += 1

            try:
                # Parse incoming request - already returns parsed dict
                raw_data = await request.json()

                # For demonstration, convert to JSON string first if we want to use load_smart
                json_string = ds.dumps_json(raw_data)
                transformed_data = ds.load_smart(json_string, config=API_CONFIG)

                # Add proxy metadata
                proxy_enhanced = {
                    "original_data": transformed_data,
                    "proxy_info": {
                        "request_id": self.request_count,
                        "processed_at": time.time(),
                        "transformation": "datason_smart_load",
                    },
                }

                # Return transformed data
                return ds.dump_api(proxy_enhanced)

            except Exception as e:
                return ds.dump_api({"error": str(e), "proxy_status": "transformation_failed"})


class RayServeDataSONDemo:
    """Demonstration of Ray Serve + DataSON integration."""

    def __init__(self):
        self.deployments = {}

    async def start_services(self):
        """Start Ray Serve services with DataSON integration."""
        if not RAY_AVAILABLE:
            print("❌ Ray is not available. Install with: pip install ray[serve]")
            return

        try:
            # Initialize Ray if not already started
            if not ray.is_initialized():
                ray.init()

            # Start Serve
            serve.start()

            # Deploy model serving
            model_deployment = DataSONModelServing.bind("advanced_classifier")
            proxy_deployment = DataSONProxyService.bind()

            # Deploy services
            serve.run(model_deployment, name="datason_model", route_prefix="/predict")
            serve.run(proxy_deployment, name="datason_proxy", route_prefix="/transform")

            print("✅ Ray Serve services started with DataSON integration!")
            print("📊 Available endpoints:")
            print("  - POST /predict - Model predictions with DataSON")
            print("  - POST /transform - Data transformation proxy")

            return {"model_service": "datason_model", "proxy_service": "datason_proxy"}

        except Exception as e:
            print(f"❌ Failed to start services: {e}")
            return None

    def create_test_requests(self) -> List[Dict[str, Any]]:
        """Create test requests demonstrating DataSON features."""

        # Single prediction request
        single_request = {
            "data": [1.0, 2.0, 3.0, 4.0],
            "metadata": {"request_type": "single_prediction", "timestamp": time.time(), "client_id": "demo_client"},
        }

        # Batch prediction request
        batch_request = {
            "batch_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "metadata": {"request_type": "batch_prediction", "batch_id": "batch_001", "priority": "high"},
        }

        # Complex nested request
        complex_request = {
            "data": {
                "features": np.array([1, 2, 3, 4, 5]).tolist(),
                "categorical": {"region": "us-west", "model_type": "production"},
                "numerical": {"confidence_threshold": 0.8, "max_results": 10},
            },
            "metadata": {
                "request_type": "complex_prediction",
                "preprocessing": ["normalize", "feature_select"],
                "postprocessing": ["calibrate", "rank"],
            },
        }

        return [single_request, batch_request, complex_request]


async def run_ray_serve_demo():
    """Run a comprehensive Ray Serve + DataSON demonstration."""
    print("🚀 Ray Serve + DataSON Integration Demo")
    print("=" * 50)

    if not RAY_AVAILABLE:
        print("❌ Ray Serve is not available.")
        print("💡 Install with: pip install ray[serve]")
        print("\nFallback: Showing DataSON serialization examples...")

        # Show DataSON features without Ray
        demo = RayServeDataSONDemo()
        test_requests = demo.create_test_requests()

        print("\n📋 Example Request Formats:")
        for i, request in enumerate(test_requests, 1):
            print(f"\n{i}️⃣ Request Type: {request['metadata']['request_type']}")

            # Serialize with DataSON API mode
            serialized = ds.dump_api(request)
            print("📤 Serialized Request:")
            serialized_str = ds.dumps_json(serialized, indent=2)
            if len(serialized_str) > 200:
                print(serialized_str[:200] + "...")
            else:
                print(serialized_str)

            # Parse back with smart loading (convert to JSON string first)
            json_string = ds.dumps_json(serialized)
            parsed = ds.load_smart(json_string, config=API_CONFIG)
            print("📥 Parsed Back Successfully:", parsed["metadata"]["request_type"])

        return {"status": "demo_completed_without_ray"}

    # Full Ray Serve demo
    demo = RayServeDataSONDemo()

    print("\n1️⃣ Starting Ray Serve services...")
    services = await demo.start_services()

    if services:
        print("\n2️⃣ Creating test requests...")
        test_requests = demo.create_test_requests()

        print("\n3️⃣ Services ready for testing!")
        print("🧪 Test with curl:")

        for i, request in enumerate(test_requests, 1):
            serialized_request = ds.dumps_json(ds.dump_api(request), indent=2)
            print(f"\nTest {i}: {request['metadata']['request_type']}")
            print("curl -X POST http://localhost:8000/predict \\")
            print("  -H 'Content-Type: application/json' \\")
            print(f"  -d '{serialized_request[:100]}...'")

        print("\n✅ Demo setup completed!")
        print("📊 Key DataSON features in Ray Serve:")
        print("- 🌐 dump_api() for clean JSON responses")
        print("- 🧠 load_smart() for intelligent request parsing")
        print("- 🔄 Automatic type handling for ML data")
        print("- 📦 Complex nested data structure support")
        print("- ⚡ High-performance serialization")

        return services

    return {"status": "failed_to_start"}


def test_ray_serve_integration():
    """Test Ray Serve + DataSON integration without starting heavy cluster."""
    print("🧪 Testing Ray Serve + DataSON Integration (Lightweight)")
    print("=" * 60)

    if not RAY_AVAILABLE:
        print("❌ Ray is not available. Install with: pip install ray[serve]")
        return {"status": "ray_not_available"}

    # Test the DataSON integration patterns without starting cluster
    print("✅ Ray available, testing DataSON patterns...")

    # 1. Test mock model
    print("\n1️⃣ Testing MockMLModel...")
    model = MockMLModel("test_classifier")

    # Test single prediction
    test_data = [1.0, 2.0, 3.0, 4.0]
    result = model.predict(test_data)
    print(f"   Single prediction: {result}")

    # 2. Test DataSON serialization patterns
    print("\n2️⃣ Testing DataSON serialization patterns...")

    # Create request data
    request_data = {"data": test_data, "metadata": {"client_id": "test", "timestamp": time.time()}}

    # Test smart loading (convert dict to JSON string first for proper load_smart usage)
    json_string = ds.dumps_json(request_data)
    smart_loaded = ds.load_smart(json_string)
    print(f"   Smart loaded: {type(smart_loaded)}")

    # Test API serialization
    api_result = ds.dump_api(result)
    print(f"   API serialized: {type(api_result)}")

    # 3. Test the deployment class logic (without deploying)
    print("\n3️⃣ Testing deployment logic...")

    # Create a test version without the decorator for testing
    class TestDataSONModelServing:
        """Test version of the model serving class without Ray decorators."""

        def __init__(self, model_type: str = "classifier"):
            self.model = MockMLModel(model_type)

        def get_health_check_data(self):
            return {
                "status": "healthy",
                "model_status": self.model.get_info(),
                "timestamp": time.time(),
                "datason_version": getattr(ds, "__version__", "unknown"),
            }

    if RAY_AVAILABLE:
        # Create test instance (not deployed)
        serving_instance = TestDataSONModelServing("test_model")
        model_info = serving_instance.model.get_info()
        print(f"   Model info: {model_info}")

        # Test health check logic
        health_data = serving_instance.get_health_check_data()
        health_result = ds.dump_api(health_data)
        print(f"   Health check: {health_result['status']}")

    # 4. Show example requests
    print("\n4️⃣ Example usage patterns...")
    demo = RayServeDataSONDemo()
    test_requests = demo.create_test_requests()

    for i, req in enumerate(test_requests[:2]):  # Just show first 2
        print(f"   Request {i + 1}: {len(str(req))} chars")
        # Process with DataSON
        processed = ds.dump_api(req)
        print(f"   Processed: {len(str(processed))} chars")

    print("\n✅ Integration test completed successfully!")
    print("\n💡 To run full Ray Serve cluster:")
    print("   python ray_serve_integration_guide.py --full")
    print("   (Warning: Requires significant system resources)")

    return {
        "status": "test_completed",
        "ray_available": True,
        "datason_integration": "working",
        "deployment_classes": "tested",
    }


if __name__ == "__main__":
    import sys

    # Check for full deployment flag
    if "--full" in sys.argv:
        print("🚀 Starting full Ray Serve demonstration...")
        print("⚠️  This will consume significant system resources!")

        if RAY_AVAILABLE:
            # Only run heavy demo if explicitly requested
            result = asyncio.run(run_ray_serve_demo())
            print(f"Demo result: {result}")
        else:
            print("❌ Ray not available for full demo")
    else:
        # Run lightweight test by default
        result = test_ray_serve_integration()
        print(f"\n🎯 Test Result: {result}")

        print("\n🔧 Key DataSON Features Demonstrated:")
        print("- 🤖 dump_api() for clean JSON responses")
        print("- 🧠 load_smart() for intelligent request parsing")
        print("- 📊 Enhanced error handling and metadata")
        print("- 🔄 Async request/response processing")
        print("- 🎯 ML model integration patterns")
        print("- 📈 Health monitoring and status checks")
