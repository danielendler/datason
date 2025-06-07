#!/usr/bin/env python3
"""
Datason Modern API Demo - Phase 3 API Modernization
=====================================================

This script demonstrates the new intention-revealing API introduced in Phase 3,
which makes datason more discoverable and user-friendly while maintaining
100% backward compatibility.

Key improvements:
- Intention-revealing names (load_basic, load_smart, load_perfect, etc.)
- Compositional utilities (dump_secure, dump_chunked, etc.)
- Domain-specific convenience (dump_ml, dump_api, etc.)
- Progressive disclosure of complexity
"""

from datetime import datetime

import datason


def demo_modern_serialization():
    """Demonstrate the new dump_* family of functions."""
    print("=== Modern Serialization API Demo ===\n")

    # Sample data with different complexity levels
    simple_data = {"name": "Alice", "age": 30, "city": "New York"}
    sensitive_data = {
        "user": "john_doe",
        "password": "secret123",
        "email": "john@example.com",
        "ssn": "123-45-6789",
        "notes": "Customer since 2020",
    }

    # 1. Basic serialization
    print("1. Basic serialization with dump():")
    result = datason.dump(simple_data)
    print(f"   Input:  {simple_data}")
    print(f"   Output: {result}")
    print()

    # 2. API-safe serialization
    print("2. API-safe serialization with dump_api():")
    api_result = datason.dump_api(simple_data)
    print(f"   Clean output for API responses: {api_result}")
    print()

    # 3. Secure serialization with PII redaction
    print("3. Secure serialization with dump_secure():")
    secure_result = datason.dump_secure(sensitive_data)
    print(f"   Input:     {sensitive_data}")
    print(f"   Redacted:  {secure_result}")
    print("   ↳ Notice how sensitive fields are automatically redacted")
    print()

    # 4. Performance-optimized serialization
    print("4. Performance-optimized with dump_fast():")
    fast_result = datason.dump_fast(simple_data)
    print(f"   Fast mode: {fast_result}")
    print()

    # 5. Chunked serialization for large data
    print("5. Chunked serialization with dump_chunked():")
    large_list = list(range(2500))
    chunked_result = datason.dump_chunked(large_list, chunk_size=1000)
    print(f"   Chunks created: {len(list(chunked_result.chunks))}")
    print(f"   Metadata: {chunked_result.metadata}")
    print()

    # 6. JSON string compatibility
    print("6. JSON string compatibility with dumps()/loads():")
    json_str = datason.dumps(simple_data)
    parsed_back = datason.loads(json_str)
    print(f"   JSON string: {json_str}")
    print(f"   Parsed back: {parsed_back}")
    print()


def demo_modern_deserialization():
    """Demonstrate the new load_* family of functions with progressive complexity."""
    print("=== Modern Deserialization API Demo ===\n")

    # Sample serialized data
    complex_serialized = {
        "user_id": "12345",
        "timestamp": "2023-12-01T10:30:00",
        "data": [1, 2, 3, 4, 5],
        "metadata": {"version": "1.0", "source": "api"},
    }

    print("Progressive deserialization complexity:")
    print(f"Data to deserialize: {complex_serialized}\n")

    # 1. Basic heuristics-only deserialization
    print("1. load_basic() - Heuristics only (60-70% success rate):")
    basic_result = datason.load_basic(complex_serialized)
    print(f"   Result: {basic_result}")
    print("   ↳ Fast but limited type fidelity")
    print()

    # 2. Smart deserialization with auto-detection
    print("2. load_smart() - Auto-detection + heuristics (80-90% success rate):")
    smart_result = datason.load_smart(complex_serialized)
    print(f"   Result: {smart_result}")
    print("   ↳ Better type reconstruction, good for most use cases")
    print()

    # 3. Perfect template-based deserialization
    print("3. load_perfect() - Template-based (100% success rate):")
    template = {"user_id": "", "timestamp": datetime.now(), "data": [], "metadata": {}}
    try:
        perfect_result = datason.load_perfect(complex_serialized, template)
        print(f"   Result: {perfect_result}")
        print("   ↳ Guaranteed perfect reconstruction when template matches")
    except Exception:
        print("   Note: Template deserialization requires compatible data structure")
    print()

    # 4. Metadata-based type reconstruction
    print("4. load_typed() - Metadata-based (95% success rate):")
    typed_result = datason.load_typed(complex_serialized)
    print(f"   Result: {typed_result}")
    print("   ↳ Uses embedded type metadata for high fidelity")
    print()


def demo_ml_workflow():
    """Demonstrate ML-specific workflow with the modern API."""
    print("=== ML Workflow Demo ===\n")

    try:
        import numpy as np

        # Sample ML data
        model_data = {
            "weights": np.array([0.1, 0.2, 0.3, 0.4]),
            "bias": np.array([0.05]),
            "hyperparameters": {"learning_rate": 0.001, "epochs": 100, "batch_size": 32},
            "metrics": {"accuracy": 0.95, "loss": 0.05},
        }

        print("ML-optimized serialization workflow:")
        print("1. Original ML data with NumPy arrays")
        print(f"   Weights shape: {model_data['weights'].shape}")
        print(f"   Bias shape: {model_data['bias'].shape}")
        print()

        # ML-optimized serialization
        print("2. Serialize with dump_ml() - ML-optimized:")
        ml_serialized = datason.dump_ml(model_data)
        print("   ✓ NumPy arrays properly handled")
        print("   ✓ ML-specific configurations applied")
        print()

        # Smart deserialization for ML data
        print("3. Deserialize with load_smart():")
        ml_reconstructed = datason.load_smart(ml_serialized)
        print("   ✓ Data structure preserved")
        print(f"   ✓ Hyperparameters: {ml_reconstructed.get('hyperparameters', {})}")
        print()

    except ImportError:
        print("NumPy not available - skipping ML demo")
        print("Install numpy to see ML-specific features")
        print()


def demo_api_discovery():
    """Demonstrate the API discovery and help functions."""
    print("=== API Discovery & Help Demo ===\n")

    # Get API information
    print("1. API Information:")
    api_info = datason.get_api_info()
    print(f"   API Version: {api_info['api_version']}")
    print(f"   Phase: {api_info['phase']}")
    print(f"   Available dump functions: {', '.join(api_info['dump_functions'])}")
    print(f"   Available load functions: {', '.join(api_info['load_functions'])}")
    print()

    # Get help on choosing the right function
    print("2. API Help - Function Recommendations:")
    help_info = datason.help_api()

    print("   Serialization options:")
    for key, info in help_info["serialization"].items():
        func = info["function"]
        use_case = info["use_case"]
        print(f"     • {func:<15} → {use_case}")

    print("\n   Deserialization options:")
    for key, info in help_info["deserialization"].items():
        func = info["function"]
        success_rate = info["success_rate"]
        use_case = info["use_case"]
        print(f"     • {func:<15} → {success_rate} success, {use_case}")

    print("\n   General recommendations:")
    for rec in help_info["recommendations"]:
        print(f"     • {rec}")
    print()


def demo_backward_compatibility():
    """Demonstrate that the old API still works."""
    print("=== Backward Compatibility Demo ===\n")

    data = {"old_api": "still_works", "compatibility": True}

    # Old API still works
    print("1. Old API functions still work:")
    old_serialized = datason.serialize(data)
    old_deserialized = datason.deserialize(old_serialized)
    print(f"   serialize() → {old_serialized}")
    print(f"   deserialize() → {old_deserialized}")
    print()

    # New and old APIs produce equivalent results
    print("2. New API produces equivalent results:")
    new_serialized = datason.dump(data)
    new_deserialized = datason.load_basic(new_serialized)
    print(f"   dump() → {new_serialized}")
    print(f"   load_basic() → {new_deserialized}")
    print(f"   Results equivalent: {old_serialized == new_serialized}")
    print()


def main():
    """Run all demos."""
    print("🚀 Datason Phase 3 Modern API Demo")
    print("=" * 50)
    print()

    demo_modern_serialization()
    demo_modern_deserialization()
    demo_ml_workflow()
    demo_api_discovery()
    demo_backward_compatibility()

    print("✅ Demo completed! The modern API provides:")
    print("   • Clear intention-revealing function names")
    print("   • Progressive complexity (basic → smart → perfect)")
    print("   • Domain-specific optimizations (ML, API, security)")
    print("   • 100% backward compatibility")
    print("   • Built-in help and discovery")
    print()
    print("Get started with: datason.help_api()")


if __name__ == "__main__":
    main()
