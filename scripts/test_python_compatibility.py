#!/usr/bin/env python3
"""Test Python version compatibility for datason.

This script tests basic functionality across different Python versions
to ensure our Python 3.8+ support claim is accurate.
"""

import sys
from typing import Any, Dict


def test_basic_functionality() -> Dict[str, Any]:
    """Test basic datason functionality.

    Returns:
        Dict with test results
    """
    results = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "tests": {},
        "errors": [],
    }

    try:
        # Test import
        import datason

        results["tests"]["import"] = "✅ PASS"

        # Test basic serialization
        test_data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        serialized = datason.serialize(test_data)
        results["tests"]["basic_serialization"] = "✅ PASS"

        # Test deserialization
        deserialized = datason.deserialize(serialized)
        assert deserialized == test_data
        results["tests"]["deserialization"] = "✅ PASS"

        # Test datetime handling
        from datetime import datetime, timezone

        dt_data = {"timestamp": datetime.now(timezone.utc)}
        dt_serialized = datason.serialize(dt_data)
        assert "timestamp" in dt_serialized
        results["tests"]["datetime_serialization"] = "✅ PASS"

        # Test configuration (if available)
        try:
            config = datason.get_ml_config()
            configured_result = datason.serialize(test_data, config=config)
            assert configured_result is not None
            results["tests"]["configuration_system"] = "✅ PASS"
        except AttributeError:
            results["tests"]["configuration_system"] = "⚠️ SKIP (not available)"

        # Test type handlers (if available)
        try:
            import uuid
            from decimal import Decimal

            complex_data = {
                "uuid": uuid.uuid4(),
                "decimal": Decimal("10.5"),
                "complex": 1 + 2j,
                "set": {1, 2, 3},
                "bytes": b"hello",
            }
            complex_serialized = datason.serialize(complex_data)
            assert complex_serialized is not None
            results["tests"]["advanced_types"] = "✅ PASS"
        except Exception as e:
            results["tests"]["advanced_types"] = f"❌ FAIL: {e}"
            results["errors"].append(f"advanced_types: {e}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        results["errors"].append(error_msg)
        results["tests"]["import"] = f"❌ FAIL: {error_msg}"

    return results


def test_optional_dependencies() -> Dict[str, str]:
    """Test optional dependency compatibility.

    Returns:
        Dict mapping dependency names to status
    """
    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "tensorflow": "tensorflow",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "Pillow": "PIL",
        "transformers": "transformers",
        "jax": "jax",
    }

    results = {}
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            results[name] = "✅ Available"
        except ImportError:
            results[name] = "❌ Not available"
        except Exception as e:
            results[name] = f"⚠️ Error: {e}"

    return results


def main() -> None:
    """Run compatibility tests and print results."""
    print("🐍 Python Version Compatibility Test")
    print(f"Python {sys.version}")
    print("=" * 60)

    # Test basic functionality
    print("\n📋 Testing Basic Functionality:")
    basic_results = test_basic_functionality()

    print(f"Python Version: {basic_results['python_version']}")
    for test_name, result in basic_results["tests"].items():
        print(f"  {test_name.replace('_', ' ').title()}: {result}")

    if basic_results["errors"]:
        print("\n❌ Errors encountered:")
        for error in basic_results["errors"]:
            print(f"  • {error}")

    # Test optional dependencies
    print("\n📦 Testing Optional Dependencies:")
    dep_results = test_optional_dependencies()
    for dep_name, status in dep_results.items():
        print(f"  {dep_name}: {status}")

    # Summary
    passed_tests = sum(1 for result in basic_results["tests"].values() if result.startswith("✅"))
    total_tests = len(basic_results["tests"])

    print("\n🏆 Summary:")
    print(f"  Core Tests: {passed_tests}/{total_tests} passed")

    available_deps = sum(1 for status in dep_results.values() if status.startswith("✅"))
    total_deps = len(dep_results)
    print(f"  Optional Dependencies: {available_deps}/{total_deps} available")

    if basic_results["errors"]:
        print(f"  ❌ {len(basic_results['errors'])} errors detected")
        sys.exit(1)
    else:
        print("  ✅ All core functionality working!")


if __name__ == "__main__":
    main()
