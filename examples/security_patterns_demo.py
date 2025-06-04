#!/usr/bin/env python3
"""
Demo: Security Patterns in Utils Module (v0.5.5)

This demonstrates how the utils module now leverages the same security
patterns learned from core.py, providing configurable limits and protection
against resource exhaustion attacks.
"""

from typing import Any, Dict

from datason.utils import (
    UtilityConfig,
    UtilitySecurityError,
    deep_compare,
    enhance_data_types,
    find_data_anomalies,
)


def main() -> None:
    """Demonstrate security patterns consistency."""
    print("Security Patterns Consistency Demo")
    print("=" * 40)

    # 1. Show default security limits from core.py
    print("\n1. Default Security Configuration:")
    default_config = UtilityConfig()
    print(f"   Max Depth: {default_config.max_depth}")
    print(f"   Max Object Size: {default_config.max_object_size:,}")
    print(f"   Max String Length: {default_config.max_string_length:,}")
    print(f"   Circular Reference Detection: {default_config.enable_circular_reference_detection}")

    # 2. Custom configuration for stricter environments
    print("\n2. Custom Configuration (Production Safe):")
    strict_config = UtilityConfig(
        max_depth=20,  # Lower depth limit
        max_object_size=50_000,  # Smaller object limit
        max_string_length=10_000,  # Shorter string limit
        enable_circular_reference_detection=True,
    )
    print(f"   Max Depth: {strict_config.max_depth}")
    print(f"   Max Object Size: {strict_config.max_object_size:,}")
    print(f"   Max String Length: {strict_config.max_string_length:,}")

    # 3. Test security enforcement
    print("\n3. Security Enforcement Examples:")

    # Depth limit test
    print("\n   a) Depth Limit Protection:")
    deep_nested: Dict[str, Any] = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": "deep"}}}}}}}}}}}
    try:
        deep_compare(deep_nested, deep_nested, config=UtilityConfig(max_depth=5))
        print("      ✗ Expected security violation but none occurred")
    except UtilitySecurityError as e:
        print(f"      ✓ Depth limit enforced: {str(e)[:80]}...")

    # Object size limit test
    print("\n   b) Object Size Protection:")
    large_obj = {f"field_{i}": f"value_{i}" for i in range(100)}
    anomalies = find_data_anomalies(large_obj, config=UtilityConfig(max_object_size=50))
    if anomalies["security_violations"]:
        print(f"      ✓ Object size violation detected: {anomalies['security_violations'][0]['violation']}")
    else:
        print("      ✗ Expected object size violation")

    # String length limit test
    print("\n   c) String Length Protection:")
    long_string_data = {"content": "x" * 1000}
    anomalies = find_data_anomalies(long_string_data, config=UtilityConfig(max_string_length=500))
    if anomalies["security_violations"]:
        print(f"      ✓ String length violation detected: {anomalies['security_violations'][0]['violation']}")
    else:
        print("      ✗ Expected string length violation")

    # Circular reference test
    print("\n   d) Circular Reference Protection:")
    circular_data: Dict[str, Any] = {"name": "test"}
    circular_data["self"] = circular_data
    try:
        enhance_data_types(circular_data, config=UtilityConfig(enable_circular_reference_detection=True))
        print("      ✗ Expected circular reference detection")
    except UtilitySecurityError as e:
        print(f"      ✓ Circular reference detected: {str(e)[:80]}...")

    # 4. Configurable security for different environments
    print("\n4. Environment-Specific Configurations:")

    # Development environment (more permissive)
    dev_config = UtilityConfig(
        max_depth=100,
        max_object_size=1_000_000,
        max_string_length=10_000_000,
        enable_circular_reference_detection=False,  # For performance
    )
    print(f"   Development: depth={dev_config.max_depth}, objects={dev_config.max_object_size:,}")

    # Production environment (strict)
    prod_config = UtilityConfig(
        max_depth=25, max_object_size=100_000, max_string_length=1_000_000, enable_circular_reference_detection=True
    )
    print(f"   Production: depth={prod_config.max_depth}, objects={prod_config.max_object_size:,}")

    # Public API environment (very strict)
    api_config = UtilityConfig(
        max_depth=10, max_object_size=10_000, max_string_length=100_000, enable_circular_reference_detection=True
    )
    print(f"   Public API: depth={api_config.max_depth}, objects={api_config.max_object_size:,}")

    print("\n" + "=" * 40)
    print("✓ Utils module now shares security patterns with core.py!")
    print("✓ Configurable limits allow adaptation to different environments")
    print("✓ Consistent protection against resource exhaustion attacks")


if __name__ == "__main__":
    main()
