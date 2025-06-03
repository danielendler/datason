#!/usr/bin/env python3
"""
Example: Enhanced Data Utilities with Security Features (v0.5.5)

This example demonstrates the security-enhanced utilities module that applies
the same security patterns learned from core.py. Users can configure limits
and the utilities provide protection against resource exhaustion attacks.
"""

from typing import Any, Dict

import pandas as pd

from datason.utils import (
    UtilityConfig,
    UtilitySecurityError,
    deep_compare,
    enhance_data_types,
    enhance_pandas_dataframe,
    find_data_anomalies,
    get_default_utility_config,
    normalize_data_structure,
)


def demonstrate_security_features() -> None:
    """Show how security limits protect against various attacks."""
    print("=== Security Features Demo ===")

    # Custom security configuration
    strict_config = UtilityConfig(
        max_depth=10, max_object_size=100, max_string_length=1000, enable_circular_reference_detection=True
    )

    print(f"Strict config limits: depth={strict_config.max_depth}, object_size={strict_config.max_object_size}")

    # Test 1: Depth limit protection
    print("\n1. Testing depth limit protection...")
    deep_nested: Dict[str, Any] = {"level": 1}
    current: Dict[str, Any] = deep_nested
    for i in range(2, 15):  # Create 14 levels of nesting
        current["nested"] = {"level": i}
        current = current["nested"]

    try:
        result = deep_compare(deep_nested, deep_nested, config=strict_config)
        if result["security_violations"]:
            print(f"✓ Security violation detected: {result['security_violations'][0]['violation']}")
        else:
            print("✗ Expected security violation but none detected")
    except UtilitySecurityError as e:
        print(f"✓ Security error caught: {e}")

    # Test 2: Object size limit protection
    print("\n2. Testing object size limit protection...")
    large_dict = {f"key_{i}": f"value_{i}" for i in range(200)}

    anomalies = find_data_anomalies(large_dict, config=strict_config)
    if anomalies["security_violations"]:
        print(f"✓ Security violation logged: {anomalies['security_violations'][0]['violation']}")
    else:
        print("✗ Expected security violation but none logged")

    # Test 3: Circular reference protection
    print("\n3. Testing circular reference protection...")
    circular_dict: Dict[str, Any] = {"name": "parent"}
    circular_dict["self_ref"] = circular_dict

    try:
        enhance_data_types(circular_dict, config=strict_config)
        print("✗ Expected circular reference error")
    except UtilitySecurityError as e:
        print(f"✓ Circular reference protection worked: {e}")


def demonstrate_configurable_utilities() -> None:
    """Show how utilities can be configured for different use cases."""
    print("\n=== Configurable Utilities Demo ===")

    # Default configuration for general use
    default_config = get_default_utility_config()
    print(f"Default limits: depth={default_config.max_depth}, string_length={default_config.max_string_length}")

    # Lenient configuration for trusted data
    lenient_config = UtilityConfig(
        max_depth=100,
        max_object_size=1_000_000,
        max_string_length=10_000_000,
        enable_circular_reference_detection=False,  # For performance
    )

    # Sample data with various issues
    messy_data = {
        "user": {
            "name": "  John Doe  ",  # Whitespace
            "age": "25",  # String number
            "salary": "50000.50",  # String float
            "active": "true",  # String boolean
            "joined": "2023-01-15",  # String date
            "tags": ["developer", "python", "ai"] * 10,  # Large list
        },
        "metadata": {
            "version": "1.0",
            "large_text": "x" * 15000,  # Large string
            "config": {"debug": "false", "timeout": "30"},
        },
    }

    print("\n1. Finding anomalies with custom rules...")
    custom_anomaly_rules = {
        "max_string_length": 5000,  # Lower than security limit
        "max_list_length": 20,  # Detect large lists
        "suspicious_patterns": [r"password", r"secret", r"api_key"],
        "detect_suspicious_patterns": True,
    }

    anomalies = find_data_anomalies(messy_data, rules=custom_anomaly_rules, config=lenient_config)

    print(f"Large strings found: {len(anomalies['large_strings'])}")
    print(f"Large collections found: {len(anomalies['large_collections'])}")
    print(f"Security violations: {len(anomalies['security_violations'])}")

    print("\n2. Enhancing data types...")
    enhanced_data, report = enhance_data_types(messy_data, config=lenient_config)

    print(f"Type conversions: {len(report['type_conversions'])}")
    print(f"Cleaned values: {len(report['cleaned_values'])}")
    print(f"Enhanced user age: {enhanced_data['user']['age']} (type: {type(enhanced_data['user']['age'])})")
    print(f"Enhanced user active: {enhanced_data['user']['active']} (type: {type(enhanced_data['user']['active'])})")

    print("\n3. Normalizing data structure...")
    flattened = normalize_data_structure(messy_data, target_structure="flat", config=lenient_config)
    print(f"Flattened keys (first 5): {list(flattened.keys())[:5]}")


def demonstrate_pandas_integration() -> None:
    """Show enhanced pandas/numpy integration with security."""
    print("\n=== Pandas Integration Demo ===")

    try:
        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "score": ["85.5", "92.0", "invalid", "88.5"],
                "active": ["true", "false", "true", "1"],
                "category": ["A", "B", "A", "C"],
            }
        )

        print("Original DataFrame:")
        print(df.dtypes)

        # Enhance with security limits
        security_config = UtilityConfig(max_depth=5, max_object_size=1000)
        enhanced_df, report = enhance_pandas_dataframe(df, config=security_config)

        print("\nEnhanced DataFrame:")
        print(enhanced_df.dtypes)
        print(f"\nType conversions made: {len(report['type_conversions'])}")
        for conversion in report["type_conversions"]:
            print(f"  {conversion['column']}: {conversion['from']} -> {conversion['to']}")

    except ImportError:
        print("Pandas not available - skipping pandas integration demo")
    except Exception as e:
        print(f"Pandas demo error: {e}")


def main() -> None:
    """Run all utility demonstrations."""
    print("DataSon Enhanced Utilities Demo (v0.5.5)")
    print("=" * 50)

    demonstrate_security_features()
    demonstrate_configurable_utilities()
    demonstrate_pandas_integration()

    print("\n" + "=" * 50)
    print("Demo complete! The utils module now leverages the same")
    print("security patterns as core.py with configurable limits.")


if __name__ == "__main__":
    main()
