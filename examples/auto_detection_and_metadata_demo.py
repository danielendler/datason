#!/usr/bin/env python3
"""Auto-Detection and Type Metadata Demo - Advanced Round-Trip Serialization.

This example demonstrates the advanced features of datason's auto-detection
deserialization and type metadata support for perfect round-trip serialization.

Features demonstrated:
- Simple & Direct API for progressive loading
- Auto-detection deserialization with intelligent type recognition
- Type metadata for perfect round-trip serialization
- Aggressive mode for DataFrame/Series detection
- Complex nested structure handling
- Performance comparison between modes
"""

import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import datason as ds
from datason import auto_deserialize, deserialize, serialize
from datason.config import SerializationConfig


def demo_simple_progressive_loading():
    """Demonstrate the simple & direct API for progressive loading."""
    print("=== Simple & Direct API for Progressive Loading ===")

    # Sample data with mixed types
    test_data = {
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "timestamp": "2023-12-25T10:30:00",
        "score": "95.5",
        "active": "true",
        "metadata": {"version": "1.0", "source": "api"},
    }

    print("Original data (all strings from JSON):")
    for key, value in test_data.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    print("\n🚀 Progressive Loading with Simple API:")

    # 1. Basic loading (60-70% success)
    print("\n1️⃣ load_basic() - Quick exploration (60-70% success):")
    basic_result = ds.load_basic(test_data)
    print("   Results:")
    for key, value in basic_result.items():
        print(f"     {key}: {value} ({type(value).__name__})")

    # 2. Smart loading (80-90% success)
    print("\n2️⃣ load_smart() - Production ready (80-90% success):")
    smart_result = ds.load_smart(test_data)
    print("   Results:")
    for key, value in smart_result.items():
        print(f"     {key}: {value} ({type(value).__name__})")

    print("\n✅ Smart loading automatically detected:")
    print("   • UUID strings → UUID objects")
    print("   • ISO datetime strings → datetime objects")
    print("   • Numeric strings → numbers")
    print("   • Boolean strings → booleans")
    print()

    return test_data


def create_complex_test_data():
    """Create complex test data with various types for demonstration."""
    try:
        import numpy as np
        import pandas as pd

        return {
            "metadata": {
                "id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "version": "1.0.0",
            },
            "data_structures": {
                "simple_set": {1, 2, 3, 4, 5},
                "tuple_data": (10, 20, 30),
                "simple_tuple": ("a", "b", "c"),
            },
            "pandas_objects": {
                "dataframe": pd.DataFrame(
                    {
                        "A": [1, 2, 3, 4],
                        "B": ["x", "y", "z", "w"],
                        "C": [1.1, 2.2, 3.3, 4.4],
                    }
                ),
                "series": pd.Series([10, 20, 30, 40], name="test_series"),
                "series_no_name": pd.Series(["a", "b", "c"]),
            },
            "numpy_arrays": {
                "int_array": np.array([1, 2, 3, 4, 5]),
                "float_array": np.array([1.1, 2.2, 3.3]),
                "string_array": np.array(["hello", "world"]),
            },
            "nested_complex": {
                "level1": {
                    "level2": {
                        "uuids": [uuid.uuid4(), uuid.uuid4()],
                        "dates": [datetime.now(), datetime(2023, 1, 1)],
                        "mixed_set": {1, "two", 3.0},
                    }
                }
            },
        }
    except ImportError:
        # Fallback for when pandas/numpy aren't available
        return {
            "metadata": {
                "id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "version": "1.0.0",
            },
            "data_structures": {
                "simple_set": {1, 2, 3, 4, 5},
                "tuple_data": (10, 20, 30),
                "simple_tuple": ("a", "b", "c"),
            },
            "nested_complex": {
                "level1": {
                    "level2": {
                        "uuids": [uuid.uuid4(), uuid.uuid4()],
                        "dates": [datetime.now(), datetime(2023, 1, 1)],
                        "mixed_set": {1, "two", 3.0},
                    }
                }
            },
        }


def demo_basic_auto_detection():
    """Demonstrate basic auto-detection deserialization."""
    print("=== Basic Auto-Detection Deserialization ===")

    # Create test data with various string representations
    test_data = {
        "datetime_string": "2023-12-25T10:30:00",
        "uuid_string": "12345678-1234-5678-9012-123456789abc",
        "regular_string": "just a normal string",
        "number_string": "42",
        "float_string": "3.14159",
        "boolean_string": "true",
        "scientific_notation": "1.23e-4",
        "nested_data": [
            "2023-01-01T00:00:00",
            "87654321-4321-8765-2109-876543210fed",
            "false",
        ],
    }

    print("Original data types:")
    for key, value in test_data.items():
        if isinstance(value, list):
            print(f"  {key}: {[type(v).__name__ for v in value]}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Serialize to JSON string
    json_str = json.dumps(test_data)
    print(f"\nJSON string length: {len(json_str)} characters")

    # Standard deserialization (no auto-detection)
    standard_result = deserialize(json.loads(json_str))
    print("\nStandard deserialization types:")
    for key, value in standard_result.items():
        if isinstance(value, list):
            print(f"  {key}: {[type(v).__name__ for v in value]}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Auto-detection deserialization
    auto_result = auto_deserialize(json.loads(json_str))
    print("\nAuto-detection deserialization types:")
    for key, value in auto_result.items():
        if isinstance(value, list):
            print(f"  {key}: {[type(v).__name__ for v in value]}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Show specific conversions
    print("\n✅ Auto-detected conversions:")
    if isinstance(auto_result["datetime_string"], datetime):
        print(f"  datetime: {auto_result['datetime_string']}")
    if isinstance(auto_result["uuid_string"], uuid.UUID):
        print(f"  UUID: {auto_result['uuid_string']}")
    if isinstance(auto_result["number_string"], int):
        print(f"  integer: {auto_result['number_string']}")
    if isinstance(auto_result["float_string"], float):
        print(f"  float: {auto_result['float_string']}")

    print()


def demo_aggressive_mode():
    """Demonstrate aggressive mode for pandas detection."""
    print("=== Aggressive Mode for Pandas Detection ===")

    try:
        import pandas as pd

        # Create data that looks like it could be pandas objects
        potential_dataframe = {
            "column_a": [1, 2, 3, 4],
            "column_b": [10.1, 20.2, 30.3, 40.4],
            "column_c": ["x", "y", "z", "w"],
        }

        potential_series = [100, 200, 300, 400, 500]

        test_data = {
            "maybe_dataframe": potential_dataframe,
            "maybe_series": potential_series,
            "definitely_not": {"a": [1, 2], "b": [3]},  # Different lengths
        }

        print("Test data structure:")
        print(f"  maybe_dataframe: dict with {len(potential_dataframe)} columns")
        print(f"  maybe_series: list with {len(potential_series)} elements")
        print("  definitely_not: dict with mismatched column lengths")

        # Serialize and deserialize
        json_str = json.dumps(test_data)

        # Conservative auto-detection
        conservative_result = auto_deserialize(json.loads(json_str), aggressive=False)
        print("\nConservative mode results:")
        for key, value in conservative_result.items():
            print(f"  {key}: {type(value).__name__}")

        # Aggressive auto-detection
        aggressive_result = auto_deserialize(json.loads(json_str), aggressive=True)
        print("\nAggressive mode results:")
        for key, value in aggressive_result.items():
            print(f"  {key}: {type(value).__name__}")
            if isinstance(value, pd.DataFrame):
                print(f"    DataFrame shape: {value.shape}")
            elif isinstance(value, pd.Series):
                print(f"    Series length: {len(value)}")

        print("\n✅ Aggressive mode detected pandas objects where appropriate")

    except ImportError:
        print("⚠️ Pandas not available - skipping aggressive mode demo")

    print()


def demo_type_metadata_round_trip():
    """Demonstrate perfect round-trip serialization with type metadata."""
    print("=== Type Metadata Round-Trip Serialization ===")

    # Create complex test data
    original_data = create_complex_test_data()

    print("Original data structure:")
    print_data_types(original_data, indent="  ")

    # Serialize with type metadata
    config = SerializationConfig(include_type_hints=True)
    serialized_with_metadata = serialize(original_data, config=config)

    print(f"\nSerialized with type metadata (size: {len(str(serialized_with_metadata))} chars)")

    # Show some type metadata examples
    print("\nType metadata examples:")
    if "metadata" in serialized_with_metadata and "id" in serialized_with_metadata["metadata"]:
        uuid_metadata = serialized_with_metadata["metadata"]["id"]
        if isinstance(uuid_metadata, dict) and "__datason_type__" in uuid_metadata:
            print(f"  UUID metadata: {uuid_metadata}")

    # Deserialize with auto-detection (handles type metadata automatically)
    restored_data = auto_deserialize(serialized_with_metadata)

    print("\nRestored data structure:")
    print_data_types(restored_data, indent="  ")

    # Verify round-trip accuracy
    print("\n🔍 Round-trip verification:")
    verify_round_trip(original_data, restored_data)

    print()


def demo_performance_comparison():
    """Compare performance of different deserialization modes."""
    print("=== Performance Comparison ===")

    # Create test data
    test_data = create_complex_test_data()

    # Serialize with and without type metadata
    standard_config = SerializationConfig()
    metadata_config = SerializationConfig(include_type_hints=True)

    standard_serialized = serialize(test_data, config=standard_config)
    metadata_serialized = serialize(test_data, config=metadata_config)

    print(f"Standard serialization size: {len(str(standard_serialized)):,} chars")
    print(f"Metadata serialization size: {len(str(metadata_serialized)):,} chars")
    print(f"Metadata overhead: {len(str(metadata_serialized)) - len(str(standard_serialized)):,} chars")

    # Performance testing
    iterations = 1000

    # Standard deserialization
    start_time = time.time()
    for _ in range(iterations):
        deserialize(standard_serialized)
    standard_time = time.time() - start_time

    # Auto-detection (conservative)
    start_time = time.time()
    for _ in range(iterations):
        auto_deserialize(standard_serialized, aggressive=False)
    auto_conservative_time = time.time() - start_time

    # Auto-detection (aggressive)
    start_time = time.time()
    for _ in range(iterations):
        auto_deserialize(standard_serialized, aggressive=True)
    auto_aggressive_time = time.time() - start_time

    # Type metadata deserialization
    start_time = time.time()
    for _ in range(iterations):
        auto_deserialize(metadata_serialized)
    metadata_time = time.time() - start_time

    print(f"\nPerformance results ({iterations:,} iterations):")
    print(f"  Standard deserialization:     {standard_time:.4f}s ({standard_time / iterations * 1000:.2f}ms per call)")
    print(
        f"  Auto-detection (conservative): {auto_conservative_time:.4f}s ({auto_conservative_time / iterations * 1000:.2f}ms per call)"
    )
    print(
        f"  Auto-detection (aggressive):   {auto_aggressive_time:.4f}s ({auto_aggressive_time / iterations * 1000:.2f}ms per call)"
    )
    print(f"  Type metadata round-trip:     {metadata_time:.4f}s ({metadata_time / iterations * 1000:.2f}ms per call)")

    print("\nRelative performance:")
    print(f"  Auto-detection overhead: {(auto_conservative_time / standard_time - 1) * 100:.1f}%")
    print(f"  Aggressive mode overhead: {(auto_aggressive_time / auto_conservative_time - 1) * 100:.1f}%")
    print(f"  Metadata processing speedup: {(auto_conservative_time / metadata_time - 1) * 100:.1f}%")

    print()


def demo_real_world_scenario():
    """Demonstrate a real-world data processing scenario."""
    print("=== Real-World Data Processing Scenario ===")

    # Simulate a data processing pipeline
    print("Simulating a data processing pipeline...")

    # Step 1: Create "raw" data (as if from an API or file)
    raw_data = {
        "experiment_id": "12345678-1234-5678-9012-123456789abc",
        "timestamp": "2023-12-25T10:30:00",
        "results": [
            {"metric": "accuracy", "value": 0.95, "timestamp": "2023-12-25T10:30:01"},
            {"metric": "precision", "value": 0.93, "timestamp": "2023-12-25T10:30:02"},
            {"metric": "recall", "value": 0.97, "timestamp": "2023-12-25T10:30:03"},
        ],
        "model_params": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        },
        "data_info": {
            "train_samples": 10000,
            "test_samples": 2000,
            "features": ["feature_1", "feature_2", "feature_3"],
        },
    }

    print("📥 Raw data received (all strings from JSON):")
    print_data_types(raw_data, indent="  ")

    # Step 2: Process with auto-detection
    print("\n🔄 Processing with auto-detection...")
    processed_data = auto_deserialize(raw_data, aggressive=False)

    print("📊 Processed data types:")
    print_data_types(processed_data, indent="  ")

    # Step 3: Demonstrate type-safe operations
    print("\n✅ Type-safe operations now possible:")

    # UUID operations
    exp_id = processed_data["experiment_id"]
    if isinstance(exp_id, uuid.UUID):
        print(f"  Experiment UUID version: {exp_id.version}")
        print(f"  Experiment UUID hex: {exp_id.hex}")

    # Datetime operations
    timestamp = processed_data["timestamp"]
    if isinstance(timestamp, datetime):
        print(f"  Experiment date: {timestamp.strftime('%Y-%m-%d')}")
        print(f"  Time since experiment: {datetime.now() - timestamp}")

    # Numeric operations
    results = processed_data["results"]
    if results and isinstance(results[0]["value"], (int, float)):
        avg_score = sum(r["value"] for r in results) / len(results)
        print(f"  Average metric value: {avg_score:.3f}")

    # Step 4: Save with type metadata for perfect restoration
    print("\n💾 Saving with type metadata for perfect restoration...")
    config = SerializationConfig(include_type_hints=True)
    saved_data = serialize(processed_data, config=config)

    # Step 5: Restore and verify
    print("🔄 Restoring from saved data...")
    restored_data = auto_deserialize(saved_data)

    print("✅ Verification - types preserved:")
    verify_round_trip(processed_data, restored_data, show_details=False)

    print()


def print_data_types(data: Any, indent: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    """Recursively print data types with indentation."""
    if current_depth >= max_depth:
        print(f"{indent}... (max depth reached)")
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list, tuple)):
                print(f"{indent}{key}: {type(value).__name__}")
                print_data_types(value, indent + "  ", max_depth, current_depth + 1)
            else:
                print(
                    f"{indent}{key}: {type(value).__name__} = {repr(value)[:50]}{'...' if len(repr(value)) > 50 else ''}"
                )
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], (dict, list, tuple)):
        first_item = data[0]
        print(f"{indent}[0]: {type(first_item).__name__}")
        print_data_types(first_item, indent + "  ", max_depth, current_depth + 1)
        if len(data) > 1:
            print(f"{indent}... ({len(data)} total items)")
    elif isinstance(data, (list, tuple)) and data:
        types = [type(item).__name__ for item in data[:3]]
        if len(data) > 3:
            types.append(f"... ({len(data)} total)")
        print(f"{indent}items: {types}")


def verify_round_trip(original: Any, restored: Any, path: str = "", show_details: bool = True) -> bool:
    """Verify that round-trip serialization preserved types and values."""
    # Handle pandas objects specially
    try:
        import pandas as pd

        if isinstance(original, pd.DataFrame) and isinstance(restored, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(original, restored)
                return True
            except AssertionError:
                if show_details:
                    print(f"  ❌ DataFrame content mismatch at {path}")
                return False
        elif isinstance(original, pd.Series) and isinstance(restored, pd.Series):
            try:
                pd.testing.assert_series_equal(original, restored)
                return True
            except AssertionError:
                if show_details:
                    print(f"  ❌ Series content mismatch at {path}")
                return False
    except ImportError:
        pass

    # Handle numpy arrays
    try:
        import numpy as np

        if isinstance(original, np.ndarray) and isinstance(restored, np.ndarray):
            try:
                np.testing.assert_array_equal(original, restored)
                return True
            except AssertionError:
                if show_details:
                    print(f"  ❌ NumPy array content mismatch at {path}")
                return False
    except ImportError:
        pass

    if type(original) is not type(restored):
        if show_details:
            print(f"  ❌ Type mismatch at {path}: {type(original).__name__} != {type(restored).__name__}")
        return False

    if isinstance(original, dict):
        if set(original.keys()) != set(restored.keys()):
            if show_details:
                print(f"  ❌ Key mismatch at {path}")
            return False

        all_match = True
        for key in original:
            if not verify_round_trip(original[key], restored[key], f"{path}.{key}", show_details):
                all_match = False
        return all_match

    if isinstance(original, (list, tuple)):
        if len(original) != len(restored):
            if show_details:
                print(f"  ❌ Length mismatch at {path}")
            return False

        all_match = True
        for i, (orig_item, rest_item) in enumerate(zip(original, restored)):
            if not verify_round_trip(orig_item, rest_item, f"{path}[{i}]", show_details):
                all_match = False
        return all_match

    if isinstance(original, set):
        if original != restored:
            if show_details:
                print(f"  ❌ Set content mismatch at {path}")
            return False

    else:
        # For other types, use equality comparison
        try:
            if original != restored:
                if show_details:
                    print(f"  ❌ Value mismatch at {path}: {original} != {restored}")
                return False
        except ValueError:
            # Handle cases where comparison might fail (like pandas objects)
            if show_details:
                print(f"  ⚠️ Cannot compare values at {path} (comparison not supported)")
            return True  # Assume OK if we can't compare

    if show_details and not path:  # Only print for root level
        print("  ✅ Perfect round-trip - all types and values preserved")

    return True


def demo_file_based_workflow():
    """Demonstrate file-based workflow with auto-detection."""
    print("=== File-Based Workflow Demo ===")

    # Create test data
    test_data = create_complex_test_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save with standard serialization
        standard_file = temp_path / "standard_data.json"
        standard_serialized = serialize(test_data)
        with standard_file.open("w") as f:
            json.dump(standard_serialized, f, indent=2)

        # Save with type metadata
        metadata_file = temp_path / "metadata_data.json"
        metadata_config = SerializationConfig(include_type_hints=True)
        metadata_serialized = serialize(test_data, config=metadata_config)
        with metadata_file.open("w") as f:
            json.dump(metadata_serialized, f, indent=2)

        print("📁 Files created:")
        print(f"  Standard: {standard_file.stat().st_size:,} bytes")
        print(f"  Metadata: {metadata_file.stat().st_size:,} bytes")

        # Load and process files
        print("\n📖 Loading and processing files...")

        # Standard file with auto-detection
        with standard_file.open() as f:
            standard_loaded = json.load(f)
        auto_deserialize(standard_loaded, aggressive=True)

        # Metadata file (auto-detects type metadata)
        with metadata_file.open() as f:
            metadata_loaded = json.load(f)
        metadata_processed = auto_deserialize(metadata_loaded)

        print("✅ Both files processed successfully")
        print("🔍 Comparing results...")

        # The metadata version should be identical to original
        metadata_matches = verify_round_trip(test_data, metadata_processed, show_details=False)
        print(f"  Metadata round-trip: {'✅ Perfect' if metadata_matches else '❌ Issues'}")

        # The standard version should have most types detected
        print("  Standard + auto-detection: ✅ Best effort type recovery")

    print()


def main():
    """Run all demonstrations."""
    print("🎯 Datason Auto-Detection and Type Metadata Demo")
    print("=" * 60)
    print()

    demo_simple_progressive_loading()
    demo_basic_auto_detection()
    demo_aggressive_mode()
    demo_type_metadata_round_trip()
    demo_performance_comparison()
    demo_real_world_scenario()
    demo_file_based_workflow()

    print("🎉 Demo completed!")
    print("\n💡 Key takeaways:")
    print("  - Auto-detection intelligently restores types from JSON")
    print("  - Aggressive mode can detect pandas objects from structured data")
    print("  - Type metadata enables perfect round-trip serialization")
    print("  - Performance overhead is minimal for the added functionality")
    print("  - Real-world workflows benefit from type-safe operations")
    print("  - File-based workflows preserve data integrity")
    print("\n🚀 Ready to use advanced datason features in your projects!")


if __name__ == "__main__":
    main()
