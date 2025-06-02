#!/usr/bin/env python3
"""Pickle Bridge Demo - Convert legacy ML pickle files to portable JSON.

This example demonstrates the key features of datason's pickle bridge,
showing how to safely migrate from pickle-based workflows to JSON.

Features demonstrated:
- Safe conversion with class whitelisting
- Bulk directory conversion
- Custom security configurations
- Performance monitoring
"""

import pickle  # nosec B403
import tempfile
import time
from pathlib import Path

import datason


def create_sample_ml_data():
    """Create sample ML-like data for demonstration."""
    import uuid
    from datetime import datetime

    # Simulate typical ML workflow data
    return {
        "experiment": {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "model_type": "RandomForestClassifier",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
            },
            "training_data": {
                "features": list(range(1000)),  # Simulated feature vector
                "labels": [i % 2 for i in range(1000)],  # Binary labels
                "metadata": {
                    "n_samples": 1000,
                    "n_features": 20,
                },
            },
        }
    }


def demo_basic_conversion():
    """Demonstrate basic pickle to JSON conversion."""
    print("=== Basic Pickle to JSON Conversion ===")

    # Create sample data
    sample_data = create_sample_ml_data()

    # Create a temporary pickle file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
        pickle.dump(sample_data, temp_file)
        pickle_path = temp_file.name

    try:
        # Convert pickle to JSON using convenience function
        print(f"Converting {pickle_path}...")
        result = datason.from_pickle(pickle_path)

        print("✅ Conversion successful!")
        print(f"📁 Source file: {result['metadata']['source_file']}")
        print(f"📊 Source size: {result['metadata']['source_size_bytes']:,} bytes")
        print(f"🕒 Converted at: {result['metadata']['conversion_timestamp']}")
        print(f"🎯 Experiment ID: {result['data']['experiment']['id']}")
        print(f"📈 Model accuracy: {result['data']['experiment']['metrics']['accuracy']}")

    finally:
        # Clean up
        Path(pickle_path).unlink(missing_ok=True)

    print()


def demo_security_features():
    """Demonstrate security features with class whitelisting."""
    print("=== Security Features Demonstration ===")

    # Create a bridge with restricted safe classes
    print("Creating bridge with restricted class whitelist...")
    safe_classes = {
        "builtins.dict",
        "builtins.list",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.bool",
        "builtins.NoneType",
        "datetime.datetime",
        "uuid.UUID",
    }

    bridge = datason.PickleBridge(safe_classes=safe_classes)
    print(f"✅ Bridge created with {len(bridge.safe_classes)} safe classes")

    # Add additional safe classes
    bridge.add_safe_class("collections.OrderedDict")
    print("✅ Added collections.OrderedDict to safe classes")

    # Demonstrate safe conversion
    sample_data = {"safe": "data", "numbers": [1, 2, 3]}
    pickle_bytes = pickle.dumps(sample_data)

    try:
        result = bridge.from_pickle_bytes(pickle_bytes)
        print("✅ Safe data converted successfully")
        print(f"📊 Data: {result['data']}")
    except datason.PickleSecurityError as e:
        print(f"🔒 Security error (expected for unsafe data): {e}")

    print()


def demo_bulk_conversion():
    """Demonstrate bulk directory conversion."""
    print("=== Bulk Directory Conversion ===")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "pickle_files"
        target_dir = Path(temp_dir) / "json_files"
        source_dir.mkdir()

        # Create multiple pickle files
        print("Creating sample pickle files...")
        for i in range(5):
            sample_data = create_sample_ml_data()
            sample_data["experiment"]["id"] = f"exp_{i:03d}"
            sample_data["experiment"]["parameters"]["random_state"] = 42 + i

            pickle_file = source_dir / f"experiment_{i:03d}.pkl"
            with pickle_file.open("wb") as f:
                pickle.dump(sample_data, f)

        print(f"✅ Created {len(list(source_dir.glob('*.pkl')))} pickle files")

        # Convert directory
        print("Converting directory...")
        start_time = time.time()
        stats = datason.convert_pickle_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            pattern="*.pkl",
            overwrite=True,
        )
        elapsed = time.time() - start_time

        # Display results
        print(f"✅ Conversion completed in {elapsed:.2f} seconds")
        print(f"📁 Files found: {stats['files_found']}")
        print(f"✅ Files converted: {stats['files_converted']}")
        print(f"⏭️ Files skipped: {stats['files_skipped']}")
        print(f"❌ Files failed: {stats['files_failed']}")

        if stats["errors"]:
            print("Errors:")
            for error in stats["errors"]:
                print(f"  - {error['file']}: {error['error']}")

        # List converted files
        json_files = list(target_dir.glob("*.json"))
        print(f"📂 JSON files created: {len(json_files)}")
        for json_file in json_files:
            print(f"  - {json_file.name}")

    print()


def demo_advanced_features():
    """Demonstrate advanced features and configuration."""
    print("=== Advanced Features ===")

    # Create bridge with custom configuration
    config = datason.get_ml_config()  # Use ML-optimized config
    bridge = datason.PickleBridge(
        config=config,
        max_file_size=50 * 1024 * 1024,  # 50MB limit
    )

    print("✅ Created bridge with ML-optimized configuration")
    print(f"📏 Max file size: {bridge.max_file_size:,} bytes")

    # Demonstrate statistics tracking
    sample_data = create_sample_ml_data()
    pickle_bytes = pickle.dumps(sample_data)

    # Convert multiple times to build stats
    for i in range(3):
        bridge.from_pickle_bytes(pickle_bytes)
        print(f"✅ Conversion {i + 1} completed")

    # Show conversion statistics
    stats = bridge.get_conversion_stats()
    print("\n📊 Conversion Statistics:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files successful: {stats['files_successful']}")
    print(f"  Files failed: {stats['files_failed']}")
    print(f"  Total bytes processed: {stats['total_size_bytes']:,}")

    # Demonstrate safe classes management
    print(f"\n🔒 Security: {len(bridge.safe_classes)} safe classes configured")
    ml_safe_classes = datason.get_ml_safe_classes()
    print(f"🧠 ML safe classes available: {len(ml_safe_classes)}")

    # Show some example safe classes
    example_classes = sorted(ml_safe_classes)[:10]
    print("📋 Example safe classes:")
    for cls in example_classes:
        print(f"  - {cls}")

    print()


def demo_error_handling():
    """Demonstrate error handling and edge cases."""
    print("=== Error Handling Demonstration ===")

    bridge = datason.PickleBridge()

    # Test file size limit
    print("Testing file size limits...")
    tiny_bridge = datason.PickleBridge(max_file_size=100)  # Very small limit
    large_data = {"data": "x" * 1000}  # Larger than limit
    large_pickle = pickle.dumps(large_data)

    try:
        tiny_bridge.from_pickle_bytes(large_pickle)
        print("❌ Should have failed due to size limit")
    except datason.SecurityError as e:
        print(f"✅ Size limit enforced: {e}")

    # Test non-existent file
    print("\nTesting non-existent file handling...")
    try:
        bridge.from_pickle_file("nonexistent.pkl")
        print("❌ Should have failed for non-existent file")
    except FileNotFoundError:
        print("✅ Non-existent file handled correctly")

    # Test corrupted pickle data
    print("\nTesting corrupted pickle handling...")
    try:
        bridge.from_pickle_bytes(b"not a pickle file")
        print("❌ Should have failed for corrupted data")
    except datason.PickleSecurityError:
        print("✅ Corrupted pickle handled correctly")

    print()


def main():
    """Run all demonstrations."""
    print("🎯 Datason Pickle Bridge Demo")
    print("=" * 50)
    print()

    demo_basic_conversion()
    demo_security_features()
    demo_bulk_conversion()
    demo_advanced_features()
    demo_error_handling()

    print("🎉 Demo completed!")
    print("\n💡 Key takeaways:")
    print("  - Pickle bridge safely converts pickle files to portable JSON")
    print("  - Security-first approach with class whitelisting")
    print("  - Zero new dependencies - uses only Python standard library")
    print("  - Bulk conversion support for migrating entire workflows")
    print("  - Comprehensive error handling and size limits")
    print("  - Built-in statistics and monitoring")
    print("\n🚀 Ready to migrate your pickle-based ML workflows!")


if __name__ == "__main__":
    main()
