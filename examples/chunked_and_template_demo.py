#!/usr/bin/env python3
"""Demonstration of datason's advanced chunked processing and template-based deserialization.

This example showcases the new features added in v0.4.0 and v0.4.5:
- Chunked processing for memory-bounded large object processing
- Streaming capabilities for multi-GB datasets
- Template-based deserialization for enhanced type fidelity
- Enhanced round-trip capabilities for ML pipeline workflows

Each feature is demonstrated with real-world use cases and performance considerations.
"""

import datetime
import json
import tempfile
from pathlib import Path

import datason

# Optional imports for more comprehensive demonstrations
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


def demonstrate_chunked_processing():
    """Show chunked processing for memory-bounded large object handling."""
    print("=" * 70)
    print("CHUNKED PROCESSING DEMO (v0.4.0)")
    print("=" * 70)

    # Create large dataset that would be challenging to process in memory
    print("📊 Creating large dataset...")
    large_dataset = [
        {
            "id": i,
            "timestamp": datetime.datetime.now(),
            "user_data": {"name": f"user_{i}", "preferences": [f"pref_{j}" for j in range(i % 5)], "score": i * 0.1},
            "metadata": {"batch": i // 100, "processed": False},
        }
        for i in range(10000)  # 10K records
    ]

    print(f"📈 Dataset size: {len(large_dataset):,} records")

    # Estimate memory usage
    memory_stats = datason.estimate_memory_usage(large_dataset)
    print(f"💾 Estimated memory usage: {memory_stats['estimated_serialized_mb']:.1f} MB")
    print(f"🔧 Recommended chunk size: {memory_stats['recommended_chunk_size']:,}")
    print(f"📦 Recommended chunks: {memory_stats['recommended_chunks']}")

    # Chunk the dataset
    print("\n🔄 Chunking dataset...")
    chunk_size = 1000
    chunked_result = datason.serialize_chunked(large_dataset, chunk_size=chunk_size)

    print(f"✅ Chunked into {chunked_result.metadata['total_chunks']} chunks")
    print(f"🔢 Strategy: {chunked_result.metadata['chunking_strategy']}")
    print(f"📋 Object type: {chunked_result.metadata['object_type']}")

    # Process chunks one at a time (memory efficient)
    print("\n⚙️ Processing chunks individually...")
    processed_count = 0

    for i, chunk in enumerate(chunked_result.chunks):
        # Process each chunk without loading all chunks into memory
        print(f"  Processing chunk {i + 1}/{chunked_result.metadata['total_chunks']}: {len(chunk)} items")
        processed_count += len(chunk)

        # Example processing: count records per batch
        batch_counts = {}
        for record in chunk:
            batch_id = record["metadata"]["batch"]
            batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1

        if i == 0:  # Show example for first chunk
            print(f"    Sample batch distribution: {dict(list(batch_counts.items())[:3])}")

    print(f"✅ Processed {processed_count:,} records using chunked approach")

    # Save chunked data to file
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "chunked_dataset.jsonl"
        print(f"\n💾 Saving chunked data to {output_file.name}...")

        chunked_result = datason.serialize_chunked(large_dataset, chunk_size=chunk_size)
        chunked_result.save_to_file(output_file, format="jsonl")

        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.1f} MB")

        # Read back and verify
        print("🔍 Reading back chunks...")
        chunk_count = 0
        record_count = 0

        for chunk in datason.deserialize_chunked_file(output_file, format="jsonl"):
            chunk_count += 1
            record_count += len(chunk)

        print(f"✅ Verified: {chunk_count} chunks, {record_count:,} records")

    return large_dataset


def demonstrate_streaming_serialization():
    """Show streaming serialization for continuous data processing."""
    print("\n" + "=" * 70)
    print("STREAMING SERIALIZATION DEMO (v0.4.0)")
    print("=" * 70)

    print("🌊 Demonstrating streaming serialization for continuous data...")

    # Simulate continuous data stream
    def generate_sensor_data(num_readings=5000):
        """Simulate IoT sensor data stream."""
        for i in range(num_readings):
            yield {
                "sensor_id": f"sensor_{i % 100}",
                "timestamp": datetime.datetime.now(),
                "temperature": 20 + (i % 30),  # Simulate temperature variation
                "humidity": 50 + (i % 40),  # Simulate humidity variation
                "battery_level": max(0, 100 - (i // 100)),  # Decreasing battery
                "location": {"lat": 40.7128 + (i % 10) * 0.001, "lon": -74.0060 + (i % 10) * 0.001},
            }

    with tempfile.TemporaryDirectory() as temp_dir:
        stream_file = Path(temp_dir) / "sensor_stream.jsonl"

        print(f"📡 Streaming sensor data to {stream_file.name}...")

        # Stream data without loading everything into memory
        with datason.stream_serialize(stream_file, format="jsonl") as stream:
            batch = []
            batch_size = 100

            for reading in generate_sensor_data(5000):
                batch.append(reading)

                if len(batch) >= batch_size:
                    # Write batch and clear memory
                    stream.write(batch)
                    batch = []
                    print(f"  📦 Streamed batch of {batch_size} readings...")

            # Write remaining readings
            if batch:
                stream.write(batch)
                print(f"  📦 Streamed final batch of {len(batch)} readings...")

        # Verify streamed data
        file_size = stream_file.stat().st_size / (1024 * 1024)  # MB
        print(f"💾 Stream file size: {file_size:.1f} MB")

        # Process streamed data efficiently
        print("\n🔄 Processing streamed data...")
        total_readings = 0
        sensor_counts = {}
        low_battery_alerts = 0

        for batch in datason.deserialize_chunked_file(stream_file, format="jsonl"):
            for reading in batch:
                total_readings += 1

                # Count readings per sensor
                sensor_id = reading["sensor_id"]
                sensor_counts[sensor_id] = sensor_counts.get(sensor_id, 0) + 1

                # Check for low battery alerts
                if reading["battery_level"] < 20:
                    low_battery_alerts += 1

        print(f"✅ Processed {total_readings:,} sensor readings")
        print(f"🔋 Low battery alerts: {low_battery_alerts}")
        print(f"📊 Unique sensors: {len(sensor_counts)}")
        print(f"📈 Avg readings per sensor: {total_readings / len(sensor_counts):.1f}")


def demonstrate_template_based_deserialization():
    """Show template-based deserialization for enhanced type fidelity."""
    print("\n" + "=" * 70)
    print("TEMPLATE-BASED DESERIALIZATION DEMO (v0.4.5)")
    print("=" * 70)

    print("🎯 Demonstrating template-based deserialization for type fidelity...")

    # Example 1: Basic template inference and application
    print("\n1️⃣ Template Inference from Sample Data")
    sample_training_data = [
        {"user_id": 1001, "name": "Alice", "age": 28, "score": 95.5, "active": True},
        {"user_id": 1002, "name": "Bob", "age": 34, "score": 87.2, "active": False},
        {"user_id": 1003, "name": "Charlie", "age": 29, "score": 91.8, "active": True},
    ]

    # Infer template from sample data
    template = datason.infer_template_from_data(sample_training_data)
    print(f"📝 Inferred template: {template}")

    # New data with mixed types (simulating data from external API)
    new_user_data = [
        {"user_id": "1004", "name": "Diana", "age": "31", "score": "88.9", "active": "true"},
        {"user_id": "1005", "name": "Eve", "age": "26", "score": "94.1", "active": "false"},
    ]

    # Deserialize with template to enforce correct types
    corrected_data = datason.deserialize_with_template(new_user_data, template)

    print("🔧 Original data types:", [type(item["age"]).__name__ for item in new_user_data])
    print("✅ Corrected data types:", [type(item["age"]).__name__ for item in corrected_data])
    print("🎯 Type conversion successful!")

    # Example 2: ML Round-trip Template
    print("\n2️⃣ ML Pipeline Round-trip Fidelity")

    ml_experiment = {
        "experiment_id": "exp_001",
        "model_config": {"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        "training_data": {
            "features": [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
            "labels": [0, 1, 0],
            "preprocessing": {"scaler": "standard", "feature_selection": True},
        },
        "results": {
            "accuracy": 0.924,
            "f1_score": 0.891,
            "training_time": datetime.timedelta(minutes=45),
            "model_size_mb": 12.5,
        },
        "metadata": {
            "created": datetime.datetime.now(),
            "researcher": "Dr. Smith",
            "git_commit": "a1b2c3d4",
            "environment": "python_3.9",
        },
    }

    # Create ML-specific template
    ml_template = datason.create_ml_round_trip_template(ml_experiment)
    print(f"🧠 ML template created: {ml_template['object_type']}")
    print(f"🏷️ Template marker: {ml_template.get('__ml_template__', False)}")

    # Serialize experiment with type metadata for perfect round-trip
    config = datason.SerializationConfig(include_type_hints=True)
    serialized_experiment = datason.serialize(ml_experiment, config=config)

    # Deserialize with template
    restored_experiment = datason.deserialize_with_template(serialized_experiment, ml_experiment)

    print("✅ ML experiment serialized and restored with type fidelity")
    print(f"🕐 Original timestamp type: {type(ml_experiment['metadata']['created'])}")
    print(f"🕐 Restored timestamp type: {type(restored_experiment['metadata']['created'])}")

    # Example 3: DataFrame Template Matching
    if pd is not None:
        print("\n3️⃣ DataFrame Template Matching")

        # Create template DataFrame with specific dtypes
        template_df = pd.DataFrame(
            {
                "id": pd.Series([1], dtype="int32"),
                "score": pd.Series([0.0], dtype="float32"),
                "category": pd.Series(["A"], dtype="category"),
                "timestamp": pd.Series([pd.Timestamp.now()], dtype="datetime64[ns]"),
            }
        )

        print("📊 Template DataFrame dtypes:")
        for col, dtype in template_df.dtypes.items():
            print(f"  {col}: {dtype}")

        # Simulate data from CSV/JSON with generic types
        raw_data = [
            {"id": 101, "score": 85.5, "category": "premium", "timestamp": "2023-12-01T10:00:00"},
            {"id": 102, "score": 92.1, "category": "standard", "timestamp": "2023-12-01T11:00:00"},
            {"id": 103, "score": 78.9, "category": "premium", "timestamp": "2023-12-01T12:00:00"},
        ]

        # Deserialize with DataFrame template to enforce types
        structured_df = datason.deserialize_with_template(raw_data, template_df)

        print("\n📊 Structured DataFrame dtypes:")
        for col, dtype in structured_df.dtypes.items():
            print(f"  {col}: {dtype}")

        print("✅ DataFrame types match template specification!")
        print(f"📏 DataFrame shape: {structured_df.shape}")

    return corrected_data, restored_experiment


def demonstrate_large_dataset_workflow():
    """Show complete workflow with large dataset using all features."""
    print("\n" + "=" * 70)
    print("INTEGRATED LARGE DATASET WORKFLOW")
    print("=" * 70)

    print("🔄 Demonstrating complete workflow with chunking, streaming, and templates...")

    # Step 1: Create large dataset with complex structure
    print("\n📊 Step 1: Generate large complex dataset...")

    def generate_ml_dataset(num_samples=10000):
        """Generate ML dataset with various data types."""
        for i in range(num_samples):
            yield {
                "sample_id": i,
                "features": {
                    "numerical": [float(j + i * 0.1) for j in range(10)],
                    "categorical": f"category_{i % 5}",
                    "text": f"sample text for item {i}",
                    "encoded": [i % 2, (i + 1) % 2, (i + 2) % 2],
                },
                "target": i % 3,  # 3-class classification
                "metadata": {
                    "created": datetime.datetime.now(),
                    "source": "synthetic",
                    "quality_score": min(1.0, 0.5 + (i % 100) * 0.005),
                    "tags": [f"tag_{j}" for j in range(i % 4)],
                },
                "preprocessing": {"scaled": True, "normalized": True, "outlier_score": float(i % 10) / 10.0},
            }

    # Step 2: Stream dataset to file with chunking
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_file = Path(temp_dir) / "large_ml_dataset.jsonl"
        template_file = Path(temp_dir) / "dataset_template.json"

        print(f"📁 Step 2: Stream dataset to {dataset_file.name}...")

        # Create template from first few samples
        first_samples = list(generate_ml_dataset(5))
        dataset_template = datason.infer_template_from_data(first_samples)

        # Save template for later use
        with template_file.open("w") as f:
            json.dump(datason.serialize(dataset_template), f, indent=2)

        # Stream full dataset
        with datason.stream_serialize(dataset_file, format="jsonl") as stream:
            batch = []
            batch_size = 500
            total_streamed = 0

            for sample in generate_ml_dataset(10000):
                batch.append(sample)

                if len(batch) >= batch_size:
                    stream.write(batch)
                    total_streamed += len(batch)
                    if total_streamed % 2000 == 0:
                        print(f"  📦 Streamed {total_streamed:,} samples...")
                    batch = []

            if batch:
                stream.write(batch)
                total_streamed += len(batch)

        print(f"✅ Streamed {total_streamed:,} samples to file")

        # Step 3: Process dataset in chunks with template validation
        print("\n🔍 Step 3: Process dataset using template validation...")

        # Load template
        with template_file.open("r") as f:
            loaded_template = json.load(f)

        # Process chunks with template validation
        processed_samples = 0
        quality_scores = []
        target_distribution = {0: 0, 1: 0, 2: 0}

        for chunk in datason.deserialize_chunked_file(dataset_file, format="jsonl"):
            # Validate chunk against template
            validated_chunk = datason.deserialize_with_template(chunk, loaded_template)

            # Process validated chunk
            for sample in validated_chunk:
                processed_samples += 1
                quality_scores.append(sample["preprocessing"]["outlier_score"])
                target_distribution[sample["target"]] += 1

        print(f"✅ Processed {processed_samples:,} samples with template validation")
        print(f"📊 Target distribution: {target_distribution}")
        print(f"📈 Avg quality score: {sum(quality_scores) / len(quality_scores):.3f}")

        # Step 4: Demonstrate memory efficiency
        file_size = dataset_file.stat().st_size / (1024 * 1024)
        print("\n💾 Memory Efficiency Results:")
        print(f"  File size: {file_size:.1f} MB")
        print("  Processed without loading full dataset into memory")
        print("  Template ensured type consistency across all chunks")
        print("  ✅ Scalable to datasets larger than available RAM")


def demonstrate_performance_comparison():
    """Compare performance of different processing approaches."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    import time

    # Create test dataset
    test_data = [{"id": i, "value": f"item_{i}", "score": i * 0.1} for i in range(5000)]

    print(f"🏃 Performance testing with {len(test_data):,} records...")

    # Test 1: Standard serialization
    start_time = time.time()
    standard_result = datason.serialize(test_data)
    standard_time = time.time() - start_time

    # Test 2: Chunked serialization
    start_time = time.time()
    chunked_result = datason.serialize_chunked(test_data, chunk_size=500)
    chunked_chunks = chunked_result.to_list()  # Force evaluation
    chunked_time = time.time() - start_time

    # Test 3: Template-based processing
    template = datason.infer_template_from_data(test_data[:10])
    start_time = time.time()
    template_result = datason.deserialize_with_template(test_data, template)
    template_time = time.time() - start_time

    print("\n⏱️ Performance Results:")
    print(f"{'Method':<20} {'Time (ms)':<12} {'Output Size':<12} {'Memory'}")
    print("-" * 60)
    print(f"{'Standard':<20} {standard_time * 1000:<12.1f} {len(str(standard_result)):<12} {'High'}")
    print(f"{'Chunked':<20} {chunked_time * 1000:<12.1f} {len(str(chunked_chunks)):<12} {'Low'}")
    print(f"{'Template':<20} {template_time * 1000:<12.1f} {len(str(template_result)):<12} {'Medium'}")

    print("\n💡 Key Insights:")
    print("  🚀 Chunked processing enables memory-bounded operations")
    print("  🎯 Template processing ensures type consistency")
    print("  ⚖️ Choose method based on dataset size and requirements")


def main():
    """Run all demonstrations."""
    print("DATASON ADVANCED FEATURES DEMONSTRATION")
    print("======================================")
    print("v0.4.0: Chunked Processing & Streaming")
    print("v0.4.5: Template-Based Deserialization")
    print()

    # Run all demonstrations
    demonstrate_chunked_processing()
    demonstrate_streaming_serialization()
    corrected_data, ml_experiment = demonstrate_template_based_deserialization()
    demonstrate_large_dataset_workflow()
    demonstrate_performance_comparison()

    print("\n" + "=" * 70)
    print("SUMMARY OF NEW CAPABILITIES")
    print("=" * 70)
    print("✅ Chunked Processing (v0.4.0):")
    print("  • Memory-bounded processing of large datasets")
    print("  • Automatic chunk size recommendations")
    print("  • Support for lists, DataFrames, numpy arrays, and dicts")
    print("  • Memory usage estimation and optimization")
    print()
    print("✅ Streaming Serialization (v0.4.0):")
    print("  • Process datasets larger than available RAM")
    print("  • JSONL and JSON output formats")
    print("  • Integrated with chunked processing")
    print("  • Efficient for continuous data streams")
    print()
    print("✅ Template-Based Deserialization (v0.4.5):")
    print("  • Automatic template inference from sample data")
    print("  • Type-guided deserialization with coercion")
    print("  • ML-specific round-trip templates")
    print("  • Perfect type fidelity for complex workflows")
    print()
    print("🎯 These features make datason ideal for:")
    print("  • Large-scale ML/AI data processing")
    print("  • IoT and streaming data workflows")
    print("  • Production systems with memory constraints")
    print("  • Data pipelines requiring type consistency")


if __name__ == "__main__":
    main()
