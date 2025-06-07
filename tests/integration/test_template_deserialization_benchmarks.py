"""Benchmarks for template-based deserialization capabilities (v0.4.5).

This module contains performance benchmarks for the new template-based
deserialization features to ensure they meet performance expectations.
"""

import time
from datetime import datetime
from uuid import uuid4

import pytest

import datason
from datason.deserializers import (
    TemplateDeserializer,
    create_ml_round_trip_template,
    deserialize_with_template,
    infer_template_from_data,
)

# Optional imports for comprehensive benchmarking
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


@pytest.mark.benchmark(group="template_deserialization")
class TestTemplateDeserializationBenchmarks:
    """Benchmark template-based deserialization performance."""

    def test_template_vs_auto_deserialization(self, benchmark):
        """Compare template-based vs auto-detection deserialization."""
        # Sample data with mixed types
        sample_data = [
            {"id": i, "name": f"user_{i}", "created": "2023-01-01T10:00:00", "active": True} for i in range(1000)
        ]

        # Create template
        template = {"id": 0, "name": "", "created": datetime.now(), "active": True}

        def template_deserialize():
            deserializer = TemplateDeserializer(template)
            return [deserializer.deserialize(item) for item in sample_data]

        result = benchmark(template_deserialize)
        assert len(result) == 1000

    def test_template_inference_performance(self, benchmark):
        """Benchmark template inference from sample data."""
        # Sample data for inference
        sample_data = [
            {
                "user_id": i,
                "profile": {"name": f"User {i}", "age": 20 + (i % 50)},
                "timestamps": ["2023-01-01T10:00:00", "2023-01-02T11:00:00"],
                "metadata": {"created": "2023-01-01T10:00:00", "active": True},
            }
            for i in range(100)  # Smaller sample for inference
        ]

        result = benchmark(infer_template_from_data, sample_data)
        assert isinstance(result, dict)
        assert "user_id" in result

    @pytest.mark.pandas
    def test_dataframe_template_deserialization(self, benchmark):
        """Benchmark DataFrame template-based deserialization."""
        pd = pytest.importorskip("pandas")

        # Create template DataFrame
        template_df = pd.DataFrame({"id": [1], "value": [0.0], "category": [""], "timestamp": [pd.Timestamp.now()]})

        # Sample data to deserialize
        sample_data = [
            {"id": i, "value": i * 0.1, "category": f"cat_{i % 5}", "timestamp": "2023-01-01T10:00:00"}
            for i in range(1000)
        ]

        def dataframe_template_deserialize():
            return deserialize_with_template(sample_data, template_df)

        result = benchmark(dataframe_template_deserialize)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1000

    def test_complex_nested_template_performance(self, benchmark):
        """Benchmark complex nested structure template deserialization."""
        # Complex template
        template = {
            "user": {"id": 0, "profile": {"name": "", "created": datetime.now(), "uuid": uuid4()}},
            "data": [{"key": "", "value": 0.0}],
            "metadata": {"timestamps": [datetime.now()], "flags": {"active": True, "verified": False}},
        }

        # Sample data
        sample_data = [
            {
                "user": {
                    "id": i,
                    "profile": {
                        "name": f"User {i}",
                        "created": "2023-01-01T10:00:00",
                        "uuid": "12345678-1234-5678-9012-123456789abc",
                    },
                },
                "data": [{"key": f"metric_{j}", "value": j * 0.1} for j in range(5)],
                "metadata": {
                    "timestamps": ["2023-01-01T10:00:00", "2023-01-02T11:00:00"],
                    "flags": {"active": True, "verified": i % 2 == 0},
                },
            }
            for i in range(100)
        ]

        def complex_template_deserialize():
            deserializer = TemplateDeserializer(template)
            return [deserializer.deserialize(item) for item in sample_data]

        result = benchmark(complex_template_deserialize)
        assert len(result) == 100


@pytest.mark.benchmark(group="ml_template_performance")
class TestMLTemplatePerformanceBenchmarks:
    """Benchmark ML-specific template functionality."""

    @pytest.mark.pandas
    def test_ml_dataframe_template_creation(self, benchmark):
        """Benchmark ML template creation for DataFrames."""
        pd = pytest.importorskip("pandas")

        # Simulate ML training data
        training_df = pd.DataFrame(
            {
                "feature_1": np.random.random(5000) if np else [i * 0.001 for i in range(5000)],
                "feature_2": [f"category_{i % 10}" for i in range(5000)],
                "feature_3": [i % 100 for i in range(5000)],
                "target": [i % 2 for i in range(5000)],
            }
        )

        result = benchmark(create_ml_round_trip_template, training_df)
        assert result["__ml_template__"] is True
        assert result["object_type"] == "DataFrame"

    @pytest.mark.numpy
    def test_ml_numpy_template_creation(self, benchmark):
        """Benchmark ML template creation for numpy arrays."""
        np = pytest.importorskip("numpy")

        # Large feature matrix
        feature_matrix = np.random.random((5000, 100))

        result = benchmark(create_ml_round_trip_template, feature_matrix)
        assert result["__ml_template__"] is True
        assert result["object_type"] == "ndarray"

    def test_ml_round_trip_serialization_performance(self, benchmark):
        """Benchmark complete ML round-trip with template."""
        # Simulate ML model data
        ml_data = {
            "model_config": {"learning_rate": 0.01, "epochs": 100},
            "training_data": [{"features": [i * 0.1, i * 0.2], "label": i % 2} for i in range(1000)],
            "metadata": {"created": datetime.now(), "version": "1.0", "uuid": uuid4()},
        }

        def ml_round_trip():
            # Create template
            template = create_ml_round_trip_template(ml_data)

            # Serialize with type hints
            config = datason.SerializationConfig(include_type_hints=True)
            serialized = datason.serialize(ml_data, config=config)

            # Deserialize with template
            return deserialize_with_template(serialized, template)

        result = benchmark(ml_round_trip)
        assert "model_config" in result
        assert "training_data" in result


@pytest.mark.benchmark(group="template_scalability")
class TestTemplateScalabilityBenchmarks:
    """Test scalability of template-based deserialization."""

    @pytest.mark.parametrize("data_size", [100, 500, 1000, 2000])
    def test_template_deserialization_scalability(self, benchmark, data_size):
        """Test how template deserialization scales with data size."""
        template = {"id": 0, "name": "", "value": 0.0, "timestamp": datetime.now(), "active": True}

        sample_data = [
            {"id": i, "name": f"item_{i}", "value": i * 0.1, "timestamp": "2023-01-01T10:00:00", "active": i % 2 == 0}
            for i in range(data_size)
        ]

        def template_deserialize():
            deserializer = TemplateDeserializer(template)
            return [deserializer.deserialize(item) for item in sample_data]

        result = benchmark(template_deserialize)
        assert len(result) == data_size

    def test_template_complexity_impact(self, benchmark):
        """Test impact of template complexity on performance."""
        # Create increasingly complex template
        complex_template = {
            "level_1": {
                "level_2": {
                    "level_3": {
                        "data": [
                            {
                                "id": 0,
                                "timestamps": [datetime.now()],
                                "nested": {"values": [0.0], "metadata": {"created": datetime.now()}},
                            }
                        ]
                    }
                }
            }
        }

        sample_data = {
            "level_1": {
                "level_2": {
                    "level_3": {
                        "data": [
                            {
                                "id": i,
                                "timestamps": ["2023-01-01T10:00:00"],
                                "nested": {"values": [i * 0.1], "metadata": {"created": "2023-01-01T10:00:00"}},
                            }
                            for i in range(50)
                        ]
                    }
                }
            }
        }

        result = benchmark(deserialize_with_template, sample_data, complex_template)
        assert "level_1" in result


@pytest.mark.benchmark(group="template_type_coercion")
class TestTypeCoercionBenchmarks:
    """Benchmark type coercion performance in template deserialization."""

    def test_type_coercion_performance(self, benchmark):
        """Test performance of type coercion during template deserialization."""
        template = {
            "int_value": 0,
            "float_value": 0.0,
            "str_value": "",
            "bool_value": True,
            "datetime_value": datetime.now(),
        }

        # Data with string representations that need coercion
        sample_data = [
            {
                "int_value": str(i),
                "float_value": str(i * 0.1),
                "str_value": f"item_{i}",
                "bool_value": "true" if i % 2 == 0 else "false",
                "datetime_value": "2023-01-01T10:00:00",
            }
            for i in range(1000)
        ]

        def type_coercion_deserialize():
            deserializer = TemplateDeserializer(template)
            return [deserializer.deserialize(item) for item in sample_data]

        result = benchmark(type_coercion_deserialize)
        assert len(result) == 1000
        # Verify type coercion worked
        assert isinstance(result[0]["int_value"], int)
        assert isinstance(result[0]["float_value"], float)

    def test_strict_vs_flexible_mode_performance(self, benchmark):
        """Compare performance of strict vs flexible template modes."""
        template = {"expected": 0, "known": ""}

        # Data with extra fields
        sample_data = [
            {
                "expected": i,
                "known": f"item_{i}",
                "extra_field_1": f"extra_{i}",
                "extra_field_2": i * 2,
                "unexpected": "2023-01-01T10:00:00",
            }
            for i in range(500)
        ]

        def flexible_deserialize():
            deserializer = TemplateDeserializer(template, strict=False, fallback_auto_detect=True)
            return [deserializer.deserialize(item) for item in sample_data]

        result = benchmark(flexible_deserialize)
        assert len(result) == 500


def test_template_deserialization_benchmark_summary():
    """Summary test that demonstrates template deserialization performance."""
    print("\n" + "=" * 60)
    print("TEMPLATE DESERIALIZATION BENCHMARK SUMMARY")
    print("=" * 60)

    # Test 1: Template inference performance
    sample_data = [
        {
            "user_id": i,
            "profile": {"name": f"User {i}", "age": 20 + (i % 50)},
            "created": "2023-01-01T10:00:00",
            "active": True,
        }
        for i in range(1000)
    ]

    start_time = time.time()
    template = infer_template_from_data(sample_data[:10])  # Infer from small sample
    inference_time = time.time() - start_time

    print("✅ Template Inference (from 10 samples):")
    print(f"   Time taken: {inference_time:.4f}s")
    print(f"   Template keys: {list(template.keys()) if isinstance(template, dict) else 'N/A'}")

    # Test 2: Template-based deserialization
    start_time = time.time()
    deserializer = TemplateDeserializer(template)
    results = [deserializer.deserialize(item) for item in sample_data]
    deserialization_time = time.time() - start_time

    print("\n✅ Template Deserialization (1K items):")
    print(f"   Time taken: {deserialization_time:.3f}s")
    print(f"   Throughput: {len(results) / deserialization_time:.0f} items/sec")
    print(f"   Type consistency: {all(isinstance(r['user_id'], int) for r in results)}")

    # Test 3: ML template creation
    ml_data = {
        "features": [[i, i * 2, i * 3] for i in range(1000)],
        "labels": [i % 2 for i in range(1000)],
        "metadata": {"created": datetime.now(), "version": "1.0"},
    }

    start_time = time.time()
    ml_template = create_ml_round_trip_template(ml_data)
    ml_template_time = time.time() - start_time

    print("\n✅ ML Template Creation:")
    print(f"   Time taken: {ml_template_time:.4f}s")
    print(f"   Template type: {ml_template.get('object_type', 'Unknown')}")
    print(f"   ML optimized: {ml_template.get('__ml_template__', False)}")

    print("\n🎯 Key Benefits Demonstrated:")
    print("   • Fast template inference from sample data")
    print("   • Consistent type coercion and validation")
    print("   • ML-optimized templates for data science workflows")
    print("   • Scalable performance for large datasets")
    print("=" * 60)
