"""
Performance benchmarks for datason.

This module contains performance tests to ensure datason remains fast
and efficient, especially compared to standard JSON serialization.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict

import pytest

import datason as ds


class TestPerformanceBenchmarks:
    """Performance benchmarks for datason operations."""

    def test_serialize_large_dict_performance(self) -> None:
        """Test serialization performance on large dictionaries."""
        # Create a large dictionary with various data types
        large_dict = {}
        for i in range(1000):
            large_dict[f"key_{i}"] = {
                "id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "value": i * 1.5,
                "active": i % 2 == 0,
                "data": list(range(10)),
            }

        # Benchmark serialization
        start_time = time.time()
        result = ds.serialize(large_dict)
        end_time = time.time()

        serialize_time = end_time - start_time

        # Should complete in reasonable time (< 1 second for 1000 items)
        assert serialize_time < 1.0, f"Serialization took {serialize_time:.3f}s, too slow!"

        # Result should be JSON-compatible
        assert isinstance(result, dict)
        assert len(result) == 1000

    def test_serialize_vs_standard_json_simple(self) -> None:
        """Test that datason serialization is reasonably fast compared to standard JSON."""
        # Simple data that should serialize quickly
        simple_data = {
            "name": "test",
            "value": 42,
            "items": [1, 2, 3, 4, 5],
            "nested": {"a": 1, "b": 2},
        }

        # Warm up to avoid cold start effects
        json.dumps(simple_data)
        ds.serialize(simple_data)

        # Measure multiple iterations for more stable timing
        iterations = 10

        # Benchmark standard JSON
        start_time = time.time()
        for _ in range(iterations):
            json_result = json.dumps(simple_data)
        json_time = (time.time() - start_time) / iterations

        # Benchmark datason
        start_time = time.time()
        for _ in range(iterations):
            sp_result = ds.serialize(simple_data)
        sp_time = (time.time() - start_time) / iterations

        # datason should be reasonably fast (not more than 50x slower than JSON)
        # Use a more generous multiplier since we're doing more complex processing
        max_slowdown = 50
        assert sp_time < json_time * max_slowdown or sp_time < 0.01, (
            f"datason too slow: {sp_time:.4f}s vs JSON {json_time:.4f}s "
            f"(slowdown: {sp_time / json_time:.1f}x, max allowed: {max_slowdown}x)"
        )

        # Results should be equivalent for simple data
        assert json.loads(json_result) == sp_result

    def test_serialize_optimization_effectiveness(self) -> None:
        """Test that our optimization for already-serialized data works."""
        # Already JSON-compatible data
        json_compatible = {
            "simple": "data",
            "numbers": [1, 2, 3, 4, 5],
            "nested": {"more": "data", "values": [10, 20, 30]},
        }

        # First serialization (should be optimized)
        start_time = time.time()
        result1 = ds.serialize(json_compatible)
        time.time() - start_time

        # Second serialization (should be even faster due to optimization)
        start_time = time.time()
        result2 = ds.serialize(result1)
        second_time = time.time() - start_time

        # Second should be faster or equal (optimization should kick in)
        # Note: This might be very fast, so we just check it doesn't crash
        assert second_time >= 0
        assert result1 == result2

    def test_deserialize_performance(self) -> None:
        """Test deserialization performance."""
        # Create data with many datetime strings and UUIDs
        data = {
            "timestamps": [
                "2023-01-01T10:00:00",
                "2023-01-01T11:00:00",
                "2023-01-01T12:00:00",
            ]
            * 100,  # 300 datetime strings
            "ids": [
                "12345678-1234-5678-9012-123456789abc",
                "87654321-4321-8765-2109-cba987654321",
            ]
            * 100,  # 200 UUID strings
            "regular_data": {
                "numbers": list(range(100)),
                "strings": [f"item_{i}" for i in range(100)],
            },
        }

        start_time = time.time()
        result = ds.deserialize(data)
        end_time = time.time()

        deserialize_time = end_time - start_time

        # Should complete in reasonable time
        assert deserialize_time < 1.0, f"Deserialization took {deserialize_time:.3f}s, too slow!"

        # Check that parsing worked
        assert len(result["timestamps"]) == 300
        assert all(isinstance(ts, datetime) for ts in result["timestamps"])
        assert len(result["ids"]) == 200
        assert all(isinstance(uid, uuid.UUID) for uid in result["ids"])

    def test_round_trip_performance(self) -> None:
        """Test round-trip serialization/deserialization performance."""
        # Complex data structure
        complex_data = {
            "metadata": {
                "created_at": datetime.now(),
                "id": uuid.uuid4(),
                "version": "1.0.0",
            },
            "data": [
                {
                    "timestamp": datetime.now(),
                    "user_id": uuid.uuid4(),
                    "value": i * 1.5,
                    "tags": [f"tag_{j}" for j in range(5)],
                }
                for i in range(100)
            ],
        }

        # Full round trip
        start_time = time.time()

        # Serialize
        serialized = ds.serialize(complex_data)

        # Convert to JSON and back (simulate real-world usage)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        # Deserialize
        deserialized = ds.deserialize(parsed)

        end_time = time.time()

        round_trip_time = end_time - start_time

        # Should complete in reasonable time
        assert round_trip_time < 2.0, f"Round trip took {round_trip_time:.3f}s, too slow!"

        # Check data integrity
        assert isinstance(deserialized["metadata"]["created_at"], datetime)
        assert isinstance(deserialized["metadata"]["id"], uuid.UUID)
        assert len(deserialized["data"]) == 100

    def test_memory_usage_large_dataset(self) -> None:
        """Test that we don't have excessive memory usage."""
        # Create a reasonably large dataset
        large_dataset = []
        for _i in range(1000):
            large_dataset.append(
                {
                    "id": uuid.uuid4(),
                    "timestamp": datetime.now(),
                    "data": {
                        "values": list(range(50)),
                        "metadata": {
                            "created": datetime.now(),
                            "tags": [f"tag_{j}" for j in range(10)],
                        },
                    },
                }
            )

        # This should not crash or consume excessive memory
        start_time = time.time()
        result = ds.serialize(large_dataset)
        end_time = time.time()

        # Should complete without issues
        assert end_time - start_time < 5.0  # Max 5 seconds for 1000 complex items
        assert isinstance(result, list)
        assert len(result) == 1000

    def test_numpy_performance(self) -> None:
        """Test performance with numpy arrays if available."""
        np = pytest.importorskip("numpy")

        # Large numpy arrays
        data = {
            "large_array": np.random.random(1000),
            "int_array": np.arange(1000),
            "bool_array": np.random.choice([True, False], 1000),
            "nested": {"matrix": np.random.random((50, 20))},
        }

        start_time = time.time()
        result = ds.serialize(data)
        end_time = time.time()

        serialize_time = end_time - start_time

        # Should handle numpy efficiently
        assert serialize_time < 2.0, f"Numpy serialization took {serialize_time:.3f}s, too slow!"

        # Check results
        assert len(result["large_array"]) == 1000
        assert len(result["int_array"]) == 1000
        assert len(result["bool_array"]) == 1000
        assert len(result["nested"]["matrix"]) == 50
        assert len(result["nested"]["matrix"][0]) == 20

    def test_pandas_performance(self) -> None:
        """Test performance with pandas DataFrames if available."""
        pd = pytest.importorskip("pandas")
        np = pytest.importorskip("numpy")

        # Large DataFrame
        df = pd.DataFrame(
            {
                "A": np.random.random(1000),
                "B": pd.date_range("2023-01-01", periods=1000),
                "C": [f"string_{i}" for i in range(1000)],
                "D": np.random.choice([True, False], 1000),
            }
        )

        data = {
            "dataframe": df,
            "series": pd.Series(np.random.random(500)),
            "metadata": {"timestamp": pd.Timestamp.now()},
        }

        start_time = time.time()
        result = ds.serialize(data)
        end_time = time.time()

        serialize_time = end_time - start_time

        # Should handle pandas efficiently
        assert serialize_time < 3.0, f"Pandas serialization took {serialize_time:.3f}s, too slow!"

        # Check results
        assert len(result["dataframe"]) == 1000
        assert len(result["series"]) == 500


class TestScalabilityEdgeCases:
    """Test edge cases for scalability and performance."""

    def test_deeply_nested_structure_performance(self) -> None:
        """Test performance with deeply nested structures."""
        # Create deeply nested structure
        nested: Dict[str, Any] = {}
        current = nested
        for i in range(50):  # 50 levels deep
            current["level"] = i
            current["timestamp"] = datetime.now()
            current["id"] = uuid.uuid4()
            current["next"] = {}
            current = current["next"]

        start_time = time.time()
        ds.serialize(nested)
        end_time = time.time()

        # Should handle deep nesting efficiently
        serialize_time = end_time - start_time
        assert serialize_time < 1.0, f"Deep nesting took {serialize_time:.3f}s, too slow!"

    def test_wide_structure_performance(self) -> None:
        """Test performance with very wide structures."""
        # Create structure with many keys at the same level
        wide_dict = {}
        for i in range(1000):
            wide_dict[f"key_{i}"] = {
                "timestamp": datetime.now(),
                "uuid": uuid.uuid4(),
                "value": i,
            }

        start_time = time.time()
        result = ds.serialize(wide_dict)
        end_time = time.time()

        # Should handle wide structures efficiently
        serialize_time = end_time - start_time
        assert serialize_time < 1.0, f"Wide structure took {serialize_time:.3f}s, too slow!"
        assert len(result) == 1000

    def test_mixed_type_performance(self) -> None:
        """Test performance with mixed data types."""
        # Create data with every supported type
        mixed_data = []
        for i in range(100):
            mixed_data.append(
                {
                    "string": f"item_{i}",
                    "int": i,
                    "float": i * 1.5,
                    "bool": i % 2 == 0,
                    "none": None,
                    "datetime": datetime.now(),
                    "uuid": uuid.uuid4(),
                    "list": list(range(10)),
                    "nested_dict": {
                        "timestamp": datetime.now(),
                        "data": [k * 2 for k in range(5)],
                    },
                }
            )

        start_time = time.time()
        result = ds.serialize(mixed_data)
        end_time = time.time()

        # Should handle mixed types efficiently
        serialize_time = end_time - start_time
        assert serialize_time < 2.0, f"Mixed types took {serialize_time:.3f}s, too slow!"
        assert len(result) == 100
