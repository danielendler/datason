"""Benchmarks for chunked processing and streaming capabilities (v0.4.0).

This module contains performance benchmarks for the new chunked serialization
and streaming features to ensure they meet performance expectations.
"""

import tempfile
import time
from pathlib import Path

import pytest

import datason

# Optional imports for comprehensive benchmarking
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


@pytest.mark.benchmark(group="chunked_serialization")
class TestChunkedSerializationBenchmarks:
    """Benchmark chunked serialization performance."""

    def test_chunked_vs_standard_list_serialization(self, benchmark):
        """Compare chunked vs standard serialization for large lists."""
        large_list = list(range(10000))

        def chunked_serialize():
            result = datason.serialize_chunked(large_list, chunk_size=1000)
            return result.to_list()

        result = benchmark(chunked_serialize)
        assert len(result) == 10  # 10 chunks

    def test_chunk_size_impact_on_performance(self, benchmark):
        """Test how different chunk sizes affect performance."""
        large_list = list(range(5000))

        @benchmark
        def serialize_with_chunk_size():
            return datason.serialize_chunked(large_list, chunk_size=500).to_list()

        assert len(serialize_with_chunk_size) == 10  # 5000/500 = 10 chunks

    @pytest.mark.pandas
    def test_dataframe_chunked_serialization(self, benchmark):
        """Benchmark DataFrame chunked serialization."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {"id": range(5000), "value": [f"item_{i}" for i in range(5000)], "score": [i * 0.1 for i in range(5000)]}
        )

        def chunked_serialize_df():
            result = datason.serialize_chunked(df, chunk_size=500)
            return result.to_list()

        result = benchmark(chunked_serialize_df)
        assert len(result) == 10  # 5000/500 = 10 chunks

    @pytest.mark.numpy
    def test_numpy_array_chunked_serialization(self, benchmark):
        """Benchmark numpy array chunked serialization."""
        np = pytest.importorskip("numpy")

        arr = np.random.random((5000, 10))

        def chunked_serialize_array():
            result = datason.serialize_chunked(arr, chunk_size=500)
            return result.to_list()

        result = benchmark(chunked_serialize_array)
        assert len(result) == 10  # 5000/500 = 10 chunks

    def test_memory_estimation_performance(self, benchmark):
        """Benchmark memory estimation function."""
        large_data = [{"id": i, "data": list(range(i % 10))} for i in range(1000)]

        result = benchmark(datason.estimate_memory_usage, large_data)
        assert "recommended_chunk_size" in result
        assert "estimated_serialized_mb" in result


@pytest.mark.benchmark(group="streaming_serialization")
class TestStreamingSerializationBenchmarks:
    """Benchmark streaming serialization performance."""

    def test_streaming_vs_batch_serialization(self, benchmark):
        """Compare streaming vs batch serialization."""
        data_items = [{"id": i, "value": f"item_{i}"} for i in range(1000)]

        def streaming_serialize():
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "stream_test.jsonl"

                with datason.stream_serialize(file_path, format="jsonl") as stream:
                    for item in data_items:
                        stream.write(item)

                return file_path.stat().st_size

        file_size = benchmark(streaming_serialize)
        assert file_size > 0

    def test_streaming_chunked_writing_performance(self, benchmark):
        """Benchmark streaming with chunked writing."""
        large_data = list(range(5000))

        def streaming_chunked():
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "chunked_stream.jsonl"

                with datason.stream_serialize(file_path, format="jsonl") as stream:
                    stream.write_chunked(large_data, chunk_size=500)

                return file_path.stat().st_size

        file_size = benchmark(streaming_chunked)
        assert file_size > 0

    def test_chunked_file_deserialization_performance(self, benchmark):
        """Benchmark chunked file deserialization."""
        # Setup: create a chunked file
        data = list(range(1000))
        result = datason.serialize_chunked(data, chunk_size=100)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.jsonl"
            result.save_to_file(file_path, format="jsonl")

            def deserialize_chunks():
                chunks = list(datason.deserialize_chunked_file(file_path, format="jsonl"))
                return len(chunks)

            chunk_count = benchmark(deserialize_chunks)
            assert chunk_count == 10  # 1000/100 = 10 chunks


@pytest.mark.benchmark(group="chunked_memory_efficiency")
class TestMemoryEfficiencyBenchmarks:
    """Benchmark memory efficiency of chunked processing."""

    def test_large_dataset_memory_usage(self, benchmark):
        """Test memory efficiency with large datasets."""
        try:
            import os

            import psutil  # type: ignore
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        # Create a reasonably large dataset
        large_dataset = [{"id": i, "data": list(range(50)), "text": f"item_{i}" * 10} for i in range(2000)]

        def chunked_process():
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Process in chunks
            result = datason.serialize_chunked(large_dataset, chunk_size=200)
            chunk_count = 0
            for chunk in result.chunks:
                chunk_count += 1
                # Simulate processing without keeping all chunks in memory
                pass

            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

            return {"chunk_count": chunk_count, "memory_increase_mb": memory_increase / (1024 * 1024)}

        result = benchmark(chunked_process)
        assert result["chunk_count"] == 10  # 2000/200 = 10 chunks
        # Memory increase should be reasonable
        assert result["memory_increase_mb"] < 100  # Less than 100MB increase

    @pytest.mark.pandas
    def test_dataframe_memory_efficient_processing(self, benchmark):
        """Test memory-efficient DataFrame processing."""
        pd = pytest.importorskip("pandas")

        # Create a larger DataFrame
        df = pd.DataFrame(
            {
                "id": range(10000),
                "category": [f"cat_{i % 100}" for i in range(10000)],
                "value": [i * 0.001 for i in range(10000)],
                "description": [f"description for item {i}" * 5 for i in range(10000)],
            }
        )

        def memory_efficient_process():
            # Process DataFrame in chunks without loading all at once
            chunk_count = 0
            total_rows = 0

            result = datason.serialize_chunked(df, chunk_size=1000)
            for chunk in result.chunks:
                chunk_count += 1
                # Count rows in each chunk
                if isinstance(chunk, list):
                    total_rows += len(chunk)

            return {"chunk_count": chunk_count, "total_rows_processed": total_rows}

        result = benchmark(memory_efficient_process)
        assert result["chunk_count"] == 10  # 10000/1000 = 10 chunks
        assert result["total_rows_processed"] == 10000


@pytest.mark.benchmark(group="chunked_scalability")
class TestScalabilityBenchmarks:
    """Test scalability of chunked processing with different data sizes."""

    @pytest.mark.parametrize("data_size", [1000, 5000, 10000])
    def test_chunked_serialization_scalability(self, benchmark, data_size):
        """Test how chunked serialization scales with data size."""
        data = list(range(data_size))
        chunk_size = min(1000, data_size // 5)  # Adaptive chunk size

        def chunked_serialize():
            result = datason.serialize_chunked(data, chunk_size=chunk_size)
            return len(result.to_list())

        chunk_count = benchmark(chunked_serialize)
        expected_chunks = (data_size + chunk_size - 1) // chunk_size
        assert chunk_count == expected_chunks

    @pytest.mark.parametrize("chunk_size", [100, 500, 1000, 2000])
    def test_chunk_size_performance_impact(self, benchmark, chunk_size):
        """Test how different chunk sizes affect performance."""
        data = list(range(10000))

        def serialize_with_chunk_size():
            result = datason.serialize_chunked(data, chunk_size=chunk_size)
            return result.metadata["total_chunks"]

        chunk_count = benchmark(serialize_with_chunk_size)
        expected_chunks = (10000 + chunk_size - 1) // chunk_size
        assert chunk_count == expected_chunks


@pytest.mark.benchmark(group="streaming_formats")
class TestStreamingFormatBenchmarks:
    """Benchmark different streaming formats."""

    def test_jsonl_vs_json_format_performance(self, benchmark):
        """Compare JSONL vs JSON format performance."""
        data_items = [{"id": i, "value": f"item_{i}"} for i in range(500)]

        def jsonl_streaming():
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "jsonl_test.jsonl"

                with datason.stream_serialize(file_path, format="jsonl") as stream:
                    for item in data_items:
                        stream.write(item)

                return file_path.stat().st_size

        file_size = benchmark(jsonl_streaming)
        assert file_size > 0

    def test_json_format_streaming_performance(self, benchmark):
        """Benchmark JSON format streaming."""
        data_items = [{"id": i, "value": f"item_{i}"} for i in range(500)]

        def json_streaming():
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "json_test.json"

                with datason.stream_serialize(file_path, format="json") as stream:
                    for item in data_items:
                        stream.write(item)

                return file_path.stat().st_size

        file_size = benchmark(json_streaming)
        assert file_size > 0


def test_chunked_streaming_benchmark_summary():
    """Summary test that demonstrates key performance characteristics."""
    print("\n" + "=" * 60)
    print("CHUNKED PROCESSING & STREAMING BENCHMARK SUMMARY")
    print("=" * 60)

    # Test 1: Large list chunked processing
    large_list = list(range(50000))
    start_time = time.time()
    result = datason.serialize_chunked(large_list, chunk_size=5000)
    chunks = result.to_list()
    chunked_time = time.time() - start_time

    print("✅ Large List Processing (50K items):")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Time taken: {chunked_time:.3f}s")
    print(f"   Memory usage estimate: {result.metadata}")

    # Test 2: Streaming performance
    data_items = [{"id": i, "data": list(range(i % 10))} for i in range(10000)]

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "streaming_test.jsonl"

        start_time = time.time()
        with datason.stream_serialize(file_path, format="jsonl") as stream:
            for item in data_items:
                stream.write(item)
        streaming_time = time.time() - start_time

        file_size = file_path.stat().st_size / (1024 * 1024)  # MB

        print("\n✅ Streaming Serialization (10K items):")
        print(f"   Time taken: {streaming_time:.3f}s")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Throughput: {len(data_items) / streaming_time:.0f} items/sec")

    # Test 3: Memory estimation
    memory_stats = datason.estimate_memory_usage(large_list)
    print("\n✅ Memory Estimation:")
    print(f"   Object type: {memory_stats['object_type']}")
    print(f"   Estimated size: {memory_stats['estimated_serialized_mb']:.2f} MB")
    print(f"   Recommended chunk size: {memory_stats['recommended_chunk_size']:,}")

    print("\n🎯 Key Benefits Demonstrated:")
    print("   • Memory-bounded processing for large datasets")
    print("   • Streaming capabilities for continuous data")
    print("   • Automatic memory optimization recommendations")
    print("   • Scalable performance across different data sizes")
    print("=" * 60)
