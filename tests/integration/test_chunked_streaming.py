"""Tests for chunked processing and streaming capabilities (v0.4.0)."""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

import datason
from datason.core_new import (
    ChunkedSerializationResult,
    StreamingSerializer,
    deserialize_chunked_file,
    estimate_memory_usage,
    serialize_chunked,
    stream_serialize,
)

# Optional imports for comprehensive testing
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


class TestChunkedSerialization:
    """Test chunked serialization functionality."""

    def test_serialize_chunked_basic_list(self):
        """Test basic chunked serialization of a list."""
        large_list = list(range(1000))
        chunk_size = 100

        result = serialize_chunked(large_list, chunk_size=chunk_size)

        # Fix for CI: Use both isinstance and string comparison for robustness
        assert (
            isinstance(result, ChunkedSerializationResult) or result.__class__.__name__ == "ChunkedSerializationResult"
        )
        assert result.metadata["total_chunks"] == 10
        assert result.metadata["total_items"] == 1000
        assert result.metadata["chunk_size"] == 100
        assert result.metadata["object_type"] == "list"
        assert result.metadata["chunking_strategy"] == "sequence"

        # Test converting to list
        chunks = result.to_list()
        assert len(chunks) == 10
        assert all(len(chunk) == 100 for chunk in chunks)

        # Verify data integrity
        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk)
        assert reconstructed == large_list

    def test_serialize_chunked_tuple(self):
        """Test chunked serialization of a tuple."""
        large_tuple = tuple(range(500))
        chunk_size = 50

        result = serialize_chunked(large_tuple, chunk_size=chunk_size)

        # Fix for CI: Use string comparison as fallback
        assert (
            isinstance(result, ChunkedSerializationResult) or result.__class__.__name__ == "ChunkedSerializationResult"
        )
        assert result.metadata["total_chunks"] == 10
        assert result.metadata["object_type"] == "tuple"

        chunks = result.to_list()
        assert len(chunks) == 10

    @pytest.mark.pandas
    def test_serialize_chunked_dataframe(self):
        """Test chunked serialization of a DataFrame."""
        pd = pytest.importorskip("pandas")

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": [f"item_{i}" for i in range(1000)],
                "timestamp": [datetime.now() for _ in range(1000)],
            }
        )

        chunk_size = 100
        result = serialize_chunked(df, chunk_size=chunk_size)

        assert result.metadata["total_chunks"] == 10
        assert result.metadata["total_rows"] == 1000
        assert result.metadata["total_columns"] == 3
        assert result.metadata["object_type"] == "pandas.DataFrame"
        assert result.metadata["chunking_strategy"] == "dataframe_rows"
        assert set(result.metadata["columns"]) == {"id", "value", "timestamp"}

        # Test chunk content
        chunks = result.to_list()
        assert len(chunks) == 10

        # Each chunk should be a list of records (default DataFrame serialization)
        for chunk in chunks:
            assert isinstance(chunk, list)
            assert len(chunk) == 100

    @pytest.mark.numpy
    def test_serialize_chunked_numpy_array(self):
        """Test chunked serialization of a numpy array."""
        np = pytest.importorskip("numpy")

        arr = np.random.random((1000, 5))
        chunk_size = 100

        result = serialize_chunked(arr, chunk_size=chunk_size)

        assert result.metadata["total_chunks"] == 10
        assert result.metadata["total_items"] == 1000
        assert result.metadata["object_type"] == "numpy.ndarray"
        assert result.metadata["chunking_strategy"] == "array_rows"
        assert result.metadata["shape"] == (1000, 5)

        chunks = result.to_list()
        assert len(chunks) == 10

        # Each chunk should be a nested list (default numpy serialization)
        for chunk in chunks:
            assert isinstance(chunk, list)
            assert len(chunk) == 100
            assert len(chunk[0]) == 5  # Second dimension preserved

    def test_serialize_chunked_dict(self):
        """Test chunked serialization of a dictionary."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(500)}
        chunk_size = 50

        result = serialize_chunked(large_dict, chunk_size=chunk_size)

        assert result.metadata["total_chunks"] == 10
        assert result.metadata["total_items"] == 500
        assert result.metadata["object_type"] == "dict"
        assert result.metadata["chunking_strategy"] == "dict_items"

        chunks = result.to_list()
        assert len(chunks) == 10

        # Verify all original data is preserved
        reconstructed = {}
        for chunk in chunks:
            assert isinstance(chunk, dict)
            reconstructed.update(chunk)

        assert reconstructed == large_dict

    def test_serialize_chunked_non_chunnable_object(self):
        """Test chunked serialization with non-chunnable objects."""
        simple_obj = {"a": 1, "b": "hello"}

        result = serialize_chunked(simple_obj, chunk_size=10)

        # Simple dicts get chunked by items now, not treated as single objects
        assert result.metadata["total_chunks"] == 1
        assert result.metadata["chunking_strategy"] == "dict_items"  # Changed expectation

        chunks = result.to_list()
        assert len(chunks) == 1
        assert chunks[0] == simple_obj

    def test_chunked_result_save_to_file_jsonl(self):
        """Test saving chunked results to JSONL file."""
        data = list(range(100))
        result = serialize_chunked(data, chunk_size=25)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_chunks.jsonl"
            result.save_to_file(file_path, format="jsonl")

            assert file_path.exists()

            # Read and verify content
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 4  # 4 chunks

            # Parse each line
            for line in lines:
                chunk = json.loads(line.strip())
                assert isinstance(chunk, list)
                assert len(chunk) == 25

    def test_chunked_result_save_to_file_json(self):
        """Test saving chunked results to JSON file."""
        data = list(range(50))
        result = serialize_chunked(data, chunk_size=25)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_chunks.json"
            result.save_to_file(file_path, format="json")

            assert file_path.exists()

            # Read and verify content
            with file_path.open("r") as f:
                content = json.load(f)

            assert "chunks" in content
            assert "metadata" in content
            assert len(content["chunks"]) == 2
            assert content["metadata"]["total_chunks"] == 2


class TestStreamingSerialization:
    """Test streaming serialization functionality."""

    def test_streaming_serializer_jsonl(self):
        """Test streaming serializer with JSONL format."""
        data_items = [{"id": i, "value": f"item_{i}", "uuid": str(uuid.uuid4())} for i in range(100)]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "stream_test.jsonl"

            with stream_serialize(file_path, format="jsonl") as stream:
                for item in data_items:
                    stream.write(item)

            assert file_path.exists()

            # Read and verify
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 100

            # Parse and verify content
            for i, line in enumerate(lines):
                item = json.loads(line.strip())
                assert item["id"] == i
                assert item["value"] == f"item_{i}"

    def test_streaming_serializer_json(self):
        """Test streaming serializer with JSON format."""
        data_items = [{"number": i} for i in range(10)]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "stream_test.json"

            with stream_serialize(file_path, format="json") as stream:
                for item in data_items:
                    stream.write(item)

            assert file_path.exists()

            # Read and verify
            with file_path.open("r") as f:
                content = json.load(f)

            assert "data" in content
            assert "metadata" in content
            assert content["metadata"]["items_written"] == 10
            assert len(content["data"]) == 10

    def test_streaming_serializer_write_chunked(self):
        """Test streaming serializer with chunked writing."""
        large_data = list(range(1000))

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "chunked_stream.jsonl"

            with stream_serialize(file_path, format="jsonl") as stream:
                stream.write_chunked(large_data, chunk_size=100)

            assert file_path.exists()

            # Count lines (should be 10 chunks)
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 10  # 10 chunks of 100 items each

    @pytest.mark.pandas
    def test_streaming_serializer_dataframe(self):
        """Test streaming serializer with DataFrames."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"x": range(100), "y": [f"text_{i}" for i in range(100)]})

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "df_stream.jsonl"

            with stream_serialize(file_path, format="jsonl") as stream:
                stream.write_chunked(df, chunk_size=25)

            assert file_path.exists()

            # Should have 4 chunks
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 4

    def test_streaming_serializer_error_handling(self):
        """Test error handling in streaming serializer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "error_test.jsonl"

            # Test writing outside context manager
            stream = StreamingSerializer(file_path)
            with pytest.raises(RuntimeError, match="not in context manager"):
                stream.write({"test": "data"})

    def test_streaming_serializer_custom_config(self):
        """Test streaming serializer with custom configuration."""
        data_items = [{"timestamp": datetime.now(), "value": 123.456}]

        config = datason.get_performance_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "config_test.jsonl"

            with stream_serialize(file_path, config=config, format="jsonl") as stream:
                for item in data_items:
                    stream.write(item)

            assert file_path.exists()

            # Verify timestamp was serialized according to performance config
            with file_path.open("r") as f:
                content = json.loads(f.readline().strip())

            # Performance config uses Unix timestamps
            assert isinstance(content["timestamp"], (int, float))


class TestDeserializeChunkedFile:
    """Test deserializing chunked files."""

    def test_deserialize_chunked_file_jsonl(self):
        """Test deserializing JSONL chunked files."""
        # Create test data and save as chunked file
        data = list(range(100))
        result = serialize_chunked(data, chunk_size=25)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.jsonl"
            result.save_to_file(file_path, format="jsonl")

            # Deserialize chunks
            chunks = list(deserialize_chunked_file(file_path, format="jsonl"))

            assert len(chunks) == 4
            assert all(len(chunk) == 25 for chunk in chunks)

            # Reconstruct original data
            reconstructed = []
            for chunk in chunks:
                reconstructed.extend(chunk)

            assert reconstructed == data

    def test_deserialize_chunked_file_json(self):
        """Test deserializing JSON chunked files."""
        data = list(range(50))
        result = serialize_chunked(data, chunk_size=25)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            result.save_to_file(file_path, format="json")

            # Deserialize chunks
            chunks = list(deserialize_chunked_file(file_path, format="json"))

            assert len(chunks) == 2
            assert all(len(chunk) == 25 for chunk in chunks)

    def test_deserialize_chunked_file_with_processor(self):
        """Test deserializing chunked files with custom chunk processor."""
        data = list(range(100))
        result = serialize_chunked(data, chunk_size=25)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.jsonl"
            result.save_to_file(file_path, format="jsonl")

            # Define processor that doubles each value
            def double_processor(chunk):
                return [x * 2 for x in chunk]

            # Deserialize with processor
            processed_chunks = list(
                deserialize_chunked_file(file_path, format="jsonl", chunk_processor=double_processor)
            )

            assert len(processed_chunks) == 4

            # Verify processing was applied
            reconstructed = []
            for chunk in processed_chunks:
                reconstructed.extend(chunk)

            expected = [x * 2 for x in data]
            assert reconstructed == expected

    def test_deserialize_chunked_file_invalid_format(self):
        """Test error handling for invalid format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("dummy content")

            with pytest.raises(ValueError, match="Unsupported format"):
                list(deserialize_chunked_file(file_path, format="txt"))


class TestMemoryEstimation:
    """Test memory usage estimation functionality."""

    def test_estimate_memory_usage_list(self):
        """Test memory estimation for lists."""
        test_list = list(range(1000))
        stats = estimate_memory_usage(test_list)

        assert stats["object_type"] == "list"
        assert stats["item_count"] == 1000
        assert stats["object_size_mb"] > 0
        assert stats["estimated_serialized_mb"] > 0
        assert stats["recommended_chunk_size"] > 0
        assert stats["recommended_chunks"] >= 1

    @pytest.mark.pandas
    def test_estimate_memory_usage_dataframe(self):
        """Test memory estimation for DataFrames."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"a": range(10000), "b": [f"text_{i}" for i in range(10000)]})

        stats = estimate_memory_usage(df)

        assert stats["object_type"] == "DataFrame"
        assert stats["item_count"] == 10000
        assert stats["object_size_mb"] > 0
        assert stats["recommended_chunk_size"] > 0

    @pytest.mark.numpy
    def test_estimate_memory_usage_numpy_array(self):
        """Test memory estimation for numpy arrays."""
        np = pytest.importorskip("numpy")

        arr = np.random.random((5000, 10))
        stats = estimate_memory_usage(arr)

        assert stats["object_type"] == "ndarray"
        assert stats["item_count"] == 5000
        assert stats["object_size_mb"] > 0

    def test_estimate_memory_usage_dict(self):
        """Test memory estimation for dictionaries."""
        test_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        stats = estimate_memory_usage(test_dict)

        assert stats["object_type"] == "dict"
        assert stats["item_count"] == 1000
        assert stats["recommended_chunk_size"] > 0

    def test_estimate_memory_usage_single_object(self):
        """Test memory estimation for single objects."""
        test_obj = {"single": "object"}
        stats = estimate_memory_usage(test_obj)

        assert stats["object_type"] == "dict"
        assert stats["item_count"] == 1
        assert stats["recommended_chunks"] == 1


class TestChunkedStreamingIntegration:
    """Test integration between chunked processing and streaming."""

    def test_full_workflow_large_dataset(self):
        """Test complete workflow with large dataset."""
        # Create large test dataset
        large_dataset = [
            {"id": i, "data": list(range(i % 10)), "timestamp": datetime.now(), "uuid": str(uuid.uuid4())}
            for i in range(1000)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Use streaming to write chunked data
            output_file = Path(temp_dir) / "large_dataset.jsonl"

            with stream_serialize(output_file, format="jsonl") as stream:
                stream.write_chunked(large_dataset, chunk_size=100)

            # Step 2: Read back and verify
            chunks = list(deserialize_chunked_file(output_file, format="jsonl"))

            assert len(chunks) == 10  # 1000 items / 100 per chunk

            # Reconstruct and verify
            reconstructed = []
            for chunk in chunks:
                reconstructed.extend(chunk)

            assert len(reconstructed) == 1000

            # Verify structure is preserved
            for i, item in enumerate(reconstructed):
                assert item["id"] == i
                assert len(item["data"]) == i % 10

    @pytest.mark.pandas
    def test_dataframe_chunked_streaming_roundtrip(self):
        """Test DataFrame chunked streaming round-trip."""
        pd = pytest.importorskip("pandas")

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "id": range(500),
                "category": [f"cat_{i % 5}" for i in range(500)],
                "value": [i * 1.5 for i in range(500)],
                "timestamp": [datetime.now() for _ in range(500)],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Stream DataFrame in chunks
            output_file = Path(temp_dir) / "dataframe_chunks.jsonl"

            with stream_serialize(output_file, format="jsonl") as stream:
                stream.write_chunked(df, chunk_size=50)

            # Read back chunks
            chunks = list(deserialize_chunked_file(output_file, format="jsonl"))

            assert len(chunks) == 10  # 500 rows / 50 per chunk

            # Each chunk should be a list of records
            for chunk in chunks:
                assert isinstance(chunk, list)
                assert len(chunk) == 50
                assert all(isinstance(record, dict) for record in chunk)

    def test_memory_bounded_processing(self):
        """Test memory-bounded processing workflow."""

        # This simulates processing a dataset larger than memory
        def generate_large_data():
            """Generator that simulates large data without storing it all."""
            for i in range(10000):
                yield {"id": i, "data": f"item_{i}", "processed": False}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "memory_bounded.jsonl"

            # Process in chunks without loading all data
            with stream_serialize(output_file, format="jsonl") as stream:
                chunk = []
                chunk_size = 100

                for item in generate_large_data():
                    # Mark as processed
                    item["processed"] = True
                    chunk.append(item)

                    if len(chunk) >= chunk_size:
                        # Write chunk and clear memory
                        stream.write(chunk)
                        chunk = []

                # Write remaining items
                if chunk:
                    stream.write(chunk)

            # Verify file was created and has correct number of chunks
            with output_file.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 100  # 10000 items / 100 per chunk = 100 chunks

            # Verify sample data
            first_chunk = json.loads(lines[0])
            assert len(first_chunk) == 100
            assert all(item["processed"] for item in first_chunk)


class TestConfigurationWithChunking:
    """Test chunked processing with different configurations."""

    def test_chunked_with_performance_config(self):
        """Test chunked processing with performance configuration."""
        data = [{"timestamp": datetime.now(), "value": i} for i in range(100)]
        config = datason.get_performance_config()

        result = serialize_chunked(data, chunk_size=25, config=config)
        chunks = result.to_list()

        # Performance config should use Unix timestamps
        sample_item = chunks[0][0]
        assert isinstance(sample_item["timestamp"], (int, float))

    def test_chunked_with_ml_config(self):
        """Test chunked processing with ML configuration."""
        data = [{"id": i, "features": [i * 0.1, i * 0.2]} for i in range(100)]
        config = datason.get_ml_config()

        result = serialize_chunked(data, chunk_size=25, config=config)
        chunks = result.to_list()

        assert len(chunks) == 4
        assert all(len(chunk) == 25 for chunk in chunks)

    def test_streaming_with_financial_config(self):
        """Test streaming with custom financial configuration."""
        from decimal import Decimal

        from datason.config import DateFormat, SerializationConfig

        financial_data = [{"price": Decimal("123.45"), "volume": 1000, "timestamp": datetime.now()} for _ in range(50)]

        # Create custom financial config (replaced removed preset)
        config = SerializationConfig(
            preserve_decimals=True,
            date_format=DateFormat.UNIX_MS,
            ensure_ascii=True,
            check_if_serialized=True,
            include_type_hints=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "financial_stream.jsonl"

            with stream_serialize(file_path, config=config, format="jsonl") as stream:
                for item in financial_data:
                    stream.write(item)

            # Verify financial precision is preserved
            with file_path.open("r") as f:
                sample_line = f.readline()
                item = json.loads(sample_line)

                # Financial config preserves decimals with new format
                assert "price" in item
                assert isinstance(item["price"], dict)  # Decimal metadata
                assert item["price"]["__datason_type__"] == "decimal.Decimal"
