"""Unit tests for the stream_load functionality in datason."""

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest

from datason import stream_load


def create_test_file(
    data: List[Dict[str, Any]],
    format: str = "jsonl",
    compress: bool = False,
) -> str:
    """Create a test file with the given data and format.

    Args:
        data: List of dictionaries to write to the file
        format: File format ('jsonl' or 'json')
        compress: Whether to gzip compress the file

    Returns:
        Path to the created file
    """
    _, path = tempfile.mkstemp(suffix=f".{format}" + (".gz" if compress else ""))
    try:
        content = "\n".join(json.dumps(item) for item in data) if format == "jsonl" else json.dumps(data)
        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return path
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise


class TestStreamLoad:
    """Test cases for the stream_load function."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> Generator[None, None, None]:
        """Set up and tear down test fixtures.

        Yields:
            None: After setting up test data
        """
        self.test_data = [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
            {"id": 3, "name": "Charlie", "active": True},
        ]
        self.test_files: List[str] = []
        yield
        # Cleanup
        for path in self.test_files:
            if os.path.exists(path):
                os.unlink(path)

    def test_stream_load_jsonl(self) -> None:
        """Test streaming load from a JSONL file."""
        path = create_test_file(self.test_data, format="jsonl")
        self.test_files.append(path)

        with stream_load(path) as stream:
            results = list(stream)

        assert len(results) == len(self.test_data)
        for result, expected in zip(results, self.test_data):
            assert result == expected

    def test_stream_load_json(self) -> None:
        """Test streaming load from a JSON array file."""
        path = create_test_file(self.test_data, format="json")
        self.test_files.append(path)

        with stream_load(path, format="json") as stream:
            results = list(stream)

        assert len(results) == len(self.test_data)
        for result, expected in zip(results, self.test_data):
            assert result == expected

    def test_stream_load_gzipped_jsonl(self) -> None:
        """Test streaming load from a gzipped JSONL file."""
        path = create_test_file(self.test_data, format="jsonl", compress=True)
        self.test_files.append(path)

        with stream_load(path) as stream:
            results = list(stream)

        assert len(results) == len(self.test_data)
        for result, expected in zip(results, self.test_data):
            assert result == expected

    def test_stream_load_with_chunk_processor(self) -> None:
        """Test streaming load with a chunk processor function."""
        path = create_test_file(self.test_data, format="jsonl")
        self.test_files.append(path)

        def process_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single chunk by adding a processed flag.

            Args:
                chunk: The input chunk to process

            Returns:
                The processed chunk with added 'processed' flag
            """
            chunk["processed"] = True
            return chunk

        with stream_load(path, chunk_processor=process_chunk) as stream:
            results = list(stream)

        assert len(results) == len(self.test_data)
        for result, expected in zip(results, self.test_data):
            assert result["processed"] is True
            assert result["id"] == expected["id"]
            assert result["name"] == expected["name"]
            assert result["active"] == expected["active"]

    def test_stream_load_empty_file(self) -> None:
        """Test streaming load from an empty file."""
        path = create_test_file([], format="jsonl")
        self.test_files.append(path)

        with stream_load(path) as stream:
            results = list(stream)

        assert len(results) == 0

    def test_stream_load_invalid_format(self) -> None:
        """Test streaming load with an invalid format."""
        path = create_test_file(self.test_data, format="jsonl")
        self.test_files.append(path)

        with pytest.raises(ValueError):
            with stream_load(path, format="invalid"):
                pass

    def test_stream_load_nonexistent_file(self) -> None:
        """Test streaming load from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            with stream_load("/nonexistent/file.jsonl"):
                pass

    def test_stream_load_items_yielded_property(self) -> None:
        """Test the items_yielded property of the streaming deserializer."""
        path = create_test_file(self.test_data, format="jsonl")
        self.test_files.append(path)

        with stream_load(path) as stream:
            items_yielded = []
            for i, item in enumerate(stream, 1):
                items_yielded.append(stream.items_yielded)
                assert stream.items_yielded == i

        assert items_yielded == [1, 2, 3]

    def test_stream_load_with_path_object(self) -> None:
        """Test streaming load with a Path object instead of string."""
        path = Path(create_test_file(self.test_data, format="jsonl"))
        self.test_files.append(str(path))

        with stream_load(path) as stream:
            results = list(stream)

        assert len(results) == len(self.test_data)
        for result, expected in zip(results, self.test_data):
            assert result == expected
