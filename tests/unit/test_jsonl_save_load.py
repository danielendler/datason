import tempfile
from pathlib import Path

import numpy as np

import datason

# Legacy tests removed - replaced with modern API tests below


def test_ml_jsonl_roundtrip():
    """Test ML-optimized JSONL serialization with type preservation."""
    ml_data = [{"weights": np.array([1.0, 2.0, 3.0]), "epoch": 1}, {"weights": np.array([1.1, 2.1, 3.1]), "epoch": 2}]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "ml_data.jsonl"

        # Save with ML optimization
        datason.save_ml(ml_data, path)

        # Load with perfect reconstruction
        template = {"weights": np.array([0.0]), "epoch": 0}
        loaded = list(datason.load_perfect_file(path, template))

        # Verify ML types preserved
        assert len(loaded) == 2
        assert isinstance(loaded[0]["weights"], np.ndarray)
        assert np.allclose(loaded[0]["weights"], np.array([1.0, 2.0, 3.0]))
        assert loaded[0]["epoch"] == 1


def test_secure_jsonl_roundtrip():
    """Test secure JSONL serialization with PII redaction."""
    sensitive_data = [
        {"name": "John", "ssn": "123-45-6789", "email": "john@example.com"},
        {"name": "Jane", "ssn": "987-65-4321", "email": "jane@example.com"},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "secure_data.jsonl"

        # Save with security features
        datason.save_secure(sensitive_data, path, redact_pii=True)

        # Load and verify redaction
        loaded = list(datason.load_smart_file(path))

        # Should be wrapped with redaction metadata
        assert "data" in loaded[0]
        assert "redaction_summary" in loaded[0]

        # Verify PII was redacted
        redacted_data = loaded[0]["data"]
        assert redacted_data[0]["ssn"] == "<REDACTED>"
        assert redacted_data[0]["email"] == "<REDACTED>"
        assert redacted_data[0]["name"] == "John"  # Name not redacted


def test_api_jsonl_roundtrip():
    """Test API-safe JSONL serialization."""
    api_data = [
        {"status": "success", "data": [1, 2, 3], "timestamp": "2024-01-01T12:00:00"},
        {"status": "error", "message": "Not found", "code": 404},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "api_data.jsonl"

        # Save API-safe format
        datason.save_api(api_data, path)

        # Load with smart reconstruction
        loaded = list(datason.load_smart_file(path))

        assert len(loaded) == 2
        assert loaded[0]["status"] == "success"
        assert loaded[1]["code"] == 404


def test_chunked_jsonl_save():
    """Test chunked JSONL saving for large datasets."""
    large_data = list(range(10000))

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "large_data.jsonl"

        # Save with chunking
        datason.save_chunked(large_data, path, chunk_size=1000)

        # Load and verify
        loaded = list(datason.load_smart_file(path))

        # Should have chunks, not the original list structure
        assert len(loaded) == 10  # 10 chunks of 1000 items each
        # ML serialization may convert lists to numpy arrays
        assert isinstance(loaded[0], (list, np.ndarray))
        assert len(loaded[0]) == 1000


def test_streaming_ml_save():
    """Test streaming ML data saving."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "streaming_ml.jsonl"

        # Streaming save
        with datason.stream_save_ml(path) as stream:
            for i in range(5):
                epoch_data = {"epoch": i, "weights": np.random.rand(3), "loss": float(i * 0.1)}
                stream.write(epoch_data)

        # Load with template
        template = {"epoch": 0, "weights": np.array([0.0]), "loss": 0.0}
        loaded = list(datason.load_perfect_file(path, template))

        assert len(loaded) == 5
        assert all(isinstance(item["weights"], np.ndarray) for item in loaded)
        assert loaded[2]["epoch"] == 2


def test_compression_support():
    """Test gzip compression with new API."""
    data = [{"test": i} for i in range(100)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "compressed.jsonl.gz"

        # Save compressed
        datason.save_ml(data, path)

        # Verify file is actually compressed
        assert path.exists()
        assert path.suffix == ".gz"

        # Load compressed
        loaded = list(datason.load_smart_file(path))
        assert len(loaded) == 100
        assert loaded[50]["test"] == 50


def test_progressive_loading():
    """Test different loading strategies."""
    mixed_data = [{"simple": "text"}, {"complex": {"nested": {"value": 42}}}, {"array": [1, 2, 3, 4, 5]}]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "mixed_data.jsonl"
        datason.save_api(mixed_data, path)

        # Test basic loading (fastest)
        basic_loaded = list(datason.load_basic_file(path))
        assert len(basic_loaded) == 3

        # Test smart loading (better accuracy)
        smart_loaded = list(datason.load_smart_file(path))
        assert len(smart_loaded) == 3

        # Compare structure (arrays may be converted to numpy arrays)
        assert basic_loaded[0] == smart_loaded[0]  # Simple text
        assert basic_loaded[1] == smart_loaded[1]  # Complex nested
        # Array comparison - handle potential numpy conversion
        basic_array = basic_loaded[2]["array"]
        smart_array = smart_loaded[2]["array"]
        if isinstance(basic_array, np.ndarray) or isinstance(smart_array, np.ndarray):
            # Convert both to lists for comparison
            basic_list = basic_array.tolist() if isinstance(basic_array, np.ndarray) else basic_array
            smart_list = smart_array.tolist() if isinstance(smart_array, np.ndarray) else smart_array
            assert basic_list == smart_list
        else:
            assert basic_array == smart_array
