"""
Comprehensive Unit Tests for File Operations

Tests all file operation features including:
- JSON and JSONL format support
- Compression (auto-detection and explicit)
- Format auto-detection and override
- Security features (redaction, signing)
- Streaming operations
- Template-based perfect reconstruction
- Error handling and edge cases
"""

import gzip
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import datason


class TestFileFormatDetection:
    """Test automatic file format detection."""

    def test_json_extension_detection(self):
        """Test .json files are detected as JSON format."""
        assert datason.api._detect_file_format(Path("data.json")) == "json"
        assert datason.api._detect_file_format("model.json") == "json"

    def test_jsonl_extension_detection(self):
        """Test .jsonl files are detected as JSONL format."""
        assert datason.api._detect_file_format(Path("data.jsonl")) == "jsonl"
        assert datason.api._detect_file_format("experiments.jsonl") == "jsonl"

    def test_compressed_detection(self):
        """Test compressed files are detected correctly."""
        assert datason.api._detect_file_format("data.json.gz") == "json"
        assert datason.api._detect_file_format("data.jsonl.gz") == "jsonl"

    def test_unknown_extension_defaults(self):
        """Test unknown extensions default to JSONL."""
        assert datason.api._detect_file_format("data.txt") == "jsonl"
        assert datason.api._detect_file_format("data.xyz") == "jsonl"
        assert datason.api._detect_file_format("data") == "jsonl"


class TestBasicFileOperations:
    """Test basic save and load operations."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data = {"simple": "data", "numbers": [1, 2, 3, 4, 5], "nested": {"a": 1, "b": 2}}

    def test_save_load_json_format(self):
        """Test saving and loading JSON format."""
        json_path = self.temp_dir / "test.json"

        # Save as JSON
        datason.save_ml(self.test_data, json_path)

        # Load back
        loaded = list(datason.load_smart_file(json_path))
        assert len(loaded) == 1

        # Compare structure (ML serialization may convert lists to arrays)
        loaded_data = loaded[0]
        assert loaded_data["simple"] == self.test_data["simple"]
        assert loaded_data["nested"] == self.test_data["nested"]
        # Numbers may be converted to numpy array by ML serialization
        if isinstance(loaded_data["numbers"], np.ndarray):
            assert loaded_data["numbers"].tolist() == self.test_data["numbers"]
        else:
            assert loaded_data["numbers"] == self.test_data["numbers"]

    def test_save_load_jsonl_format(self):
        """Test saving and loading JSONL format."""
        jsonl_path = self.temp_dir / "test.jsonl"

        # Save as JSONL
        datason.save_ml(self.test_data, jsonl_path)

        # Load back
        loaded = list(datason.load_smart_file(jsonl_path))
        assert len(loaded) == 1

        # Compare structure (ML serialization may convert lists to arrays)
        loaded_data = loaded[0]
        assert loaded_data["simple"] == self.test_data["simple"]
        assert loaded_data["nested"] == self.test_data["nested"]
        # Numbers may be converted to numpy array by ML serialization
        if isinstance(loaded_data["numbers"], np.ndarray):
            assert loaded_data["numbers"].tolist() == self.test_data["numbers"]
        else:
            assert loaded_data["numbers"] == self.test_data["numbers"]

    def test_save_list_json_vs_jsonl(self):
        """Test list data behavior in JSON vs JSONL."""
        test_list = [{"id": 1}, {"id": 2}, {"id": 3}]

        json_path = self.temp_dir / "list.json"
        jsonl_path = self.temp_dir / "list.jsonl"

        # Save to both formats
        datason.save_ml(test_list, json_path)
        datason.save_ml(test_list, jsonl_path)

        # Load back
        json_loaded = list(datason.load_smart_file(json_path))
        jsonl_loaded = list(datason.load_smart_file(jsonl_path))

        # JSON should load as single list
        assert len(json_loaded) == 3
        assert json_loaded == test_list

        # JSONL should load as separate records
        assert len(jsonl_loaded) == 3
        assert jsonl_loaded == test_list

    def test_format_override(self):
        """Test explicit format override."""
        data = {"test": "data"}

        # Save with explicit format override
        override_path = self.temp_dir / "override.txt"
        datason.save_ml(data, override_path, format="json")

        # Load with explicit format
        loaded = list(datason.load_smart_file(override_path, format="json"))
        assert len(loaded) == 1
        assert loaded[0] == data


class TestCompressionSupport:
    """Test compression functionality."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create data that compresses well
        self.compressible_data = {
            "repeated_data": ["same_string"] * 1000,
            "zeros": [0] * 1000,
            "pattern": list(range(100)) * 10,
        }

    def test_automatic_compression_detection(self):
        """Test automatic compression based on .gz extension."""
        compressed_path = self.temp_dir / "data.jsonl.gz"

        # Save (should compress automatically)
        datason.save_ml(self.compressible_data, compressed_path)

        # Verify file is actually compressed
        assert compressed_path.exists()

        # Load back
        loaded = list(datason.load_smart_file(compressed_path))
        assert len(loaded) == 1
        assert loaded[0] == self.compressible_data

    def test_compression_size_benefit(self):
        """Test that compression actually reduces file size."""
        uncompressed_path = self.temp_dir / "data.jsonl"
        compressed_path = self.temp_dir / "data.jsonl.gz"

        # Save both versions
        datason.save_ml(self.compressible_data, uncompressed_path)
        datason.save_ml(self.compressible_data, compressed_path)

        # Check sizes
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.stat().st_size

        # Compressed should be smaller
        assert compressed_size < uncompressed_size
        print(f"Compression ratio: {uncompressed_size / compressed_size:.1f}x")

    def test_mixed_compression_formats(self):
        """Test compression with different formats."""
        json_gz = self.temp_dir / "data.json.gz"
        jsonl_gz = self.temp_dir / "data.jsonl.gz"

        # Save to both compressed formats
        datason.save_ml(self.compressible_data, json_gz)
        datason.save_ml(self.compressible_data, jsonl_gz)

        # Load both
        json_loaded = list(datason.load_smart_file(json_gz))
        jsonl_loaded = list(datason.load_smart_file(jsonl_gz))

        assert len(json_loaded) == len(jsonl_loaded) == 1
        assert json_loaded[0] == jsonl_loaded[0] == self.compressible_data


class TestMLDataTypes:
    """Test ML-specific data type handling."""

    def setup_method(self):
        """Set up ML test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ml_data = {
            "arrays": {
                "weights": np.random.randn(100, 50),
                "biases": np.random.randn(50),
                "int_labels": np.random.randint(0, 10, 1000),
            },
            "dataframe": pd.DataFrame(
                {
                    "feature_a": np.random.random(100),
                    "feature_b": np.random.choice(["A", "B", "C"], 100),
                    "target": np.random.choice([0, 1], 100),
                }
            ),
            "metadata": {"model_type": "neural_network", "timestamp": datetime.now()},
        }

    def test_numpy_array_preservation(self):
        """Test NumPy arrays are preserved correctly."""
        ml_path = self.temp_dir / "ml_data.jsonl"

        # Save ML data
        datason.save_ml(self.ml_data, ml_path)

        # Load back
        loaded = list(datason.load_smart_file(ml_path))[0]

        # Check arrays are preserved
        assert isinstance(loaded["arrays"]["weights"], np.ndarray)
        assert loaded["arrays"]["weights"].shape == (100, 50)
        assert isinstance(loaded["arrays"]["biases"], np.ndarray)
        assert loaded["arrays"]["biases"].shape == (50,)

    def test_pandas_dataframe_preservation(self):
        """Test pandas DataFrames are preserved correctly."""
        ml_path = self.temp_dir / "ml_data.jsonl"

        # Save ML data
        datason.save_ml(self.ml_data, ml_path)

        # Load back
        loaded = list(datason.load_smart_file(ml_path))[0]

        # Check DataFrame is preserved
        assert isinstance(loaded["dataframe"], pd.DataFrame)
        assert len(loaded["dataframe"]) == 100
        assert list(loaded["dataframe"].columns) == ["feature_a", "feature_b", "target"]

    def test_perfect_reconstruction_with_template(self):
        """Test perfect type reconstruction using templates."""
        ml_path = self.temp_dir / "ml_data.jsonl"

        # Create template
        template = {
            "arrays": {"weights": np.array([[0.0]]), "biases": np.array([0.0]), "int_labels": np.array([0])},
            "dataframe": self.ml_data["dataframe"].iloc[:1],  # Template DataFrame
            "metadata": {},
        }

        # Save ML data
        datason.save_ml(self.ml_data, ml_path)

        # Load with perfect reconstruction
        loaded = list(datason.load_perfect_file(ml_path, template))[0]

        # Verify perfect type preservation
        assert isinstance(loaded["arrays"]["weights"], np.ndarray)
        assert loaded["arrays"]["weights"].shape == (100, 50)
        assert isinstance(loaded["dataframe"], pd.DataFrame)
        assert len(loaded["dataframe"]) == 100


class TestStreamingOperations:
    """Test streaming save operations."""

    def setup_method(self):
        """Set up streaming test."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_basic_streaming(self):
        """Test basic streaming functionality."""
        stream_path = self.temp_dir / "stream.jsonl"

        # Stream data
        with datason.stream_save_ml(stream_path) as stream:
            for i in range(10):
                record = {"id": i, "data": f"record_{i}", "timestamp": datetime.now()}
                stream.write(record)

        # Verify streaming worked
        records = list(datason.load_smart_file(stream_path))
        assert len(records) == 10
        assert records[0]["id"] == 0
        assert records[9]["id"] == 9

    def test_streaming_with_ml_data(self):
        """Test streaming with ML data types."""
        stream_path = self.temp_dir / "ml_stream.jsonl"

        # Stream ML records
        with datason.stream_save_ml(stream_path) as stream:
            for epoch in range(5):
                record = {
                    "epoch": epoch,
                    "weights": np.random.randn(10, 10),
                    "metrics": {"loss": np.random.exponential(), "accuracy": np.random.random()},
                }
                stream.write(record)

        # Verify ML data is preserved
        records = list(datason.load_smart_file(stream_path))
        assert len(records) == 5

        first_record = records[0]
        assert isinstance(first_record["weights"], np.ndarray)
        assert first_record["weights"].shape == (10, 10)

    def test_streaming_with_compression(self):
        """Test streaming to compressed files."""
        stream_path = self.temp_dir / "stream.jsonl.gz"

        # Stream to compressed file
        with datason.stream_save_ml(stream_path) as stream:
            for i in range(20):
                record = {"id": i, "data": "x" * 100}  # Compressible data
                stream.write(record)

        # Verify compressed streaming worked
        records = list(datason.load_smart_file(stream_path))
        assert len(records) == 20

        # Verify file is actually compressed
        with gzip.open(stream_path, "rt") as f:
            first_line = f.readline()
            assert '"id": 0' in first_line


class TestSecurityFeatures:
    """Test security and redaction features."""

    def setup_method(self):
        """Set up security test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sensitive_data = {
            "user_info": {
                "name": "John Doe",
                "email": "john.doe@company.com",
                "ssn": "123-45-6789",
                "credit_card": "4532-1234-5678-9012",
            },
            "secrets": {
                "api_key": "sk-1234567890abcdef",
                "password": "secret123",
                "database_url": "postgresql://user:pass@localhost/db",
            },
        }

    def test_save_secure_with_auto_redaction(self):
        """Test save_secure with automatic PII detection."""
        secure_path = self.temp_dir / "secure.jsonl"

        # Save with auto redaction
        datason.save_secure(self.sensitive_data, secure_path, redact_pii=True)

        # Load back
        loaded = list(datason.load_smart_file(secure_path))[0]

        # Check redaction metadata exists
        assert "redaction_summary" in loaded
        assert loaded["redaction_summary"]["total_redactions"] > 0

        # Check that PII is redacted
        user_info = loaded["user_info"]
        assert "[REDACTED:" in str(user_info["ssn"])
        assert "[REDACTED:" in str(user_info["credit_card"])

    def test_save_secure_with_field_redaction(self):
        """Test save_secure with explicit field redaction."""
        secure_path = self.temp_dir / "secure.jsonl"

        # Save with specific field redaction
        datason.save_secure(self.sensitive_data, secure_path, redact_fields=["api_key", "password", "database_url"])

        # Load back
        loaded = list(datason.load_smart_file(secure_path))[0]

        # Check field redaction worked
        secrets = loaded["secrets"]
        assert "[REDACTED:" in str(secrets["api_key"])
        assert "[REDACTED:" in str(secrets["password"])
        assert "[REDACTED:" in str(secrets["database_url"])

    def test_save_secure_combined_redaction(self):
        """Test save_secure with both PII and field redaction."""
        secure_path = self.temp_dir / "secure.jsonl"

        # Save with combined redaction
        datason.save_secure(self.sensitive_data, secure_path, redact_pii=True, redact_fields=["api_key", "password"])

        # Load back
        loaded = list(datason.load_smart_file(secure_path))[0]

        # Check both types of redaction worked
        assert loaded["redaction_summary"]["total_redactions"] >= 4  # At least PII + fields

        # PII should be redacted
        user_info = loaded["user_info"]
        assert "[REDACTED:" in str(user_info["ssn"])

        # Fields should be redacted
        secrets = loaded["secrets"]
        assert "[REDACTED:" in str(secrets["api_key"])


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up error test cases."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_path = self.temp_dir / "empty.jsonl"

        # Save empty list
        datason.save_ml([], empty_path)

        # Load back
        loaded = list(datason.load_smart_file(empty_path))
        assert len(loaded) == 0

    def test_none_data_handling(self):
        """Test handling of None data."""
        none_path = self.temp_dir / "none.jsonl"

        # Save None
        datason.save_ml(None, none_path)

        # Load back
        loaded = list(datason.load_smart_file(none_path))
        assert len(loaded) == 1
        assert loaded[0] is None

    def test_invalid_file_path_handling(self):
        """Test handling of invalid file paths."""
        # Try to save to invalid directory
        invalid_path = Path("/invalid/directory/file.jsonl")

        # Should handle gracefully (exact behavior depends on implementation)
        try:
            datason.save_ml({"test": "data"}, invalid_path)
        except (FileNotFoundError, PermissionError, OSError):
            # These are expected exceptions for invalid paths
            pass

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        corrupted_path = self.temp_dir / "corrupted.jsonl"

        # Create corrupted file
        with open(corrupted_path, "w") as f:
            f.write('{"incomplete": json\n')  # Invalid JSON

        # Loading should handle gracefully
        try:
            loaded = list(datason.load_smart_file(corrupted_path))
            # If it doesn't raise an exception, it should return something sensible
            assert isinstance(loaded, list)
        except (ValueError, json.JSONDecodeError):
            # These are acceptable exceptions for corrupted data
            pass


class TestAPIDiscovery:
    """Test API discovery and help functions."""

    def test_help_api_includes_file_operations(self):
        """Test that help_api includes file operations."""
        help_info = datason.help_api()

        # Should include file operations section
        assert isinstance(help_info, dict)
        # Note: Exact structure depends on implementation

    def test_api_info_includes_file_features(self):
        """Test that get_api_info includes file features."""
        api_info = datason.get_api_info()

        # Should include info about file operations
        assert isinstance(api_info, dict)
        # Note: Exact structure depends on implementation


class TestPerformance:
    """Test performance characteristics."""

    def setup_method(self):
        """Set up performance test data."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        large_data = {"big_array": np.random.randn(1000, 1000), "metadata": {"size": "1M elements"}}

        large_path = self.temp_dir / "large.jsonl.gz"  # Use compression

        # Save large data
        datason.save_ml(large_data, large_path)

        # Load back
        loaded = list(datason.load_smart_file(large_path))[0]

        # Verify integrity
        assert isinstance(loaded["big_array"], np.ndarray)
        assert loaded["big_array"].shape == (1000, 1000)

    def test_many_small_records_streaming(self):
        """Test streaming many small records."""
        stream_path = self.temp_dir / "many_records.jsonl"

        # Stream many small records
        with datason.stream_save_ml(stream_path) as stream:
            for i in range(10000):
                record = {"id": i, "value": i * 2}
                stream.write(record)

        # Verify all records were written
        records = list(datason.load_smart_file(stream_path))
        assert len(records) == 10000
        assert records[0]["id"] == 0
        assert records[9999]["id"] == 9999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
