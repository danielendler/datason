"""Tests for the pickle bridge functionality.

Tests the safe conversion of pickle files to datason JSON format,
including security, performance, and compatibility testing.
"""

import json
import os
import pickle
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from datason import (
    PickleBridge,
    PickleSecurityError,
    convert_pickle_directory,
    from_pickle,
    get_ml_safe_classes,
)
from datason.config import get_ml_config
from datason.core_new import SecurityError


# Test data fixtures
@pytest.fixture
def safe_test_data():
    """Create safe test data that should convert successfully."""
    return {
        "string": "hello world",
        "number": 42,
        "float": 3.14159,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3, "test"],
        "dict": {"nested": "value", "count": 123},
        "datetime": datetime(2023, 1, 15, 10, 30, 0),
        "uuid": uuid.uuid4(),
    }


class UnsafeTestClass:
    """A test class that should not be in the safe classes list."""

    def __init__(self, value):
        self.value = value


@pytest.fixture
def temp_pickle_file(safe_test_data):
    """Create a temporary pickle file with safe test data."""
    import tempfile

    # Create a temporary file that won't be automatically deleted
    fd, temp_path = tempfile.mkstemp(suffix=".pkl")
    try:
        # Write pickle data to the file
        with os.fdopen(fd, "wb") as f:
            pickle.dump(safe_test_data, f)

        yield temp_path

    finally:
        # Clean up the temporary file
        Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestPickleBridge:
    """Test the main PickleBridge class functionality."""

    def test_init_with_defaults(self):
        """Test PickleBridge initialization with default parameters."""
        bridge = PickleBridge()

        assert bridge.safe_classes == PickleBridge.DEFAULT_SAFE_CLASSES
        assert bridge.config is not None
        assert bridge.max_file_size == 100 * 1024 * 1024  # 100MB
        assert bridge._conversion_stats["files_processed"] == 0

    def test_init_with_custom_params(self):
        """Test PickleBridge initialization with custom parameters."""
        custom_classes = {"builtins.dict", "builtins.str"}
        custom_config = get_ml_config()

        bridge = PickleBridge(
            safe_classes=custom_classes,
            config=custom_config,
            max_file_size=50 * 1024 * 1024,
        )

        assert bridge.safe_classes == custom_classes
        assert bridge.config == custom_config
        assert bridge.max_file_size == 50 * 1024 * 1024

    def test_add_safe_class(self):
        """Test adding individual safe classes."""
        bridge = PickleBridge(safe_classes=set())

        bridge.add_safe_class("my.custom.Class")
        assert "my.custom.Class" in bridge.safe_classes

    def test_add_safe_module(self):
        """Test adding safe module with warning."""
        bridge = PickleBridge(safe_classes=set())

        with pytest.warns(UserWarning, match="Adding entire module"):
            bridge.add_safe_module("mymodule")

        assert "mymodule.*" in bridge.safe_classes

    def test_safe_unpickler_allowed_class(self, temp_pickle_file):
        """Test that safe unpickler allows whitelisted classes."""
        bridge = PickleBridge()

        with Path(temp_pickle_file).open("rb") as f:
            unpickler = bridge._safe_unpickler(f)
            # This should not raise since we're using safe test data
            data = unpickler.load()

        assert isinstance(data, dict)

    def test_safe_unpickler_blocked_class(self):
        """Test that safe unpickler blocks non-whitelisted classes."""
        bridge = PickleBridge(safe_classes={"builtins.dict"})  # Only allow dict

        # Create pickle with unsafe class
        unsafe_data = UnsafeTestClass("dangerous")

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(unsafe_data, f)
            f.seek(0)

            unpickler = bridge._safe_unpickler(f)
            with pytest.raises(PickleSecurityError, match="unauthorized class"):
                unpickler.load()

    def test_from_pickle_file_success(self, temp_pickle_file):
        """Test successful conversion of a pickle file."""
        bridge = PickleBridge()
        result = bridge.from_pickle_file(temp_pickle_file)

        # Check structure
        assert "data" in result
        assert "metadata" in result

        # Check metadata
        metadata = result["metadata"]
        assert "source_file" in metadata
        assert "source_size_bytes" in metadata
        assert "conversion_timestamp" in metadata
        assert "datason_version" in metadata
        assert "safe_classes_used" in metadata

        # Check that basic data was preserved
        assert result["data"]["string"] == "hello world"
        assert result["data"]["number"] == 42

        # Check stats were updated
        stats = bridge.get_conversion_stats()
        assert stats["files_processed"] == 1
        assert stats["files_successful"] == 1
        assert stats["files_failed"] == 0

    def test_from_pickle_file_too_large(self, temp_pickle_file):
        """Test that oversized files are rejected."""
        bridge = PickleBridge(max_file_size=100)  # Very small limit

        with pytest.raises(SecurityError, match="file size.*exceeds maximum"):
            bridge.from_pickle_file(temp_pickle_file)

    def test_from_pickle_file_not_found(self):
        """Test handling of non-existent files."""
        bridge = PickleBridge()

        with pytest.raises(FileNotFoundError):
            bridge.from_pickle_file("nonexistent.pkl")

    def test_from_pickle_bytes_success(self, safe_test_data):
        """Test successful conversion of pickle bytes."""
        bridge = PickleBridge()

        # Create pickle bytes
        pickle_bytes = pickle.dumps(safe_test_data)

        result = bridge.from_pickle_bytes(pickle_bytes)

        # Check structure
        assert "data" in result
        assert "metadata" in result
        assert result["data"]["string"] == "hello world"

    def test_from_pickle_bytes_too_large(self, safe_test_data):
        """Test that oversized pickle bytes are rejected."""
        bridge = PickleBridge(max_file_size=100)  # Very small limit

        pickle_bytes = pickle.dumps(safe_test_data)

        with pytest.raises(SecurityError, match="data size.*exceeds maximum"):
            bridge.from_pickle_bytes(pickle_bytes)

    def test_convert_directory_success(self, temp_directory, safe_test_data):
        """Test successful directory conversion."""
        bridge = PickleBridge()

        # Create test pickle files
        source_dir = temp_directory / "source"
        target_dir = temp_directory / "target"
        source_dir.mkdir()

        for i in range(3):
            pickle_file = source_dir / f"test_{i}.pkl"
            with pickle_file.open("wb") as f:
                pickle.dump({**safe_test_data, "file_id": i}, f)

        # Convert directory
        stats = bridge.convert_directory(source_dir, target_dir)

        # Check statistics
        assert stats["files_found"] == 3
        assert stats["files_converted"] == 3
        assert stats["files_skipped"] == 0
        assert stats["files_failed"] == 0
        assert len(stats["errors"]) == 0

        # Check that JSON files were created
        json_files = list(target_dir.glob("*.json"))
        assert len(json_files) == 3

        # Verify content of one file
        with json_files[0].open() as f:
            data = json.load(f)
            assert "data" in data
            assert "metadata" in data

    def test_convert_directory_with_overwrite(self, temp_directory, safe_test_data):
        """Test directory conversion with overwrite behavior."""
        bridge = PickleBridge()

        source_dir = temp_directory / "source"
        target_dir = temp_directory / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Create a pickle file
        pickle_file = source_dir / "test.pkl"
        with pickle_file.open("wb") as f:
            pickle.dump(safe_test_data, f)

        # Create existing JSON file
        json_file = target_dir / "test.json"
        with json_file.open("w") as f:
            json.dump({"existing": "data"}, f)

        # Convert without overwrite (should skip)
        stats = bridge.convert_directory(source_dir, target_dir, overwrite=False)
        assert stats["files_skipped"] == 1
        assert stats["files_converted"] == 0

        # Convert with overwrite (should convert)
        stats = bridge.convert_directory(source_dir, target_dir, overwrite=True)
        assert stats["files_skipped"] == 0
        assert stats["files_converted"] == 1

    def test_convert_directory_source_not_found(self, temp_directory):
        """Test error handling for non-existent source directory."""
        bridge = PickleBridge()

        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            bridge.convert_directory("nonexistent", temp_directory)

    def test_get_conversion_stats(self, temp_pickle_file):
        """Test getting conversion statistics."""
        bridge = PickleBridge()

        # Initial stats
        stats = bridge.get_conversion_stats()
        assert stats["files_processed"] == 0

        # Convert a file
        bridge.from_pickle_file(temp_pickle_file)

        # Updated stats
        stats = bridge.get_conversion_stats()
        assert stats["files_processed"] == 1
        assert stats["files_successful"] == 1
        assert stats["total_size_bytes"] > 0


class TestConvenienceFunctions:
    """Test the convenience functions for pickle bridge."""

    def test_from_pickle_function(self, temp_pickle_file):
        """Test the from_pickle convenience function."""
        result = from_pickle(temp_pickle_file)

        assert "data" in result
        assert "metadata" in result
        assert result["data"]["string"] == "hello world"

    def test_from_pickle_with_custom_config(self, temp_pickle_file):
        """Test from_pickle with custom configuration."""
        custom_classes = {
            "builtins.dict",
            "builtins.str",
            "builtins.int",
            "builtins.list",
            "builtins.bool",
            "builtins.NoneType",
            "builtins.float",
            "datetime.datetime",
            "uuid.UUID",
        }
        config = get_ml_config()

        result = from_pickle(temp_pickle_file, safe_classes=custom_classes, config=config)

        assert "data" in result
        assert "metadata" in result

    def test_convert_pickle_directory_function(self, temp_directory, safe_test_data):
        """Test the convert_pickle_directory convenience function."""
        source_dir = temp_directory / "source"
        target_dir = temp_directory / "target"
        source_dir.mkdir()

        # Create test file
        pickle_file = source_dir / "test.pkl"
        with pickle_file.open("wb") as f:
            pickle.dump(safe_test_data, f)

        stats = convert_pickle_directory(source_dir, target_dir)

        assert stats["files_found"] == 1
        assert stats["files_converted"] == 1

    def test_get_ml_safe_classes_function(self):
        """Test the get_ml_safe_classes convenience function."""
        safe_classes = get_ml_safe_classes()

        assert isinstance(safe_classes, set)
        assert "numpy.ndarray" in safe_classes
        assert "pandas.core.frame.DataFrame" in safe_classes
        assert "sklearn.linear_model._base.LinearRegression" in safe_classes
        assert "builtins.dict" in safe_classes


class TestSecurityFeatures:
    """Test security-related functionality."""

    def test_default_safe_classes_comprehensive(self):
        """Test that default safe classes include expected ML libraries."""
        safe_classes = PickleBridge.DEFAULT_SAFE_CLASSES

        # Check numpy classes
        assert "numpy.ndarray" in safe_classes
        assert "numpy.dtype" in safe_classes

        # Check pandas classes
        assert "pandas.core.frame.DataFrame" in safe_classes
        assert "pandas.core.series.Series" in safe_classes

        # Check sklearn classes
        assert "sklearn.linear_model._base.LinearRegression" in safe_classes
        assert "sklearn.ensemble._forest.RandomForestClassifier" in safe_classes

        # Check basic Python types
        assert "builtins.dict" in safe_classes
        assert "builtins.list" in safe_classes
        assert "datetime.datetime" in safe_classes

    def test_module_wildcard_matching(self):
        """Test that module wildcards work correctly."""
        bridge = PickleBridge(safe_classes={"mymodule.*"})

        with tempfile.NamedTemporaryFile() as f:
            unpickler = bridge._safe_unpickler(f)

            # This should work with module wildcard
            # Note: We can't actually test this without creating a real class,
            # but we can test the logic in find_class
            try:
                # This will fail because the class doesn't exist,
                # but it should pass the security check
                unpickler.find_class("mymodule.submodule", "SomeClass")
            except (ImportError, AttributeError):
                pass  # Expected - class doesn't exist
            except PickleSecurityError:
                pytest.fail("Should have allowed mymodule.* classes")

    def test_security_error_inheritance(self):
        """Test that PickleSecurityError inherits from SecurityError."""
        assert issubclass(PickleSecurityError, SecurityError)

        # Test that it can be caught as SecurityError
        def _raise_error():
            raise PickleSecurityError("test error")

        with pytest.raises(SecurityError):
            _raise_error()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_pickle_file(self, temp_directory):
        """Test handling of empty pickle files."""
        bridge = PickleBridge()

        empty_file = temp_directory / "empty.pkl"
        empty_file.touch()  # Create empty file

        with pytest.raises(PickleSecurityError):
            bridge.from_pickle_file(empty_file)

    def test_corrupted_pickle_file(self, temp_directory):
        """Test handling of corrupted pickle files."""
        bridge = PickleBridge()

        corrupted_file = temp_directory / "corrupted.pkl"
        with corrupted_file.open("wb") as f:
            f.write(b"not a pickle file")

        with pytest.raises(PickleSecurityError):
            bridge.from_pickle_file(corrupted_file)

    def test_none_data_serialization(self):
        """Test serialization of None data."""
        bridge = PickleBridge()

        pickle_bytes = pickle.dumps(None)
        result = bridge.from_pickle_bytes(pickle_bytes)

        assert result["data"] is None

    def test_complex_nested_structures(self):
        """Test serialization of complex nested data structures."""
        bridge = PickleBridge()

        complex_data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"deep": "value"}],
                    "list_of_dicts": [{"a": 1}, {"b": 2}],
                },
                "datetime": datetime.now(),
                "uuid": uuid.uuid4(),
            }
        }

        pickle_bytes = pickle.dumps(complex_data)
        result = bridge.from_pickle_bytes(pickle_bytes)

        # Check that nested structure was preserved
        assert result["data"]["level1"]["level2"]["level3"][2]["deep"] == "value"
        assert len(result["data"]["level1"]["level2"]["list_of_dicts"]) == 2


class TestPerformance:
    """Test performance characteristics of pickle bridge."""

    def test_large_data_conversion(self):
        """Test conversion of reasonably large data structures."""
        bridge = PickleBridge()

        # Create moderately large data
        large_data = {
            "numbers": list(range(10000)),
            "strings": [f"item_{i}" for i in range(1000)],
            "nested": {f"key_{i}": {"value": i * 2} for i in range(1000)},
        }

        pickle_bytes = pickle.dumps(large_data)

        import time

        start_time = time.time()
        result = bridge.from_pickle_bytes(pickle_bytes)
        elapsed = time.time() - start_time

        # Should complete reasonably quickly (adjust threshold as needed)
        assert elapsed < 5.0  # 5 seconds max for large conversion
        assert len(result["data"]["numbers"]) == 10000

    def test_stats_tracking_performance(self, temp_directory, safe_test_data):
        """Test that statistics tracking doesn't significantly impact performance."""
        bridge = PickleBridge()

        source_dir = temp_directory / "source"
        source_dir.mkdir()

        # Create multiple pickle files
        num_files = 50
        for i in range(num_files):
            pickle_file = source_dir / f"test_{i}.pkl"
            with pickle_file.open("wb") as f:
                pickle.dump({**safe_test_data, "id": i}, f)

        import time

        start_time = time.time()
        stats = bridge.convert_directory(source_dir, temp_directory / "target")
        elapsed = time.time() - start_time

        # Should process files efficiently
        assert elapsed < 10.0  # Should be fast for 50 small files
        assert stats["files_converted"] == num_files

        # Verify stats are accurate
        final_stats = bridge.get_conversion_stats()
        assert final_stats["files_processed"] == num_files
        assert final_stats["files_successful"] == num_files


if __name__ == "__main__":
    pytest.main([__file__])
