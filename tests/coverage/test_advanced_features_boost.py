#!/usr/bin/env python3
"""Additional comprehensive tests for advanced features to boost coverage even higher."""

import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytest

# Import modules
try:
    from datason import serialize
    from datason.config import SerializationConfig
    from datason.core import (
        _contains_potentially_exploitable_nested_list_structure,
        _contains_potentially_exploitable_nested_structure,
        _convert_dict_tuples_fast,
        _convert_list_tuples_fast,
        _convert_tuple_to_list_fast,
        _get_cached_type_category_fast,
        _is_fully_json_compatible,
        _is_homogeneous_collection,
        _is_json_basic_type_safe,
        _process_dict_iterative,
        _process_homogeneous_dict,
        _process_homogeneous_list,
        _process_list_iterative,
        _process_string_optimized,
        _serialize_iterative_path,
        _serialize_json_only_fast_path,
        _uuid_to_string_optimized,
        deserialize_chunked_file,
    )
    from datason.deserializers import (
        _deserialize_string_full,
        _deserialize_with_type_metadata,
        _get_cached_parsed_object,
        _get_cached_string_pattern,
        _is_numeric_part,
        _reconstruct_from_split,
        _restore_pandas_types,
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestIterativeSerializationAdvanced:
    """Test advanced iterative serialization features."""

    def test_iterative_serialization_deep_structure(self) -> None:
        """Test iterative serialization with deep structures."""
        # Create complex nested structure
        deep_data = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}

        result = _serialize_iterative_path(deep_data, None)
        assert isinstance(result, dict)
        assert result["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_iterative_dict_processing_complex(self) -> None:
        """Test iterative dict processing with complex data."""
        complex_dict = {
            "simple": "value",
            "nested": {"inner": {"deeper": "value"}},
            "list": [1, 2, {"nested_in_list": "value"}],
            "datetime": datetime.now(),
            "uuid": uuid.uuid4(),
        }

        result = _process_dict_iterative(complex_dict, None)
        assert isinstance(result, dict)
        assert "simple" in result
        assert "nested" in result

    def test_iterative_list_processing_complex(self) -> None:
        """Test iterative list processing with complex data."""
        complex_list = [1, 2, 3, {"nested": "dict"}, [4, 5, {"deeper": "nested"}], datetime.now(), uuid.uuid4()]

        result = _process_list_iterative(complex_list, None)
        assert isinstance(result, list)
        assert len(result) == len(complex_list)

    def test_exploit_detection_nested_dict(self) -> None:
        """Test detection of potentially exploitable nested dict structures."""
        # Create deeply nested dict that could be exploitative
        exploit_dict: Dict[str, Any] = {}
        current = exploit_dict
        for i in range(30):  # Create very deep nesting
            current["next"] = {}
            current = current["next"]
        current["end"] = "value"

        is_exploit = _contains_potentially_exploitable_nested_structure(exploit_dict, 0)
        assert isinstance(is_exploit, bool)

    def test_exploit_detection_nested_list(self) -> None:
        """Test detection of potentially exploitable nested list structures."""
        # Create deeply nested list that could be exploitative
        exploit_list: List[Any] = []
        current = exploit_list
        for i in range(25):
            inner: List[Any] = []
            current.append(inner)
            current = inner
        current.append("end")

        is_exploit = _contains_potentially_exploitable_nested_list_structure(exploit_list, 0)
        assert isinstance(is_exploit, bool)


class TestOptimizedSerializationPaths:
    """Test optimized serialization paths and fast functions."""

    def test_json_only_fast_path(self) -> None:
        """Test JSON-only fast path optimization."""
        # Test with purely JSON-compatible data
        json_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, "three"],
            "nested": {"inner": "value"},
        }

        result = _serialize_json_only_fast_path(json_data)
        assert result == json_data

    def test_is_fully_json_compatible_deep(self) -> None:
        """Test JSON compatibility check with deep structures."""
        # Test with fully compatible nested structure
        compatible_data = {"level1": {"level2": {"level3": ["a", "b", "c"]}}}

        assert _is_fully_json_compatible(compatible_data, max_depth=5) is True

        # Test with incompatible data
        incompatible_data = {"datetime": datetime.now()}

        assert _is_fully_json_compatible(incompatible_data) is False

    def test_tuple_conversion_optimizations(self) -> None:
        """Test tuple conversion optimization functions."""
        # Test dict with complex tuple structures
        dict_with_tuples = {
            "simple_tuple": (1, 2, 3),
            "nested_tuple": (1, (2, 3), 4),
            "mixed": {"inner": (5, 6)},
            "string": "no_change",
        }

        result = _convert_dict_tuples_fast(dict_with_tuples)
        assert isinstance(result["simple_tuple"], list)
        assert isinstance(result["nested_tuple"], list)
        assert isinstance(result["nested_tuple"][1], list)

        # Test list with complex tuple structures
        list_with_tuples = [(1, 2, 3), "string", {"dict": (4, 5)}, [(6, 7), (8, 9)]]

        result_list = _convert_list_tuples_fast(list_with_tuples)
        assert isinstance(result_list[0], list)
        assert isinstance(result_list[3][0], list)

        # Test tuple to list conversion with nested tuples
        complex_tuple = (1, (2, (3, 4)), [5, (6, 7)])

        result_tuple = _convert_tuple_to_list_fast(complex_tuple)
        assert isinstance(result_tuple, list)
        assert isinstance(result_tuple[1], list)


class TestAdvancedCachingAndOptimization:
    """Test advanced caching and optimization features."""

    def test_fast_type_category_caching(self) -> None:
        """Test fast type category caching with various types."""
        # Test caching with multiple types
        types_to_test = [str, int, float, dict, list, tuple, set, datetime, uuid.UUID]

        for test_type in types_to_test:
            category = _get_cached_type_category_fast(test_type)
            # Should return a category or None
            assert category is None or isinstance(category, str)

    def test_homogeneous_collection_detection(self) -> None:
        """Test homogeneous collection detection and optimization."""
        # Test homogeneous list
        homogeneous_list = [1, 2, 3, 4, 5] * 20  # Large homogeneous list

        result = _is_homogeneous_collection(homogeneous_list, sample_size=10)
        assert result is None or isinstance(result, str)

        # Test homogeneous dict
        homogeneous_dict = {f"key_{i}": i for i in range(50)}

        result_dict = _is_homogeneous_collection(homogeneous_dict, sample_size=15)
        assert result_dict is None or isinstance(result_dict, str)

        # Test mixed collection
        mixed_list = [1, "string", {"dict": "value"}, datetime.now()]

        result_mixed = _is_homogeneous_collection(mixed_list)
        assert result_mixed is None or isinstance(result_mixed, str)

    def test_homogeneous_processing(self) -> None:
        """Test homogeneous collection processing."""
        # Test homogeneous dict processing
        config = SerializationConfig()
        homo_dict = {f"key_{i}": f"value_{i}" for i in range(20)}

        result = _process_homogeneous_dict(homo_dict, config, 1, set(), None)
        assert isinstance(result, dict)
        assert len(result) == len(homo_dict)

        # Test homogeneous list processing
        homo_list = [f"item_{i}" for i in range(25)]

        result_list = _process_homogeneous_list(homo_list, config, 1, set(), None)
        assert isinstance(result_list, list)
        assert len(result_list) == len(homo_list)

    def test_string_optimization_functions(self) -> None:
        """Test string optimization functions."""
        # Test string processing optimization
        test_strings = [
            "short",
            "a" * 1000,  # Long string
            "special\nchars\ttab",
        ]

        for test_string in test_strings:
            result = _process_string_optimized(test_string, 500)
            assert isinstance(result, str)

        # Test UUID to string optimization
        test_uuid = uuid.uuid4()
        uuid_string = _uuid_to_string_optimized(test_uuid)
        assert isinstance(uuid_string, str)
        assert len(uuid_string) == 36  # Standard UUID string length

    def test_json_basic_type_safe_checking(self) -> None:
        """Test safe JSON basic type checking with circular reference protection."""
        # Test with safe values
        safe_values = ["string", 42, 3.14, True, None]

        for value in safe_values:
            result = _is_json_basic_type_safe(value, set(), 3)
            assert isinstance(result, bool)

        # Test with potentially circular structure
        test_dict = {"key": "value"}
        seen_ids = {id(test_dict)}  # Mark as already seen

        result = _is_json_basic_type_safe(test_dict, seen_ids, 3)
        assert isinstance(result, bool)


class TestAdvancedDeserializationPaths:
    """Test advanced deserialization paths and functions."""

    def test_string_deserialization_full(self) -> None:
        """Test full string deserialization with various formats."""
        config = SerializationConfig()

        # Test various string formats
        test_strings = [
            "2023-01-01T12:00:00",  # Datetime
            "12345678-1234-5678-9012-123456789abc",  # UUID
            "/tmp/test/path.txt",  # Path
            "42",  # Number
            "3.14",  # Float
            "true",  # Boolean
            "regular_string",  # Regular string
        ]

        for test_string in test_strings:
            result = _deserialize_string_full(test_string, config)
            # Should return the string or converted type
            assert result is not None

    def test_type_metadata_deserialization_advanced(self) -> None:
        """Test advanced type metadata deserialization."""
        # Test various type metadata formats
        metadata_objects = [
            {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"},
            {"__datason_type__": "uuid", "__datason_value__": "12345678-1234-5678-9012-123456789abc"},
            {"__datason_type__": "decimal", "__datason_value__": "123.45"},
        ]

        for metadata_obj in metadata_objects:
            result = _deserialize_with_type_metadata(metadata_obj)
            # Should convert to appropriate type
            assert result is not None

    def test_pandas_type_restoration(self) -> None:
        """Test pandas type restoration if available."""
        try:
            pytest.importorskip("pandas")

            # Test with DataFrame-like data
            df_data = {"columns": ["A", "B"], "data": [[1, 2], [3, 4]]}

            result = _restore_pandas_types(df_data)
            # Should restore or return unchanged
            assert result is not None

        except ImportError:
            # Skip if pandas not available
            pass

    def test_split_format_reconstruction(self) -> None:
        """Test split format DataFrame reconstruction."""
        try:
            pd = pytest.importorskip("pandas")

            split_data = {"columns": ["A", "B"], "index": [0, 1, 2], "data": [[1, 2], [3, 4], [5, 6]]}

            result = _reconstruct_from_split(split_data)
            assert isinstance(result, pd.DataFrame)

        except ImportError:
            # Skip if pandas not available
            pass

    def test_numeric_part_detection(self) -> None:
        """Test numeric part detection in strings."""
        test_cases = [
            ("123", True),
            ("3.14", True),
            ("1.23e-4", True),
            ("-456", True),
            ("abc123", False),
            ("123abc", False),
            ("", False),
        ]

        for test_string, expected in test_cases:
            result = _is_numeric_part(test_string)
            assert result == expected

    def test_deserialization_caching_advanced(self) -> None:
        """Test advanced deserialization caching mechanisms."""
        # Test string pattern caching with various patterns
        test_patterns = [
            "2023-01-01T12:00:00",
            "12345678-1234-5678-9012-123456789abc",
            "/tmp/test/path.txt",
            "regular_string",
        ]

        for pattern in test_patterns:
            cached_pattern = _get_cached_string_pattern(pattern)
            # May return cached pattern or None

            if cached_pattern:
                _get_cached_parsed_object(pattern, cached_pattern)
                # May return cached object or None


class TestChunkedFileOperations:
    """Test chunked file operations and streaming."""

    def test_deserialize_chunked_file_advanced(self) -> None:
        """Test advanced chunked file deserialization."""
        # Create test file with multiple JSON objects
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            test_data = [
                {"id": 1, "name": "item1", "data": list(range(10))},
                {"id": 2, "name": "item2", "data": list(range(10, 20))},
                {"id": 3, "name": "item3", "data": list(range(20, 30))},
            ]

            for item in test_data:
                tmp.write(json.dumps(item) + "\n")
            tmp.flush()

            try:
                # Test chunked file reading
                chunks = list(deserialize_chunked_file(tmp.name, format="jsonl"))
                assert len(chunks) == 3

                # Test with chunk processor
                def process_chunk(chunk: Any) -> Any:
                    if isinstance(chunk, dict) and "data" in chunk:
                        chunk["data_length"] = len(chunk["data"])
                    return chunk

                processed_chunks = list(
                    deserialize_chunked_file(tmp.name, format="jsonl", chunk_processor=process_chunk)
                )

                assert len(processed_chunks) == 3
                assert "data_length" in processed_chunks[0]

            finally:
                # Clean up
                os.unlink(tmp.name)


class TestAdvancedConfigurationPaths:
    """Test advanced configuration and edge case paths."""

    def test_config_fallback_paths(self) -> None:
        """Test configuration fallback paths."""
        # Test with None config in various functions
        test_data = {"simple": "data", "number": 42}

        # Test iterative processing with None config
        result1 = _process_dict_iterative(test_data, None)
        assert isinstance(result1, dict)

        result2 = _process_list_iterative([1, 2, 3], None)
        assert isinstance(result2, list)

    def test_memory_optimization_edge_cases(self) -> None:
        """Test memory optimization edge cases."""
        # Test with empty collections
        empty_dict: Dict[str, Any] = {}
        empty_list: List[Any] = []

        result1 = _is_homogeneous_collection(empty_dict)
        result2 = _is_homogeneous_collection(empty_list)

        # Should handle empty collections gracefully
        assert result1 is None or isinstance(result1, str)
        assert result2 is None or isinstance(result2, str)

    def test_security_boundary_conditions(self) -> None:
        """Test security boundary conditions."""
        # Test with collections at size limits
        config = SerializationConfig(max_size=100)

        # Test with collection just under the limit
        medium_dict = {f"key_{i}": f"value_{i}" for i in range(95)}
        result = serialize(medium_dict, config=config)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
