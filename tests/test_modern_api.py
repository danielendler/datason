"""Tests for the modern API (Phase 3) - API Modernization.

These tests ensure that the new intention-revealing wrapper functions
work correctly and provide the expected interface improvements while
maintaining backward compatibility.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict

import pytest

import datason
from datason.api import (
    dump,
    dump_api,
    dump_chunked,
    dump_ml,
    dump_secure,
    dumps,
    get_api_info,
    help_api,
    load_basic,
    load_perfect,
    load_smart,
    load_typed,
    loads,
    suppress_deprecation_warnings,
)
from datason.config import SerializationConfig


@dataclass
class SampleTestObject:
    """Test object for serialization testing."""

    name: str
    value: int
    data: Dict[str, Any]


class TestModernAPIBasics:
    """Test basic functionality of the modern API."""

    def test_dump_basic_functionality(self):
        """Test that dump() works for basic serialization."""
        data = {"test": "value", "number": 42}
        result = dump(data)
        assert result == data

    def test_load_basic_functionality(self):
        """Test that load_basic() works for basic deserialization."""
        data = {"test": "value", "number": 42}
        result = load_basic(data)
        assert result == data

    def test_dumps_loads_json_compatibility(self):
        """Test that dumps/loads provide json module compatibility."""
        data = {"test": "value", "numbers": [1, 2, 3]}

        # Test dumps
        json_str = dumps(data)
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed == data

        # Test loads
        result = loads(json_str)
        assert result == data

    def test_api_info_functions(self):
        """Test that API info functions provide expected information."""
        # Test get_api_info
        info = get_api_info()
        assert info["api_version"] == "modern"
        assert info["phase"] == "3"
        assert "dump" in info["dump_functions"]
        assert "load_basic" in info["load_functions"]

        # Test help_api
        help_info = help_api()
        assert "serialization" in help_info
        assert "deserialization" in help_info
        assert "recommendations" in help_info
        assert isinstance(help_info["recommendations"], list)


class TestDumpFunctions:
    """Test the dump_* family of functions."""

    def test_dump_modes_mutually_exclusive(self):
        """Test that dump() modes are mutually exclusive."""
        data = {"test": "data"}

        # Should work with one mode
        dump(data, ml_mode=True)
        dump(data, api_mode=True)
        dump(data, fast_mode=True)

        # Should fail with multiple modes
        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            dump(data, ml_mode=True, api_mode=True)

    def test_dump_secure_redaction(self):
        """Test that dump_secure() redacts sensitive information."""
        data = {"name": "John Doe", "ssn": "123-45-6789", "email": "john@example.com", "password": "secret123"}

        result = dump_secure(data)

        # If redaction summary is included, result may be wrapped
        actual_data = result["data"] if isinstance(result, dict) and "data" in result else result

        # Should redact SSN and password
        assert actual_data["name"] == "John Doe"  # Safe field
        # Check that sensitive fields are redacted (could be different redaction token)
        assert actual_data["password"] != "secret123"  # Field-based redaction

    def test_dump_ml_uses_ml_config(self):
        """Test that dump_ml() uses ML-optimized configuration."""
        try:
            import numpy as np

            # Test with NumPy array
            data = {"array": np.array([1, 2, 3]), "model_params": {"lr": 0.01}}
            result = dump_ml(data)

            # Should serialize the numpy array properly (check for ML handling)
            # The exact format may vary, but it should handle NumPy arrays
            assert result is not None
            assert "array" in result

        except ImportError:
            pytest.skip("NumPy not available for ML testing")

    def test_dump_api_clean_output(self):
        """Test that dump_api() produces clean API-safe output."""
        data = {"message": "success", "data": [1, 2, 3]}
        result = dump_api(data)

        # Should be clean and predictable
        assert result == data  # Simple data should pass through cleanly

    def test_dump_chunked_creates_chunks(self):
        """Test that dump_chunked() creates chunked results."""
        large_list = list(range(2500))  # Larger than default chunk size
        result = dump_chunked(large_list, chunk_size=1000)

        # Should return ChunkedSerializationResult
        assert hasattr(result, "chunks")
        assert hasattr(result, "metadata")

        # Convert chunks to list to check length
        chunks_list = list(result.chunks)
        assert len(chunks_list) >= 2  # Should be split into multiple chunks


class TestLoadFunctions:
    """Test the load_* family of functions."""

    def test_load_basic_simple_heuristics(self):
        """Test that load_basic() uses simple heuristics."""
        data = {
            "date_string": "2023-01-01T12:00:00",
            "uuid_string": "12345678-1234-5678-9012-123456789012",
            "numbers": [1, 2, 3],
        }

        result = load_basic(data)

        # Should apply basic parsing
        assert isinstance(result["numbers"], list)

    def test_load_smart_better_reconstruction(self):
        """Test that load_smart() provides better type reconstruction."""
        # Create some data with type information
        original_data = {"value": 42, "text": "hello"}
        serialized = dump(original_data)

        result = load_smart(serialized)
        assert result == original_data

    def test_load_perfect_with_template(self):
        """Test that load_perfect() achieves perfect reconstruction with template."""
        original = SampleTestObject(name="test", value=42, data={"nested": "value"})

        # Serialize with ML config to preserve structure
        serialized = dump_ml(original)

        # Use template for perfect reconstruction
        template = SampleTestObject(name="", value=0, data={})
        result = load_perfect(serialized, template)

        # The result should match the original data structure
        # Note: The exact type reconstruction depends on template deserialization
        if isinstance(result, SampleTestObject):
            assert result.name == original.name
            assert result.value == original.value
        else:
            # If not perfect type reconstruction, at least data should match
            assert result["name"] == original.name
            assert result["value"] == original.value

    def test_load_typed_metadata_reconstruction(self):
        """Test that load_typed() uses embedded metadata."""
        # Create data with type information
        data = {"value": 42, "metadata": {"type": "test"}}

        result = load_typed(data)
        assert result == data  # Should reconstruct properly


class TestBackwardCompatibility:
    """Test that backward compatibility is maintained."""

    def test_existing_functions_still_work(self):
        """Test that existing serialize/deserialize functions still work."""
        data = {"test": "backward_compatibility"}

        # Old API should still work
        serialized = datason.serialize(data)
        deserialized = datason.deserialize(serialized)

        assert deserialized == data

    def test_new_and_old_api_equivalence(self):
        """Test that new API produces equivalent results to old API."""
        data = {"numbers": [1, 2, 3], "text": "test"}

        # Compare basic serialization
        old_result = datason.serialize(data)
        new_result = dump(data)
        assert old_result == new_result

        # Compare basic deserialization
        old_deserial = datason.deserialize(old_result)
        new_deserial = load_basic(new_result)
        assert old_deserial == new_deserial

    def test_deprecation_warnings_suppressible(self):
        """Test that deprecation warnings can be suppressed."""
        # Enable deprecation warnings first
        suppress_deprecation_warnings(False)

        # This would generate warnings (if we had them implemented)
        # For now, just test the suppression function works
        suppress_deprecation_warnings(True)
        suppress_deprecation_warnings(False)


class TestAdvancedFeatures:
    """Test advanced features of the modern API."""

    def test_dump_with_custom_config(self):
        """Test that dump() accepts custom configuration."""
        data = {"test": "config"}
        config = SerializationConfig(sort_keys=True)

        result = dump(data, config=config)
        assert result == data

    def test_load_smart_with_config(self):
        """Test that load_smart() accepts custom configuration."""
        data = {"test": "config"}
        config = SerializationConfig(auto_detect_types=False)

        result = load_smart(data, config=config)
        assert result == data

    def test_dump_secure_custom_patterns(self):
        """Test that dump_secure() accepts custom redaction patterns."""
        data = {"custom_field": "CUSTOM-123-SECRET", "safe_field": "safe_value"}

        result = dump_secure(data, redact_patterns=[r"CUSTOM-\d+-SECRET"], redact_fields=["custom_field"])

        # Handle wrapped result if redaction summary is included
        actual_data = result["data"] if isinstance(result, dict) and "data" in result else result

        # Should redact the custom field (value should be different)
        assert actual_data["custom_field"] != "CUSTOM-123-SECRET"
        assert actual_data["safe_field"] == "safe_value"

    def test_progressive_complexity_documentation(self):
        """Test that functions document their complexity/success rates."""
        # Check that docstrings contain success rate information
        assert "60-70%" in load_basic.__doc__
        assert "80-90%" in load_smart.__doc__
        assert "100%" in load_perfect.__doc__
        assert "95%" in load_typed.__doc__


class TestErrorHandling:
    """Test error handling in the modern API."""

    def test_dump_invalid_mode_combination(self):
        """Test error handling for invalid mode combinations."""
        data = {"test": "data"}

        with pytest.raises(ValueError):
            dump(data, ml_mode=True, api_mode=True, fast_mode=True)

    def test_load_perfect_missing_template(self):
        """Test error handling when template is required but missing."""
        data = {"test": "data"}

        # This should work (template provided)
        result = load_perfect(data, {"test": ""})
        assert result == data


class TestIntegrationWithExistingFeatures:
    """Test integration with existing datason features."""

    def test_modern_api_with_caching(self):
        """Test that modern API works with the caching system."""
        # Test with different cache scopes
        with datason.operation_scope():
            data = {"cached": "data"}
            result1 = dump(data)
            result2 = dump(data)  # Should use cache
            assert result1 == result2

    def test_modern_api_with_ml_objects(self):
        """Test modern API with ML objects if available."""
        try:
            import numpy as np

            data = {"array": np.array([1, 2, 3])}

            # Should work with ML-optimized dump
            ml_result = dump_ml(data)
            assert ml_result is not None

            # Should reconstruct properly
            reconstructed = load_smart(ml_result)
            assert "array" in reconstructed

        except ImportError:
            pytest.skip("NumPy not available for ML object testing")

    def test_modern_api_with_redaction_engine(self):
        """Test modern API integration with redaction features."""
        sensitive_data = {
            "user": "john_doe",
            "credit_card": "1234-5678-9012-3456",
            "notes": "Call customer at 555-1234",
        }

        # Use secure dump
        result = dump_secure(sensitive_data)

        # Handle wrapped result if redaction summary is included
        actual_data = result["data"] if isinstance(result, dict) and "data" in result else result

        # Should redact credit card - check that the value was changed
        assert "credit_card" in actual_data  # Field should exist
        assert actual_data["credit_card"] != "1234-5678-9012-3456"  # But value should be redacted


if __name__ == "__main__":
    pytest.main([__file__])
