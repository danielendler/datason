"""
End-to-end idempotency tests for datason serialization and deserialization.

This module tests the complete idempotency implementation across both
serialization (core_new) and deserialization (deserializers_new) to ensure
that double serialization/deserialization is prevented while maintaining
data integrity and performance.
"""

import uuid
from datetime import datetime
from decimal import Decimal

import pytest

import datason.core_new as core_new
import datason.deserializers_new as deserializers_new
from datason.config import SerializationConfig


class TestSerializationIdempotency:
    """Test serialization idempotency implementation."""

    def test_basic_serialization_idempotency(self):
        """Test that serializing already serialized data is idempotent."""
        config = SerializationConfig(include_type_hints=True)

        # Test with a set that creates type metadata
        original = {1, 2, 3}

        first = core_new.serialize(original, config)
        second = core_new.serialize(first, config)

        assert first == second, "Serialization should be idempotent"
        assert "__datason_type__" in str(first), "Should contain type metadata"

    def test_complex_serialization_idempotency(self):
        """Test serialization idempotency with complex nested data."""
        config = SerializationConfig(include_type_hints=True)

        complex_data = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "uuid": uuid.uuid4(),
            "set": {1, 2, 3},
            "tuple": (4, 5, 6),
            "nested": {"decimal": Decimal("123.45"), "complex_num": complex(1, 2)},
        }

        first = core_new.serialize(complex_data, config)
        second = core_new.serialize(first, config)
        third = core_new.serialize(second, config)

        assert first == second == third, "Multiple serializations should be idempotent"


class TestDeserializationIdempotency:
    """Test deserialization idempotency implementation."""

    def test_basic_deserialization_idempotency(self):
        """Test that deserializing already deserialized data is idempotent."""
        # Already deserialized datetime
        dt = datetime(2023, 1, 1, 12, 0, 0)

        first = deserializers_new.deserialize(dt)
        second = deserializers_new.deserialize(first)

        assert first is dt, "Should return same object for already deserialized data"
        assert second is dt, "Should remain idempotent"

    def test_complex_deserialization_idempotency(self):
        """Test deserialization idempotency with complex objects."""
        mixed_data = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "uuid": uuid.uuid4(),
            "decimal": Decimal("123.45"),
            "string": "normal_string",
            "number": 42,
        }

        first = deserializers_new.deserialize(mixed_data)
        second = deserializers_new.deserialize(first)

        assert first == second, "Deserialization should be idempotent"
        assert first["datetime"] is second["datetime"], "Objects should be identical"

    def test_type_metadata_processing(self):
        """Test that type metadata is correctly processed and becomes idempotent."""
        type_metadata = {"__datason_type__": "datetime", "__datason_value__": "2023-01-01T12:00:00"}

        first = deserializers_new.deserialize(type_metadata)
        second = deserializers_new.deserialize(first)

        assert isinstance(first, datetime), "Should correctly process type metadata"
        assert first is second, "Should be idempotent after processing"


class TestEndToEndIdempotency:
    """Test complete end-to-end idempotency scenarios."""

    def test_complete_round_trip_idempotency(self):
        """Test complete serialize -> deserialize -> serialize -> deserialize cycle."""
        config = SerializationConfig(include_type_hints=True)

        original_data = {
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "uuid": uuid.uuid4(),
            "decimal": Decimal("123.45"),
            "set": {1, 2, 3},
            "tuple": (4, 5, 6),
            "nested": {"list": [1, 2, {"inner": "value"}], "complex_num": complex(1, 2)},
        }

        # Step 1: First serialization
        first_serialized = core_new.serialize(original_data, config)

        # Step 2: Second serialization (should be idempotent)
        second_serialized = core_new.serialize(first_serialized, config)
        assert first_serialized == second_serialized, "Serialization should be idempotent"

        # Step 3: First deserialization
        first_deserialized = deserializers_new.deserialize(first_serialized)

        # Step 4: Second deserialization (should be idempotent)
        second_deserialized = deserializers_new.deserialize(first_deserialized)
        assert first_deserialized == second_deserialized, "Deserialization should be idempotent"

        # Step 5: Third serialization (should match first)
        third_serialized = core_new.serialize(first_deserialized, config)
        assert first_serialized == third_serialized, "Round-trip should be consistent"

        # Step 6: Verify complete round-trip
        final_deserialized = deserializers_new.deserialize(third_serialized)

        # Check type preservation
        assert isinstance(final_deserialized["datetime"], datetime)
        assert isinstance(final_deserialized["uuid"], uuid.UUID)
        assert isinstance(final_deserialized["decimal"], Decimal)
        assert isinstance(final_deserialized["set"], set)
        assert isinstance(final_deserialized["tuple"], tuple)
        assert isinstance(final_deserialized["nested"]["complex_num"], complex)

        # Check data integrity
        assert final_deserialized["datetime"] == original_data["datetime"]
        assert final_deserialized["uuid"] == original_data["uuid"]
        assert final_deserialized["decimal"] == original_data["decimal"]
        assert final_deserialized["set"] == original_data["set"]
        assert final_deserialized["tuple"] == original_data["tuple"]
        assert final_deserialized["nested"]["complex_num"] == original_data["nested"]["complex_num"]

    def test_performance_with_idempotency(self):
        """Test that idempotency provides performance benefits."""
        import time

        config = SerializationConfig(include_type_hints=True)

        # Create test data
        test_data = {"datetime": datetime(2023, 1, 1, 12, 0, 0), "uuid": uuid.uuid4(), "set": {1, 2, 3}}

        # First serialization (will do actual work)
        first_serialized = core_new.serialize(test_data, config)
        first_deserialized = deserializers_new.deserialize(first_serialized)

        # Time multiple idempotent operations
        start = time.perf_counter()
        for _ in range(100):
            # These should all be very fast due to idempotency
            temp1 = core_new.serialize(first_deserialized, config)
            temp2 = deserializers_new.deserialize(temp1)
            temp3 = core_new.serialize(temp2, config)
            deserializers_new.deserialize(temp3)  # Final deserialization
        end = time.perf_counter()

        avg_time_ms = (end - start) * 1000 / 100

        # Should be very fast due to idempotency (less than 1ms per 4-step cycle)
        assert avg_time_ms < 1.0, f"Idempotent operations should be fast, got {avg_time_ms:.2f}ms"

    def test_mixed_serialized_and_raw_data(self):
        """Test handling of mixed data with both serialized and raw components."""
        config = SerializationConfig(include_type_hints=True)

        # Mix of already serialized and raw data
        mixed_data = {
            "raw_datetime": datetime(2023, 1, 1, 12, 0, 0),
            "serialized_set": {"__datason_type__": "set", "__datason_value__": [1, 2, 3]},
            "raw_string": "hello",
            "raw_number": 42,
        }

        # Serialize the mixed data
        serialized = core_new.serialize(mixed_data, config)

        # Should handle both raw and already-serialized components correctly
        assert isinstance(serialized, dict)

        # Deserialize back
        deserialized = deserializers_new.deserialize(serialized)

        # Check that everything is properly restored
        assert isinstance(deserialized["raw_datetime"], datetime)
        assert isinstance(deserialized["serialized_set"], set)
        assert deserialized["serialized_set"] == {1, 2, 3}
        assert deserialized["raw_string"] == "hello"
        assert deserialized["raw_number"] == 42


class TestIdempotencyEdgeCases:
    """Test edge cases for idempotency implementation."""

    def test_empty_containers_idempotency(self):
        """Test idempotency with empty containers."""
        config = SerializationConfig(include_type_hints=True)

        empty_data = {"empty_dict": {}, "empty_list": [], "empty_set": set(), "empty_tuple": ()}

        first = core_new.serialize(empty_data, config)
        second = core_new.serialize(first, config)

        assert first == second, "Empty containers should be idempotent"

        deserialized = deserializers_new.deserialize(first)
        re_serialized = core_new.serialize(deserialized, config)

        assert first == re_serialized, "Round-trip with empty containers should be consistent"

    def test_none_values_idempotency(self):
        """Test idempotency with None values."""
        config = SerializationConfig(include_type_hints=True)

        none_data = {"none_value": None, "list_with_none": [1, None, 3], "dict_with_none": {"a": None, "b": 2}}

        first = core_new.serialize(none_data, config)
        second = core_new.serialize(first, config)

        assert first == second, "None values should be idempotent"

    def test_deeply_nested_idempotency(self):
        """Test idempotency with deeply nested structures."""
        config = SerializationConfig(include_type_hints=True)

        # Create deeply nested structure
        nested = {"level": 1}
        for i in range(2, 10):
            nested = {"level": i, "nested": nested, "set": {i, i + 1}}

        first = core_new.serialize(nested, config)
        second = core_new.serialize(first, config)

        assert first == second, "Deeply nested structures should be idempotent"

        deserialized = deserializers_new.deserialize(first)
        re_serialized = core_new.serialize(deserialized, config)

        assert first == re_serialized, "Deep nesting round-trip should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
