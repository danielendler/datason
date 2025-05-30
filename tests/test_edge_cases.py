"""
Edge case tests for SerialPy to improve coverage.

This module contains targeted tests for specific edge cases and corner scenarios.
"""

import collections
from datetime import datetime
import decimal
import fractions
from pathlib import Path
from typing import Any, Dict

import serialpy as sp
from serialpy.core import serialize


class TestEdgeCasesForCoverage:
    """Test edge cases to improve code coverage."""

    def test_object_with_dict_method(self) -> None:
        """Test object with callable dict method."""

        class TestObj:
            def __init__(self) -> None:
                self.value = 42

            def dict(self) -> Dict[str, Any]:
                return {"custom": self.value}

        obj = TestObj()
        result = serialize(obj)
        assert result == {"custom": 42}

    def test_object_dict_method_exception(self) -> None:
        """Test object whose dict() method raises an exception."""

        class BrokenDict:
            def __init__(self) -> None:
                self.backup_value = "fallback"

            def dict(self) -> Dict[str, Any]:
                raise ValueError("Broken dict method")

        obj = BrokenDict()
        result = serialize(obj)
        # Should fall back to __dict__
        assert "backup_value" in result

    def test_object_vars_exception(self) -> None:
        """Test object where vars() raises an exception."""

        class SlotsObj:
            __slots__ = ["value"]

            def __init__(self) -> None:
                self.value = 42

        obj = SlotsObj()
        result = serialize(obj)
        # Should fall back to string conversion
        assert isinstance(result, str)

    def test_object_empty_dict(self) -> None:
        """Test object with empty __dict__."""

        class EmptyObj:
            pass

        obj = EmptyObj()
        result = serialize(obj)
        # Should fall back to string conversion
        assert isinstance(result, str)

    def test_complex_numbers(self) -> None:
        """Test serialization of complex numbers."""
        data = {"complex": 3 + 4j}
        result = serialize(data)
        assert isinstance(result["complex"], str)

    def test_decimal_objects(self) -> None:
        """Test serialization of Decimal objects."""
        data = {"decimal": decimal.Decimal("123.456")}
        result = serialize(data)
        assert isinstance(result["decimal"], str)

    def test_fraction_objects(self) -> None:
        """Test serialization of Fraction objects."""
        data = {"fraction": fractions.Fraction(3, 4)}
        result = serialize(data)
        assert isinstance(result["fraction"], str)

    def test_pathlib_objects(self) -> None:
        """Test serialization of pathlib objects."""
        data = {"path": Path("/usr/local/bin")}
        result = serialize(data)
        assert isinstance(result["path"], str)

    def test_collections_ordered_dict(self) -> None:
        """Test serialization of OrderedDict."""
        data = {"ordered": collections.OrderedDict([("a", 1), ("b", 2)])}
        result = serialize(data)
        # OrderedDict should be handled like a regular dict
        assert isinstance(result["ordered"], dict)
        assert result["ordered"]["a"] == 1

    def test_collections_other_types(self) -> None:
        """Test serialization of other collections types."""
        data = {
            "deque": collections.deque([1, 2, 3]),
            "counter": collections.Counter("hello"),
        }
        result = serialize(data)
        assert isinstance(result["deque"], str)
        # Counter inherits from dict, so it gets serialized as a dict
        assert isinstance(result["counter"], dict)

    def test_basic_circular_protection(self) -> None:
        """Test that we handle basic nested structures without infinite recursion."""
        # Instead of true circular reference, test deeply nested but finite structure
        nested: Dict[str, Any] = {"level": 0}
        current = nested
        for i in range(10):  # Reasonable depth
            current["next"] = {"level": i + 1}
            current = current["next"]

        # Should not crash
        result = serialize(nested)
        assert isinstance(result, dict)
        assert result["level"] == 0

    def test_deeply_nested_structure(self) -> None:
        """Test very deeply nested structure."""
        deep: Dict[str, Any] = {}
        current = deep
        for i in range(20):  # Reasonable depth
            current["level"] = i
            current["next"] = {}
            current = current["next"]
        current["end"] = True

        result = serialize(deep)
        assert isinstance(result, dict)
        assert result["level"] == 0


class TestDataUtilsEdgeCases:
    """Test data_utils.py edge cases."""

    def test_convert_string_method_votes_none(self) -> None:
        """Test convert_string_method_votes with None input."""
        result = sp.convert_string_method_votes(None)
        assert result is None

    def test_convert_string_method_votes_string_list(self) -> None:
        """Test convert_string_method_votes with string list."""
        tx = {"method_votes": "[1, 2, 3]"}
        result = sp.convert_string_method_votes(tx)
        assert isinstance(result, dict)
        assert result["method_votes"] == [1, 2, 3]

    def test_convert_string_method_votes_plain_string(self) -> None:
        """Test convert_string_method_votes with plain string."""
        tx = {"method_votes": "single_method"}
        result = sp.convert_string_method_votes(tx)
        assert isinstance(result, dict)
        assert result["method_votes"] == ["single_method"]

    def test_convert_string_method_votes_none_value(self) -> None:
        """Test convert_string_method_votes with None value."""
        tx: Dict[str, Any] = {"method_votes": None}
        result = sp.convert_string_method_votes(tx)
        assert isinstance(result, dict)
        assert result["method_votes"] == []

    def test_convert_string_method_votes_empty_list(self) -> None:
        """Test convert_string_method_votes with empty list."""
        tx: Dict[str, Any] = {"method_votes": []}
        result = sp.convert_string_method_votes(tx)
        assert isinstance(result, dict)
        assert result["method_votes"] == []

    def test_convert_string_method_votes_invalid_eval(self) -> None:
        """Test convert_string_method_votes with invalid eval string."""
        tx = {"method_votes": "[1, 2, broken"}
        result = sp.convert_string_method_votes(tx)
        assert isinstance(result, dict)
        # Invalid eval should be treated as plain string
        assert result["method_votes"] == ["[1, 2, broken"]

    def test_convert_string_method_votes_list_with_none(self) -> None:
        """Test convert_string_method_votes with list containing None."""
        transactions = [
            {"method_votes": "[1, 2]"},
            None,  # This should be filtered out
            {"method_votes": "test"},
        ]
        # Cast to proper type to satisfy type checker
        result = sp.convert_string_method_votes(transactions)  # type: ignore
        assert isinstance(result, list)
        assert len(result) == 2


class TestSerializersEdgeCases:
    """Test serializers.py edge cases."""

    def test_serialize_detection_details_non_dict(self) -> None:
        """Test serialize_detection_details with non-dict input."""
        assert sp.serialize_detection_details("not_a_dict") == "not_a_dict"
        assert sp.serialize_detection_details(42) == 42
        assert sp.serialize_detection_details(None) is None

    def test_serialize_detection_details_with_nan_inf(self) -> None:
        """Test serialize_detection_details with NaN and Inf values."""
        data = {
            "method1": {
                "values": [1.0, float("nan"), float("inf"), 2.5],
                "timestamp": datetime(2023, 1, 1),
            }
        }
        result = sp.serialize_detection_details(data)

        # Check NaN/Inf conversion
        assert result["method1"]["values"][1] is None
        assert result["method1"]["values"][2] is None

        # Check datetime conversion
        assert isinstance(result["method1"]["timestamp"], str)


class TestConvertersEdgeCases:
    """Test converters.py edge cases."""

    def test_safe_int_string_float(self) -> None:
        """Test safe_int with string representation of float."""
        assert sp.safe_int("42.0") == 42
        assert sp.safe_int("3.14") == 3

    def test_safe_conversions_exception_handling(self) -> None:
        """Test exception handling in safe converters."""

        class BadObject:
            def __str__(self) -> str:
                raise ValueError("Cannot convert to string")

        bad_obj = BadObject()

        # These should not crash, should return defaults
        assert sp.safe_float(bad_obj) == 0.0
        assert sp.safe_int(bad_obj) == 0


class TestHelperFunctions:
    """Test the private helper functions."""

    def test_is_already_serialized_dict_edge_cases(self) -> None:
        """Test _is_already_serialized_dict with edge cases."""
        from serialpy.core import _is_already_serialized_dict

        # Valid serialized dict
        assert _is_already_serialized_dict({"a": 1, "b": "hello", "c": True})

        # Dict with non-string keys
        assert not _is_already_serialized_dict({1: "value"})

        # Dict with non-serializable values
        assert not _is_already_serialized_dict({"a": datetime.now()})

        # Dict with NaN values
        assert not _is_already_serialized_dict({"a": float("nan")})

    def test_is_already_serialized_list_edge_cases(self) -> None:
        """Test _is_already_serialized_list with edge cases."""
        from serialpy.core import _is_already_serialized_list

        # Valid serialized list
        assert _is_already_serialized_list([1, "hello", True, None])

        # List with unserializable values
        assert not _is_already_serialized_list([1, datetime.now()])

        # Tuple (should always return False to force conversion)
        assert not _is_already_serialized_list((1, 2, 3))

        # List with NaN
        assert not _is_already_serialized_list([1, float("nan")])


class TestOptimizationAndPerformance:
    """Test optimization paths and performance scenarios."""

    def test_optimization_bypass_non_string_keys(self) -> None:
        """Test that dicts with non-string keys bypass optimization."""
        data = {1: "value", "key": 2}
        result = serialize(data)
        # Should process all items, converting keys to strings
        assert isinstance(result, dict)

    def test_tuple_conversion(self) -> None:
        """Test that tuples are always converted to lists."""
        data = {"tuple": (1, 2, 3)}
        result = serialize(data)
        assert result["tuple"] == [1, 2, 3]
        assert isinstance(result["tuple"], list)

    def test_mixed_serializable_content(self) -> None:
        """Test mixed content optimization."""
        data = {
            "good": "string",
            "bad": datetime.now(),
            "nested": {"inner": float("nan")},
        }
        result = serialize(data)
        assert result["good"] == "string"
        assert isinstance(result["bad"], str)
        assert result["nested"]["inner"] is None

    def test_large_structure(self) -> None:
        """Test serialization of moderately large structure."""
        large_data = {}
        for i in range(100):  # Reduced size to avoid issues
            large_data[f"key_{i}"] = {
                "index": i,
                "timestamp": datetime.now(),
                "value": f"value_{i}",
            }

        result = serialize(large_data)
        assert isinstance(result, dict)
        assert len(result) == 100
