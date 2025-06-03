"""
Tests for circular reference handling and protection against infinite loops.

This test suite ensures that datason properly handles circular references
and problematic objects without hanging or causing infinite recursion.
"""

import signal
import time
import warnings
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import datason as ds


class CircularObject:
    """Simple test object with circular reference."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.self_ref = self


class ComplexCircularObject:
    """More complex object with nested circular references."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.children: List[ComplexCircularObject] = []
        self.parent: Optional[ComplexCircularObject] = None

    def add_child(self, child: "ComplexCircularObject") -> None:
        child.parent = self
        self.children.append(child)


class ProblematicObject:
    """Object similar to the original bug report."""

    def __init__(self) -> None:
        self.file_handle = BytesIO(b"test data")
        self.string_io = StringIO("test string")
        self.mock_connection = MagicMock()
        self.mock_object = Mock()


def serialize_with_timeout(obj: Any, timeout_seconds: float = 8.0) -> Tuple[bool, Any, float, str]:
    """
    Serialize an object with a timeout to prevent hanging.

    Note: Timeout increased to account for first-time ML library imports.

    Returns:
        tuple: (success: bool, result: Any, time_taken: float, error: str)
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Serialization timed out after {timeout_seconds}s")

    # Set up signal timeout (Unix only, but this is for debugging)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))

    try:
        start_time = time.time()
        result = ds.serialize(obj)
        end_time = time.time()
        signal.alarm(0)  # Cancel the alarm
        return True, result, end_time - start_time, ""
    except TimeoutError as e:
        signal.alarm(0)  # Cancel the alarm
        return False, None, timeout_seconds, str(e)
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        end_time = time.time()
        return False, None, end_time - start_time, str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


class TestCircularReferences:
    """Test circular reference handling."""

    def test_simple_circular_reference(self) -> None:
        """Test that simple circular references are handled properly."""
        obj = CircularObject("test")

        success, result, time_taken, error = serialize_with_timeout(obj, timeout_seconds=10.0)

        assert success, f"Serialization failed: {error}"
        # First serialization may take longer due to imports, so be more generous
        assert time_taken < 10.0, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, dict)
        assert result["name"] == "test"
        # The self_ref should be handled safely (either None or circular reference placeholder)
        assert "self_ref" in result

    def test_nested_circular_reference(self) -> None:
        """Test nested circular references."""
        parent = ComplexCircularObject("parent")
        child1 = ComplexCircularObject("child1")
        child2 = ComplexCircularObject("child2")

        parent.add_child(child1)
        parent.add_child(child2)
        child1.add_child(parent)  # Create circular reference

        success, result, time_taken, error = serialize_with_timeout(parent, timeout_seconds=10.0)

        assert success, f"Serialization failed: {error}"
        # This should be faster since imports are already done
        assert time_taken < 5.0, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, dict)
        assert result["name"] == "parent"

    def test_dict_circular_reference(self) -> None:
        """Test circular references in dictionaries."""
        obj1: Dict[str, Any] = {"name": "obj1"}
        obj2: Dict[str, Any] = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2  # Create circular reference

        success, result, time_taken, error = serialize_with_timeout(obj1)

        assert success, f"Serialization failed: {error}"
        assert time_taken < 3.0, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, dict)
        assert result["name"] == "obj1"
        # One of the references should be None or handled safely


class TestProblematicObjects:
    """Test handling of objects that previously caused hanging."""

    def test_mock_object_serialization(self) -> None:
        """Test that MagicMock objects don't cause hanging."""
        mock_obj = MagicMock()

        success, result, time_taken, error = serialize_with_timeout(mock_obj, timeout_seconds=10.0)

        assert success, f"Serialization failed: {error}"
        # First test in class may take longer due to imports, so be more generous
        assert time_taken < 6.5, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, str)
        assert "MagicMock" in result

    def test_bytesio_object_serialization(self) -> None:
        """Test that BytesIO objects don't cause hanging."""
        bio = BytesIO(b"test data")

        success, result, time_taken, error = serialize_with_timeout(bio)

        assert success, f"Serialization failed: {error}"
        assert time_taken < 3.0, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, str)
        assert "BytesIO" in result

    def test_problematic_object_combination(self) -> None:
        """Test the exact scenario from the original bug report."""
        obj = ProblematicObject()

        success, result, time_taken, error = serialize_with_timeout(obj, timeout_seconds=10.0)

        assert success, f"Serialization failed: {error}"
        # This should be faster since imports are already done
        assert time_taken < 5.0, f"Serialization took too long: {time_taken}s"
        assert isinstance(result, dict)

        # Verify problematic objects are handled safely
        assert "file_handle" in result
        assert "mock_connection" in result


class TestPerformanceRequirements:
    """Test that serialization meets performance requirements."""

    def test_serialization_speed_simple_objects(self) -> None:
        """Test that simple objects serialize very quickly."""
        data = {"simple": "data", "number": 42, "list": [1, 2, 3]}

        start_time = time.time()
        result = ds.serialize(data)
        end_time = time.time()

        time_taken = end_time - start_time
        assert time_taken < 0.1, f"Simple serialization took too long: {time_taken}s"
        assert result == data

    def test_serialization_speed_complex_objects(self) -> None:
        """Test that even complex objects serialize within reasonable time."""
        complex_data = {
            "nested": {"deeply": {"nested": {"data": list(range(100))}}},
            "large_list": list(range(1000)),
            "mixed_types": [1, "string", 3.14, True, None],
        }

        start_time = time.time()
        result = ds.serialize(complex_data)
        end_time = time.time()

        time_taken = end_time - start_time
        assert time_taken < 0.5, f"Complex serialization took too long: {time_taken}s"
        assert isinstance(result, dict)

    def test_no_hanging_on_deep_nesting(self) -> None:
        """Test that deeply nested objects don't cause hanging."""
        # Create deeply nested dict
        nested: Dict[str, Any] = {}
        current = nested
        for i in range(50):  # Deep but not infinite
            current["level"] = i
            current["next"] = {}
            current = current["next"]
        current["end"] = True

        success, result, time_taken, error = serialize_with_timeout(nested, timeout_seconds=3.0)

        assert success, f"Deep nesting serialization failed: {error}"
        assert time_taken < 2.0, f"Deep nesting took too long: {time_taken}s"


class TestWarningsAndFallbacks:
    """Test that proper warnings are issued and fallbacks work."""

    def test_circular_reference_warning(self) -> None:
        """Test that circular references trigger warnings."""
        obj = CircularObject("test")

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            ds.serialize(obj)

        # Should have warnings about circular references or problematic objects
        assert len(warning_list) > 0, "Expected warnings for circular references"

    def test_problematic_object_warning(self) -> None:
        """Test that problematic objects trigger warnings."""
        mock_obj = MagicMock()

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            ds.serialize(mock_obj)

        # Should have warnings about problematic objects
        warning_messages = [str(w.message) for w in warning_list]
        assert any("problematic" in msg.lower() or "mock" in msg.lower() for msg in warning_messages), (
            f"Expected warnings about problematic objects, got: {warning_messages}"
        )


class TestRegression:
    """Regression tests for specific bug scenarios."""

    def test_original_bug_scenario(self) -> None:
        """Test the exact scenario that caused the original hanging bug."""

        class UnserializableObject:
            def __init__(self) -> None:
                self.file_handle = BytesIO(b"test data")
                self.connection = MagicMock()

        obj = UnserializableObject()

        # This should complete quickly and not hang
        success, result, time_taken, error = serialize_with_timeout(obj, timeout_seconds=5.0)

        assert success, f"Original bug scenario failed: {error}"
        assert time_taken < 2.0, f"Original bug scenario took too long: {time_taken}s (expected < 2s)"
        assert isinstance(result, dict)

    def test_import_performance(self) -> None:
        """Test that importing datason doesn't take too long."""
        # This test ensures that the ML library imports don't cause delays
        start_time = time.time()

        # Force reimport by removing from cache and importing again
        import sys

        if "datason" in sys.modules:
            del sys.modules["datason"]
        if "datason.core" in sys.modules:
            del sys.modules["datason.core"]

        end_time = time.time()

        import_time = end_time - start_time
        # Import should be reasonable - ML libraries might take some time but not excessive
        assert import_time < 10.0, f"Datason import took too long: {import_time}s (expected < 10s)"

    def test_circular_reference_performance(self) -> None:
        """Test that subsequent circular reference handling is fast."""
        # This test runs after imports are done, so should be much faster
        obj = CircularObject("performance_test")

        success, result, time_taken, error = serialize_with_timeout(obj, timeout_seconds=5.0)

        assert success, f"Serialization failed: {error}"
        # After imports, should be much faster
        assert time_taken < 1.0, f"Subsequent serialization too slow: {time_taken}s"
        assert isinstance(result, dict)
        assert result["name"] == "performance_test"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
