"""Security tests for Datason.

Tests for security protections including circular reference detection,
depth limits, size limits, and resource exhaustion prevention.
"""

import warnings

import pytest

from datason import serialize
from datason.core import (
    MAX_OBJECT_SIZE,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    SecurityError,
)


class TestCircularReferenceProtection:
    """Test protection against circular references."""

    def test_simple_circular_reference(self):
        """Test that simple circular references are handled safely."""
        a = {}
        b = {"a": a}
        a["b"] = b  # Create circular reference

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(a)

            # Should get a warning about circular reference
            assert len(w) >= 1
            assert "circular reference" in str(w[0].message).lower()

            # Result should be safe (no infinite recursion)
            assert isinstance(result, dict)

    def test_list_circular_reference(self):
        """Test circular references in lists."""
        a = []
        b = [a]
        a.append(b)  # Create circular reference

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(a)

            # Should handle gracefully
            assert isinstance(result, list)

    def test_self_reference(self):
        """Test object referencing itself."""
        a = {}
        a["self"] = a  # Self-reference

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(a)

            # Should handle gracefully
            assert isinstance(result, dict)
            assert result["self"] is None  # Circular ref replaced with None


class TestDepthLimits:
    """Test protection against excessive recursion depth."""

    def test_deep_nesting_within_limits(self):
        """Test that reasonable nesting depth works."""
        # Create nested dict within reasonable limits (100 levels)
        nested = {}
        current = nested
        for i in range(100):
            current["next"] = {}
            current = current["next"]
        current["value"] = "deep"

        # Should serialize successfully
        result = serialize(nested)
        assert isinstance(result, dict)

    def test_excessive_depth_raises_error(self):
        """Test that excessive depth raises SecurityError."""
        # Create deeply nested structure beyond our limits but not Python's recursion limit
        # Use our MAX_SERIALIZATION_DEPTH + a smaller amount to avoid hitting Python's limit
        test_depth = (
            MAX_SERIALIZATION_DEPTH + 50
        )  # Much smaller than Python's ~1000 default

        nested = {}
        current = nested
        for i in range(test_depth):
            current["next"] = {}
            current = current["next"]
        current["value"] = "too_deep"

        # Should raise SecurityError
        with pytest.raises(SecurityError) as exc_info:
            serialize(nested)

        assert "Maximum serialization depth" in str(exc_info.value)


class TestSizeLimits:
    """Test protection against resource exhaustion."""

    def test_large_dict_within_limits(self):
        """Test that reasonably large dicts work."""
        large_dict = {f"key_{i}": i for i in range(1000)}
        result = serialize(large_dict)
        assert isinstance(result, dict)
        assert len(result) == 1000

    def test_excessive_dict_size_raises_error(self):
        """Test that excessively large dicts raise SecurityError."""
        # Create dict larger than MAX_OBJECT_SIZE
        huge_dict = {f"key_{i}": i for i in range(MAX_OBJECT_SIZE + 1000)}

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_dict)

        assert "Dictionary size" in str(exc_info.value)
        assert "exceeds maximum" in str(exc_info.value)

    def test_large_list_within_limits(self):
        """Test that reasonably large lists work."""
        large_list = list(range(1000))
        result = serialize(large_list)
        assert isinstance(result, list)
        assert len(result) == 1000

    def test_excessive_list_size_raises_error(self):
        """Test that excessively large lists raise SecurityError."""
        huge_list = list(range(MAX_OBJECT_SIZE + 1000))

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_list)

        assert "List/tuple size" in str(exc_info.value)

    def test_large_string_truncation(self):
        """Test that very large strings are truncated safely."""
        large_string = "x" * (MAX_STRING_LENGTH + 1000)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(large_string)

            # Should get warning about truncation
            assert len(w) >= 1
            assert "exceeds maximum" in str(w[0].message)

            # Result should be truncated
            assert isinstance(result, str)
            assert len(result) == MAX_STRING_LENGTH + len("...[TRUNCATED]")
            assert result.endswith("...[TRUNCATED]")


class TestNumpySecurityLimits:
    """Test security limits for numpy arrays."""

    def test_normal_numpy_array(self):
        """Test that normal numpy arrays serialize correctly."""
        np = pytest.importorskip("numpy")

        arr = np.array([1, 2, 3, 4, 5])
        result = serialize(arr)
        assert result == [1, 2, 3, 4, 5]

    def test_large_numpy_array_raises_error(self):
        """Test that excessively large numpy arrays raise SecurityError."""
        np = pytest.importorskip("numpy")

        # Create array larger than MAX_OBJECT_SIZE
        huge_array = np.zeros(MAX_OBJECT_SIZE + 1000)

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_array)

        assert "NumPy array size" in str(exc_info.value)

    def test_numpy_string_truncation(self):
        """Test that large numpy strings are truncated."""
        np = pytest.importorskip("numpy")

        large_numpy_str = np.str_("x" * (MAX_STRING_LENGTH + 100))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(large_numpy_str)

            # Should get warning and truncation
            assert len(w) >= 1
            assert isinstance(result, str)
            assert result.endswith("...[TRUNCATED]")


class TestPandasSecurityLimits:
    """Test security limits for pandas objects."""

    def test_normal_dataframe(self):
        """Test that normal DataFrames serialize correctly."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = serialize(df)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_large_dataframe_raises_error(self):
        """Test that excessively large DataFrames raise SecurityError."""
        pd = pytest.importorskip("pandas")

        # Create DataFrame larger than MAX_OBJECT_SIZE
        # Use sqrt to create reasonable dimensions
        import math

        size = int(math.sqrt(MAX_OBJECT_SIZE)) + 100

        # Create large DataFrame
        df = pd.DataFrame({"col": range(size * size)})

        with pytest.raises(SecurityError) as exc_info:
            serialize(df)

        assert "DataFrame size" in str(exc_info.value)

    def test_large_series_raises_error(self):
        """Test that excessively large Series raise SecurityError."""
        pd = pytest.importorskip("pandas")

        large_series = pd.Series(range(MAX_OBJECT_SIZE + 1000))

        with pytest.raises(SecurityError) as exc_info:
            serialize(large_series)

        assert "Series/Index size" in str(exc_info.value)


class TestErrorHandling:
    """Test improved error handling without information leakage."""

    def test_object_with_failing_dict_method(self):
        """Test handling of objects with failing .dict() method."""

        class BadDictObject:
            def dict(self):
                raise RuntimeError("Simulated failure")

        obj = BadDictObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(obj)

            # Should get warning about failure
            assert len(w) >= 1
            assert "Failed to serialize object using .dict() method" in str(
                w[0].message
            )

            # Should fall back to string representation
            assert isinstance(result, str)

    def test_object_with_failing_vars(self):
        """Test handling of objects with failing vars() call."""

        class BadVarsObject:
            def __init__(self):
                # Create an object that vars() might fail on
                pass

            def __getattribute__(self, name):
                if name == "__dict__":
                    raise RuntimeError("Simulated __dict__ failure")
                return super().__getattribute__(name)

        obj = BadVarsObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = serialize(obj)

            # Should handle gracefully and return safe representation
            assert isinstance(result, str)

    def test_unprintable_object(self):
        """Test handling of objects that can't be converted to string."""

        class UnprintableObject:
            def __str__(self):
                raise RuntimeError("Cannot convert to string")

            def __repr__(self):
                raise RuntimeError("Cannot represent")

        obj = UnprintableObject()
        result = serialize(obj)

        # Should return safe fallback representation
        assert isinstance(result, str)
        assert "UnprintableObject object at" in result


class TestSecurityConstants:
    """Test that security constants are reasonable."""

    def test_security_constants_exist(self):
        """Test that security constants are defined."""
        from datason.core import (
            MAX_OBJECT_SIZE,
            MAX_SERIALIZATION_DEPTH,
            MAX_STRING_LENGTH,
        )

        assert isinstance(MAX_SERIALIZATION_DEPTH, int)
        assert isinstance(MAX_OBJECT_SIZE, int)
        assert isinstance(MAX_STRING_LENGTH, int)

        # Should be reasonable values
        assert MAX_SERIALIZATION_DEPTH > 100
        assert MAX_OBJECT_SIZE > 1000
        assert MAX_STRING_LENGTH > 1000

    def test_security_error_class(self):
        """Test that SecurityError class is available."""
        from datason.core import SecurityError

        # Should be a proper exception class
        assert issubclass(SecurityError, Exception)

        # Should be raisable
        with pytest.raises(SecurityError):
            raise SecurityError("Test error")
