"""Security tests for datason.

Tests for security protections including circular reference detection,
depth limits, size limits, and resource exhaustion prevention.
"""

import os
import sys
import warnings

import pytest

from datason import serialize
from datason.core import (
    MAX_OBJECT_SIZE,
    MAX_SERIALIZATION_DEPTH,
    MAX_STRING_LENGTH,
    SecurityError,
)

# Environment detection for CI vs local testing
IS_CI = any(
    [
        os.getenv("CI"),
        os.getenv("GITHUB_ACTIONS"),
        os.getenv("TRAVIS"),
        os.getenv("CIRCLECI"),
        os.getenv("JENKINS_URL"),
        os.getenv("GITLAB_CI"),
    ]
)

# Test parameters based on environment
if IS_CI:
    # Conservative limits for CI - test the functionality without resource exhaustion
    TEST_DEPTH_LIMIT = min(MAX_SERIALIZATION_DEPTH + 10, 200)
    TEST_SIZE_LIMIT = min(MAX_OBJECT_SIZE + 1000, 50_000)
    SKIP_INTENSIVE = True
else:
    # Local testing - always ensure we exceed the security limit for meaningful tests
    TEST_DEPTH_LIMIT = MAX_SERIALIZATION_DEPTH + 50
    TEST_SIZE_LIMIT = MAX_OBJECT_SIZE + 1000
    SKIP_INTENSIVE = False


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
        """Test that excessive depth raises SecurityError.

        Note: Uses conservative limits in CI environments to avoid timeouts,
        but tests full limits locally for thorough validation.
        """
        # For the depth test, we need to be more careful about Python's recursion limit
        # Use a depth that definitively exceeds our security limit but is safe for recursion
        if IS_CI:
            # In CI, use small but meaningful test
            test_depth = min(MAX_SERIALIZATION_DEPTH + 10, 150)
        else:
            # Locally, we can test closer to the limit, but be conservative
            # Make sure we exceed our security depth but stay well under Python's limit
            import sys

            python_limit = sys.getrecursionlimit()
            # Calculate safe depth that leaves room for Python's internal operations
            safe_max_depth = python_limit - 300

            # Only test if we can safely exceed our security limit
            if safe_max_depth > MAX_SERIALIZATION_DEPTH:
                test_depth = min(MAX_SERIALIZATION_DEPTH + 50, safe_max_depth)
            else:
                # If we can't safely test the real limit, use a smaller test
                pytest.skip(
                    f"Cannot safely test depth limit: Python recursion limit ({python_limit}) too close to security limit ({MAX_SERIALIZATION_DEPTH})"
                )

        # Build nested structure
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
        """Test that excessively large dicts raise SecurityError.

        Note: Uses conservative size in CI environments to avoid resource exhaustion,
        but tests actual limits locally.
        """
        huge_dict = {f"key_{i}": i for i in range(TEST_SIZE_LIMIT)}

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
        """Test that excessively large lists raise SecurityError.

        Note: Uses conservative size in CI environments to avoid resource exhaustion,
        but tests actual limits locally.
        """
        huge_list = list(range(TEST_SIZE_LIMIT))

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

    @pytest.mark.skipif(
        SKIP_INTENSIVE, reason="Intensive test skipped in CI environment"
    )
    def test_memory_intensive_dict_limits(self):
        """Test dictionary limits with memory-intensive scenarios (local only)."""
        if MAX_OBJECT_SIZE < 5_000_000:
            pytest.skip("MAX_OBJECT_SIZE too small for this test")

        # Test with larger dict that might stress memory
        large_size = MAX_OBJECT_SIZE + 5000
        huge_dict = {f"key_{i}": f"value_{i}" * 10 for i in range(large_size)}

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_dict)

        assert "Dictionary size" in str(exc_info.value)

    @pytest.mark.skipif(
        SKIP_INTENSIVE, reason="Intensive test skipped in CI environment"
    )
    def test_memory_intensive_list_limits(self):
        """Test list limits with memory-intensive scenarios (local only)."""
        if MAX_OBJECT_SIZE < 5_000_000:
            pytest.skip("MAX_OBJECT_SIZE too small for this test")

        # Test with larger list that might stress memory
        large_size = MAX_OBJECT_SIZE + 5000
        huge_list = [f"item_{i}" * 10 for i in range(large_size)]

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_list)

        assert "List/tuple size" in str(exc_info.value)


class TestNumpySecurityLimits:
    """Test security limits for numpy arrays."""

    def test_normal_numpy_array(self):
        """Test that normal numpy arrays serialize correctly."""
        np = pytest.importorskip("numpy")

        arr = np.array([1, 2, 3, 4, 5])
        result = serialize(arr)
        assert result == [1, 2, 3, 4, 5]

    def test_large_numpy_array_raises_error(self):
        """Test that excessively large numpy arrays raise SecurityError.

        Note: Uses conservative size in CI environments.
        """
        np = pytest.importorskip("numpy")

        huge_array = np.zeros(TEST_SIZE_LIMIT)

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

    @pytest.mark.skipif(
        SKIP_INTENSIVE, reason="Intensive test skipped in CI environment"
    )
    def test_memory_intensive_numpy_limits(self):
        """Test numpy limits with memory-intensive scenarios (local only)."""
        np = pytest.importorskip("numpy")

        if MAX_OBJECT_SIZE < 5_000_000:
            pytest.skip("MAX_OBJECT_SIZE too small for this test")

        # Test with larger array that might stress memory
        large_size = MAX_OBJECT_SIZE + 10000
        huge_array = np.ones(large_size, dtype=np.float64)

        with pytest.raises(SecurityError) as exc_info:
            serialize(huge_array)

        assert "NumPy array size" in str(exc_info.value)


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
        """Test that excessively large DataFrames raise SecurityError.

        Note: Uses conservative size in CI environments.
        """
        pd = pytest.importorskip("pandas")

        # Create large DataFrame with single column to avoid complex size calculations
        df = pd.DataFrame({"col": range(TEST_SIZE_LIMIT)})

        with pytest.raises(SecurityError) as exc_info:
            serialize(df)

        assert "DataFrame size" in str(exc_info.value)

    def test_large_series_raises_error(self):
        """Test that excessively large Series raise SecurityError.

        Note: Uses conservative size in CI environments.
        """
        pd = pytest.importorskip("pandas")

        large_series = pd.Series(range(TEST_SIZE_LIMIT))

        with pytest.raises(SecurityError) as exc_info:
            serialize(large_series)

        assert "Series/Index size" in str(exc_info.value)

    @pytest.mark.skipif(
        SKIP_INTENSIVE, reason="Intensive test skipped in CI environment"
    )
    def test_memory_intensive_pandas_limits(self):
        """Test pandas limits with memory-intensive scenarios (local only)."""
        pd = pytest.importorskip("pandas")

        if MAX_OBJECT_SIZE < 5_000_000:
            pytest.skip("MAX_OBJECT_SIZE too small for this test")

        # Test with larger DataFrame that might stress memory
        large_size = MAX_OBJECT_SIZE + 10000
        df = pd.DataFrame(
            {
                "col1": range(large_size),
                "col2": [f"text_{i}" for i in range(large_size)],
            }
        )

        with pytest.raises(SecurityError) as exc_info:
            serialize(df)

        assert "DataFrame size" in str(exc_info.value)


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

    def test_environment_configuration(self):
        """Test that environment configuration is working correctly."""
        # Test that we have different configurations for CI vs local
        if IS_CI:
            assert TEST_DEPTH_LIMIT <= 200, "CI depth limit should be conservative"
            assert TEST_SIZE_LIMIT <= 50_000, "CI size limit should be conservative"
            assert SKIP_INTENSIVE is True, "Intensive tests should be skipped in CI"
        else:
            # Local testing should use more thorough limits
            assert TEST_DEPTH_LIMIT > MAX_SERIALIZATION_DEPTH, (
                "Local depth should exceed security limit"
            )
            assert TEST_SIZE_LIMIT > MAX_OBJECT_SIZE, (
                "Local size should exceed security limit"
            )
            assert SKIP_INTENSIVE is False, "Intensive tests should run locally"

        # Verify our test limits are still meaningful
        assert TEST_DEPTH_LIMIT > MAX_SERIALIZATION_DEPTH, (
            "Test depth should exceed limit to trigger error"
        )
        assert TEST_SIZE_LIMIT > MAX_OBJECT_SIZE, (
            "Test size should exceed limit to trigger error"
        )
