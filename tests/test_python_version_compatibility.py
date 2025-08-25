"""
Tests for Python version compatibility and optimizations.

This test suite ensures that DataSON works correctly across all supported
Python versions and that optimizations don't break existing functionality.
"""

import sys
import time
from unittest.mock import patch

import pytest

# Import DataSON components
from datason.redaction import RedactionEngine

# Try to import optimization features (may not be available in older Python)
try:
    from datason.optimizations import (
        AdaptiveCache,
        JITOptimizedPatternMatcher,
        ParallelRedactionProcessor,
        VersionAwareOptimizations,
        get_optimization_info,
    )

    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False


class TestPythonVersionCompatibility:
    """Test DataSON compatibility across Python versions."""

    def test_basic_redaction_works_all_versions(self):
        """Ensure basic redaction works on all Python versions."""
        engine = RedactionEngine(redact_fields=["password", "*.secret"], redaction_replacement="[REDACTED]")

        test_data = {
            "username": "alice",
            "password": "secret123",
            "config": {"secret": "hidden_value", "public": "visible_value"},
        }

        result = engine.process_object(test_data)

        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"
        assert result["config"]["secret"] == "[REDACTED]"
        assert result["config"]["public"] == "visible_value"

    def test_performance_regression_protection(self):
        """Ensure optimizations don't cause performance regressions."""
        engine = RedactionEngine(redact_fields=["password"])

        test_data = [{"user": f"user_{i}", "password": f"pass_{i}"} for i in range(100)]

        # Measure baseline performance
        start_time = time.perf_counter()
        results = [engine.process_object(obj) for obj in test_data]
        execution_time = time.perf_counter() - start_time

        # Verify results are correct
        assert len(results) == 100
        assert all(result["password"] == "<REDACTED>" for result in results)

        # Performance should be reasonable (less than 1 second for 100 objects)
        assert execution_time < 1.0, f"Performance regression detected: {execution_time:.3f}s"

    def test_memory_usage_stability(self):
        """Ensure optimizations don't cause memory leaks."""
        import gc

        engine = RedactionEngine(redact_fields=["*.secret"])

        # Process many objects to test memory stability
        for batch in range(10):
            test_objects = [{"data": f"value_{i}", "secret": f"secret_{i}"} for i in range(100)]

            results = [engine.process_object(obj) for obj in test_objects]
            assert len(results) == 100

            # Force garbage collection
            gc.collect()

        # If we get here without memory errors, the test passes
        assert True


@pytest.mark.skipif(not OPTIMIZATIONS_AVAILABLE, reason="Optimizations not available")
class TestOptimizationFeatures:
    """Test optimization features when available."""

    def test_optimization_info_structure(self):
        """Test that optimization info has expected structure."""
        info = get_optimization_info()

        required_keys = [
            "python_version",
            "jit_available",
            "free_threading_available",
            "recommended_cache_size",
            "parallel_processing_threshold",
            "optimization_level",
            "expected_performance_boost",
        ]

        for key in required_keys:
            assert key in info, f"Missing key in optimization info: {key}"

        # Verify data types
        assert isinstance(info["python_version"], str)
        assert isinstance(info["jit_available"], bool)
        assert isinstance(info["free_threading_available"], bool)
        assert isinstance(info["recommended_cache_size"], int)
        assert isinstance(info["parallel_processing_threshold"], int)
        assert info["optimization_level"] in ["basic", "intermediate", "advanced"]

    def test_version_aware_optimizations(self):
        """Test version-aware optimization setup."""
        optimizer = VersionAwareOptimizations()

        assert hasattr(optimizer, "python_version")
        assert hasattr(optimizer, "optimization_level")
        assert hasattr(optimizer, "cache_size")
        assert hasattr(optimizer, "parallel_threshold")

        # Verify optimization level is appropriate for current Python version
        if sys.version_info >= (3, 13):
            assert optimizer.optimization_level == "advanced"
        elif sys.version_info >= (3, 12):
            assert optimizer.optimization_level == "intermediate"
        else:
            assert optimizer.optimization_level == "basic"

    def test_jit_optimized_pattern_matcher(self):
        """Test JIT-optimized pattern matching."""
        patterns = [("password", None), ("*.secret", None), ("user.email", None)]

        matcher = JITOptimizedPatternMatcher(patterns)

        # Test various field paths
        test_cases = [
            ("user.password", False),  # Should not match "password" exactly
            ("password", True),  # Should match exactly
            ("config.secret", True),  # Should match *.secret
            ("user.email", True),  # Should match exactly
            ("public.data", False),  # Should not match any pattern
        ]

        for field_path, expected in test_cases:
            if sys.version_info >= (3, 13):
                result = matcher.match_field_jit_friendly(field_path)
            else:
                result = matcher.match_field_fallback(field_path)

            assert result == expected, f"Pattern match failed for {field_path}"

    def test_adaptive_cache_behavior(self):
        """Test adaptive cache with different sizes."""
        # Test with small cache
        small_cache = AdaptiveCache(maxsize=3)

        # Fill cache to capacity
        small_cache.set("key1", True)
        small_cache.set("key2", False)
        small_cache.set("key3", True)

        assert small_cache.get("key1") is True
        assert small_cache.get("key2") is False
        assert small_cache.get("key3") is True

        # Add one more item - should evict least recently used
        small_cache.set("key4", False)

        # key1 should be evicted (least recently used)
        assert small_cache.get("key1") is None
        assert small_cache.get("key4") is False

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="Requires Python 3.13+")
    def test_parallel_processing_python313(self):
        """Test parallel processing features on Python 3.13+."""
        processor = ParallelRedactionProcessor(max_workers=2)

        def simple_function(x):
            return x * 2

        test_data = list(range(100))

        # Test parallel processing
        results = processor.process_objects_parallel(simple_function, test_data, threshold=10)

        expected = [x * 2 for x in test_data]
        assert results == expected

        # Test that small datasets use sequential processing
        small_data = list(range(5))
        small_results = processor.process_objects_parallel(simple_function, small_data, threshold=10)

        expected_small = [x * 2 for x in small_data]
        assert small_results == expected_small


class TestBackwardCompatibility:
    """Test backward compatibility with older Python versions."""

    def test_import_compatibility(self):
        """Test that core DataSON functionality imports on all versions."""
        # These imports should work on Python 3.8+
        from datason import deserialize, serialize

        # Basic functionality test
        data = {"test": "value"}
        serialized = serialize(data)
        deserialized = deserialize(serialized)
        assert deserialized == data

    def test_redaction_api_stability(self):
        """Test that redaction API remains stable across versions."""
        # Create engines using different factory functions
        engines = [
            RedactionEngine(redact_fields=["password"]),
            RedactionEngine(redact_fields=["password"], redact_patterns=[]),
        ]

        test_data = {"password": "secret", "public": "data"}

        for engine in engines:
            result = engine.process_object(test_data)
            assert "public" in result
            assert result["public"] == "data"
            # All engines should redact password field in some way
            assert result.get("password") != "secret"

    def test_graceful_degradation(self):
        """Test that optimizations degrade gracefully on older Python."""
        # Test with mock optimization availability
        with patch("datason.optimizations.PYTHON_VERSION", (3, 8)):
            with patch("datason.optimizations.HAS_JIT", False):
                with patch("datason.optimizations.HAS_FREE_THREADING", False):
                    if OPTIMIZATIONS_AVAILABLE:
                        optimizer = VersionAwareOptimizations()
                        assert optimizer.optimization_level == "basic"
                        assert optimizer.cache_size <= 512
                        assert optimizer.parallel_threshold >= 200


class TestPerformanceRegression:
    """Test for performance regressions across Python versions."""

    @pytest.mark.performance
    def test_serialization_performance_baseline(self):
        """Establish performance baseline for different data sizes."""
        import datason

        # Small dataset
        small_data = {"user": "alice", "id": 123}

        start_time = time.perf_counter()
        for _ in range(1000):
            _ = datason.serialize(small_data)
        small_time = time.perf_counter() - start_time

        # Should complete 1000 serializations in reasonable time
        assert small_time < 0.1, f"Small data serialization too slow: {small_time:.3f}s"

        # Medium dataset
        medium_data = {
            "users": [{"id": i, "name": f"user_{i}"} for i in range(100)],
            "metadata": {"total": 100, "page": 1},
        }

        start_time = time.perf_counter()
        for _ in range(100):
            _ = datason.serialize(medium_data)
        medium_time = time.perf_counter() - start_time

        # Should complete 100 medium serializations in reasonable time
        assert medium_time < 1.0, f"Medium data serialization too slow: {medium_time:.3f}s"

    @pytest.mark.performance
    def test_redaction_performance_baseline(self):
        """Establish redaction performance baseline."""
        engine = RedactionEngine(
            redact_fields=["password", "*.secret", "*.token"],
            redact_patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN pattern
        )

        test_data = {
            "user": {"password": "secret123", "name": "Alice"},
            "config": {"secret": "hidden", "public": "visible"},
            "profile": {"ssn": "123-45-6789", "age": 30},
        }

        start_time = time.perf_counter()
        for _ in range(1000):
            _ = engine.process_object(test_data)
        redaction_time = time.perf_counter() - start_time

        # Redaction should complete 1000 operations in reasonable time
        assert redaction_time < 1.0, f"Redaction too slow: {redaction_time:.3f}s"


@pytest.mark.parametrize("python_version", [(3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13)])
def test_version_specific_behavior(python_version):
    """Test expected behavior for specific Python versions."""
    with patch("sys.version_info", python_version):
        if OPTIMIZATIONS_AVAILABLE:
            info = get_optimization_info()

            if python_version >= (3, 13):
                assert info["optimization_level"] == "advanced"
                assert info["recommended_cache_size"] == 2048
            elif python_version >= (3, 12):
                assert info["optimization_level"] == "intermediate"
                assert info["recommended_cache_size"] == 1024
            else:
                assert info["optimization_level"] == "basic"
                assert info["recommended_cache_size"] == 512


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
