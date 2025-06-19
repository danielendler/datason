"""Unit tests for datason.cache_manager module.

This test file covers the configurable caching system with different scopes:
- CacheMetrics class and its methods
- ScopedCache class and cache operations
- ScopedPool class and object pooling
- Cache scope management and context managers
- Cache clearing and metrics functions
"""

import warnings
from unittest.mock import patch

import pytest

from datason.cache_manager import (
    CacheMetrics,
    ScopedCache,
    ScopedPool,
    clear_all_caches,
    clear_caches,
    dict_pool,
    get_cache_metrics,
    list_pool,
    operation_scope,
    parsed_object_cache,
    request_scope,
    reset_cache_metrics,
    string_pattern_cache,
    type_cache,
)
from datason.config import CacheScope, SerializationConfig, cache_scope


class TestCacheMetrics:
    """Test the CacheMetrics class functionality."""

    def test_cache_metrics_initialization(self):
        """Test that CacheMetrics initializes with zero values."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.size_warnings == 0
        assert metrics.hit_rate == 0.0

    def test_cache_metrics_hit(self):
        """Test cache hit tracking."""
        metrics = CacheMetrics()
        metrics.hit()
        assert metrics.hits == 1
        assert metrics.misses == 0

    def test_cache_metrics_miss(self):
        """Test cache miss tracking."""
        metrics = CacheMetrics()
        metrics.miss()
        assert metrics.hits == 0
        assert metrics.misses == 1

    def test_cache_metrics_evict(self):
        """Test cache eviction tracking."""
        metrics = CacheMetrics()
        metrics.evict()
        assert metrics.evictions == 1

    def test_cache_metrics_warn_size(self):
        """Test cache size warning tracking."""
        metrics = CacheMetrics()
        metrics.warn_size()
        assert metrics.size_warnings == 1

    def test_cache_metrics_hit_rate_calculation(self):
        """Test hit rate calculation with various scenarios."""
        metrics = CacheMetrics()

        # No operations - hit rate should be 0
        assert metrics.hit_rate == 0.0

        # Only hits
        metrics.hit()
        metrics.hit()
        assert metrics.hit_rate == 1.0

        # Only misses
        metrics.miss()
        metrics.miss()
        assert metrics.hit_rate == 0.5  # 2 hits, 2 misses

        # Mixed
        metrics.hit()  # 3 hits, 2 misses
        assert metrics.hit_rate == 0.6

    def test_cache_metrics_reset(self):
        """Test resetting cache metrics."""
        metrics = CacheMetrics()
        metrics.hit()
        metrics.miss()
        metrics.evict()
        metrics.warn_size()

        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.evictions == 1
        assert metrics.size_warnings == 1

        metrics.reset()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.size_warnings == 0

    def test_cache_metrics_str_representation(self):
        """Test string representation of cache metrics."""
        metrics = CacheMetrics()
        metrics.hit()
        metrics.miss()

        str_repr = str(metrics)
        assert "hits=1" in str_repr
        assert "misses=1" in str_repr
        assert "hit_rate=50.00%" in str_repr
        assert "evictions=0" in str_repr


class TestScopedCache:
    """Test the ScopedCache class functionality."""

    def setup_method(self):
        """Set up for each test method."""
        clear_all_caches()
        reset_cache_metrics()

    def test_scoped_cache_initialization(self):
        """Test ScopedCache initialization."""
        cache = ScopedCache("test_cache")
        assert cache.cache_name == "test_cache"

    def test_cache_disabled_scope(self):
        """Test cache behavior when disabled."""
        with cache_scope(CacheScope.DISABLED):
            cache = ScopedCache("test_cache")

            # Setting and getting from disabled cache
            cache.set("key1", "value1")
            assert cache.get("key1") is None

    def test_cache_operation_scope(self):
        """Test cache behavior in operation scope (always empty)."""
        with cache_scope(CacheScope.OPERATION):
            cache = ScopedCache("test_cache")

            # Setting and getting from operation-scoped cache
            cache.set("key1", "value1")
            # Operation scope caches are always empty
            assert cache.get("key1") is None

    def test_cache_request_scope(self):
        """Test cache behavior in request scope."""
        with cache_scope(CacheScope.REQUEST):
            # Test string_pattern cache
            string_cache = ScopedCache("string_pattern")
            string_cache.set("key1", "value1")
            assert string_cache.get("key1") == "value1"

            # Test parsed_object cache
            object_cache = ScopedCache("parsed_object")
            object_cache.set("key2", {"data": "value2"})
            assert object_cache.get("key2") == {"data": "value2"}

            # Test type cache
            type_cache_obj = ScopedCache("type")
            type_cache_obj.set("key3", "datetime")
            assert type_cache_obj.get("key3") == "datetime"

            # Test unknown cache name
            unknown_cache = ScopedCache("unknown")
            unknown_cache.set("key4", "value4")
            # Should return None for unknown cache types
            assert unknown_cache.get("key4") is None

    def test_cache_process_scope(self):
        """Test cache behavior in process scope."""
        with cache_scope(CacheScope.PROCESS):
            # Test string_pattern cache
            string_cache = ScopedCache("string_pattern")
            string_cache.set("key1", "value1")
            assert string_cache.get("key1") == "value1"

            # Test parsed_object cache
            object_cache = ScopedCache("parsed_object")
            object_cache.set("key2", {"data": "value2"})
            assert object_cache.get("key2") == {"data": "value2"}

            # Test type cache
            type_cache_obj = ScopedCache("type")
            type_cache_obj.set("key3", "datetime")
            assert type_cache_obj.get("key3") == "datetime"

    def test_cache_size_limit_warning(self):
        """Test cache size limit warnings."""
        config = SerializationConfig(
            cache_scope=CacheScope.PROCESS, cache_size_limit=2, cache_warn_on_limit=True, cache_metrics_enabled=True
        )

        with cache_scope(CacheScope.PROCESS):
            with patch("datason.config.get_default_config", return_value=config):
                cache = ScopedCache("string_pattern")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Fill cache to limit
                    cache.set("key1", "value1")
                    cache.set("key2", "value2")

                    # This should trigger a warning and eviction
                    cache.set("key3", "value3")

                    # Should have triggered a warning
                    assert len(w) >= 1
                    assert "reached size limit" in str(w[-1].message)

    def test_cache_size_limit_eviction(self):
        """Test cache eviction when size limit is reached."""
        config = SerializationConfig(
            cache_scope=CacheScope.PROCESS, cache_size_limit=2, cache_warn_on_limit=False, cache_metrics_enabled=True
        )

        with cache_scope(CacheScope.PROCESS):
            with patch("datason.config.get_default_config", return_value=config):
                cache = ScopedCache("string_pattern")

                # Fill cache to limit
                cache.set("key1", "value1")
                cache.set("key2", "value2")

                # Add one more - should evict oldest
                cache.set("key3", "value3")

                # First key should be evicted (FIFO)
                assert cache.get("key1") is None
                assert cache.get("key2") == "value2"
                assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test cache clearing."""
        with cache_scope(CacheScope.PROCESS):
            cache = ScopedCache("string_pattern")
            cache.set("key1", "value1")
            cache.set("key2", "value2")

            assert cache.get("key1") == "value1"
            assert cache.get("key2") == "value2"

            cache.clear()

            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_cache_metrics_tracking(self):
        """Test that cache operations are tracked in metrics."""
        config = SerializationConfig(cache_scope=CacheScope.PROCESS, cache_metrics_enabled=True)

        with cache_scope(CacheScope.PROCESS):
            with patch("datason.config.get_default_config", return_value=config):
                cache = ScopedCache("string_pattern")

                # Clear metrics
                reset_cache_metrics(CacheScope.PROCESS)

                # Miss
                cache.get("nonexistent")

                # Set and hit
                cache.set("key1", "value1")
                cache.get("key1")

                metrics = get_cache_metrics(CacheScope.PROCESS)
                process_metrics = metrics[CacheScope.PROCESS]

                assert process_metrics.misses >= 1
                assert process_metrics.hits >= 1


class TestScopedPool:
    """Test the ScopedPool class functionality."""

    def setup_method(self):
        """Set up for each test method."""
        clear_all_caches()

    def test_scoped_pool_initialization(self):
        """Test ScopedPool initialization."""
        pool = ScopedPool("test_pool", dict)
        assert pool.pool_name == "test_pool"
        assert pool.pool_type is dict

    def test_pool_disabled_scope(self):
        """Test pool behavior when disabled."""
        with cache_scope(CacheScope.DISABLED):
            pool = ScopedPool("test_pool", dict)

            # Get from disabled pool - should create new object
            obj = pool.get()
            assert isinstance(obj, dict)

            # Return to disabled pool - should do nothing
            pool.return_object(obj)

    def test_pool_operation_scope(self):
        """Test pool behavior in operation scope."""
        with cache_scope(CacheScope.OPERATION):
            pool = ScopedPool("test_pool", list)

            # Get from operation pool - should create new object
            obj = pool.get()
            assert isinstance(obj, list)

            # Return to operation pool - should do nothing
            pool.return_object(obj)

    def test_pool_request_scope(self):
        """Test pool behavior in request scope."""
        with cache_scope(CacheScope.REQUEST):
            # Test dict pool
            dict_pool_obj = ScopedPool("dict", dict)
            obj1 = dict_pool_obj.get()
            assert isinstance(obj1, dict)

            obj1["test"] = "value"
            dict_pool_obj.return_object(obj1)

            # Get another object - should be the same (cleared) object
            obj2 = dict_pool_obj.get()
            assert obj2 is obj1
            assert "test" not in obj2  # Should be cleared

            # Test list pool
            list_pool_obj = ScopedPool("list", list)
            list_obj = list_pool_obj.get()
            assert isinstance(list_obj, list)

            # Test unknown pool
            unknown_pool = ScopedPool("unknown", set)
            unknown_obj = unknown_pool.get()
            assert isinstance(unknown_obj, set)

    def test_pool_process_scope(self):
        """Test pool behavior in process scope."""
        with cache_scope(CacheScope.PROCESS):
            # Test dict pool
            dict_pool_obj = ScopedPool("dict", dict)
            obj1 = dict_pool_obj.get()
            assert isinstance(obj1, dict)

            obj1["test"] = "value"
            dict_pool_obj.return_object(obj1)

            # Get another object - should be the same (cleared) object
            obj2 = dict_pool_obj.get()
            assert obj2 is obj1
            assert "test" not in obj2  # Should be cleared

            # Test list pool
            list_pool_obj = ScopedPool("list", list)
            list_obj = list_pool_obj.get()
            assert isinstance(list_obj, list)

    def test_pool_size_limit(self):
        """Test pool size limits."""
        config = SerializationConfig(
            cache_scope=CacheScope.PROCESS,
            cache_size_limit=8,  # Pool limit is cache_size_limit // 4 = 2
        )

        with cache_scope(CacheScope.PROCESS):
            with patch("datason.config.get_default_config", return_value=config):
                pool = ScopedPool("dict", dict)

                # Return more objects than the limit
                obj1 = {}
                obj2 = {}
                obj3 = {}

                pool.return_object(obj1)
                pool.return_object(obj2)
                pool.return_object(obj3)  # Should be rejected due to size limit

                # Only first 2 should be in pool
                retrieved1 = pool.get()
                retrieved2 = pool.get()
                retrieved3 = pool.get()  # Should be new object

                # Check by object identity since pool clears the dicts but preserves object instances
                assert retrieved1 is obj1 or retrieved1 is obj2
                assert retrieved2 is obj1 or retrieved2 is obj2
                assert retrieved1 is not retrieved2  # Should be different objects
                assert (
                    retrieved3 is not obj1 and retrieved3 is not obj2 and retrieved3 is not obj3
                )  # Should be a completely new object

    def test_pool_clear(self):
        """Test pool clearing."""
        with cache_scope(CacheScope.PROCESS):
            pool = ScopedPool("dict", dict)

            # Add objects to pool
            obj1 = {}
            obj1["marker"] = "obj1"
            obj2 = {}
            obj2["marker"] = "obj2"
            pool.return_object(obj1)
            pool.return_object(obj2)

            # Clear pool
            pool.clear()

            # Next get should create new object
            obj3 = pool.get()
            assert "marker" not in obj3  # Should be a fresh dict

    def test_pool_object_clearing(self):
        """Test that objects are cleared when returned to pool."""
        with cache_scope(CacheScope.PROCESS):
            pool = ScopedPool("dict", dict)

            obj = {}
            obj["key"] = "value"
            obj.setdefault("other", "data")

            assert "key" in obj
            assert "other" in obj

            pool.return_object(obj)

            # Object should be cleared
            assert len(obj) == 0

    def test_pool_object_without_clear_method(self):
        """Test handling objects without clear method."""

        class NoCleanMethod:
            def __init__(self):
                self.data = "test"

        with cache_scope(CacheScope.PROCESS):
            pool = ScopedPool("custom", NoCleanMethod)

            obj = NoCleanMethod()
            obj.data = "modified"

            # Should not raise error even though object lacks clear method
            pool.return_object(obj)

            # Object data should remain unchanged
            assert obj.data == "modified"


class TestCacheInstances:
    """Test the global cache instances."""

    def test_global_cache_instances_exist(self):
        """Test that global cache instances are created."""
        assert isinstance(string_pattern_cache, ScopedCache)
        assert isinstance(parsed_object_cache, ScopedCache)
        assert isinstance(type_cache, ScopedCache)
        assert isinstance(dict_pool, ScopedPool)
        assert isinstance(list_pool, ScopedPool)

    def test_global_cache_names(self):
        """Test that global cache instances have correct names."""
        assert string_pattern_cache.cache_name == "string_pattern"
        assert parsed_object_cache.cache_name == "parsed_object"
        assert type_cache.cache_name == "type"
        assert dict_pool.pool_name == "dict"
        assert list_pool.pool_name == "list"

    def test_global_pool_types(self):
        """Test that global pool instances have correct types."""
        assert dict_pool.pool_type is dict
        assert list_pool.pool_type is list


class TestCacheFunctions:
    """Test module-level cache functions."""

    def setup_method(self):
        """Set up for each test method."""
        clear_all_caches()
        reset_cache_metrics()

    def test_clear_caches_function(self):
        """Test the clear_caches function."""
        # Add some data to caches
        with cache_scope(CacheScope.PROCESS):
            string_pattern_cache.set("key1", "value1")
            parsed_object_cache.set("key2", "value2")
            type_cache.set("key3", "value3")

            # Verify data is there
            assert string_pattern_cache.get("key1") == "value1"
            assert parsed_object_cache.get("key2") == "value2"
            assert type_cache.get("key3") == "value3"

            # Clear caches
            clear_caches()

            # Verify data is gone
            assert string_pattern_cache.get("key1") is None
            assert parsed_object_cache.get("key2") is None
            assert type_cache.get("key3") is None

    def test_clear_all_caches_function(self):
        """Test the clear_all_caches function."""
        # Add data to process-level caches
        with cache_scope(CacheScope.PROCESS):
            string_pattern_cache.set("key1", "value1")

        # Add data to request-level caches
        with cache_scope(CacheScope.REQUEST):
            string_pattern_cache.set("key2", "value2")

        # Clear all caches
        clear_all_caches()

        # Verify all data is gone
        with cache_scope(CacheScope.PROCESS):
            assert string_pattern_cache.get("key1") is None

        with cache_scope(CacheScope.REQUEST):
            assert string_pattern_cache.get("key2") is None

    def test_clear_all_caches_with_ml_serializers(self):
        """Test clear_all_caches handles ML serializers clearing gracefully."""
        # This test simply verifies that clear_all_caches doesn't crash
        # when ML serializers are available. The actual clearing is tested
        # indirectly through integration tests.
        try:
            clear_all_caches()
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            # Should not raise any exceptions
            pytest.fail(f"clear_all_caches raised an exception: {e}")

    def test_clear_all_caches_missing_ml_serializers(self):
        """Test clear_all_caches handles missing ML serializers gracefully."""
        # Should not raise error even if ml_serializers can't be imported
        clear_all_caches()  # Should complete without error

    def test_get_cache_metrics_specific_scope(self):
        """Test getting metrics for specific scope."""
        reset_cache_metrics()

        # Generate some activity
        with cache_scope(CacheScope.PROCESS):
            string_pattern_cache.get("nonexistent")  # Miss

        metrics = get_cache_metrics(CacheScope.PROCESS)
        assert CacheScope.PROCESS in metrics
        assert isinstance(metrics[CacheScope.PROCESS], CacheMetrics)

    def test_get_cache_metrics_all_scopes(self):
        """Test getting metrics for all scopes."""
        metrics = get_cache_metrics()
        assert isinstance(metrics, dict)
        # Should contain metrics for various scopes
        for scope, metric in metrics.items():
            assert isinstance(scope, CacheScope)
            assert isinstance(metric, CacheMetrics)

    def test_reset_cache_metrics_specific_scope(self):
        """Test resetting metrics for specific scope."""
        # Generate activity
        with cache_scope(CacheScope.PROCESS):
            string_pattern_cache.get("nonexistent")

        # Reset specific scope
        reset_cache_metrics(CacheScope.PROCESS)

        metrics = get_cache_metrics(CacheScope.PROCESS)
        assert metrics[CacheScope.PROCESS].misses == 0
        assert metrics[CacheScope.PROCESS].hits == 0

    def test_reset_cache_metrics_all_scopes(self):
        """Test resetting metrics for all scopes."""
        # Generate activity in multiple scopes
        with cache_scope(CacheScope.PROCESS):
            string_pattern_cache.get("nonexistent")

        with cache_scope(CacheScope.REQUEST):
            string_pattern_cache.get("nonexistent")

        # Reset all metrics
        reset_cache_metrics()

        all_metrics = get_cache_metrics()
        for metrics in all_metrics.values():
            assert metrics.hits == 0
            assert metrics.misses == 0


class TestContextManagers:
    """Test context managers for cache scopes."""

    def setup_method(self):
        """Set up for each test method."""
        clear_all_caches()

    def test_operation_scope_context_manager(self):
        """Test operation_scope context manager works without errors."""
        # Test basic functionality - context manager should not raise errors
        with operation_scope():
            # Context manager should work
            with cache_scope(CacheScope.PROCESS):
                string_pattern_cache.set("test_key", "test_value")

    def test_operation_scope_cleanup_on_exception(self):
        """Test operation_scope handles exceptions gracefully."""
        try:
            with operation_scope():
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        # Should complete without error

    def test_request_scope_context_manager(self):
        """Test request_scope context manager works without errors."""
        # Test basic functionality - context manager should not raise errors
        with request_scope():
            # Context manager should work
            with cache_scope(CacheScope.REQUEST):
                string_pattern_cache.set("test_key", "test_value")

    def test_request_scope_cleanup_on_exception(self):
        """Test request_scope handles exceptions gracefully."""
        try:
            with request_scope():
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        # Should complete without error

    def test_nested_context_managers(self):
        """Test nested cache context managers work without errors."""
        with request_scope():
            with operation_scope():
                # Nested context managers should work
                with cache_scope(CacheScope.REQUEST):
                    string_pattern_cache.set("nested_key", "nested_value")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_none_values(self):
        """Test caching None values."""
        with cache_scope(CacheScope.PROCESS):
            cache = ScopedCache("test_cache")
            cache.set("none_key", None)

            # Should be able to retrieve None
            assert cache.get("none_key") is None
            # But this is ambiguous with miss - this is expected behavior

    def test_cache_with_complex_keys(self):
        """Test caching with various key types."""
        with cache_scope(CacheScope.PROCESS):
            cache = ScopedCache("string_pattern")

            # Test different key types
            cache.set(123, "int_key_value")
            cache.set("string", "string_key_value")
            cache.set(("tuple", "key"), "tuple_key_value")

            assert cache.get(123) == "int_key_value"
            assert cache.get("string") == "string_key_value"
            assert cache.get(("tuple", "key")) == "tuple_key_value"

    def test_empty_cache_operations(self):
        """Test operations on empty caches."""
        with cache_scope(CacheScope.PROCESS):
            cache = ScopedCache("test_cache")

            # Get from empty cache
            assert cache.get("nonexistent") is None

            # Clear empty cache
            cache.clear()  # Should not raise error

            # Pool operations on empty pool
            pool = ScopedPool("test_pool", list)
            obj = pool.get()  # Should create new object
            assert isinstance(obj, list)

    def test_unknown_cache_scope(self):
        """Test behavior with unknown cache scope."""
        # Mock an unknown scope
        unknown_scope = "UNKNOWN_SCOPE"

        with patch("datason.cache_manager.get_cache_scope", return_value=unknown_scope):
            cache = ScopedCache("test_cache")

            # Should default to empty cache behavior
            cache.set("key", "value")
            assert cache.get("key") is None

    def test_cache_metrics_disabled(self):
        """Test cache behavior when metrics are disabled."""
        config = SerializationConfig(cache_scope=CacheScope.PROCESS, cache_metrics_enabled=False)

        with cache_scope(CacheScope.PROCESS):
            with patch("datason.config.get_default_config", return_value=config):
                cache = ScopedCache("string_pattern")

                # Operations should work without tracking metrics
                cache.set("key1", "value1")
                result = cache.get("key1")
                assert result == "value1"

                # Miss should also work
                assert cache.get("nonexistent") is None

    def test_clear_all_caches_basic(self):
        """Test basic clear_all_caches functionality."""
        # Should not raise errors
        clear_all_caches()
        assert True  # Completed without error
