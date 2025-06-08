"""Cache management system for datason.

This module provides configurable caching with different scopes:
- Operation-scoped: Cache cleared after each operation (default, safest)
- Request-scoped: Cache persists within a request context
- Process-scoped: Cache persists for the entire process (fastest, potential cross-contamination)
- Disabled: No caching (slowest, most predictable)
"""

import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional, Tuple

from .config import CacheScope, SerializationConfig, get_cache_scope


# Cache metrics for monitoring
class CacheMetrics:
    """Tracks cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_warnings = 0

    def hit(self):
        self.hits += 1

    def miss(self):
        self.misses += 1

    def evict(self):
        self.evictions += 1

    def warn_size(self):
        self.size_warnings += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_warnings = 0

    def __str__(self) -> str:
        return f"CacheMetrics(hits={self.hits}, misses={self.misses}, hit_rate={self.hit_rate:.2%}, evictions={self.evictions})"


# Context variables for request-scoped caches
_request_string_pattern_cache: ContextVar[Dict[int, str]] = ContextVar("request_string_pattern_cache", default=None)
_request_parsed_object_cache: ContextVar[Dict[str, Any]] = ContextVar("request_parsed_object_cache", default=None)
_request_type_cache: ContextVar[Dict[str, str]] = ContextVar("request_type_cache", default=None)
_request_dict_pool: ContextVar[List[Dict]] = ContextVar("request_dict_pool", default=None)
_request_list_pool: ContextVar[List[List]] = ContextVar("request_list_pool", default=None)

# Process-level caches (global)
_process_string_pattern_cache: Dict[int, str] = {}
_process_parsed_object_cache: Dict[str, Any] = {}
_process_type_cache: Dict[str, str] = {}
_process_dict_pool: List[Dict] = []
_process_list_pool: List[List] = []

# Cache metrics by scope
_cache_metrics: Dict[CacheScope, CacheMetrics] = defaultdict(CacheMetrics)


class ScopedCache:
    """A cache that respects the current cache scope configuration."""

    def __init__(self, cache_name: str):
        self.cache_name = cache_name

    def _get_current_cache_and_config(self) -> Tuple[Dict, SerializationConfig]:
        """Get the current cache dict and configuration based on scope."""
        from .config import get_default_config

        scope = get_cache_scope()
        config = get_default_config()

        if scope == CacheScope.DISABLED:
            return {}, config
        elif scope == CacheScope.OPERATION:
            # Operation-scoped caches are always empty (cleared after each operation)
            return {}, config
        elif scope == CacheScope.REQUEST:
            # Get or create request-scoped cache
            if self.cache_name == "string_pattern":
                cache = _request_string_pattern_cache.get()
                if cache is None:
                    cache = {}
                    _request_string_pattern_cache.set(cache)
            elif self.cache_name == "parsed_object":
                cache = _request_parsed_object_cache.get()
                if cache is None:
                    cache = {}
                    _request_parsed_object_cache.set(cache)
            elif self.cache_name == "type":
                cache = _request_type_cache.get()
                if cache is None:
                    cache = {}
                    _request_type_cache.set(cache)
            else:
                cache = {}
            return cache, config
        elif scope == CacheScope.PROCESS:
            # Use process-level global caches
            if self.cache_name == "string_pattern":
                return _process_string_pattern_cache, config
            elif self.cache_name == "parsed_object":
                return _process_parsed_object_cache, config
            elif self.cache_name == "type":
                return _process_type_cache, config
            else:
                return {}, config
        else:
            return {}, config

    def get(self, key: Any) -> Optional[Any]:
        """Get a value from the cache."""
        cache, config = self._get_current_cache_and_config()
        scope = get_cache_scope()

        if scope == CacheScope.DISABLED:
            if config.cache_metrics_enabled:
                _cache_metrics[scope].miss()
            return None

        value = cache.get(key)
        if config.cache_metrics_enabled:
            if value is not None:
                _cache_metrics[scope].hit()
            else:
                _cache_metrics[scope].miss()

        return value

    def set(self, key: Any, value: Any) -> None:
        """Set a value in the cache."""
        cache, config = self._get_current_cache_and_config()
        scope = get_cache_scope()

        if scope == CacheScope.DISABLED:
            return

        # Check size limit
        if len(cache) >= config.cache_size_limit:
            if config.cache_warn_on_limit:
                warnings.warn(
                    f"Cache '{self.cache_name}' reached size limit ({config.cache_size_limit}). "
                    f"Consider increasing cache_size_limit or using a different cache_scope.",
                    stacklevel=3,
                )
                if config.cache_metrics_enabled:
                    _cache_metrics[scope].warn_size()

            # Evict oldest item (simple FIFO eviction)
            if cache:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                if config.cache_metrics_enabled:
                    _cache_metrics[scope].evict()

        cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        cache, _ = self._get_current_cache_and_config()
        cache.clear()


class ScopedPool:
    """A pool that respects the current cache scope configuration."""

    def __init__(self, pool_name: str, pool_type: type):
        self.pool_name = pool_name
        self.pool_type = pool_type

    def _get_current_pool_and_config(self) -> Tuple[List, SerializationConfig]:
        """Get the current pool and configuration based on scope."""
        from .config import get_default_config

        scope = get_cache_scope()
        config = get_default_config()

        if scope == CacheScope.DISABLED or scope == CacheScope.OPERATION:
            return [], config
        elif scope == CacheScope.REQUEST:
            # Get or create request-scoped pool
            if self.pool_name == "dict":
                pool = _request_dict_pool.get()
                if pool is None:
                    pool = []
                    _request_dict_pool.set(pool)
            elif self.pool_name == "list":
                pool = _request_list_pool.get()
                if pool is None:
                    pool = []
                    _request_list_pool.set(pool)
            else:
                pool = []
            return pool, config
        elif scope == CacheScope.PROCESS:
            # Use process-level global pools
            if self.pool_name == "dict":
                return _process_dict_pool, config
            elif self.pool_name == "list":
                return _process_list_pool, config
            else:
                return [], config
        else:
            return [], config

    def get(self):
        """Get an object from the pool or create a new one."""
        pool, config = self._get_current_pool_and_config()

        if pool:
            return pool.pop()
        else:
            return self.pool_type()

    def return_object(self, obj) -> None:
        """Return an object to the pool."""
        pool, config = self._get_current_pool_and_config()
        scope = get_cache_scope()

        if scope == CacheScope.DISABLED or scope == CacheScope.OPERATION:
            return

        # Clear the object before returning to pool
        if hasattr(obj, "clear"):
            obj.clear()

        # Check pool size limit
        if len(pool) < config.cache_size_limit // 4:  # Use 1/4 of cache limit for pools
            pool.append(obj)

    def clear(self) -> None:
        """Clear the pool."""
        pool, _ = self._get_current_pool_and_config()
        pool.clear()


# Create scoped cache instances
string_pattern_cache = ScopedCache("string_pattern")
parsed_object_cache = ScopedCache("parsed_object")
type_cache = ScopedCache("type")

# Create scoped pool instances
dict_pool = ScopedPool("dict", dict)
list_pool = ScopedPool("list", list)


def clear_caches() -> None:
    """Clear all caches for the current scope."""
    string_pattern_cache.clear()
    parsed_object_cache.clear()
    type_cache.clear()
    dict_pool.clear()
    list_pool.clear()

    # Clear ML serializers lazy import cache
    try:
        from . import ml_serializers

        for key in ml_serializers._LAZY_IMPORTS:
            ml_serializers._LAZY_IMPORTS[key] = None
    except ImportError:
        pass


def clear_all_caches() -> None:
    """Clear all caches across all scopes (for testing/debugging)."""
    # Clear process-level caches
    _process_string_pattern_cache.clear()
    _process_parsed_object_cache.clear()
    _process_type_cache.clear()
    _process_dict_pool.clear()
    _process_list_pool.clear()

    # Clear ML serializers lazy import cache
    try:
        from . import ml_serializers

        for key in ml_serializers._LAZY_IMPORTS:
            ml_serializers._LAZY_IMPORTS[key] = None
    except ImportError:
        pass

    # Clear request-level caches if they exist
    try:
        if _request_string_pattern_cache.get() is not None:
            _request_string_pattern_cache.get().clear()
    except LookupError:
        pass

    try:
        if _request_parsed_object_cache.get() is not None:
            _request_parsed_object_cache.get().clear()
    except LookupError:
        pass

    try:
        if _request_type_cache.get() is not None:
            _request_type_cache.get().clear()
    except LookupError:
        pass

    try:
        if _request_dict_pool.get() is not None:
            _request_dict_pool.get().clear()
    except LookupError:
        pass

    try:
        if _request_list_pool.get() is not None:
            _request_list_pool.get().clear()
    except LookupError:
        pass


def get_cache_metrics(scope: Optional[CacheScope] = None) -> Dict[CacheScope, CacheMetrics]:
    """Get cache metrics for a specific scope or all scopes."""
    if scope is not None:
        return {scope: _cache_metrics[scope]}
    return dict(_cache_metrics)


def reset_cache_metrics(scope: Optional[CacheScope] = None) -> None:
    """Reset cache metrics for a specific scope or all scopes."""
    if scope is not None:
        _cache_metrics[scope].reset()
    else:
        for metrics in _cache_metrics.values():
            metrics.reset()


@contextmanager
def operation_scope() -> Generator[None, None, None]:
    """Context manager that ensures operation-scoped caching and cleanup."""
    # Clear caches at start of operation
    clear_caches()
    try:
        yield
    finally:
        # Clear caches at end of operation
        clear_caches()


@contextmanager
def request_scope() -> Generator[None, None, None]:
    """Context manager that manages request-scoped caching."""
    # Initialize request-scoped caches
    _request_string_pattern_cache.set({})
    _request_parsed_object_cache.set({})
    _request_type_cache.set({})
    _request_dict_pool.set([])
    _request_list_pool.set([])

    try:
        yield
    finally:
        # Clear request-scoped caches
        clear_caches()
