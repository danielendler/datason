"""Thread-safe caching for datason.

All global mutable state uses threading.Lock to ensure safety
in multi-threaded applications and web servers.
"""

from __future__ import annotations

import threading
from typing import Any


class ThreadSafeCache:
    """Simple thread-safe dict cache with size limit and FIFO eviction."""

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: Any) -> Any | None:
        """Get a value from cache, or None if not found."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set a value in cache, evicting oldest if at capacity."""
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                # FIFO eviction: remove first inserted key
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)


# Global caches (thread-safe)
type_cache = ThreadSafeCache(max_size=500)
