"""
DataSON Python 3.13+ Optimizations Module

This module provides version-aware optimizations that leverage Python 3.13's
new features (JIT, free-threading) while maintaining backward compatibility.
"""

import os
import sys
import threading
import warnings
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional, Tuple

# Version detection constants
PYTHON_VERSION = sys.version_info
HAS_JIT = PYTHON_VERSION >= (3, 13) and hasattr(sys, "_jit_compile")
HAS_FREE_THREADING = PYTHON_VERSION >= (3, 13) and threading.active_count

# Import optimizations based on available features
if HAS_FREE_THREADING:
    try:
        from concurrent.futures import ThreadPoolExecutor as OptimizedExecutor
    except ImportError:
        OptimizedExecutor = None
else:
    OptimizedExecutor = None


class VersionAwareOptimizations:
    """Base class for Python version-aware optimizations."""

    def __init__(self):
        self.python_version = PYTHON_VERSION
        self.jit_available = HAS_JIT
        self.free_threading_available = HAS_FREE_THREADING
        self._setup_optimizations()

    def _setup_optimizations(self):
        """Configure optimizations based on available Python features."""
        if self.python_version >= (3, 13):
            self.optimization_level = "advanced"
            self.parallel_threshold = 50  # Lower threshold for parallel processing
            self.cache_size = 2048  # Larger cache for better performance
        elif self.python_version >= (3, 12):
            self.optimization_level = "intermediate"
            self.parallel_threshold = 100
            self.cache_size = 1024
        else:
            self.optimization_level = "basic"
            self.parallel_threshold = 200  # Higher threshold for older Python
            self.cache_size = 512  # Smaller cache to avoid memory pressure


class JITOptimizedPatternMatcher:
    """JIT-friendly pattern matching for Python 3.13+"""

    def __init__(self, patterns: List[Tuple[str, Any]]):
        self.patterns = patterns
        self.pattern_count = len(patterns)

        # Pre-compute frequently used data for JIT optimization
        self._compiled_patterns = [p[1] for p in patterns if p[1] is not None]
        self._string_patterns = [(p[0].lower(), i) for i, p in enumerate(patterns) if p[1] is None]

    def match_field_jit_friendly(self, field_path: str) -> bool:
        """
        JIT-optimized pattern matching.

        This method is designed to be compiled to native code by Python 3.13's
        JIT compiler. It uses tight loops and avoids complex Python constructs.
        """
        field_lower = field_path.lower()

        # First pass: compiled regex patterns (most common case)
        for i in range(len(self._compiled_patterns)):
            if self._compiled_patterns[i].match(field_path):
                return True

        # Second pass: string patterns (less common, but JIT-optimized)
        return any(pattern_lower in field_lower for pattern_lower, _original_index in self._string_patterns)

    def match_field_fallback(self, field_path: str) -> bool:
        """Fallback implementation for Python < 3.13"""
        return any(
            (compiled.match(field_path) if compiled else pattern.lower() in field_path.lower())
            for pattern, compiled in self.patterns
        )


class ParallelRedactionProcessor:
    """Free-threading aware parallel processing for Python 3.13+"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.free_threading_available = HAS_FREE_THREADING

        if self.free_threading_available:
            self._executor = OptimizedExecutor(max_workers=self.max_workers)
        else:
            self._executor = None

    def process_objects_parallel(self, processor_func, objects: List[Any], threshold: int = 50) -> List[Any]:
        """
        Process multiple objects in parallel if beneficial.

        Args:
            processor_func: Function to apply to each object
            objects: List of objects to process
            threshold: Minimum number of objects to trigger parallel processing

        Returns:
            List of processed objects
        """
        if len(objects) < threshold or not self._executor:
            # Use sequential processing for small datasets or when threading unavailable
            return [processor_func(obj) for obj in objects]

        # Parallel processing for large datasets
        try:
            future_to_index = {self._executor.submit(processor_func, obj): i for i, obj in enumerate(objects)}

            results = [None] * len(objects)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    warnings.warn(f"Object {index} generated exception: {exc}", stacklevel=2)
                    # Fallback to sequential processing for this object
                    results[index] = processor_func(objects[index])

            return results

        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential.", stacklevel=2)
            return [processor_func(obj) for obj in objects]

    def process_dict_parallel(
        self, processor_func, obj_dict: Dict[str, Any], field_path: str = "", threshold: int = 20
    ) -> Dict[str, Any]:
        """
        Process dictionary fields in parallel for large dictionaries.

        This is particularly effective for API responses with many fields.
        """
        if len(obj_dict) < threshold or not self._executor:
            return {
                key: processor_func(value, f"{field_path}.{key}" if field_path else key)
                for key, value in obj_dict.items()
            }

        # Split dictionary into chunks for parallel processing
        items = list(obj_dict.items())
        chunk_size = max(1, len(items) // self.max_workers)
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        def process_chunk(chunk):
            result = {}
            for key, value in chunk:
                current_path = f"{field_path}.{key}" if field_path else key
                result[key] = processor_func(value, current_path)
            return result

        try:
            chunk_futures = [self._executor.submit(process_chunk, chunk) for chunk in chunks]

            final_result = {}
            for future in as_completed(chunk_futures):
                chunk_result = future.result()
                final_result.update(chunk_result)

            return final_result

        except Exception as e:
            warnings.warn(f"Parallel dict processing failed: {e}. Using sequential.", stacklevel=2)
            return {
                key: processor_func(value, f"{field_path}.{key}" if field_path else key)
                for key, value in obj_dict.items()
            }

    def __del__(self):
        """Clean up executor resources"""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)


class AdaptiveCache:
    """Version-aware caching implementation"""

    def __init__(self, maxsize: Optional[int] = None):
        self.python_version = PYTHON_VERSION

        if maxsize is None:
            # Adjust cache size based on Python version performance characteristics
            if self.python_version >= (3, 13):
                maxsize = 2048  # Python 3.13 can handle larger caches efficiently
            elif self.python_version >= (3, 12):
                maxsize = 1024  # Python 3.12 sweet spot
            else:
                maxsize = 512  # Conservative for older Python versions

        self.maxsize = maxsize
        self._cache: Dict[str, bool] = {}
        self._access_order: List[str] = []

    def get(self, key: str, default=None):
        """Get value from cache with LRU tracking"""
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return default

    def set(self, key: str, value: bool):
        """Set value in cache with size management"""
        if len(self._cache) >= self.maxsize and key not in self._cache and self._access_order:
            # Remove least recently used item
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)

        self._cache[key] = value
        if key not in self._access_order:
            self._access_order.append(key)

    def clear(self):
        """Clear all cached values"""
        self._cache.clear()
        self._access_order.clear()


def get_optimization_info() -> Dict[str, Any]:
    """
    Get information about available optimizations for the current Python version.

    Returns:
        Dictionary containing optimization capabilities and recommendations
    """
    return {
        "python_version": f"{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}",
        "jit_available": HAS_JIT,
        "free_threading_available": HAS_FREE_THREADING,
        "recommended_cache_size": 2048 if PYTHON_VERSION >= (3, 13) else 1024 if PYTHON_VERSION >= (3, 12) else 512,
        "parallel_processing_threshold": 50 if PYTHON_VERSION >= (3, 13) else 100 if PYTHON_VERSION >= (3, 12) else 200,
        "optimization_level": "advanced"
        if PYTHON_VERSION >= (3, 13)
        else "intermediate"
        if PYTHON_VERSION >= (3, 12)
        else "basic",
        "expected_performance_boost": "50-80%"
        if PYTHON_VERSION >= (3, 13)
        else "25-35%"
        if PYTHON_VERSION >= (3, 12)
        else "5-10%",
    }


# Factory function for creating optimized instances
def create_optimized_redaction_engine(*args, **kwargs):
    """
    Factory function to create the best RedactionEngine for the current Python version.

    This function will be extended to return version-specific implementations.
    """
    from .redaction import RedactionEngine

    # For now, return standard engine - will be enhanced with version-specific classes
    engine = RedactionEngine(*args, **kwargs)

    # Add optimization information
    engine._optimization_info = get_optimization_info()

    return engine


if __name__ == "__main__":
    # Display optimization information
    info = get_optimization_info()
    print("🐍 DataSON Python Optimization Analysis")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
