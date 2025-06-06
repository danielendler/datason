#!/usr/bin/env python3
"""Benchmark cache scopes in datason.

Measures repeated deserialization to highlight performance
differences across cache scopes.
"""
import time
import uuid

import datason
from datason.config import CacheScope, SerializationConfig

ITERATIONS = 1000
DATA = [str(uuid.uuid4()) for _ in range(ITERATIONS)]
CONFIG = SerializationConfig()


def benchmark(scope: CacheScope) -> float:
    """Return elapsed milliseconds for deserializing DATA under given scope."""
    datason.set_cache_scope(scope)
    datason.clear_all_caches()
    start = time.perf_counter()
    for item in DATA:
        datason.deserialize_fast(item, CONFIG)
    end = time.perf_counter()
    datason.clear_all_caches()
    return (end - start) * 1000


if __name__ == "__main__":
    for scope in CacheScope:
        elapsed = benchmark(scope)
        print(f"{scope.name:8s} {elapsed:.2f} ms")
