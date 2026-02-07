"""Integration tests for thread safety.

Tests concurrent serialization using ThreadPoolExecutor to verify
that global state (registry, cache, config) is thread-safe.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

import datason


class TestConcurrentSerialization:
    """Parallel serialization with ThreadPoolExecutor."""

    def test_parallel_dumps_same_data(self) -> None:
        """10 threads each serialize the same data 100 times."""
        data = {"name": "test", "count": 42, "ratio": 3.14}

        def worker() -> list[str]:
            return [datason.dumps(data) for _ in range(100)]

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # All should produce identical JSON
        expected = datason.dumps(data)
        for batch in results:
            assert len(batch) == 100  # noqa: PLR2004
            for s in batch:
                assert s == expected

    def test_parallel_dumps_different_types(self) -> None:
        """Each thread serializes a different type."""
        datasets = [
            {"ts": dt.datetime(2024, 1, i + 1)}  # noqa: DTZ001
            for i in range(10)
        ]

        def worker(data: dict[str, Any]) -> str:
            return datason.dumps(data, include_type_hints=True)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(worker, d): i for i, d in enumerate(datasets)}
            results = {}
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()

        # Each should be valid JSON
        for s in results.values():
            parsed = json.loads(s)
            assert isinstance(parsed, dict)

    def test_parallel_roundtrip(self) -> None:
        """Each thread does dumps + loads in a loop."""
        data = {
            "ts": dt.datetime(2024, 6, 15),  # noqa: DTZ001
            "arr": np.array([1.0, 2.0, 3.0]),
            "name": "test",
        }

        def worker() -> bool:
            for _ in range(50):
                s = datason.dumps(data, include_type_hints=True)
                result = datason.loads(s)
                if not isinstance(result["ts"], dt.datetime):
                    return False
                if not isinstance(result["arr"], np.ndarray):
                    return False
            return True

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker) for _ in range(8)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)


class TestConcurrentConfigContext:
    """ContextVar thread isolation for config."""

    def test_config_context_thread_isolation(self) -> None:
        """Different configs per thread don't cross-contaminate."""

        def worker_sorted(thread_id: int) -> list[str]:
            batch = []
            with datason.config(sort_keys=True):
                for _ in range(50):
                    s = datason.dumps({"z": 1, "a": 2})
                    batch.append(s)
            return batch

        def worker_unsorted(thread_id: int) -> list[str]:
            batch = []
            # No config context — default sort_keys=False
            for _ in range(50):
                s = datason.dumps({"z": 1, "a": 2})
                batch.append(s)
            return batch

        with ThreadPoolExecutor(max_workers=6) as pool:
            sorted_futures = [pool.submit(worker_sorted, i) for i in range(3)]
            unsorted_futures = [pool.submit(worker_unsorted, i) for i in range(3)]

            sorted_results = [f.result() for f in as_completed(sorted_futures)]
            unsorted_results = [f.result() for f in as_completed(unsorted_futures)]

        # Sorted threads should have keys in sorted order
        for batch in sorted_results:
            for s in batch:
                keys = list(json.loads(s).keys())
                assert keys == ["a", "z"], f"Expected sorted keys, got {keys}"

        # Unsorted threads should preserve insertion order
        for batch in unsorted_results:
            for s in batch:
                keys = list(json.loads(s).keys())
                assert keys == ["z", "a"], f"Expected unsorted keys, got {keys}"

    def test_different_type_hints_per_thread(self) -> None:
        """One thread uses type hints, another doesn't."""
        data = {"ts": dt.datetime(2024, 1, 1)}  # noqa: DTZ001

        def worker_with_hints() -> str:
            with datason.config(include_type_hints=True):
                return datason.dumps(data)

        def worker_without_hints() -> str:
            with datason.config(include_type_hints=False):
                return datason.dumps(data)

        with ThreadPoolExecutor(max_workers=4) as pool:
            hint_futures = [pool.submit(worker_with_hints) for _ in range(2)]
            no_hint_futures = [pool.submit(worker_without_hints) for _ in range(2)]

            hint_results = [f.result() for f in hint_futures]
            no_hint_results = [f.result() for f in no_hint_futures]

        # With hints: contains __datason_type__
        for s in hint_results:
            assert "__datason_type__" in s

        # Without hints: no __datason_type__
        for s in no_hint_results:
            assert "__datason_type__" not in s
