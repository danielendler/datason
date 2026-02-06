"""Root conftest for datason v2 tests.

Single conftest with explicit fixtures. No global autouse —
prefer proper isolation in each test.
"""

from __future__ import annotations

import pytest

from datason._cache import type_cache
from datason._registry import default_registry


@pytest.fixture()
def clean_state():
    """Reset all datason global state. Use explicitly in tests that need it."""
    type_cache.clear()
    default_registry.clear()
    yield
    type_cache.clear()
    default_registry.clear()


@pytest.fixture()
def sample_data() -> dict:
    """Provide a basic nested dict for serialization tests."""
    return {
        "name": "test",
        "count": 42,
        "ratio": 3.14,
        "active": True,
        "nothing": None,
        "tags": ["a", "b", "c"],
        "nested": {"x": 1, "y": 2},
    }
