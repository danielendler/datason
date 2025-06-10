"""
Test configuration for datason plugin architecture.

This module defines pytest markers to categorize tests based on dependency requirements.
"""

import importlib.util
from typing import Any

import pytest

# Dependency availability flags

# Check for numpy
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

# Check for pandas
HAS_PANDAS = importlib.util.find_spec("pandas") is not None

# Check for scikit-learn
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

# Check for PyTorch
HAS_TORCH = importlib.util.find_spec("torch") is not None

# Check for TensorFlow
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# Check for JAX
HAS_JAX = importlib.util.find_spec("jax") is not None

# Check for PIL
HAS_PIL = importlib.util.find_spec("PIL") is not None

# Check for transformers
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

try:
    from PIL import Image  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@pytest.fixture(autouse=True)
def restore_ml_serializer(request):
    """Automatically restore ML serializer state before each test to fix test isolation issues."""
    # Use a marker to skip tests that manage their own state. This is more robust.
    if request.node.get_closest_marker("no_autofixture"):
        yield
        return

    # Clear all caches before each test - use clear_all_caches for complete isolation
    # Wrap in try-except to prevent SecurityErrors during fixture setup
    try:
        import datason

        datason.clear_all_caches()
    except Exception:
        # If clearing caches fails (e.g., due to SecurityError), continue anyway
        pass

    # Reset default config to clean state for each test
    # Wrap in try-except to prevent SecurityErrors during config reset
    try:
        import datason
        from datason.config import SerializationConfig

        datason.set_default_config(SerializationConfig())
    except Exception:
        # If setting config fails, continue anyway
        pass

    # Ensure ML serializer is properly available
    try:
        import datason.core
        from datason.ml_serializers import detect_and_serialize_ml_object

        datason.core._ml_serializer = detect_and_serialize_ml_object
    except Exception:
        # If ML serializer setup fails, continue anyway
        pass

    yield

    # Clean up after test - use clear_all_caches for complete cleanup
    # Wrap in try-except to prevent SecurityErrors during fixture teardown
    try:
        import datason

        datason.clear_all_caches()
    except Exception:
        # If clearing caches fails during cleanup, continue anyway
        pass


def pytest_configure(config: Any) -> None:
    """Register custom markers for dependency-based test categorization."""
    config.addinivalue_line(
        "markers", "no_autofixture: Tests that should not use the autouse restore_ml_serializer fixture."
    )
    config.addinivalue_line("markers", "core: Core functionality tests (no optional dependencies required)")
    config.addinivalue_line("markers", "numpy: Tests requiring numpy")
    config.addinivalue_line("markers", "pandas: Tests requiring pandas")
    config.addinivalue_line("markers", "sklearn: Tests requiring scikit-learn")
    config.addinivalue_line("markers", "ml: Tests requiring ML dependencies (torch, tensorflow, etc.)")
    config.addinivalue_line("markers", "optional: Tests for optional dependency functionality")
    config.addinivalue_line("markers", "fallback: Tests for fallback behavior when dependencies are missing")
    config.addinivalue_line("markers", "intensive: Memory/CPU intensive tests (skipped in CI environments)")


# Convenience skip decorators
requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
requires_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
requires_tensorflow = pytest.mark.skipif(not HAS_TENSORFLOW, reason="tensorflow not available")
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="jax not available")
requires_pil = pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
requires_transformers = pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")

# Combined requirements
requires_ml_basic = pytest.mark.skipif(
    not (HAS_NUMPY and HAS_SKLEARN),
    reason="Basic ML dependencies (numpy, sklearn) not available",
)
requires_data_science = pytest.mark.skipif(
    not (HAS_NUMPY and HAS_PANDAS),
    reason="Data science dependencies (numpy, pandas) not available",
)
