"""
Test configuration for datason plugin architecture.

This module defines pytest markers to categorize tests based on dependency requirements.
"""

import pytest

# Dependency availability flags
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import sklearn

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def pytest_configure(config):
    """Register custom markers for dependency-based test categorization."""
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
