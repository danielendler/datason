"""
Global test configuration for comprehensive test isolation and state management.

This module ensures proper test isolation by clearing all caches and resetting
state between tests to prevent interference in CI environments.
"""

import pytest


@pytest.fixture(autouse=True, scope="function")
def ensure_clean_test_state():
    """
    Ensure complete test isolation by clearing all caches and resetting state.

    This fixture runs automatically before and after every test to prevent:
    - Cache contamination between tests
    - ML import state persistence
    - Configuration state leakage
    - Global variable interference
    """
    # Clear all state before test
    _clear_all_datason_state()

    yield

    # Clear all state after test
    _clear_all_datason_state()


def _clear_all_datason_state():
    """Clear all possible datason state and caches."""
    try:
        import datason

        # Clear all caches using the comprehensive clear function
        datason.clear_all_caches()

        # Additional cleanup for ML state - PROPERLY reinitialize lazy imports
        try:
            from datason import ml_serializers

            # Don't just clear - reinitialize with the proper structure!
            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                ml_serializers._LAZY_IMPORTS.clear()
                # Reinitialize with ALL the required keys that are actually used in the code
                ml_serializers._LAZY_IMPORTS.update(
                    {
                        "torch": None,
                        "tensorflow": None,
                        "jax": None,
                        "jnp": None,  # JAX numpy alias
                        "sklearn": None,
                        "BaseEstimator": None,  # sklearn base estimator class
                        "scipy": None,
                        "PIL_Image": None,  # This was the missing key causing the KeyError!
                        "PIL": None,
                        "transformers": None,
                        "catboost": None,
                        "keras": None,
                        "optuna": None,
                        "plotly": None,
                        "polars": None,
                        "pandas": None,
                        "numpy": None,
                    }
                )
        except ImportError:
            pass

        # Reset global configuration state
        try:
            from datason.config import SerializationConfig

            datason.set_default_config(SerializationConfig())
        except (ImportError, AttributeError):
            pass

    except ImportError:
        # datason not available, nothing to clear
        pass


@pytest.fixture
def ml_config():
    """Provide a fresh ML configuration for ML-specific tests."""
    try:
        from datason.config import get_ml_config

        return get_ml_config()
    except ImportError:
        # Fallback if config module is not available
        return None


@pytest.fixture(scope="function")
def isolated_test_environment():
    """Provide completely isolated test environment for critical tests."""
    # Pre-test cleanup
    _clear_all_datason_state()

    yield

    # Post-test cleanup
    _clear_all_datason_state()


@pytest.fixture(autouse=True, scope="session")
def configure_test_environment():
    """Configure the test environment for optimal CI performance."""
    try:
        import os

        # Set environment variables for reduced ML library verbosity
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA

        # Suppress warnings
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    except ImportError:
        pass

    yield
