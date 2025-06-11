"""
Edge case test configuration for datason.

This module ensures proper cleanup after edge case tests that may patch or
manipulate the ML serializer, preventing contamination of subsequent tests.
"""

import pytest


@pytest.fixture(autouse=True, scope="function")
def restore_ml_serializer_after_edge_cases():
    """
    Ensure ML serializer is properly restored after edge case tests.

    Edge case tests may patch datason.core.detect_and_serialize_ml_object
    which can leave the ML serializer in a corrupted state for subsequent tests.
    This fixture ensures proper restoration.
    """
    yield  # Let the test run first

    # Force restore ML serializer after any potential patching
    try:
        import datason.core
        from datason.ml_serializers import detect_and_serialize_ml_object

        # Explicitly restore the proper ML serializer function
        datason.core._ml_serializer = detect_and_serialize_ml_object

        # Also clear any cached imports that might be stale
        try:
            from datason import ml_serializers

            if hasattr(ml_serializers, "_LAZY_IMPORTS"):
                # Clear all lazy imports to force re-initialization
                ml_serializers._LAZY_IMPORTS.clear()
        except (ImportError, AttributeError):
            pass

    except (ImportError, AttributeError):
        # If ML serializers are not available, set to None
        try:
            import datason.core

            datason.core._ml_serializer = None
        except ImportError:
            pass


@pytest.fixture(autouse=True, scope="function")
def clear_module_patches():
    """
    Clear any module-level patches that edge case tests might leave behind.

    This ensures that mocks and patches from edge case tests don't persist
    and affect subsequent integration tests.
    """
    yield

    # Force garbage collection to clear any hanging references
    import gc

    gc.collect()

    # Clear datason caches
    try:
        import datason

        datason.clear_all_caches()
    except ImportError:
        pass
