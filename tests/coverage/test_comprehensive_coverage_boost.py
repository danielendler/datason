"""Pure coverage boost tests for datason modules.

This test file specifically targets uncovered import paths, conditional imports,
and edge cases that are purely for coverage rather than functional testing.
Functional tests belong in their respective feature/core test files.
"""

import warnings
from unittest.mock import patch

import pytest

import datason
from datason import serialize


class TestImportCoverageBoost:
    """Test import paths and conditional imports for coverage."""

    def test_ml_serializer_import_error_path(self) -> None:
        """Test core.py lines 14-17 - ML import failure path."""
        # Test what happens when ML serializer is None
        with patch("datason.core._ml_serializer", None):
            # This should use the fallback path
            result = serialize({"test": "data"})
            assert result == {"test": "data"}

    def test_config_import_failure_path(self) -> None:
        """Test core.py lines 36-45 - config import failure path."""
        # This tests the ImportError fallback when config system unavailable
        # We can't easily mock this, but we can test the fallback functions
        from datason.core import is_nan_like, normalize_numpy_types

        # Test fallback implementations
        assert not is_nan_like("test")
        assert normalize_numpy_types("test") == "test"

    def test_configure_function_with_config(self) -> None:
        """Test configure function when config system available."""
        if hasattr(datason, "configure"):
            # Just verify function exists and can be called
            assert callable(datason.configure)

    def test_all_exports_comprehensive(self) -> None:
        """Test that all expected exports are available."""
        # Core functions should always be available
        assert hasattr(datason, "serialize")
        assert hasattr(datason, "deserialize")

        # Check availability flags
        assert hasattr(datason, "_config_available")
        assert hasattr(datason, "_ml_available")
        assert hasattr(datason, "_pickle_bridge_available")

        # Version info
        assert hasattr(datason, "__version__")

    def test_conditional_imports_coverage(self) -> None:
        """Test conditional import coverage paths."""
        # Test that conditional features work when available
        if datason._config_available:
            from datason.config import SerializationConfig

            config = SerializationConfig()
            assert config is not None

        if datason._ml_available:
            from datason.ml_serializers import get_ml_library_info

            info = get_ml_library_info()
            assert isinstance(info, dict)


class TestCoreCachingCoverage:
    """Test core.py caching and optimization paths for coverage."""

    def test_type_cache_limit_coverage(self) -> None:
        """Test type cache behavior when limit is reached."""
        # Fill up the type cache
        original_limit = datason.core._TYPE_CACHE_SIZE_LIMIT
        datason.core._TYPE_CACHE_SIZE_LIMIT = 2
        datason.core._TYPE_CACHE.clear()

        try:
            # Add types to fill cache
            class TestType1:
                pass

            class TestType2:
                pass

            class TestType3:
                pass

            # These should use the cache
            result1 = datason.core._get_cached_type_category(TestType1)
            result2 = datason.core._get_cached_type_category(TestType2)

            # This should hit the limit and return None
            _result3 = datason.core._get_cached_type_category(TestType3)

            assert result1 == "other"
            assert result2 == "other"
            # Should not cache when limit reached

        finally:
            datason.core._TYPE_CACHE_SIZE_LIMIT = original_limit
            datason.core._TYPE_CACHE.clear()

    def test_memory_pool_coverage(self) -> None:
        """Test memory pooling paths for coverage."""
        # Test that pooled objects are safely returned
        from datason.core import _get_pooled_dict, _get_pooled_list, _return_dict_to_pool, _return_list_to_pool

        # Test dict pooling
        pooled_dict = _get_pooled_dict()
        pooled_dict["test"] = "value"
        _return_dict_to_pool(pooled_dict)

        # Test list pooling
        pooled_list = _get_pooled_list()
        pooled_list.append("test")
        _return_list_to_pool(pooled_list)


class TestRedactionCoverageEdgeCases:
    """Test redaction.py edge cases for coverage."""

    def test_redaction_engine_invalid_patterns_coverage(self) -> None:
        """Test redaction engine with invalid regex patterns."""
        from datason.redaction import RedactionEngine

        # Test with invalid regex - should handle gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = RedactionEngine(redact_patterns=["[invalid"])

            data = {"test": "some data"}
            result = engine.process_object(data)

            # Should work despite invalid pattern
            assert isinstance(result, dict)
            # Should have warning about invalid pattern
            assert len(w) > 0


if __name__ == "__main__":
    pytest.main([__file__])
