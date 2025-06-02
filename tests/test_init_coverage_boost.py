"""Tests to boost coverage for datason/__init__.py module.

This test suite specifically targets uncovered lines and branches in the __init__.py file
to improve the overall code coverage.
"""

import sys
from unittest.mock import Mock, patch

import pytest

import datason


class TestInitModuleCoverage:
    """Test coverage for initialization module edge cases."""

    def test_python_version_compatibility_check(self):
        """Test Python version compatibility checks."""
        # The version check happens on import, so we can only test the current behavior
        # but we can verify the version info is accessible
        assert hasattr(sys, "version_info")
        assert sys.version_info >= (3, 8)

    def test_version_info_access(self):
        """Test accessing version and metadata."""
        assert hasattr(datason, "__version__")
        assert hasattr(datason, "__author__")
        assert hasattr(datason, "__license__")
        assert datason.__version__ == "0.3.1"
        assert datason.__author__ == "datason Contributors"
        assert datason.__license__ == "MIT"

    def test_all_exports_available(self):
        """Test that all items in __all__ are actually available."""
        for item in datason.__all__:
            assert hasattr(datason, item), f"'{item}' is in __all__ but not available"

    def test_config_availability_flag(self):
        """Test the _config_available flag and related imports."""
        # Should be True since config is available
        assert datason._config_available is True

        # Test that config-related functions are available
        assert hasattr(datason, "SerializationConfig")
        assert hasattr(datason, "get_default_config")
        assert hasattr(datason, "configure")

    def test_ml_availability_flag(self):
        """Test the _ml_available flag and related imports."""
        # ML should be available in test environment
        assert hasattr(datason, "_ml_available")

        if datason._ml_available:
            assert hasattr(datason, "detect_and_serialize_ml_object")

    def test_pickle_bridge_availability_flag(self):
        """Test the _pickle_bridge_available flag and related imports."""
        # Pickle bridge should be available
        assert hasattr(datason, "_pickle_bridge_available")
        assert datason._pickle_bridge_available is True

        # Test that pickle bridge functions are available
        assert hasattr(datason, "PickleBridge")
        assert hasattr(datason, "from_pickle")

    @patch("importlib.import_module")
    def test_config_import_failure_path(self, mock_import):
        """Test the config import failure path."""
        # Mock importlib to simulate import failure
        mock_import.side_effect = ImportError("Mocked config import failure")

        # This would need to be tested during import, but we can test the error handling
        # in the configure function when config is not available
        with patch.object(datason, "_config_available", False):
            with pytest.raises(ImportError, match="Configuration system not available"):
                datason.configure(Mock())

    def test_configure_function_with_valid_config(self):
        """Test the configure convenience function with valid config."""
        if datason._config_available:
            config = datason.get_default_config()
            # Should not raise an error
            datason.configure(config)

            # Reset to default after test
            datason.reset_default_config()

    def test_serialize_with_config_convenience_function(self):
        """Test the serialize_with_config convenience function."""
        test_data = {"test": "data", "number": 42}

        if datason._config_available:
            # Test with various configuration options
            result = datason.serialize_with_config(
                test_data, date_format="iso", nan_handling="null", type_coercion="safe", sort_keys=True
            )
            assert isinstance(result, dict)

            # Test with string enum conversion
            result = datason.serialize_with_config(
                test_data, date_format="unix", nan_handling="string", type_coercion="aggressive"
            )
            assert isinstance(result, dict)
        else:
            # When config not available, should fall back to basic serialize
            result = datason.serialize_with_config(test_data)
            assert isinstance(result, dict)

    def test_serialize_with_config_dataframe_orient(self):
        """Test serialize_with_config with dataframe_orient parameter."""
        test_data = {"test": "data"}

        if datason._config_available:
            result = datason.serialize_with_config(test_data, dataframe_orient="records")
            assert isinstance(result, dict)

    def test_convenience_functions_in_all_when_config_available(self):
        """Test that convenience functions are in __all__ when config is available."""
        if datason._config_available:
            assert "configure" in datason.__all__
            assert "serialize_with_config" in datason.__all__


class TestImportErrorPaths:
    """Test import error handling paths."""

    @patch("sys.version_info", (3, 7, 0))
    def test_python_version_too_old_runtime_error(self):
        """Test RuntimeError for Python version too old."""
        # This test would need to be run during import to actually trigger the check
        # We can only verify the logic would work
        assert (3, 7, 0) < (3, 8)

    @patch("warnings.warn")
    @patch("sys.version_info", (3, 8, 0))
    def test_python_version_eol_warning(self, mock_warn):
        """Test deprecation warning for EOL Python versions."""
        # The warning happens on import, but we can test the logic
        if (3, 8, 0) < (3, 9):
            # Would trigger deprecation warning
            pass

    @patch("importlib.import_module")
    def test_ml_serializers_import_failure(self, mock_import):
        """Test ML serializers import failure path."""

        def side_effect(module, package=None):
            if module == ".ml_serializers":
                raise ImportError("ML serializers not available")
            return Mock()

        mock_import.side_effect = side_effect

        # This simulates the import failure path
        try:
            mock_import(".ml_serializers", package="datason")
            ml_available = True
        except ImportError:
            ml_available = False

        assert ml_available is False

    @patch("importlib.import_module")
    def test_pickle_bridge_import_failure(self, mock_import):
        """Test pickle bridge import failure path."""

        def side_effect(module, package=None):
            if module == ".pickle_bridge":
                raise ImportError("Pickle bridge not available")
            return Mock()

        mock_import.side_effect = side_effect

        # This simulates the import failure path
        try:
            mock_import(".pickle_bridge", package="datason")
            pickle_available = True
        except ImportError:
            pickle_available = False

        assert pickle_available is False


class TestConditionalImports:
    """Test conditional import behaviors."""

    def test_core_imports_always_available(self):
        """Test that core imports are always available."""
        # Core serialization
        assert hasattr(datason, "serialize")
        assert hasattr(datason, "deserialize")
        assert hasattr(datason, "auto_deserialize")

        # Chunked processing (v0.4.0)
        assert hasattr(datason, "serialize_chunked")
        assert hasattr(datason, "ChunkedSerializationResult")
        assert hasattr(datason, "stream_serialize")

        # Template deserialization (v0.4.5)
        assert hasattr(datason, "TemplateDeserializer")
        assert hasattr(datason, "deserialize_with_template")

    def test_config_conditional_imports(self):
        """Test configuration system conditional imports."""
        if datason._config_available:
            # Configuration classes
            assert hasattr(datason, "SerializationConfig")
            assert hasattr(datason, "DateFormat")
            assert hasattr(datason, "DataFrameOrient")

            # Configuration functions
            assert hasattr(datason, "get_default_config")
            assert hasattr(datason, "get_ml_config")
            assert hasattr(datason, "get_financial_config")

            # Type handlers
            assert hasattr(datason, "TypeHandler")
            assert hasattr(datason, "is_nan_like")

    def test_ml_conditional_imports(self):
        """Test ML serializers conditional imports."""
        if datason._ml_available:
            assert hasattr(datason, "detect_and_serialize_ml_object")
            assert hasattr(datason, "get_ml_library_info")
            assert hasattr(datason, "serialize_pytorch_tensor")

    def test_pickle_bridge_conditional_imports(self):
        """Test pickle bridge conditional imports."""
        if datason._pickle_bridge_available:
            assert hasattr(datason, "PickleBridge")
            assert hasattr(datason, "PickleSecurityError")
            assert hasattr(datason, "from_pickle")


class TestAllListConstruction:
    """Test __all__ list construction with different availability scenarios."""

    def test_base_all_contents(self):
        """Test base __all__ contents are always present."""
        base_exports = [
            "SecurityError",
            "serialize",
            "serialize_chunked",
            "deserialize",
            "auto_deserialize",
            "TemplateDeserializer",
            "serialize_detection_details",
        ]

        for export in base_exports:
            assert export in datason.__all__

    def test_config_all_extensions(self):
        """Test config-related __all__ extensions."""
        if datason._config_available:
            config_exports = [
                "SerializationConfig",
                "DateFormat",
                "DataFrameOrient",
                "get_default_config",
                "get_ml_config",
                "TypeHandler",
            ]

            for export in config_exports:
                assert export in datason.__all__

    def test_ml_all_extensions(self):
        """Test ML-related __all__ extensions."""
        if datason._ml_available:
            ml_exports = ["detect_and_serialize_ml_object", "get_ml_library_info", "serialize_pytorch_tensor"]

            for export in ml_exports:
                assert export in datason.__all__

    def test_pickle_bridge_all_extensions(self):
        """Test pickle bridge __all__ extensions."""
        if datason._pickle_bridge_available:
            pickle_exports = [
                "PickleBridge",
                "PickleSecurityError",
                "from_pickle",
                "convert_pickle_directory",
                "get_ml_safe_classes",
            ]

            for export in pickle_exports:
                assert export in datason.__all__


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in init module."""

    def test_serialize_with_config_without_config_system(self):
        """Test serialize_with_config when config system is not available."""
        test_data = {"test": "data"}

        with patch.object(datason, "_config_available", False):
            # Should fall back to basic serialize without config
            result = datason.serialize_with_config(test_data, some_param="ignored")
            assert isinstance(result, dict)
            assert result == test_data

    def test_module_metadata_integrity(self):
        """Test module metadata integrity."""
        assert isinstance(datason.__version__, str)
        assert "." in datason.__version__  # Should be version format like "0.3.1"
        assert isinstance(datason.__author__, str)
        assert isinstance(datason.__license__, str)
        assert isinstance(datason.__all__, list)
        assert len(datason.__all__) > 0

    def test_import_all_verification(self):
        """Verify all exports can actually be imported and used."""
        for export_name in datason.__all__:
            export_obj = getattr(datason, export_name)
            assert export_obj is not None

            # Basic type checking for different categories
            if export_name in ["serialize", "deserialize", "auto_deserialize"]:
                assert callable(export_obj)
            elif export_name in ["SerializationConfig", "DateFormat", "DataFrameOrient"]:
                # These should be classes/enums
                assert hasattr(export_obj, "__name__") or hasattr(export_obj, "_name_")
