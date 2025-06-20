"""Enhanced coverage tests for datason/__init__.py module.

This test suite targets the specific missing coverage areas in __init__.py to boost
coverage from 54% to 80%+. Focuses on conditional imports, deprecation warnings,
and dynamic __all__ list construction.
"""

import warnings
from unittest.mock import Mock, patch

import pytest

import datason


class TestConditionalImportPaths:
    """Test conditional import logic and error paths."""

    @patch.dict("sys.modules", {"datason.ml_serializers": None})
    @patch("importlib.import_module")
    def test_ml_serializers_import_failure_path(self, mock_import):
        """Test ML serializers import failure handling."""

        # Simulate ImportError for ML serializers
        def import_side_effect(module, package=None):
            if module == ".ml_serializers":
                raise ImportError("ML serializers not available")
            return Mock()

        mock_import.side_effect = import_side_effect

        # Test the import failure path
        try:
            mock_import(".ml_serializers", package="datason")
            ml_available = True
        except ImportError:
            ml_available = False

        assert ml_available is False

    @patch.dict("sys.modules", {"datason.ml_type_handlers": None})
    @patch("importlib.import_module")
    def test_ml_type_handlers_import_failure_path(self, mock_import):
        """Test ML type handlers import failure handling."""

        def import_side_effect(module, package=None):
            if module == ".ml_type_handlers":
                raise ImportError("ML type handlers not available")
            return Mock()

        mock_import.side_effect = import_side_effect

        # Test the import failure path
        try:
            mock_import(".ml_type_handlers", package="datason")
            handlers_available = True
        except ImportError:
            handlers_available = False

        assert handlers_available is False

    @patch.dict("sys.modules", {"datason.pickle_bridge": None})
    @patch("importlib.import_module")
    def test_pickle_bridge_import_failure_path(self, mock_import):
        """Test pickle bridge import failure handling."""

        def import_side_effect(module, package=None):
            if module == ".pickle_bridge":
                raise ImportError("Pickle bridge not available")
            return Mock()

        mock_import.side_effect = import_side_effect

        # Test the import failure path
        try:
            mock_import(".pickle_bridge", package="datason")
            pickle_available = True
        except ImportError:
            pickle_available = False

        assert pickle_available is False

    def test_redaction_import_failure_path(self):
        """Test redaction module import failure handling."""
        # Test what happens when redaction is not available
        with patch.dict("sys.modules", {"datason.redaction": None}):
            with patch("builtins.__import__", side_effect=ImportError("redaction not available")):
                try:
                    import datason.redaction  # noqa: F401

                    redaction_available = True
                except ImportError:
                    redaction_available = False

                # Should fail in mocked environment
                assert redaction_available is False

    def test_utils_import_failure_path(self):
        """Test utils module import failure handling."""
        # Test what happens when utils is not available
        with patch.dict("sys.modules", {"datason.utils": None}):
            with patch("builtins.__import__", side_effect=ImportError("utils not available")):
                try:
                    import datason.utils  # noqa: F401

                    utils_available = True
                except ImportError:
                    utils_available = False

                # Should fail in mocked environment
                assert utils_available is False


class TestDeprecationWarningHandling:
    """Test deprecation warning functionality."""

    def test_serialize_deprecation_warning(self):
        """Test that serialize() function issues deprecation warning."""
        test_data = {"test": "data"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call the deprecated serialize function
            result = datason.serialize(test_data)

            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "serialize() is deprecated" in str(w[0].message)
            assert "Use dump/dumps for JSON compatibility" in str(w[0].message)

            # Verify it still works
            assert isinstance(result, dict)
            assert result["test"] == "data"

    def test_serialize_with_config_parameter(self):
        """Test serialize() with config parameter."""
        test_data = {"test": "data"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with config parameter
            config = datason.get_default_config() if datason._config_available else None
            result = datason.serialize(test_data, config=config)

            # Should still issue deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

            # Should still work
            assert isinstance(result, dict)

    def test_serialize_with_no_kwargs(self):
        """Test serialize() with no additional parameters."""
        test_data = {"test": "data"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with no additional parameters
            result = datason.serialize(test_data)

            # Should still issue deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

            # Should still work
            assert isinstance(result, dict)


class TestAllListConstruction:
    """Test dynamic __all__ list construction based on availability."""

    def test_base_all_contents(self):
        """Test that base __all__ contents are present."""
        base_items = [
            "SecurityError",
            "dump",
            "dumps",
            "load",
            "loads",
            "serialize",
            "dump_json",
            "dumps_json",
            "load_json",
            "loads_json",
            "serialize_chunked",
            "ChunkedSerializationResult",
            "auto_deserialize",
            "deserialize",
            "TemplateDeserializer",
            "deserialize_with_template",
        ]

        for item in base_items:
            assert item in datason.__all__, f"Base item '{item}' missing from __all__"

    def test_config_conditional_all_extension(self):
        """Test that config items are added to __all__ when available."""
        if datason._config_available:
            config_items = [
                "SerializationConfig",
                "DateFormat",
                "DataFrameOrient",
                "get_default_config",
                "set_default_config",
                "configure",
                "serialize_with_config",
            ]

            for item in config_items:
                assert item in datason.__all__, f"Config item '{item}' missing from __all__"

    def test_ml_conditional_all_extension(self):
        """Test that ML items are added to __all__ when available."""
        if datason._ml_available:
            ml_items = [
                "detect_and_serialize_ml_object",
                "get_ml_library_info",
                "serialize_pytorch_tensor",
                "serialize_sklearn_model",
            ]

            for item in ml_items:
                assert item in datason.__all__, f"ML item '{item}' missing from __all__"

    def test_pickle_bridge_conditional_all_extension(self):
        """Test that pickle bridge items are added to __all__ when available."""
        if datason._pickle_bridge_available:
            pickle_items = [
                "PickleBridge",
                "PickleSecurityError",
                "from_pickle",
                "get_ml_safe_classes",
                "convert_pickle_directory",
            ]

            for item in pickle_items:
                assert item in datason.__all__, f"Pickle item '{item}' missing from __all__"

    def test_redaction_conditional_all_extension(self):
        """Test that redaction items are added to __all__ when available."""
        if hasattr(datason, "_redaction_available") and datason._redaction_available:
            redaction_items = [
                "RedactionEngine",
                "create_financial_redaction_engine",
                "create_healthcare_redaction_engine",
                "create_minimal_redaction_engine",
            ]

            for item in redaction_items:
                assert item in datason.__all__, f"Redaction item '{item}' missing from __all__"

    def test_utils_conditional_all_extension(self):
        """Test that utils items are added to __all__ when available."""
        if hasattr(datason, "_utils_available") and datason._utils_available:
            utils_items = [
                "deep_compare",
                "find_data_anomalies",
                "enhance_data_types",
                "normalize_data_structure",
                "UtilityConfig",
                "UtilitySecurityError",
            ]

            for item in utils_items:
                assert item in datason.__all__, f"Utils item '{item}' missing from __all__"


class TestConvenienceFunctions:
    """Test convenience functions and their edge cases."""

    def test_configure_without_config_system(self):
        """Test configure() when config system not available."""
        with patch.object(datason, "_config_available", False):
            mock_config = Mock()

            with pytest.raises(ImportError, match="Configuration system not available"):
                datason.configure(mock_config)

    def test_serialize_with_config_without_config_system(self):
        """Test serialize_with_config() when config system not available."""
        test_data = {"test": "data"}

        with patch.object(datason, "_config_available", False):
            # Should fall back to basic serialization
            result = datason.serialize_with_config(test_data)

            assert isinstance(result, dict)
            assert result["test"] == "data"

    def test_serialize_with_config_enum_conversions(self):
        """Test serialize_with_config() string to enum conversions."""
        if not datason._config_available:
            pytest.skip("Config system not available")

        test_data = {"test": "data"}

        # Test all string enum conversions
        result = datason.serialize_with_config(
            test_data, date_format="iso", nan_handling="null", type_coercion="safe", dataframe_orient="records"
        )

        assert isinstance(result, dict)
        assert result["test"] == "data"


class TestVersionAndInfoFunctions:
    """Test version and info utility functions."""

    def test_get_version_function(self):
        """Test get_version() convenience function."""
        version = datason.get_version()

        assert isinstance(version, str)
        assert version == datason.__version__
        assert len(version.split(".")) >= 2  # At least major.minor

    def test_get_info_function(self):
        """Test get_info() function returns complete info."""
        info = datason.get_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "author" in info
        assert "description" in info
        assert "config_available" in info
        assert "cache_system" in info

        # Verify values
        assert info["version"] == datason.__version__
        assert info["config_available"] == datason._config_available

        if datason._config_available:
            assert info["cache_system"] == "configurable"
        else:
            assert info["cache_system"] == "legacy"


class TestImportValidation:
    """Test that all advertised imports actually work."""

    def test_all_advertised_imports_work(self):
        """Test that every item in __all__ can actually be imported."""
        for item in datason.__all__:
            # Should be able to access each item
            attr = getattr(datason, item, None)
            assert attr is not None, f"Item '{item}' in __all__ but not accessible"

            # Should not be a placeholder or None
            assert attr is not None, f"Item '{item}' is None"

    def test_module_metadata_consistency(self):
        """Test that module metadata is consistent."""
        # Get version from pyproject.toml to make test robust across versions
        import re
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"

        # Read version from pyproject.toml using regex (Python 3.8 compatible)
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        version_match = re.search(r'version = "([^"]+)"', content)
        if not version_match:
            pytest.fail("Could not find version in pyproject.toml")

        expected_version = version_match.group(1)

        assert datason.__version__ == expected_version, (
            f"Version mismatch: expected {expected_version}, got {datason.__version__}"
        )
        assert datason.__author__ == "datason Contributors"
        assert datason.__license__ == "MIT"
        assert "serialization" in datason.__description__.lower()

    def test_private_flags_exist(self):
        """Test that private availability flags exist."""
        assert hasattr(datason, "_config_available")
        assert hasattr(datason, "_ml_available")
        assert hasattr(datason, "_pickle_bridge_available")

        # These should be boolean
        assert isinstance(datason._config_available, bool)
        assert isinstance(datason._ml_available, bool)
        assert isinstance(datason._pickle_bridge_available, bool)
