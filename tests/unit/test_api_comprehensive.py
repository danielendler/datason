"""Comprehensive tests for datason.api module.

This module tests the Phase 3 API Modernization functions including
all dump/load variants, migration helpers, and utility functions.
"""

import warnings
from unittest.mock import Mock, patch

import pytest

import datason.api as api
from datason.config import SerializationConfig


class TestDeprecationWarnings:
    """Test deprecation warning functionality."""

    def test_suppress_deprecation_warnings_enable(self):
        """Test enabling deprecation warning suppression."""
        api.suppress_deprecation_warnings(True)
        assert api._suppress_deprecation_warnings is True

    def test_suppress_deprecation_warnings_disable(self):
        """Test disabling deprecation warning suppression."""
        api.suppress_deprecation_warnings(False)
        assert api._suppress_deprecation_warnings is False

    def test_suppress_deprecation_warnings_default(self):
        """Test default behavior of deprecation warning suppression."""
        api.suppress_deprecation_warnings()
        assert api._suppress_deprecation_warnings is True


class TestDumpFunction:
    """Test the main dump() function."""

    def test_dump_basic_usage(self):
        """Test basic dump functionality."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"test": "data"}

            result = api.dump({"input": "data"})

            mock_serialize.assert_called_once()
            assert result == {"test": "data"}

    def test_dump_with_config(self):
        """Test dump with explicit config."""
        config = SerializationConfig()

        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"test": "data"}

            api.dump({"input": "data"}, config=config)

            mock_serialize.assert_called_once_with({"input": "data"}, config=config)

    def test_dump_ml_mode(self):
        """Test dump with ML mode enabled."""
        with patch("datason.api.serialize") as mock_serialize, patch("datason.api.get_ml_config") as mock_get_ml_config:
            mock_config = SerializationConfig()
            mock_get_ml_config.return_value = mock_config
            mock_serialize.return_value = {"test": "data"}

            api.dump({"input": "data"}, ml_mode=True)

            mock_get_ml_config.assert_called_once()
            mock_serialize.assert_called_once_with({"input": "data"}, config=mock_config)

    def test_dump_api_mode(self):
        """Test dump with API mode enabled."""
        with patch("datason.api.serialize") as mock_serialize, patch(
            "datason.api.get_api_config"
        ) as mock_get_api_config:
            mock_config = SerializationConfig()
            mock_get_api_config.return_value = mock_config
            mock_serialize.return_value = {"test": "data"}

            api.dump({"input": "data"}, api_mode=True)

            mock_get_api_config.assert_called_once()
            mock_serialize.assert_called_once_with({"input": "data"}, config=mock_config)

    def test_dump_fast_mode(self):
        """Test dump with fast mode enabled."""
        with patch("datason.api.serialize") as mock_serialize, patch(
            "datason.api.get_performance_config"
        ) as mock_get_performance_config:
            mock_config = SerializationConfig()
            mock_get_performance_config.return_value = mock_config
            mock_serialize.return_value = {"test": "data"}

            api.dump({"input": "data"}, fast_mode=True)

            mock_get_performance_config.assert_called_once()
            mock_serialize.assert_called_once_with({"input": "data"}, config=mock_config)

    def test_dump_multiple_modes_error(self):
        """Test error when multiple modes are enabled."""
        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            api.dump({"input": "data"}, ml_mode=True, api_mode=True)

        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            api.dump({"input": "data"}, ml_mode=True, fast_mode=True)

        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            api.dump({"input": "data"}, api_mode=True, fast_mode=True)

        with pytest.raises(ValueError, match="Only one mode can be enabled"):
            api.dump({"input": "data"}, ml_mode=True, api_mode=True, fast_mode=True)

    def test_dump_secure_mode(self):
        """Test dump with secure mode enabled."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"test": "data"}

            api.dump({"input": "sensitive@email.com"}, secure=True)

            # Verify serialize was called
            mock_serialize.assert_called_once()

            # Verify config has security settings
            call_args = mock_serialize.call_args
            config = call_args[1]["config"]

            assert config.redact_patterns is not None
            assert any("@" in pattern for pattern in config.redact_patterns)
            assert config.redact_fields is not None
            assert "password" in config.redact_fields
            assert config.include_redaction_summary is True

    def test_dump_chunked_mode(self):
        """Test dump with chunked mode enabled."""
        with patch("datason.api.serialize_chunked") as mock_serialize_chunked:
            mock_serialize_chunked.return_value = {"chunks": ["data"]}

            result = api.dump({"input": "data"}, chunked=True, chunk_size=500)

            mock_serialize_chunked.assert_called_once_with({"input": "data"}, chunk_size=500, config=None)
            assert result == {"chunks": ["data"]}

    def test_dump_with_kwargs(self):
        """Test dump with additional kwargs."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"test": "data"}

            # Test that kwargs are passed but don't necessarily create valid config
            try:
                api.dump({"input": "data"}, custom_option=True, another_option="value")
            except TypeError:
                # This is expected behavior - invalid kwargs should raise TypeError
                pass


class TestDumpVariants:
    """Test dump variant functions."""

    def test_dump_ml(self):
        """Test dump_ml function."""
        with patch("datason.api.serialize") as mock_serialize, patch("datason.api.get_ml_config") as mock_get_ml_config:
            mock_config = SerializationConfig()
            mock_get_ml_config.return_value = mock_config
            mock_serialize.return_value = {"ml": "data"}

            result = api.dump_ml({"model": "data"})

            mock_get_ml_config.assert_called_once()
            mock_serialize.assert_called_once_with({"model": "data"}, config=mock_config)
            assert result == {"ml": "data"}

    def test_dump_api(self):
        """Test dump_api function."""
        with patch("datason.api.serialize") as mock_serialize, patch(
            "datason.api.get_api_config"
        ) as mock_get_api_config:
            mock_config = SerializationConfig()
            mock_get_api_config.return_value = mock_config
            mock_serialize.return_value = {"api": "data"}

            result = api.dump_api({"response": "data"})

            mock_get_api_config.assert_called_once()
            mock_serialize.assert_called_once_with({"response": "data"}, config=mock_config)
            assert result == {"api": "data"}

    def test_dump_secure_default_settings(self):
        """Test dump_secure with default settings."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"secure": "data"}

            result = api.dump_secure({"sensitive": "data"})

            mock_serialize.assert_called_once()
            call_args = mock_serialize.call_args
            obj, config = call_args[0][0], call_args[1]["config"]

            assert obj == {"sensitive": "data"}
            assert config.redact_patterns is not None
            assert config.redact_fields is not None
            assert "password" in config.redact_fields
            assert result == {"secure": "data"}

    def test_dump_secure_custom_settings(self):
        """Test dump_secure with custom settings."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"secure": "data"}

            api.dump_secure(
                {"sensitive": "data"}, redact_pii=False, redact_fields=["custom_field"], redact_patterns=[r"\bcustom\b"]
            )

            mock_serialize.assert_called_once()

    def test_dump_fast(self):
        """Test dump_fast function."""
        with patch("datason.api.serialize") as mock_serialize, patch(
            "datason.api.get_performance_config"
        ) as mock_get_performance_config:
            mock_config = SerializationConfig()
            mock_get_performance_config.return_value = mock_config
            mock_serialize.return_value = {"fast": "data"}

            result = api.dump_fast({"speed": "data"})

            mock_get_performance_config.assert_called_once()
            mock_serialize.assert_called_once_with({"speed": "data"}, config=mock_config)
            assert result == {"fast": "data"}

    def test_dump_chunked(self):
        """Test dump_chunked function."""
        with patch("datason.api.serialize_chunked") as mock_serialize_chunked:
            mock_serialize_chunked.return_value = {"chunks": ["data1", "data2"]}

            result = api.dump_chunked({"large": "data"}, chunk_size=2000)

            mock_serialize_chunked.assert_called_once_with({"large": "data"}, chunk_size=2000)
            assert result == {"chunks": ["data1", "data2"]}

    def test_stream_dump(self):
        """Test stream_dump function."""
        with patch("datason.api.stream_serialize") as mock_stream_serialize:
            mock_stream_serialize.return_value = Mock()

            result = api.stream_dump("output.jsonl")

            mock_stream_serialize.assert_called_once_with("output.jsonl")
            assert result is not None


class TestLoadFunctions:
    """Test load variant functions."""

    def test_load_basic(self):
        """Test load_basic function."""
        with patch("datason.api.deserialize") as mock_deserialize:
            mock_deserialize.return_value = {"loaded": "data"}

            result = api.load_basic({"serialized": "data"})

            mock_deserialize.assert_called_once_with({"serialized": "data"})
            assert result == {"loaded": "data"}

    def test_load_basic_with_kwargs(self):
        """Test load_basic with additional kwargs."""
        with patch("datason.api.deserialize") as mock_deserialize:
            mock_deserialize.return_value = {"loaded": "data"}

            api.load_basic({"serialized": "data"}, parse_dates=True, custom_option="value")

            mock_deserialize.assert_called_once_with({"serialized": "data"}, parse_dates=True, custom_option="value")

    def test_load_smart_default_config(self):
        """Test load_smart with default config."""
        with patch("datason.api.deserialize_fast") as mock_deserialize_fast:
            mock_deserialize_fast.return_value = {"smart": "data"}

            result = api.load_smart({"serialized": "data"})

            mock_deserialize_fast.assert_called_once()
            call_args = mock_deserialize_fast.call_args
            assert call_args[0][0] == {"serialized": "data"}

            config = call_args[1]["config"]
            assert config.auto_detect_types is True
            assert result == {"smart": "data"}

    def test_load_smart_custom_config(self):
        """Test load_smart with custom config."""
        custom_config = SerializationConfig(auto_detect_types=False)

        with patch("datason.api.deserialize_fast") as mock_deserialize_fast:
            mock_deserialize_fast.return_value = {"smart": "data"}

            api.load_smart({"serialized": "data"}, config=custom_config)

            mock_deserialize_fast.assert_called_once_with({"serialized": "data"}, config=custom_config)

    def test_load_perfect(self):
        """Test load_perfect function."""
        template = {"template": "structure"}

        with patch("datason.api.deserialize_with_template") as mock_deserialize_template:
            mock_deserialize_template.return_value = {"perfect": "data"}

            result = api.load_perfect({"serialized": "data"}, template)

            mock_deserialize_template.assert_called_once_with({"serialized": "data"}, template)
            assert result == {"perfect": "data"}

    def test_load_perfect_with_kwargs(self):
        """Test load_perfect with additional kwargs."""
        template = {"template": "structure"}

        with patch("datason.api.deserialize_with_template") as mock_deserialize_template:
            mock_deserialize_template.return_value = {"perfect": "data"}

            api.load_perfect({"serialized": "data"}, template, strict_mode=True, validate=True)

            mock_deserialize_template.assert_called_once_with(
                {"serialized": "data"}, template, strict_mode=True, validate=True
            )

    def test_load_typed_default_config(self):
        """Test load_typed with default config."""
        with patch("datason.api.deserialize_fast") as mock_deserialize_fast, patch(
            "datason.api.get_strict_config"
        ) as mock_get_strict_config:
            mock_config = SerializationConfig()
            mock_get_strict_config.return_value = mock_config
            mock_deserialize_fast.return_value = {"typed": "data"}

            result = api.load_typed({"serialized": "data"})

            mock_get_strict_config.assert_called_once()
            mock_deserialize_fast.assert_called_once_with({"serialized": "data"}, config=mock_config)
            assert result == {"typed": "data"}

    def test_load_typed_custom_config(self):
        """Test load_typed with custom config."""
        custom_config = SerializationConfig()

        with patch("datason.api.deserialize_fast") as mock_deserialize_fast:
            mock_deserialize_fast.return_value = {"typed": "data"}

            api.load_typed({"serialized": "data"}, config=custom_config)

            mock_deserialize_fast.assert_called_once_with({"serialized": "data"}, config=custom_config)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_loads(self):
        """Test loads function."""
        json_string = '{"key": "value"}'

        with patch("datason.api.load_basic") as mock_load_basic:
            mock_load_basic.return_value = {"loaded": "data"}

            result = api.loads(json_string)

            mock_load_basic.assert_called_once_with({"key": "value"})
            assert result == {"loaded": "data"}

    def test_loads_with_kwargs(self):
        """Test loads with additional kwargs."""
        json_string = '{"key": "value"}'

        with patch("datason.api.load_basic") as mock_load_basic:
            mock_load_basic.return_value = {"loaded": "data"}

            api.loads(json_string, parse_dates=True)

            mock_load_basic.assert_called_once_with({"key": "value"}, parse_dates=True)

    def test_loads_invalid_json(self):
        """Test loads with invalid JSON."""
        invalid_json = '{"key": invalid}'

        with pytest.raises(Exception):  # Should raise JSON decode error
            api.loads(invalid_json)

    def test_dumps(self):
        """Test dumps function."""
        input_obj = {"key": "value"}

        with patch("datason.api.dump") as mock_dump:
            mock_dump.return_value = {"serialized": "data"}

            result = api.dumps(input_obj)

            mock_dump.assert_called_once_with(input_obj)
            assert result == '{"serialized": "data"}'

    def test_dumps_with_kwargs(self):
        """Test dumps with additional kwargs."""
        input_obj = {"key": "value"}

        with patch("datason.api.dump") as mock_dump:
            mock_dump.return_value = {"serialized": "data"}

            api.dumps(input_obj, secure=True, ml_mode=False)

            mock_dump.assert_called_once_with(input_obj, secure=True, ml_mode=False)


class TestMigrationHelpers:
    """Test migration helper functions."""

    def test_serialize_modern_with_warnings(self):
        """Test serialize_modern shows deprecation warning."""
        api._suppress_deprecation_warnings = False

        with patch("datason.api.serialize") as mock_serialize, pytest.warns(
            DeprecationWarning, match="serialize\\(\\) is deprecated"
        ):
            mock_serialize.return_value = {"data": "serialized"}

            result = api.serialize_modern({"input": "data"})

            mock_serialize.assert_called_once_with({"input": "data"})
            assert result == {"data": "serialized"}

    def test_serialize_modern_suppressed_warnings(self):
        """Test serialize_modern with suppressed warnings."""
        api._suppress_deprecation_warnings = True

        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"data": "serialized"}

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = api.serialize_modern({"input": "data"})

                # Should not have any deprecation warnings
                deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
                assert len(deprecation_warnings) == 0

            mock_serialize.assert_called_once_with({"input": "data"})
            assert result == {"data": "serialized"}

    def test_deserialize_modern_with_warnings(self):
        """Test deserialize_modern shows deprecation warning."""
        api._suppress_deprecation_warnings = False

        with patch("datason.api.deserialize") as mock_deserialize, pytest.warns(
            DeprecationWarning, match="deserialize\\(\\) is deprecated"
        ):
            mock_deserialize.return_value = {"data": "deserialized"}

            result = api.deserialize_modern({"input": "data"})

            mock_deserialize.assert_called_once_with({"input": "data"})
            assert result == {"data": "deserialized"}

    def test_deserialize_modern_suppressed_warnings(self):
        """Test deserialize_modern with suppressed warnings."""
        api._suppress_deprecation_warnings = True

        with patch("datason.api.deserialize") as mock_deserialize:
            mock_deserialize.return_value = {"data": "deserialized"}

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = api.deserialize_modern({"input": "data"})

                # Should not have any deprecation warnings
                deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
                assert len(deprecation_warnings) == 0

            mock_deserialize.assert_called_once_with({"input": "data"})
            assert result == {"data": "deserialized"}


class TestUtilityFunctions:
    """Test utility functions."""

    def test_help_api(self):
        """Test help_api function."""
        result = api.help_api()

        assert isinstance(result, dict)
        assert "serialization" in result
        assert "deserialization" in result
        assert "recommendations" in result

        # Check serialization section
        assert "basic" in result["serialization"]
        assert "ml_optimized" in result["serialization"]
        assert "api_safe" in result["serialization"]
        assert "secure" in result["serialization"]

        # Check deserialization section
        assert "basic" in result["deserialization"]
        assert "smart" in result["deserialization"]
        assert "perfect" in result["deserialization"]
        assert "typed" in result["deserialization"]

        # Check that functions are properly named
        assert result["serialization"]["basic"]["function"] == "dump()"
        assert result["deserialization"]["basic"]["function"] == "load_basic()"

    def test_help_api_structure(self):
        """Test help_api returns expected structure."""
        result = api.help_api()

        # Verify each serialization method has required fields
        for method_name, method_info in result["serialization"].items():
            assert "function" in method_info
            assert "use_case" in method_info
            if method_name != "chunked":  # chunked doesn't have example
                assert "example" in method_info

        # Verify each deserialization method has required fields
        for method_name, method_info in result["deserialization"].items():
            assert "function" in method_info
            assert "success_rate" in method_info
            assert "speed" in method_info
            assert "use_case" in method_info

        # Verify recommendations is a list
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_get_api_info(self):
        """Test get_api_info function."""
        result = api.get_api_info()

        assert isinstance(result, dict)
        assert result["api_version"] == "modern"
        assert result["phase"] == "3"

        # Check features
        assert isinstance(result["features"], dict)
        assert result["features"]["intention_revealing_names"] is True
        assert result["features"]["backward_compatibility"] is True

        # Check function lists
        expected_dump_functions = [
            "dump",
            "dump_ml",
            "dump_api",
            "dump_secure",
            "dump_fast",
            "dump_chunked",
            "stream_dump",
        ]
        assert result["dump_functions"] == expected_dump_functions

        expected_load_functions = ["load_basic", "load_smart", "load_perfect", "load_typed"]
        assert result["load_functions"] == expected_load_functions

        expected_convenience = ["loads", "dumps"]
        assert result["convenience"] == expected_convenience

        expected_help = ["help_api", "get_api_info"]
        assert result["help"] == expected_help


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_dump_with_none_input(self):
        """Test dump function with None input."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = None

            result = api.dump(None)

            mock_serialize.assert_called_once_with(None, config=None)
            assert result is None

    def test_dump_secure_with_none_redact_fields(self):
        """Test dump_secure handles None redact_fields gracefully."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"secure": "data"}

            api.dump_secure({"data": "test"}, redact_fields=None)

            mock_serialize.assert_called_once()

    def test_load_functions_with_empty_data(self):
        """Test load functions with empty data."""
        with patch("datason.api.deserialize") as mock_deserialize:
            mock_deserialize.return_value = {}

            result = api.load_basic({})
            assert result == {}

        with patch("datason.api.deserialize_fast") as mock_deserialize_fast:
            mock_deserialize_fast.return_value = {}

            result = api.load_smart({})
            assert result == {}

    def test_complex_kwargs_handling(self):
        """Test handling of complex kwargs combinations."""
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"test": "data"}

            # Test that invalid kwargs cause TypeError
            with pytest.raises(TypeError):
                api.dump(
                    {"input": "data"},
                    custom_bool=True,
                    custom_str="value",
                    custom_int=42,
                    custom_list=[1, 2, 3],
                    custom_dict={"nested": "value"},
                )


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_ml_workflow_simulation(self):
        """Test a complete ML workflow using the API."""
        # Simulate ML model object
        mock_model = {"type": "sklearn", "model_data": [1, 2, 3]}

        # Serialize with ML optimization
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"serialized_model": "data"}

            serialized = api.dump_ml(mock_model)
            assert serialized == {"serialized_model": "data"}

        # Deserialize with perfect reconstruction
        template = {"type": "sklearn", "model_data": []}
        with patch("datason.api.deserialize_with_template") as mock_deserialize:
            mock_deserialize.return_value = mock_model

            restored = api.load_perfect(serialized, template)
            assert restored == mock_model

    def test_api_workflow_simulation(self):
        """Test a complete API response workflow."""
        api_data = {"users": [{"id": 1, "email": "test@example.com"}], "count": 1}

        # Serialize for API
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"api_response": "data"}

            api_response = api.dump_api(api_data)
            assert api_response == {"api_response": "data"}

        # Convert to JSON string
        with patch("datason.api.dump") as mock_dump:
            mock_dump.return_value = {"json_ready": "data"}

            json_str = api.dumps(api_data)
            assert json_str == '{"json_ready": "data"}'

    def test_secure_workflow_simulation(self):
        """Test a complete secure data workflow."""
        # Serialize securely
        with patch("datason.api.serialize") as mock_serialize:
            mock_serialize.return_value = {"redacted": "data"}

            # Actually call the function to trigger the mock
            secure_data = {"password": "secret", "email": "user@example.com"}
            result = api.dump_secure(secure_data)

            assert result == {"redacted": "data"}
            mock_serialize.assert_called_once()
            call_args = mock_serialize.call_args
            config = call_args[1]["config"]

            # Verify security settings were applied
            assert "password" in config.redact_fields
            assert any("@" in pattern for pattern in config.redact_patterns)
            assert config.include_redaction_summary is True
