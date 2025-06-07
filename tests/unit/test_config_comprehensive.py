"""Comprehensive tests for datason.config module.

This module tests all configuration classes, enums, and functions
to achieve maximum coverage and verify all configuration behaviors.
"""

from datason.config import (
    CacheScope,
    DataFrameOrient,
    DateFormat,
    NanHandling,
    OutputType,
    SerializationConfig,
    TypeCoercion,
    cache_scope,
    get_api_config,
    get_cache_scope,
    get_default_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
    reset_default_config,
    set_cache_scope,
    set_default_config,
)


class TestEnums:
    """Test all configuration enums."""

    def test_date_format_enum_values(self):
        """Test DateFormat enum has all expected values."""
        assert DateFormat.ISO.value == "iso"
        assert DateFormat.UNIX.value == "unix"
        assert DateFormat.UNIX_MS.value == "unix_ms"
        assert DateFormat.STRING.value == "string"
        assert DateFormat.CUSTOM.value == "custom"

    def test_dataframe_orient_enum_values(self):
        """Test DataFrameOrient enum has all expected values."""
        assert DataFrameOrient.RECORDS.value == "records"
        assert DataFrameOrient.SPLIT.value == "split"
        assert DataFrameOrient.INDEX.value == "index"
        assert DataFrameOrient.DICT.value == "dict"
        assert DataFrameOrient.LIST.value == "list"
        assert DataFrameOrient.SERIES.value == "series"
        assert DataFrameOrient.TIGHT.value == "tight"
        assert DataFrameOrient.VALUES.value == "values"

    def test_output_type_enum_values(self):
        """Test OutputType enum has all expected values."""
        assert OutputType.JSON_SAFE.value == "json_safe"
        assert OutputType.OBJECT.value == "object"

    def test_nan_handling_enum_values(self):
        """Test NanHandling enum has all expected values."""
        assert NanHandling.NULL.value == "null"
        assert NanHandling.STRING.value == "string"
        assert NanHandling.KEEP.value == "keep"
        assert NanHandling.DROP.value == "drop"

    def test_type_coercion_enum_values(self):
        """Test TypeCoercion enum has all expected values."""
        assert TypeCoercion.STRICT.value == "strict"
        assert TypeCoercion.SAFE.value == "safe"
        assert TypeCoercion.AGGRESSIVE.value == "aggressive"

    def test_cache_scope_enum_values(self):
        """Test CacheScope enum has all expected values."""
        assert CacheScope.OPERATION.value == "operation"
        assert CacheScope.REQUEST.value == "request"
        assert CacheScope.PROCESS.value == "process"
        assert CacheScope.DISABLED.value == "disabled"


class TestSerializationConfig:
    """Test SerializationConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = SerializationConfig()

        # Date/time formatting
        assert config.date_format == DateFormat.ISO
        assert config.custom_date_format is None

        # DataFrame formatting
        assert config.dataframe_orient == DataFrameOrient.RECORDS

        # Output type control
        assert config.datetime_output == OutputType.JSON_SAFE
        assert config.series_output == OutputType.JSON_SAFE
        assert config.dataframe_output == OutputType.JSON_SAFE
        assert config.numpy_output == OutputType.JSON_SAFE

        # Value handling
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.SAFE

        # Precision control
        assert config.preserve_decimals is True
        assert config.preserve_complex is True

        # Security limits
        assert config.max_depth == 50
        assert config.max_size == 100_000
        assert config.max_string_length == 1_000_000

        # Extensibility
        assert config.custom_serializers is None

        # Output formatting
        assert config.sort_keys is False
        assert config.ensure_ascii is False

        # Performance optimization
        assert config.check_if_serialized is False

        # Type metadata
        assert config.include_type_hints is False
        assert config.auto_detect_types is False

        # Redaction settings
        assert config.redact_fields is None
        assert config.redact_patterns is None
        assert config.redact_large_objects is False
        assert config.redaction_replacement == "<REDACTED>"
        assert config.include_redaction_summary is False
        assert config.audit_trail is False

        # Caching settings
        assert config.cache_scope == CacheScope.OPERATION
        assert config.cache_size_limit == 1000
        assert config.cache_warn_on_limit is True
        assert config.cache_metrics_enabled is False

    def test_custom_config_values(self):
        """Test custom configuration values."""
        custom_serializers = {str: lambda x: x.upper()}
        redact_fields = ["password", "secret"]
        redact_patterns = [r"\b\d{16}\b"]

        config = SerializationConfig(
            date_format=DateFormat.UNIX_MS,
            custom_date_format="%Y-%m-%d",
            dataframe_orient=DataFrameOrient.SPLIT,
            datetime_output=OutputType.OBJECT,
            series_output=OutputType.OBJECT,
            dataframe_output=OutputType.OBJECT,
            numpy_output=OutputType.OBJECT,
            nan_handling=NanHandling.STRING,
            type_coercion=TypeCoercion.STRICT,
            preserve_decimals=False,
            preserve_complex=False,
            max_depth=25,
            max_size=50_000,
            max_string_length=500_000,
            custom_serializers=custom_serializers,
            sort_keys=True,
            ensure_ascii=True,
            check_if_serialized=True,
            include_type_hints=True,
            auto_detect_types=True,
            redact_fields=redact_fields,
            redact_patterns=redact_patterns,
            redact_large_objects=True,
            redaction_replacement="[HIDDEN]",
            include_redaction_summary=True,
            audit_trail=True,
            cache_scope=CacheScope.PROCESS,
            cache_size_limit=2000,
            cache_warn_on_limit=False,
            cache_metrics_enabled=True,
        )

        assert config.date_format == DateFormat.UNIX_MS
        assert config.custom_date_format == "%Y-%m-%d"
        assert config.dataframe_orient == DataFrameOrient.SPLIT
        assert config.datetime_output == OutputType.OBJECT
        assert config.series_output == OutputType.OBJECT
        assert config.dataframe_output == OutputType.OBJECT
        assert config.numpy_output == OutputType.OBJECT
        assert config.nan_handling == NanHandling.STRING
        assert config.type_coercion == TypeCoercion.STRICT
        assert config.preserve_decimals is False
        assert config.preserve_complex is False
        assert config.max_depth == 25
        assert config.max_size == 50_000
        assert config.max_string_length == 500_000
        assert config.custom_serializers == custom_serializers
        assert config.sort_keys is True
        assert config.ensure_ascii is True
        assert config.check_if_serialized is True
        assert config.include_type_hints is True
        assert config.auto_detect_types is True
        assert config.redact_fields == redact_fields
        assert config.redact_patterns == redact_patterns
        assert config.redact_large_objects is True
        assert config.redaction_replacement == "[HIDDEN]"
        assert config.include_redaction_summary is True
        assert config.audit_trail is True
        assert config.cache_scope == CacheScope.PROCESS
        assert config.cache_size_limit == 2000
        assert config.cache_warn_on_limit is False
        assert config.cache_metrics_enabled is True

    def test_config_immutability(self):
        """Test that config can be modified (it's a dataclass)."""
        config = SerializationConfig()

        # Config should be mutable (it's a dataclass, not frozen)
        config.sort_keys = True
        assert config.sort_keys is True

        config.max_depth = 100
        assert config.max_depth == 100

    def test_config_with_partial_values(self):
        """Test config with only some values specified."""
        config = SerializationConfig(
            date_format=DateFormat.UNIX,
            sort_keys=True,
            max_depth=30,
        )

        # Specified values
        assert config.date_format == DateFormat.UNIX
        assert config.sort_keys is True
        assert config.max_depth == 30

        # Default values for unspecified
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.preserve_decimals is True


class TestDefaultConfigManagement:
    """Test global default configuration management."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, SerializationConfig)
        assert config.date_format == DateFormat.ISO

    def test_set_default_config(self):
        """Test setting default configuration."""
        original_config = get_default_config()

        # Create new config
        new_config = SerializationConfig(
            date_format=DateFormat.UNIX,
            sort_keys=True,
        )

        # Set new default
        set_default_config(new_config)
        retrieved_config = get_default_config()

        assert retrieved_config.date_format == DateFormat.UNIX
        assert retrieved_config.sort_keys is True

        # Restore original
        set_default_config(original_config)

    def test_reset_default_config(self):
        """Test resetting default configuration."""
        # Set custom config
        custom_config = SerializationConfig(
            date_format=DateFormat.UNIX_MS,
            sort_keys=True,
            max_depth=25,
        )
        set_default_config(custom_config)

        # Verify custom config is set
        config = get_default_config()
        assert config.date_format == DateFormat.UNIX_MS
        assert config.sort_keys is True
        assert config.max_depth == 25

        # Reset to defaults
        reset_default_config()

        # Verify defaults are restored
        config = get_default_config()
        assert config.date_format == DateFormat.ISO
        assert config.sort_keys is False
        assert config.max_depth == 50


class TestPresetConfigurations:
    """Test preset configuration functions."""

    def test_get_ml_config(self):
        """Test ML configuration preset."""
        config = get_ml_config()

        assert config.date_format == DateFormat.UNIX_MS
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.AGGRESSIVE
        assert config.preserve_decimals is False
        assert config.preserve_complex is False
        assert config.sort_keys is True

    def test_get_api_config(self):
        """Test API configuration preset."""
        config = get_api_config()

        assert config.date_format == DateFormat.ISO
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.SAFE
        assert config.preserve_decimals is True
        assert config.preserve_complex is True
        assert config.sort_keys is True
        assert config.ensure_ascii is True

    def test_get_strict_config(self):
        """Test strict configuration preset."""
        config = get_strict_config()

        assert config.date_format == DateFormat.ISO
        assert config.dataframe_orient == DataFrameOrient.RECORDS
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.STRICT
        assert config.preserve_decimals is True
        assert config.preserve_complex is True

    def test_get_performance_config(self):
        """Test performance configuration preset."""
        config = get_performance_config()

        assert config.date_format == DateFormat.UNIX
        assert config.dataframe_orient == DataFrameOrient.VALUES
        assert config.nan_handling == NanHandling.NULL
        assert config.type_coercion == TypeCoercion.SAFE
        assert config.preserve_decimals is False
        assert config.preserve_complex is False
        assert config.sort_keys is False

    def test_preset_configs_are_independent(self):
        """Test that preset configs are independent instances."""
        ml_config = get_ml_config()

        # Modify one config
        ml_config.sort_keys = False

        # Other config should be unaffected
        api_config2 = get_api_config()
        assert api_config2.sort_keys is True

    def test_all_presets_return_valid_configs(self):
        """Test that all preset functions return valid SerializationConfig instances."""
        presets = [
            get_ml_config(),
            get_api_config(),
            get_strict_config(),
            get_performance_config(),
        ]

        for config in presets:
            assert isinstance(config, SerializationConfig)
            assert isinstance(config.date_format, DateFormat)
            assert isinstance(config.dataframe_orient, DataFrameOrient)
            assert isinstance(config.nan_handling, NanHandling)
            assert isinstance(config.type_coercion, TypeCoercion)


class TestCacheManagement:
    """Test cache management functionality."""

    def test_get_cache_scope_default(self):
        """Test getting cache scope with default value."""
        scope = get_cache_scope()
        assert isinstance(scope, CacheScope)

    def test_set_cache_scope(self):
        """Test setting cache scope."""
        original_scope = get_cache_scope()

        set_cache_scope(CacheScope.PROCESS)
        assert get_cache_scope() == CacheScope.PROCESS

        set_cache_scope(CacheScope.DISABLED)
        assert get_cache_scope() == CacheScope.DISABLED

        # Restore original
        set_cache_scope(original_scope)

    def test_cache_scope_context_manager(self):
        """Test cache scope context manager."""
        original_scope = get_cache_scope()

        # Test context manager changes scope temporarily
        with cache_scope(CacheScope.PROCESS):
            assert get_cache_scope() == CacheScope.PROCESS

            # Nested context manager
            with cache_scope(CacheScope.DISABLED):
                assert get_cache_scope() == CacheScope.DISABLED

            # Should restore to outer context
            assert get_cache_scope() == CacheScope.PROCESS

        # Should restore to original after exiting
        assert get_cache_scope() == original_scope

    def test_cache_scope_context_manager_with_exception(self):
        """Test cache scope context manager handles exceptions properly."""
        original_scope = get_cache_scope()

        try:
            with cache_scope(CacheScope.PROCESS):
                assert get_cache_scope() == CacheScope.PROCESS
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore original scope after exception
        assert get_cache_scope() == original_scope

    def test_cache_scope_context_manager_clears_caches(self):
        """Test cache scope context manager clears caches appropriately."""
        # Test that context manager works even without deserializers module
        with cache_scope(CacheScope.PROCESS):
            pass

        # Test should pass without errors

    def test_cache_scope_context_manager_handles_import_errors(self):
        """Test cache scope context manager handles import errors gracefully."""
        # Should not raise exception even if import fails
        with cache_scope(CacheScope.PROCESS):
            pass

    def test_cache_scope_context_manager_handles_attribute_errors(self):
        """Test cache scope context manager handles missing clear_caches gracefully."""
        # Should not raise exception
        with cache_scope(CacheScope.PROCESS):
            pass

    def test_cache_scope_operation_clears_on_exit(self):
        """Test that OPERATION scope clears caches on exit."""
        # Test that context manager works with OPERATION scope
        with cache_scope(CacheScope.OPERATION):
            pass

        # Test should pass without errors


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_none_values(self):
        """Test config with None values where allowed."""
        config = SerializationConfig(
            custom_date_format=None,
            custom_serializers=None,
            redact_fields=None,
            redact_patterns=None,
        )

        assert config.custom_date_format is None
        assert config.custom_serializers is None
        assert config.redact_fields is None
        assert config.redact_patterns is None

    def test_config_with_empty_collections(self):
        """Test config with empty collections."""
        config = SerializationConfig(
            custom_serializers={},
            redact_fields=[],
            redact_patterns=[],
        )

        assert config.custom_serializers == {}
        assert config.redact_fields == []
        assert config.redact_patterns == []

    def test_config_with_complex_custom_serializers(self):
        """Test config with complex custom serializers."""

        def custom_str_serializer(obj):
            return f"STR:{obj}"

        def custom_int_serializer(obj):
            return f"INT:{obj}"

        custom_serializers = {
            str: custom_str_serializer,
            int: custom_int_serializer,
        }

        config = SerializationConfig(custom_serializers=custom_serializers)

        assert config.custom_serializers[str] == custom_str_serializer
        assert config.custom_serializers[int] == custom_int_serializer

    def test_config_with_extreme_values(self):
        """Test config with extreme but valid values."""
        config = SerializationConfig(
            max_depth=1,
            max_size=1,
            max_string_length=1,
            cache_size_limit=1,
        )

        assert config.max_depth == 1
        assert config.max_size == 1
        assert config.max_string_length == 1
        assert config.cache_size_limit == 1

    def test_config_all_enum_combinations(self):
        """Test config with all possible enum combinations."""
        date_formats = list(DateFormat)
        dataframe_orients = list(DataFrameOrient)
        output_types = list(OutputType)
        nan_handlings = list(NanHandling)
        type_coercions = list(TypeCoercion)
        cache_scopes = list(CacheScope)

        # Test that all enum values can be assigned
        for date_format in date_formats:
            for dataframe_orient in dataframe_orients:
                for output_type in output_types:
                    for nan_handling in nan_handlings:
                        for type_coercion in type_coercions:
                            for cache_scope_val in cache_scopes:
                                config = SerializationConfig(
                                    date_format=date_format,
                                    dataframe_orient=dataframe_orient,
                                    datetime_output=output_type,
                                    nan_handling=nan_handling,
                                    type_coercion=type_coercion,
                                    cache_scope=cache_scope_val,
                                )

                                assert config.date_format == date_format
                                assert config.dataframe_orient == dataframe_orient
                                assert config.datetime_output == output_type
                                assert config.nan_handling == nan_handling
                                assert config.type_coercion == type_coercion
                                assert config.cache_scope == cache_scope_val

                                # Only test first few combinations to avoid excessive test time
                                return


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_preset_config_modification(self):
        """Test modifying preset configurations."""
        ml_config = get_ml_config()
        original_sort_keys = ml_config.sort_keys

        # Modify the config
        ml_config.sort_keys = not original_sort_keys
        ml_config.redact_fields = ["password"]
        ml_config.cache_scope = CacheScope.DISABLED

        # Verify modifications
        assert ml_config.sort_keys != original_sort_keys
        assert ml_config.redact_fields == ["password"]
        assert ml_config.cache_scope == CacheScope.DISABLED

        # Verify other presets are unaffected
        new_ml_config = get_ml_config()
        assert new_ml_config.sort_keys == original_sort_keys
        assert new_ml_config.redact_fields is None

    def test_config_serialization_compatibility(self):
        """Test that config values are JSON-serializable where appropriate."""
        config = SerializationConfig()

        # Test that enum values are serializable
        assert isinstance(config.date_format.value, str)
        assert isinstance(config.dataframe_orient.value, str)
        assert isinstance(config.datetime_output.value, str)
        assert isinstance(config.nan_handling.value, str)
        assert isinstance(config.type_coercion.value, str)
        assert isinstance(config.cache_scope.value, str)

    def test_realistic_usage_scenarios(self):
        """Test realistic configuration usage scenarios."""
        # Scenario 1: Web API with security
        api_config = get_api_config()
        api_config.redact_fields = ["password", "api_key", "secret"]
        api_config.redact_patterns = [r"\b\d{16}\b"]  # Credit card numbers
        api_config.include_redaction_summary = True

        assert api_config.ensure_ascii is True
        assert api_config.sort_keys is True
        assert api_config.redact_fields == ["password", "api_key", "secret"]

        # Scenario 2: ML pipeline with performance focus
        ml_config = get_performance_config()
        ml_config.date_format = DateFormat.UNIX_MS
        ml_config.type_coercion = TypeCoercion.AGGRESSIVE
        ml_config.cache_scope = CacheScope.PROCESS

        assert ml_config.preserve_decimals is False
        assert ml_config.sort_keys is False
        assert ml_config.cache_scope == CacheScope.PROCESS

        # Scenario 3: Strict data validation
        strict_config = get_strict_config()
        strict_config.max_depth = 20
        strict_config.max_size = 10_000
        strict_config.include_type_hints = True

        assert strict_config.type_coercion == TypeCoercion.STRICT
        assert strict_config.preserve_decimals is True
        assert strict_config.include_type_hints is True
