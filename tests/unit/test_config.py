"""Tests for datason configuration system."""

from __future__ import annotations

from datason._config import (
    DateFormat,
    NanHandling,
    SerializationConfig,
    api_config,
    ml_config,
    performance_config,
    strict_config,
)


class TestSerializationConfig:
    """Test the config dataclass defaults and presets."""

    def test_default_values(self):
        cfg = SerializationConfig()
        assert cfg.date_format == DateFormat.ISO
        assert cfg.nan_handling == NanHandling.NULL
        assert cfg.max_depth == 50
        assert cfg.max_size == 100_000
        assert cfg.include_type_hints is True
        assert cfg.fallback_to_string is False
        assert cfg.strict is True
        assert cfg.allow_plugin_deserialization is True

    def test_frozen(self):
        """Config is immutable."""
        cfg = SerializationConfig()
        with __import__("pytest").raises(AttributeError):
            cfg.max_depth = 999  # type: ignore[misc]


class TestPresets:
    """Test preset factory functions."""

    def test_ml_config(self):
        cfg = ml_config()
        assert cfg.date_format == DateFormat.UNIX_MS
        assert cfg.fallback_to_string is True

    def test_api_config(self):
        cfg = api_config()
        assert cfg.date_format == DateFormat.ISO
        assert cfg.sort_keys is True
        assert cfg.include_type_hints is False

    def test_strict_config(self):
        cfg = strict_config()
        assert cfg.strict is True
        assert cfg.allow_plugin_deserialization is True
        assert cfg.fallback_to_string is False

    def test_performance_config(self):
        cfg = performance_config()
        assert cfg.include_type_hints is False
        assert cfg.nan_handling == NanHandling.KEEP

    def test_preset_with_overrides(self):
        cfg = ml_config(max_depth=10)
        assert cfg.max_depth == 10
        assert cfg.date_format == DateFormat.UNIX_MS
