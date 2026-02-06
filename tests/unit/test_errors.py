"""Tests for datason error hierarchy."""

from __future__ import annotations

import pytest

from datason._errors import (
    DatasonError,
    DeserializationError,
    PluginError,
    SecurityError,
    SerializationError,
)


class TestErrorHierarchy:
    """Verify all errors inherit from DatasonError."""

    @pytest.mark.parametrize(
        "error_class",
        [SecurityError, SerializationError, DeserializationError, PluginError],
    )
    def test_inherits_from_datason_error(self, error_class):
        assert issubclass(error_class, DatasonError)

    @pytest.mark.parametrize(
        "error_class",
        [SecurityError, SerializationError, DeserializationError, PluginError],
    )
    def test_inherits_from_exception(self, error_class):
        assert issubclass(error_class, Exception)

    def test_can_catch_all_with_base(self):
        with pytest.raises(DatasonError):
            raise SecurityError("test")

    def test_error_message_preserved(self):
        with pytest.raises(SecurityError, match="depth exceeded"):
            raise SecurityError("depth exceeded")
