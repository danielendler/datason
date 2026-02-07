"""Integration tests for config presets across plugin types.

Tests that config presets (ml, api, strict, performance) correctly
affect serialization behavior across all registered plugins.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

np = pytest.importorskip("numpy")
import datason
from datason._config import (
    DateFormat,
    NanHandling,
    api_config,
    ml_config,
    performance_config,
    strict_config,
)
from datason._errors import SerializationError


class TestMlConfigPreset:
    """ML preset: UNIX_MS dates, fallback, NaN→null, type hints on."""

    def test_datetime_unix_ms(self) -> None:
        config = ml_config()
        assert config.date_format == DateFormat.UNIX_MS
        data = {"ts": dt.datetime(2024, 6, 15, 12, 0)}  # noqa: DTZ001
        s = datason.dumps(data, date_format=config.date_format, include_type_hints=config.include_type_hints)
        result = datason.loads(s)
        assert isinstance(result["ts"], dt.datetime)

    def test_nan_null_with_numpy(self) -> None:
        config = ml_config()
        data = {"values": [1.0, float("nan"), 3.0]}
        s = datason.dumps(data, nan_handling=config.nan_handling)
        parsed = json.loads(s)
        assert parsed["values"][1] is None  # NaN → null

    def test_fallback_to_string_for_unknown_type(self) -> None:
        config = ml_config()

        class CustomObj:
            def __str__(self) -> str:
                return "custom"

        data = {"obj": CustomObj(), "arr": np.array([1.0])}
        s = datason.dumps(
            data, fallback_to_string=config.fallback_to_string, include_type_hints=config.include_type_hints
        )
        parsed = json.loads(s)
        assert parsed["obj"] == "custom"

    def test_type_hints_enabled(self) -> None:
        config = ml_config()
        assert config.include_type_hints is True


class TestApiConfigPreset:
    """API preset: ISO dates, sorted keys, no type hints."""

    def test_iso_dates(self) -> None:
        config = api_config()
        assert config.date_format == DateFormat.ISO
        data = {"ts": dt.datetime(2024, 6, 15, 12, 0)}  # noqa: DTZ001
        s = datason.dumps(data, date_format=config.date_format, include_type_hints=config.include_type_hints)
        parsed = json.loads(s)
        # Without type hints, datetime becomes ISO string
        assert isinstance(parsed["ts"], str)
        assert "2024" in parsed["ts"]

    def test_sorted_keys(self) -> None:
        config = api_config()
        data = {"z": 1, "a": 2, "m": 3}
        s = datason.dumps(data, sort_keys=config.sort_keys)
        parsed = json.loads(s)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_no_type_hints_lossy(self) -> None:
        config = api_config()
        assert config.include_type_hints is False
        data = {"arr": np.array([1.0, 2.0])}
        s = datason.dumps(data, include_type_hints=config.include_type_hints)
        result = datason.loads(s)
        # Without type hints, array becomes plain list — no roundtrip fidelity
        assert isinstance(result["arr"], list)


class TestStrictConfigPreset:
    """Strict preset: no fallbacks, type hints on."""

    def test_unknown_type_raises(self) -> None:
        config = strict_config()

        class CustomObj:
            pass

        with pytest.raises(SerializationError, match="Cannot serialize"):
            datason.dumps(CustomObj(), fallback_to_string=config.fallback_to_string)

    def test_type_hints_on(self) -> None:
        config = strict_config()
        assert config.include_type_hints is True
        data = {"arr": np.array([1.0, 2.0])}
        s = datason.dumps(data, include_type_hints=config.include_type_hints)
        result = datason.loads(s)
        assert isinstance(result["arr"], np.ndarray)


class TestPerformanceConfigPreset:
    """Performance preset: no hints, NaN keep, fallback."""

    def test_no_type_hints(self) -> None:
        config = performance_config()
        assert config.include_type_hints is False

    def test_nan_keep(self) -> None:
        config = performance_config()
        assert config.nan_handling == NanHandling.KEEP

    def test_fallback_enabled(self) -> None:
        config = performance_config()
        assert config.fallback_to_string is True


class TestConfigContextManager:
    """Config context manager scoping behavior."""

    def test_config_scope_applied(self) -> None:
        data = {"z": 1, "a": 2}
        with datason.config(sort_keys=True):
            s = datason.dumps(data)
        parsed = json.loads(s)
        assert list(parsed.keys()) == ["a", "z"]

    def test_config_scope_resets_after_exit(self) -> None:
        with datason.config(sort_keys=True):
            pass
        # After exiting, default config should be active (sort_keys=False)
        data = {"z": 1, "a": 2}
        s = datason.dumps(data)
        parsed = json.loads(s)
        # Original insertion order preserved
        assert list(parsed.keys()) == ["z", "a"]

    def test_nested_config_scopes(self) -> None:
        data = {"z": 1, "a": 2}
        with datason.config(sort_keys=True):
            s1 = datason.dumps(data)
            with datason.config(sort_keys=False):
                s2 = datason.dumps(data)
            s3 = datason.dumps(data)

        # Outer scope: sorted
        assert list(json.loads(s1).keys()) == ["a", "z"]
        # Inner scope: unsorted
        assert list(json.loads(s2).keys()) == ["z", "a"]
        # Back to outer: sorted again
        assert list(json.loads(s3).keys()) == ["a", "z"]

    def test_inline_override_wins_over_context(self) -> None:
        data = {"z": 1, "a": 2}
        with datason.config(sort_keys=True):
            s = datason.dumps(data, sort_keys=False)
        parsed = json.loads(s)
        assert list(parsed.keys()) == ["z", "a"]

    def test_config_context_with_multi_type_data(self) -> None:
        data = {
            "ts": dt.datetime(2024, 1, 1),  # noqa: DTZ001
            "arr": np.array([1.0, 2.0]),
            "name": "test",
        }
        with datason.config(include_type_hints=True, sort_keys=True):
            s = datason.dumps(data)
        result = datason.loads(s)
        assert isinstance(result["ts"], dt.datetime)
        assert isinstance(result["arr"], np.ndarray)
        assert list(json.loads(s).keys()) == sorted(data.keys())
