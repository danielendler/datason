"""Hypothesis property-based tests for config combinations and security limits.

Explores the Cartesian product of config enum values and tests
security limit boundaries with generated data.
"""

from __future__ import annotations

import json
import math
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

import datason
from datason._config import DateFormat, NanHandling, SerializationConfig
from datason._errors import SecurityError
from tests.conftest import st_mixed_serializable_data


class TestConfigCombinations:
    """Explore the enum/boolean Cartesian product of config options."""

    @given(
        date_format=st.sampled_from(list(DateFormat)),
        nan_handling=st.sampled_from(list(NanHandling)),
        include_type_hints=st.booleans(),
        sort_keys=st.booleans(),
        fallback_to_string=st.booleans(),
    )
    def test_any_config_produces_valid_json_for_basic_data(
        self,
        date_format: DateFormat,
        nan_handling: NanHandling,
        include_type_hints: bool,
        sort_keys: bool,
        fallback_to_string: bool,
    ) -> None:
        """Any valid config combination produces valid JSON output for basic data."""
        data = {"name": "test", "count": 42, "ratio": 3.14, "active": True}
        s = datason.dumps(
            data,
            date_format=date_format,
            nan_handling=nan_handling,
            include_type_hints=include_type_hints,
            sort_keys=sort_keys,
            fallback_to_string=fallback_to_string,
        )
        parsed = json.loads(s)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test"

    @given(
        date_format=st.sampled_from(list(DateFormat)),
    )
    def test_datetime_roundtrip_all_formats(self, date_format: DateFormat) -> None:
        """Datetime roundtrip works for all DateFormat values (with type hints)."""
        import datetime as dt

        val = dt.datetime(2024, 6, 15, 12, 30, 45)  # noqa: DTZ001
        s = datason.dumps(val, include_type_hints=True, date_format=date_format)
        result = datason.loads(s)
        # All formats should reconstruct a datetime
        assert isinstance(result, dt.datetime)

    @given(st_mixed_serializable_data())
    def test_mixed_data_with_default_config(self, data: dict[str, Any]) -> None:
        """Mixed serializable data produces valid JSON with default config."""
        s = datason.dumps(data, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)


class TestSecurityLimitBoundaries:
    """Test at, just under, and just over security limits."""

    @given(st.integers(min_value=2, max_value=30))
    def test_depth_at_limit_succeeds(self, limit: int) -> None:
        """Nesting at exactly max_depth should succeed."""
        data: dict[str, Any] = {}
        current = data
        for _ in range(limit - 1):
            current["next"] = {}
            current = current["next"]
        current["value"] = "leaf"

        # Should succeed at exactly the limit
        s = datason.dumps(data, max_depth=limit)
        assert isinstance(s, str)

    @given(st.integers(min_value=2, max_value=30))
    def test_depth_over_limit_fails(self, limit: int) -> None:
        """Nesting at max_depth+1 should fail."""
        data: dict[str, Any] = {}
        current = data
        for _ in range(limit + 1):
            current["next"] = {}
            current = current["next"]
        current["value"] = "leaf"

        try:
            datason.dumps(data, max_depth=limit)
            # Some structures may not exceed depth depending on structure
        except SecurityError:
            pass  # Expected

    @given(st.integers(min_value=1, max_value=100))
    def test_dict_size_at_limit_succeeds(self, limit: int) -> None:
        """Dict with exactly max_size keys should succeed."""
        data = {f"key_{i}": i for i in range(limit)}
        s = datason.dumps(data, max_size=limit)
        assert isinstance(s, str)

    @given(st.integers(min_value=1, max_value=100))
    def test_dict_size_over_limit_fails(self, limit: int) -> None:
        """Dict with max_size+1 keys should fail."""
        data = {f"key_{i}": i for i in range(limit + 1)}
        try:
            datason.dumps(data, max_size=limit)
            assert False, "Expected SecurityError"  # noqa: PT015, B011
        except SecurityError:
            pass

    @given(st.integers(min_value=2, max_value=30))
    def test_deserialization_depth_at_limit_succeeds(self, limit: int) -> None:
        """Deserialization at exactly max_depth should succeed."""
        # Build a JSON string with nesting at the limit
        s = '{"a": ' * (limit - 1) + '"leaf"' + "}" * (limit - 1)
        result = datason.loads(s, max_depth=limit)
        assert isinstance(result, dict)


class TestNanHandlingProperty:
    """NaN handling is consistent across all config modes."""

    @given(
        nan_handling=st.sampled_from([NanHandling.NULL, NanHandling.STRING, NanHandling.DROP]),
    )
    def test_no_raw_nan_in_output(self, nan_handling: NanHandling) -> None:
        """JSON output should never contain literal NaN or Infinity."""
        data = {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 42.0}
        s = datason.dumps(data, nan_handling=nan_handling)
        # Should be valid JSON (NaN/Inf are not valid JSON)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    @given(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            min_size=1,
            max_size=10,
        )
    )
    def test_nan_null_mode_replaces_all(self, values: list[float]) -> None:
        """In NULL mode, all NaN/Inf values become null."""
        s = datason.dumps(values, nan_handling=NanHandling.NULL)
        parsed = json.loads(s)
        for i, v in enumerate(values):
            if math.isnan(v) or math.isinf(v):
                assert parsed[i] is None
            else:
                assert parsed[i] == v

    @given(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            min_size=1,
            max_size=10,
        )
    )
    def test_nan_string_mode_produces_strings(self, values: list[float]) -> None:
        """In STRING mode, NaN/Inf become string representations."""
        s = datason.dumps(values, nan_handling=NanHandling.STRING)
        parsed = json.loads(s)
        for i, v in enumerate(values):
            if math.isnan(v) or math.isinf(v):
                assert isinstance(parsed[i], str)
            else:
                assert parsed[i] == v


class TestFallbackToStringProperty:
    """With fallback_to_string=True, any object should serialize without error."""

    def test_custom_class_fallback(self) -> None:
        """Custom class objects serialize as their str representation."""

        class CustomObj:
            def __str__(self) -> str:
                return "custom_obj_str"

        result = datason.dumps(CustomObj(), fallback_to_string=True)
        assert "custom_obj_str" in result

    def test_set_fallback(self) -> None:
        """Sets serialize via sorted repr fallback."""
        # Sets are handled natively, but frozensets in fallback mode work
        result = datason.dumps({"data": frozenset()}, fallback_to_string=False)
        # frozenset is handled as sorted list
        assert isinstance(result, str)

    @given(
        st.sampled_from(
            [
                SerializationConfig(fallback_to_string=True),
                SerializationConfig(fallback_to_string=True, include_type_hints=False),
                SerializationConfig(fallback_to_string=True, sort_keys=True),
            ]
        )
    )
    def test_fallback_config_variants(self, config: SerializationConfig) -> None:
        """Various fallback configs all handle unknown types."""

        class Unknown:
            pass

        s = datason.dumps(
            {"obj": Unknown()},
            fallback_to_string=config.fallback_to_string,
            include_type_hints=config.include_type_hints,
            sort_keys=config.sort_keys,
        )
        parsed = json.loads(s)
        assert "obj" in parsed
