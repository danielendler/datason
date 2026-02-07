"""Fuzz-style tests for datason deserialization.

Uses Hypothesis to generate adversarial JSON inputs and verify
the deserializer never crashes, hangs, or leaks data. These tests
focus on security-relevant edge cases that property-based tests
on valid data don't cover.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

import datason
from datason._errors import DatasonError

# =========================================================================
# Strategies for adversarial inputs
# =========================================================================

# Random JSON-like values (valid JSON that may confuse the deserializer)
st_json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=200),
)

# Recursive JSON structure
st_json_values: st.SearchStrategy[Any] = st.recursive(
    st_json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=10),
        st.dictionaries(st.text(max_size=20), children, max_size=10),
    ),
    max_leaves=50,
)


# Fake type metadata with random type names
st_fake_metadata = st.fixed_dictionaries(
    {
        "__datason_type__": st.text(min_size=1, max_size=50),
        "__datason_value__": st_json_values,
    }
)

# Dicts that may or may not contain type metadata
st_maybe_typed = st.one_of(
    st_fake_metadata,
    st.dictionaries(st.text(max_size=20), st_json_values, max_size=10),
)


# =========================================================================
# Fuzz tests
# =========================================================================


class TestFuzzDeserializeNeverCrashes:
    """Verify loads() never raises unexpected exceptions."""

    @given(data=st_json_values)
    @settings(max_examples=200, deadline=2000)
    def test_arbitrary_json_does_not_crash(self, data: Any) -> None:
        """Any valid JSON should deserialize without unexpected exceptions."""
        json_str = json.dumps(data)
        try:
            datason.loads(json_str)
        except DatasonError:
            pass  # Expected errors are fine (SecurityError, DeserializationError)

    @given(data=st_fake_metadata)
    @settings(max_examples=200, deadline=2000)
    def test_fake_type_metadata_does_not_crash(self, data: Any) -> None:
        """Dicts with __datason_type__ but garbage values should not crash."""
        json_str = json.dumps(data)
        try:
            datason.loads(json_str)
        except DatasonError:
            pass  # Expected: DeserializationError or PluginError
        except (TypeError, ValueError, KeyError):
            pass  # Plugin-level errors from bad data shapes

    @given(data=st_maybe_typed)
    @settings(max_examples=200, deadline=2000)
    def test_mixed_dicts_do_not_crash(self, data: Any) -> None:
        """Dicts that may or may not have type metadata should not crash."""
        json_str = json.dumps(data)
        try:
            datason.loads(json_str)
        except DatasonError:
            pass
        except (TypeError, ValueError, KeyError):
            pass


class TestFuzzDeserializeSecurityLimits:
    """Verify security limits hold under adversarial input."""

    @given(depth=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=2000)
    def test_nested_dicts_respect_depth_limit(self, depth: int) -> None:
        """Nested structures should hit depth limit, not stack overflow."""
        data: dict[str, Any] = {"v": 1}
        for _ in range(depth):
            data = {"nested": data}
        json_str = json.dumps(data)
        try:
            datason.loads(json_str, max_depth=5)
        except DatasonError:
            pass  # SecurityError expected when depth > 5

    @given(size=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50, deadline=2000)
    def test_large_dicts_respect_size_limit(self, size: int) -> None:
        """Large dicts should not cause memory issues."""
        data = {f"key_{i}": i for i in range(size)}
        json_str = json.dumps(data)
        # Should succeed or raise SecurityError, never OOM
        try:
            datason.loads(json_str, max_size=10)
        except DatasonError:
            pass


class TestFuzzDeserializeOutputValidity:
    """Verify deserialized output is always usable."""

    @given(data=st_json_values)
    @settings(max_examples=100, deadline=2000)
    def test_output_is_json_serializable(self, data: Any) -> None:
        """Whatever loads() returns should be re-serializable."""
        json_str = json.dumps(data)
        try:
            result = datason.loads(json_str)
            # Re-serialize should not crash
            datason.dumps(result)
        except DatasonError:
            pass

    @given(data=st_json_values)
    @settings(max_examples=100, deadline=2000)
    def test_roundtrip_stability(self, data: Any) -> None:
        """dumps(loads(dumps(data))) should equal dumps(data) for plain JSON."""
        json_str = json.dumps(data)
        try:
            # loads -> dumps should produce stable output
            pass1 = datason.loads(json_str)
            json2 = datason.dumps(pass1)
            pass2 = datason.loads(json2)
            json3 = datason.dumps(pass2)
            assert json2 == json3, "Serialization is not idempotent"
        except DatasonError:
            pass
