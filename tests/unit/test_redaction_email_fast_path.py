"""Impossible stock-email scans retain redaction and extension semantics."""

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

import datason
from datason.security.redaction import BUILTIN_PATTERNS, redact_string


def test_long_non_email_string_does_not_enter_regex_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stock email regex ran on a string without its required literal")

    monkeypatch.setattr("datason.security.redaction.re.sub", forbidden)
    text = "x" * 900_000
    assert redact_string(text, ("email",)) == text
    assert datason.loads(datason.dumps({"text": text}, redact_patterns=("email",))) == {"text": text}


@pytest.mark.parametrize("patterns", [("email", "ssn"), ("ssn", "email")])
def test_skipped_email_scan_does_not_skip_other_patterns(patterns):
    assert redact_string("SSN 123-45-6789", patterns) == "SSN [REDACTED]"


def test_modified_builtin_email_pattern_is_respected(monkeypatch):
    monkeypatch.setitem(BUILTIN_PATTERNS, "email", "secret")
    assert redact_string("secret text", ("email",)) == "[REDACTED] text"


def test_custom_pattern_with_optional_at_is_not_skipped():
    assert redact_string("alice", (r"[a-z]+@?",)) == "[REDACTED]"


def test_str_subclass_cannot_hide_an_email_from_redaction():
    class MisleadingContains(str):
        def __contains__(self, item: str) -> bool:
            return False

    assert redact_string(MisleadingContains("user@example.com"), ("email",)) == "[REDACTED]"


@given(
    st.one_of(st.text(max_size=200), st.sampled_from(["user@example.com", "SSN 123-45-6789", "x" * 500])),
    st.lists(st.sampled_from(["email", "ssn", "ipv4", r"secret", r"x{2,}"]), max_size=5),
)
def test_output_matches_sequential_regex_reference(text, patterns):
    expected = text
    for pattern in patterns:
        expected = re.sub(BUILTIN_PATTERNS.get(pattern, pattern), "[REDACTED]", expected)
    assert redact_string(text, tuple(patterns)) == expected
