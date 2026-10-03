"""Regression benchmarks for stock-email literal checks and matching text."""

import pytest

import datason


@pytest.mark.parametrize("text", ["x" * 8000, "contact user@example.com"], ids=["no-email-8k", "matching-email"])
def test_bench_email_redaction(benchmark, text):
    wire = benchmark(datason.dumps, {"text": text}, redact_patterns=("email",))
    expected = "x" * 8000 if "@" not in text else "contact [REDACTED]"
    assert datason.loads(wire) == {"text": expected}
