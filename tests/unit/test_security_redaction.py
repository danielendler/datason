"""Tests for the security redaction module."""

from __future__ import annotations

import datason
from datason.security.redaction import (
    redact_string,
    redact_value,
    should_redact_field,
)


class TestShouldRedactField:
    def test_exact_match(self) -> None:
        assert should_redact_field("password", ("password",))

    def test_case_insensitive(self) -> None:
        assert should_redact_field("PASSWORD", ("password",))
        assert should_redact_field("Password", ("password",))

    def test_substring_match(self) -> None:
        assert should_redact_field("user_password", ("password",))
        assert should_redact_field("password_hash", ("password",))

    def test_multiple_fields(self) -> None:
        assert should_redact_field("api_key", ("password", "key", "secret"))
        assert should_redact_field("secret_token", ("password", "key", "secret"))

    def test_no_match(self) -> None:
        assert not should_redact_field("username", ("password", "key"))

    def test_empty_redact_fields(self) -> None:
        assert not should_redact_field("password", ())


class TestRedactString:
    def test_email_builtin(self) -> None:
        result = redact_string("contact me at user@example.com please", ("email",))
        assert "[REDACTED]" in result
        assert "user@example.com" not in result

    def test_ssn_builtin(self) -> None:
        result = redact_string("SSN: 123-45-6789", ("ssn",))
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_credit_card_builtin(self) -> None:
        result = redact_string("Card: 4111-1111-1111-1111", ("credit_card",))
        assert "[REDACTED]" in result
        assert "4111" not in result

    def test_custom_regex(self) -> None:
        result = redact_string("token: abc-123-def", (r"[a-z]+-\d+-[a-z]+",))
        assert "[REDACTED]" in result
        assert "abc-123-def" not in result

    def test_multiple_patterns(self) -> None:
        text = "Email: user@test.com, SSN: 123-45-6789"
        result = redact_string(text, ("email", "ssn"))
        assert "user@test.com" not in result
        assert "123-45-6789" not in result

    def test_no_match(self) -> None:
        result = redact_string("hello world", ("email",))
        assert result == "hello world"

    def test_empty_patterns(self) -> None:
        result = redact_string("user@example.com", ())
        assert result == "user@example.com"


class TestRedactValue:
    def test_string_with_patterns(self) -> None:
        result = redact_value("call 555-123-4567", ("phone_us",))
        assert "555-123-4567" not in result

    def test_non_string_passthrough(self) -> None:
        assert redact_value(42, ("email",)) == 42
        assert redact_value(None, ("email",)) is None
        assert redact_value([1, 2], ("email",)) == [1, 2]


class TestIntegration:
    """Test redaction through the full datason.dumps() pipeline."""

    def test_field_redaction(self) -> None:
        data = {"username": "alice", "password": "s3cret!", "email": "a@b.com"}
        result = datason.dumps(data, redact_fields=("password",))
        assert "s3cret!" not in result
        assert "[REDACTED]" in result
        assert "alice" in result

    def test_pattern_redaction(self) -> None:
        data = {"message": "Contact user@test.com for help"}
        result = datason.dumps(data, redact_patterns=("email",))
        assert "user@test.com" not in result
        assert "[REDACTED]" in result

    def test_nested_field_redaction(self) -> None:
        data = {"user": {"name": "Alice", "api_key": "sk-123abc"}}
        result = datason.dumps(data, redact_fields=("key",))
        assert "sk-123abc" not in result
        assert "Alice" in result

    def test_combined_field_and_pattern(self) -> None:
        data = {
            "password": "hunter2",
            "bio": "Email me at ceo@company.com",
        }
        result = datason.dumps(
            data,
            redact_fields=("password",),
            redact_patterns=("email",),
        )
        assert "hunter2" not in result
        assert "ceo@company.com" not in result

    def test_no_redaction_by_default(self) -> None:
        data = {"password": "s3cret!", "email": "user@test.com"}
        result = datason.dumps(data)
        assert "s3cret!" in result
        assert "user@test.com" in result

    def test_redaction_with_context_manager(self) -> None:
        data = {"api_secret": "token123", "name": "test"}
        with datason.config(redact_fields=("secret",)):
            result = datason.dumps(data)
        assert "token123" not in result
        assert "test" in result

    def test_list_of_dicts_redaction(self) -> None:
        data = [
            {"name": "Alice", "ssn": "123-45-6789"},
            {"name": "Bob", "ssn": "987-65-4321"},
        ]
        result = datason.dumps(data, redact_fields=("ssn",))
        assert "123-45-6789" not in result
        assert "987-65-4321" not in result
        assert "Alice" in result
