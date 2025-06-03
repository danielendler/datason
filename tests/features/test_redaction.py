"""Tests for redaction functionality (v0.5.5)."""

from typing import Any, Dict

from datason.redaction import (
    RedactionEngine,
    create_financial_redaction_engine,
    create_healthcare_redaction_engine,
    create_minimal_redaction_engine,
)


class TestRedactionEngine:
    """Test the core redaction engine functionality."""

    def test_field_redaction(self) -> None:
        """Test redaction based on field patterns."""
        engine = RedactionEngine(
            redact_fields=["password", "*.secret", "user.email"],
            redaction_replacement="<REDACTED>",
        )

        test_data = {
            "password": "secret123",
            "username": "alice",
            "config": {"secret": "api_key_123"},
            "user": {"email": "alice@example.com", "name": "Alice"},
        }

        result = engine.process_object(test_data)

        assert result["password"] == "<REDACTED>"
        assert result["username"] == "alice"  # Should not be redacted
        assert result["config"]["secret"] == "<REDACTED>"
        assert result["user"]["email"] == "<REDACTED>"
        assert result["user"]["name"] == "Alice"  # Should not be redacted

    def test_pattern_redaction(self) -> None:
        """Test redaction based on regex patterns."""
        engine = RedactionEngine(
            redact_patterns=[
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
            ],
            redaction_replacement="[REDACTED]",
        )

        test_text = "Contact alice@example.com or use card 1234-5678-9012-3456"
        result, was_redacted = engine.redact_text(test_text)

        assert was_redacted
        assert "alice@example.com" not in result
        assert "1234-5678-9012-3456" not in result
        assert "[REDACTED]" in result

    def test_large_object_redaction(self) -> None:
        """Test redaction of large objects."""
        engine = RedactionEngine(
            redact_large_objects=True,
            large_object_threshold=100,  # Very small threshold for testing
        )

        # Create a large object (string larger than threshold)
        large_string = "x" * 200
        result = engine.process_object(large_string)

        # Should be redacted due to size
        assert "LARGE_OBJECT_REDACTED" in str(result)

    def test_circular_reference_handling(self) -> None:
        """Test handling of circular references."""
        engine = RedactionEngine()

        # Create circular reference
        data: Dict[str, Any] = {"a": 1}
        data["self"] = data

        result = engine.process_object(data)

        # Should handle circular reference gracefully
        assert result["a"] == 1
        assert result["self"] == "<CIRCULAR_REFERENCE>"

    def test_audit_trail(self) -> None:
        """Test audit trail functionality."""
        engine = RedactionEngine(
            redact_fields=["password"],
            audit_trail=True,
        )

        test_data = {"password": "secret123", "username": "alice"}
        engine.process_object(test_data)

        audit_trail = engine.get_audit_trail()
        assert audit_trail is not None
        assert len(audit_trail) > 0
        assert audit_trail[0]["redaction_type"] == "field"
        assert audit_trail[0]["target"] == "password"

    def test_redaction_summary(self) -> None:
        """Test redaction summary functionality."""
        engine = RedactionEngine(
            redact_fields=["password"],
            include_redaction_summary=True,
        )

        test_data = {"password": "secret123", "username": "alice"}
        engine.process_object(test_data)

        summary = engine.get_redaction_summary()
        assert summary is not None
        assert "redaction_summary" in summary
        assert summary["redaction_summary"]["total_redactions"] > 0
        assert "password" in summary["redaction_summary"]["fields_redacted"]


class TestPresetRedactionEngines:
    """Test the preset redaction engines."""

    def test_financial_redaction_engine(self) -> None:
        """Test the financial redaction engine."""
        engine = create_financial_redaction_engine()

        test_data = {
            "account_number": "1234567890",
            "ssn": "123-45-6789",
            "credit_card": "4532-1234-5678-9012",
            "name": "John Doe",  # Should not be redacted
        }

        result = engine.process_object(test_data)

        assert result["account_number"] == "<REDACTED>"
        assert result["ssn"] == "<REDACTED>"
        assert result["credit_card"] == "<REDACTED>"
        assert result["name"] == "John Doe"  # Should not be redacted

    def test_healthcare_redaction_engine(self) -> None:
        """Test the healthcare redaction engine."""
        engine = create_healthcare_redaction_engine()

        test_data = {
            "record": {"patient_id": "P123456"},  # Nested to match *.patient_id
            "contact": {"email": "patient@example.com"},  # Should be redacted by pattern
            "chart": {"diagnosis": "Common cold"},  # Nested to match *.diagnosis
            "treatment": "Rest and fluids",  # Should not be redacted
        }

        result = engine.process_object(test_data)

        assert result["record"]["patient_id"] == "<REDACTED>"
        # Email should be redacted by pattern, not field name
        assert "patient@example.com" not in str(result)
        assert result["chart"]["diagnosis"] == "<REDACTED>"
        assert result["treatment"] == "Rest and fluids"  # Should not be redacted

    def test_minimal_redaction_engine(self) -> None:
        """Test the minimal redaction engine."""
        engine = create_minimal_redaction_engine()

        test_data = {
            "user": {"password": "secret123"},  # Nested to match *.password pattern
            "api": {"key": "key_abc123"},  # Nested to match *.key pattern
            "contact": {"email": "user@example.com"},  # Should be redacted by pattern
            "username": "alice",  # Should not be redacted
        }

        result = engine.process_object(test_data)

        assert result["user"]["password"] == "<REDACTED>"
        assert result["api"]["key"] == "<REDACTED>"
        # Email should be redacted by pattern
        assert "user@example.com" not in str(result)
        assert result["username"] == "alice"  # Should not be redacted


class TestRedactionEngineEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_regex_pattern(self) -> None:
        """Test handling of invalid regex patterns."""
        # Should warn but not crash
        engine = RedactionEngine(redact_patterns=["[invalid"])

        test_text = "This is a test"
        result, was_redacted = engine.redact_text(test_text)

        assert result == test_text
        assert not was_redacted

    def test_empty_configuration(self) -> None:
        """Test engine with no redaction configured."""
        engine = RedactionEngine()

        test_data = {"password": "secret123", "email": "user@example.com"}
        result = engine.process_object(test_data)

        # Should return unchanged
        assert result == test_data

    def test_nested_structure_redaction(self) -> None:
        """Test redaction in deeply nested structures."""
        engine = RedactionEngine(redact_fields=["*.password"])

        test_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "password": "deep_secret",
                        "data": "safe_data",
                    }
                }
            }
        }

        result = engine.process_object(test_data)

        assert result["level1"]["level2"]["level3"]["password"] == "<REDACTED>"
        assert result["level1"]["level2"]["level3"]["data"] == "safe_data"

    def test_list_redaction(self) -> None:
        """Test redaction in lists."""
        engine = RedactionEngine(redact_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"])

        test_data = [
            "Contact alice@example.com",
            "Or bob@test.org",
            "Normal text here",
        ]

        result = engine.process_object(test_data)

        assert "<REDACTED>" in result[0]
        assert "alice@example.com" not in result[0]
        assert "<REDACTED>" in result[1]
        assert "bob@test.org" not in result[1]
        assert result[2] == "Normal text here"
