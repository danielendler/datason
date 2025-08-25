"""Unit tests for redaction engine performance optimizations."""

from unittest.mock import MagicMock, patch

import pytest

from datason.redaction import RedactionEngine


class TestRedactionOptimizations:
    """Test the performance optimization features of RedactionEngine."""

    def test_compiled_field_patterns_initialization(self) -> None:
        """Test that field patterns are pre-compiled during initialization."""
        engine = RedactionEngine(redact_fields=["password", "*.secret", "user.email"])

        # Verify compiled patterns were created
        assert hasattr(engine, "_compiled_field_patterns")
        assert len(engine._compiled_field_patterns) == 3

        # Check pattern structure
        for field_pattern, compiled_pattern in engine._compiled_field_patterns:
            assert isinstance(field_pattern, str)
            # compiled_pattern should be either a Pattern object or None (for invalid regex)
            assert compiled_pattern is None or hasattr(compiled_pattern, "match")

    def test_compiled_field_patterns_invalid_regex(self) -> None:
        """Test handling of invalid regex patterns in field compilation."""
        # Create patterns that would be invalid as regex
        engine = RedactionEngine(redact_fields=["valid_field", "[invalid_regex"])

        # Should still create entries for invalid patterns (with None compiled pattern)
        assert len(engine._compiled_field_patterns) == 2

        # Check that at least one pattern is None (fallback for invalid regex)
        has_none_pattern = any(compiled is None for _, compiled in engine._compiled_field_patterns)
        has_valid_pattern = any(compiled is not None for _, compiled in engine._compiled_field_patterns)

        # We should have both valid and invalid patterns
        assert has_none_pattern  # Invalid regex should result in None
        assert has_valid_pattern  # Valid field should compile

    def test_field_cache_initialization(self) -> None:
        """Test that field cache is initialized properly."""
        engine = RedactionEngine()

        assert hasattr(engine, "_field_cache")
        assert isinstance(engine._field_cache, dict)
        assert len(engine._field_cache) == 0

    def test_should_redact_field_cached_functionality(self) -> None:
        """Test the cached field redaction decision functionality."""
        engine = RedactionEngine(redact_fields=["password", "*.secret"])

        # Test cache miss and population
        field_path = "user.password"
        result1 = engine._should_redact_field_cached(field_path)

        # Should be cached now
        assert field_path in engine._field_cache
        assert engine._field_cache[field_path] == result1

        # Test cache hit
        result2 = engine._should_redact_field_cached(field_path)
        assert result1 == result2

    def test_field_cache_pattern_matching(self) -> None:
        """Test that cached function correctly matches patterns."""
        engine = RedactionEngine(redact_fields=["password", "*.secret", "user.email"])

        # Test exact match
        assert engine._should_redact_field_cached("password") is True

        # Test wildcard match
        assert engine._should_redact_field_cached("config.secret") is True
        assert engine._should_redact_field_cached("nested.deep.secret") is True

        # Test specific path match
        assert engine._should_redact_field_cached("user.email") is True

        # Test non-matching patterns
        assert engine._should_redact_field_cached("username") is False
        assert engine._should_redact_field_cached("user.name") is False
        assert engine._should_redact_field_cached("config.public") is False

    def test_field_cache_size_limitation(self) -> None:
        """Test that field cache respects size limitations."""
        engine = RedactionEngine(redact_fields=["test"])

        # Fill cache to just under the limit
        for i in range(1023):
            field_path = f"field_{i}"
            engine._should_redact_field_cached(field_path)

        assert len(engine._field_cache) == 1023

        # Add one more - should still fit
        engine._should_redact_field_cached("field_1023")
        assert len(engine._field_cache) == 1024

        # Try to add beyond limit - should not grow beyond 1024
        engine._should_redact_field_cached("field_1024")
        assert len(engine._field_cache) == 1024

    def test_field_cache_fallback_string_matching(self) -> None:
        """Test fallback to string matching for invalid regex patterns."""
        engine = RedactionEngine(redact_fields=["[invalid_regex", "valid_field"])

        # The invalid regex should fallback to string matching
        # This should use the fallback logic in _should_redact_field_cached
        result = engine._should_redact_field_cached("test[invalid_regextest")

        # Fallback should do case-insensitive substring matching
        assert result is True  # "[invalid_regex" should be found in "test[invalid_regextest"

        # Test case insensitive matching
        result2 = engine._should_redact_field_cached("TEST[INVALID_REGEXTEST")
        assert result2 is True

    def test_early_exit_optimization_primitives(self) -> None:
        """Test early exit optimization for primitive types."""
        engine = RedactionEngine(redact_fields=["password"], redact_patterns=[r"secret_\w+"])

        # Test primitive types that should exit early
        primitives = [
            42,  # int
            3.14,  # float
            True,  # bool
            False,  # bool
            None,  # NoneType
        ]

        for primitive in primitives:
            result = engine.process_object(primitive)
            # Should return the primitive unchanged (early exit)
            assert result == primitive
            assert result is primitive  # Should be the same object (not processed)

    def test_early_exit_does_not_affect_complex_types(self) -> None:
        """Test that early exit doesn't affect processing of complex types."""
        engine = RedactionEngine(redact_fields=["password"])

        # Complex types should still be processed normally
        test_data = {
            "password": "secret",
            "number": 42,  # This primitive inside dict should still be processed
            "public": "data",
        }

        result = engine.process_object(test_data)

        # Dict should be processed
        assert result["password"] == "<REDACTED>"
        assert result["number"] == 42  # Primitive inside should be preserved
        assert result["public"] == "data"

    def test_integration_optimized_vs_legacy_behavior(self) -> None:
        """Test that optimized functions produce same results as legacy behavior."""
        # Create engine with both field and pattern redaction
        engine = RedactionEngine(
            redact_fields=["password", "*.secret"], redact_patterns=[r"api_key_\w+"], redaction_replacement="<REDACTED>"
        )

        test_data = {
            "password": "secret123",
            "config": {
                "secret": "hidden",
                "api_token": "api_key_abc123",  # Should be redacted by pattern
            },
            "public": "data",
            "numbers": [1, 2, 3],  # Primitives should be preserved
            "nested": {"deep": {"secret": "very_hidden"}},
        }

        result = engine.process_object(test_data)

        # Verify all expected redactions occurred
        assert result["password"] == "<REDACTED>"
        assert result["config"]["secret"] == "<REDACTED>"
        assert result["nested"]["deep"]["secret"] == "<REDACTED>"
        assert "<REDACTED>" in result["config"]["api_token"]  # Pattern redaction

        # Verify non-sensitive data is preserved
        assert result["public"] == "data"
        assert result["numbers"] == [1, 2, 3]

    def test_cache_improves_performance(self) -> None:
        """Test that caching actually improves performance for repeated field checks."""
        engine = RedactionEngine(redact_fields=["password", "*.secret"])

        # Test field that should be cached
        test_field = "user.password"

        # First call should populate cache
        with patch.object(engine, "_compiled_field_patterns") as mock_patterns:
            # Mock the compiled patterns to track calls
            mock_patterns.__iter__ = MagicMock(return_value=iter([]))

            result1 = engine._should_redact_field_cached(test_field)

        # Second call should use cache
        with patch.object(engine, "_compiled_field_patterns") as mock_patterns:
            mock_patterns.__iter__ = MagicMock(return_value=iter([]))

            result2 = engine._should_redact_field_cached(test_field)
            second_call_count = mock_patterns.__iter__.call_count

        # Results should be the same
        assert result1 == result2

        # Second call should not iterate through patterns (cache hit)
        assert second_call_count == 0

        # Cache should contain the result
        assert test_field in engine._field_cache


class TestRegexPrecompilation:
    """Test regex pre-compilation optimization."""

    def test_pattern_compilation_during_init(self) -> None:
        """Test that regex patterns are compiled during initialization."""
        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
        ]

        engine = RedactionEngine(redact_patterns=patterns)

        # Verify patterns were compiled
        assert hasattr(engine, "_compiled_patterns")
        assert len(engine._compiled_patterns) == 2

        # Each should be a compiled pattern object
        for pattern in engine._compiled_patterns:
            assert hasattr(pattern, "match")
            assert hasattr(pattern, "findall")

    def test_invalid_pattern_handling_during_compilation(self) -> None:
        """Test handling of invalid regex patterns during compilation."""
        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Valid
            r"[invalid_pattern",  # Invalid
        ]

        # Should not raise exception, but should warn
        with pytest.warns(UserWarning, match="Invalid regex pattern"):
            engine = RedactionEngine(redact_patterns=patterns)

        # Should have fewer compiled patterns than input patterns
        assert len(engine._compiled_patterns) == 1  # Only valid pattern compiled

    def test_compiled_patterns_used_in_text_redaction(self) -> None:
        """Test that pre-compiled patterns are used for text redaction."""
        engine = RedactionEngine(redact_patterns=[r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"])

        test_text = "My card number is 1234-5678-9012-3456 for payment"
        result, was_redacted = engine.redact_text(test_text)

        assert was_redacted
        assert "1234-5678-9012-3456" not in result
        assert "<REDACTED>" in result

    def test_field_pattern_compilation_wildcard_support(self) -> None:
        """Test that field patterns with wildcards are properly compiled."""
        engine = RedactionEngine(redact_fields=["*.password", "config.*.secret", "user?email"])

        # Test the compiled field patterns
        field_tests = [
            ("user.password", True),  # Should match *.password
            ("admin.password", True),  # Should match *.password
            ("config.db.secret", True),  # Should match config.*.secret
            ("user_email", True),  # Should match user?email
            ("userXemail", True),  # Should match user?email
            ("user.email", True),  # Should match user?email (? becomes . in regex, matches any char)
            ("password", False),  # Should NOT match *.password (no prefix)
            ("config.secret", False),  # Should NOT match config.*.secret (missing middle)
            ("userTOOLONGemail", False),  # Should NOT match user?email (? matches only 1 char)
        ]

        for field_path, expected in field_tests:
            result = engine._should_redact_field_cached(field_path)
            assert result == expected, f"Field '{field_path}' expected {expected}, got {result}"


class TestIntegrationWithExistingFunctionality:
    """Test that optimizations don't break existing functionality."""

    def test_optimized_engine_with_all_features(self) -> None:
        """Test optimized engine with all redaction features enabled."""
        engine = RedactionEngine(
            redact_fields=["*.password", "*.secret"],  # Use wildcards to match nested fields
            redact_patterns=[r"api_key_\w+"],
            redact_large_objects=True,
            large_object_threshold=1000,  # High threshold so only large_text gets redacted
            redaction_replacement="[REDACTED]",
            include_redaction_summary=True,
            audit_trail=True,
        )

        test_data = {
            "user": {
                "password": "secret123",  # Field redaction
                "email": "user@example.com",  # Not redacted
            },
            "config": {
                "secret": "hidden",  # Field redaction (wildcard)
                "token": "api_key_xyz789",  # Pattern redaction
            },
            "large_text": "x" * 2000,  # Large object redaction
            "primitives": [1, 2.5, True, None],  # Should use early exit
            "normal": "public_data",
        }

        result = engine.process_object(test_data)

        # Verify all redaction types work
        assert result["user"]["password"] == "[REDACTED]"
        assert result["user"]["email"] == "user@example.com"
        assert result["config"]["secret"] == "[REDACTED]"
        assert "[REDACTED]" in result["config"]["token"]
        assert "LARGE_OBJECT_REDACTED" in result["large_text"]  # large_text is replaced with string
        assert result["primitives"] == [1, 2.5, True, None]
        assert result["normal"] == "public_data"

        # Verify summary and audit trail still work
        summary = engine.get_redaction_summary()
        assert summary is not None

        audit = engine.get_audit_trail()
        assert audit is not None
        assert len(audit) > 0

    def test_backward_compatibility(self) -> None:
        """Test that optimized engine maintains backward compatibility."""
        # Test with old-style usage
        engine = RedactionEngine(redact_fields=["sensitive_field"], redaction_replacement="***")

        old_style_data = {"sensitive_field": "secret_value", "public_field": "public_value"}

        result = engine.process_object(old_style_data)

        # Should work exactly as before
        assert result["sensitive_field"] == "***"
        assert result["public_field"] == "public_value"

    def test_process_object_method_delegation(self) -> None:
        """Test that _should_redact_field properly delegates to cached version."""
        engine = RedactionEngine(redact_fields=["password"])

        # Verify the delegation works
        field_path = "user.password"

        # Call through the public interface
        result1 = engine._should_redact_field(field_path)

        # Call the cached version directly
        result2 = engine._should_redact_field_cached(field_path)

        # Results should be identical
        assert result1 == result2

        # Cache should be populated from either call
        assert field_path in engine._field_cache
