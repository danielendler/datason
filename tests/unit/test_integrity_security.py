"""Test security features of the integrity module."""

import pytest

import datason
from datason.integrity import ALLOWED_HASH_ALGORITHMS, validate_hash_algorithm


class TestHashAlgorithmSecurity:
    """Test that insecure hash algorithms are properly rejected."""

    def test_allowed_algorithms_are_secure(self):
        """Test that only secure hash algorithms are allowed."""
        # Verify that only strong algorithms are in the allowed list
        assert "sha256" in ALLOWED_HASH_ALGORITHMS
        assert "sha3_256" in ALLOWED_HASH_ALGORITHMS
        assert "sha3_512" in ALLOWED_HASH_ALGORITHMS
        assert "sha512" in ALLOWED_HASH_ALGORITHMS

        # Verify that insecure algorithms are NOT in the allowed list
        assert "md5" not in ALLOWED_HASH_ALGORITHMS
        assert "sha1" not in ALLOWED_HASH_ALGORITHMS

    def test_validate_hash_algorithm_secure(self):
        """Test that validation passes for secure algorithms."""
        # These should not raise any exception
        validate_hash_algorithm("sha256")
        validate_hash_algorithm("sha512")
        validate_hash_algorithm("sha3_256")
        validate_hash_algorithm("sha3_512")

    def test_validate_hash_algorithm_insecure(self):
        """Test that validation rejects insecure algorithms."""
        # These should raise ValueError
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            validate_hash_algorithm("md5")

        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: sha1"):
            validate_hash_algorithm("sha1")

        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: unknown"):
            validate_hash_algorithm("unknown")

    def test_hash_object_rejects_insecure_algorithms(self):
        """Test that hash_object rejects insecure algorithms."""
        test_data = {"test": "data"}

        # Secure algorithms should work
        hash_result = datason.hash_object(test_data, hash_algo="sha256")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length

        # Insecure algorithms should be rejected
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            datason.hash_object(test_data, hash_algo="md5")

        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: sha1"):
            datason.hash_object(test_data, hash_algo="sha1")

    def test_hash_json_rejects_insecure_algorithms(self):
        """Test that hash_json rejects insecure algorithms."""
        test_data = {"test": "data"}

        # Secure algorithms should work
        hash_result = datason.hash_json(test_data, hash_algo="sha256")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length

        # Insecure algorithms should be rejected
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            datason.hash_json(test_data, hash_algo="md5")

        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: sha1"):
            datason.hash_json(test_data, hash_algo="sha1")

    def test_verify_object_rejects_insecure_algorithms(self):
        """Test that verify_object rejects insecure algorithms."""
        test_data = {"test": "data"}

        # This should work with secure algorithm
        test_hash = datason.hash_object(test_data, hash_algo="sha256")
        assert datason.verify_object(test_data, test_hash, hash_algo="sha256")

        # Insecure algorithms should be rejected
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            datason.verify_object(test_data, "fake_hash", hash_algo="md5")

    def test_verify_json_rejects_insecure_algorithms(self):
        """Test that verify_json rejects insecure algorithms."""
        test_data = {"test": "data"}

        # This should work with secure algorithm
        test_hash = datason.hash_json(test_data, hash_algo="sha256")
        assert datason.verify_json(test_data, test_hash, hash_algo="sha256")

        # Insecure algorithms should be rejected
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            datason.verify_json(test_data, "fake_hash", hash_algo="md5")

    def test_hash_and_redact_rejects_insecure_algorithms(self):
        """Test that hash_and_redact rejects insecure algorithms."""
        test_data = {"test": "data", "secret": "value"}

        # This should work with secure algorithm
        redacted, hash_result = datason.hash_and_redact(test_data, redact={"fields": ["secret"]}, hash_algo="sha256")
        assert isinstance(hash_result, str)
        # Check that the secret value was redacted, not just the key
        assert "value" not in str(redacted)
        assert redacted["secret"] != "value"  # Secret value should be replaced

        # Insecure algorithms should be rejected
        with pytest.raises(ValueError, match="Unsupported or insecure hash algorithm: md5"):
            datason.hash_and_redact(test_data, hash_algo="md5")

    def test_different_secure_algorithms_produce_different_hashes(self):
        """Test that different secure algorithms produce different hash values."""
        test_data = {"test": "data"}

        hash_sha256 = datason.hash_object(test_data, hash_algo="sha256")
        hash_sha512 = datason.hash_object(test_data, hash_algo="sha512")
        hash_sha3_256 = datason.hash_object(test_data, hash_algo="sha3_256")
        hash_sha3_512 = datason.hash_object(test_data, hash_algo="sha3_512")

        # All should be strings but different values
        assert isinstance(hash_sha256, str)
        assert isinstance(hash_sha512, str)
        assert isinstance(hash_sha3_256, str)
        assert isinstance(hash_sha3_512, str)

        # They should all be different
        hashes = {hash_sha256, hash_sha512, hash_sha3_256, hash_sha3_512}
        assert len(hashes) == 4, "All hash algorithms should produce different results"

        # Different lengths for different algorithms
        assert len(hash_sha256) == 64  # SHA256
        assert len(hash_sha512) == 128  # SHA512
        assert len(hash_sha3_256) == 64  # SHA3-256
        assert len(hash_sha3_512) == 128  # SHA3-512


class TestCanonicalizeExposure:
    """Test that canonicalize function is properly exposed."""

    def test_canonicalize_accessible_via_datason(self):
        """Test that canonicalize is accessible through main datason module."""
        test_data = {"b": 2, "a": 1}

        # Should be accessible as datason.canonicalize
        canonical = datason.canonicalize(test_data)
        assert isinstance(canonical, str)

        # Should produce sorted JSON
        assert canonical == '{"a":1,"b":2}'

    def test_canonicalize_accessible_via_integrity_module(self):
        """Test that canonicalize is accessible through integrity module."""
        from datason.integrity import canonicalize

        test_data = {"b": 2, "a": 1}
        canonical = canonicalize(test_data)
        assert isinstance(canonical, str)
        assert canonical == '{"a":1,"b":2}'

    def test_canonicalize_in_all_exports(self):
        """Test that canonicalize is in the __all__ list."""
        assert "canonicalize" in datason.__all__

    def test_canonicalize_with_redaction(self):
        """Test canonicalize with redaction (if available)."""
        test_data = {"public": "data", "private": "secret"}

        try:
            canonical = datason.canonicalize(test_data, redact={"fields": ["private"], "replacement": "[REDACTED]"})
            # Should contain redacted version
            assert "[REDACTED]" in canonical
            assert "secret" not in canonical
            assert "public" in canonical
            assert "data" in canonical
        except RuntimeError as e:
            if "redaction module unavailable" in str(e):
                pytest.skip("Redaction module not available")
            else:
                raise

    def test_canonicalize_deterministic(self):
        """Test that canonicalize produces deterministic output."""
        test_data = {"z": 3, "a": 1, "m": 2}

        # Multiple calls should produce identical results
        canonical1 = datason.canonicalize(test_data)
        canonical2 = datason.canonicalize(test_data)
        canonical3 = datason.canonicalize(test_data)

        assert canonical1 == canonical2 == canonical3

        # Should be sorted JSON
        assert canonical1 == '{"a":1,"m":2,"z":3}'


class TestSecurityDocumentation:
    """Test that security features are properly documented."""

    def test_hash_algorithm_error_message_helpful(self):
        """Test that error messages provide helpful guidance."""
        with pytest.raises(ValueError) as exc_info:
            validate_hash_algorithm("md5")

        error_message = str(exc_info.value)
        assert "md5" in error_message
        assert "sha256" in error_message
        assert "Must be one of:" in error_message

    def test_all_allowed_algorithms_in_error_message(self):
        """Test that error message lists all allowed algorithms."""
        with pytest.raises(ValueError) as exc_info:
            validate_hash_algorithm("invalid")

        error_message = str(exc_info.value)
        for algo in ALLOWED_HASH_ALGORITHMS:
            assert algo in error_message

    def test_security_upgrade_guidance(self):
        """Test that the module provides clear security guidance."""
        # Verify that the allowed algorithms are well-documented
        assert len(ALLOWED_HASH_ALGORITHMS) >= 4

        # All allowed algorithms should be cryptographically strong
        strong_algorithms = {"sha256", "sha512", "sha3_256", "sha3_512"}
        assert ALLOWED_HASH_ALGORITHMS.issubset(strong_algorithms) or strong_algorithms.issubset(
            ALLOWED_HASH_ALGORITHMS
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
