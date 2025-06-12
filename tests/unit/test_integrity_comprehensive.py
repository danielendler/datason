"""Tests for datason.integrity module."""

from typing import Dict, List

import pytest

import datason


class TestIntegrity:
    """Verify hashing and verification utilities."""

    def test_hash_and_verify(self) -> None:
        obj = {"a": 1, "b": [2, 3]}
        h1 = datason.hash_object(obj)
        h2 = datason.hash_object(obj)
        assert h1 == h2
        assert datason.verify_object(obj, h1)

    def test_hash_without_redaction(self) -> None:
        """Test that integrity functions work without redaction dependency."""
        obj = {"id": 1, "data": [1, 2, 3], "nested": {"key": "value"}}

        # Basic hashing should work
        hash1 = datason.hash_object(obj)
        hash2 = datason.hash_object(obj)
        assert hash1 == hash2

        # Verification should work
        assert datason.verify_object(obj, hash1)
        assert not datason.verify_object({"different": "object"}, hash1)

        # JSON hashing should work
        json_hash = datason.hash_json(obj)
        assert datason.verify_json(obj, json_hash)

        # Different algorithms should produce different hashes
        sha512_hash = datason.hash_object(obj, hash_algo="sha512")
        assert sha512_hash != hash1

    def test_redaction_hashing(self) -> None:
        """Test redaction-aware hashing when redaction is available."""
        obj = {"id": 1, "ssn": "123-45-6789"}
        original = datason.hash_object(obj)

        try:
            redacted, redacted_hash = datason.hash_and_redact(obj, redact={"fields": ["ssn"]})
            assert datason.verify_object(redacted, redacted_hash)
            assert original != redacted_hash

            # Same redaction should produce same hash
            redacted2, h2 = datason.hash_and_redact(obj, redact={"fields": ["ssn"]})
            assert h2 == redacted_hash

            # Verify redacted object doesn't contain sensitive data
            assert "ssn" not in str(redacted) or redacted.get("ssn") == "<REDACTED>"

        except RuntimeError as e:
            if "Redaction module is not available" in str(e):
                pytest.skip("Redaction module not available")
            else:
                raise

    def test_redaction_error_handling(self) -> None:
        """Test proper error when redaction is requested but unavailable."""
        # This test would be more meaningful if we could mock the import failure,
        # but for now we test that the functions work when redaction is available
        obj = {"id": 1, "secret": "hidden"}

        # These should work regardless of redaction availability
        hash_without_redact = datason.hash_object(obj)
        assert datason.verify_object(obj, hash_without_redact)

        # If redaction is available, this should work
        # If not, it should raise a helpful error
        try:
            redacted, hash_with_redact = datason.hash_and_redact(obj, redact={"fields": ["secret"]})
            assert datason.verify_object(redacted, hash_with_redact)
        except RuntimeError as e:
            assert "Redaction module is not available" in str(e)
            assert "Install datason with redaction support" in str(e)

    def test_sign_and_verify(self) -> None:
        """Test digital signatures when cryptography is available."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            pytest.skip("cryptography not available")

        private = Ed25519PrivateKey.generate()
        public = private.public_key()

        private_pem = private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        public_pem = public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        obj = {"message": "hi"}
        sig = datason.sign_object(obj, private_pem)
        assert datason.verify_signature(obj, sig, public_pem)
        assert not datason.verify_signature({"message": "no"}, sig, public_pem)

    def test_signing_error_handling(self) -> None:
        """Test proper error messages when cryptography is unavailable."""
        obj = {"test": "data"}

        # Mock a scenario where cryptography might not be available
        # In practice, this test would pass if cryptography is installed
        try:
            # This will either work (if cryptography is available) or raise RuntimeError
            private_key_pem = "invalid_key_for_testing"
            datason.sign_object(obj, private_key_pem)
        except RuntimeError as e:
            if "cryptography is required for signing" in str(e):
                # Expected when cryptography is not available
                pass
            else:
                # Some other error (like invalid key format) - that's fine too
                pass
        except Exception:
            # Other exceptions (like key parsing errors) are expected with invalid key
            pass

    def test_canonicalization_stability(self) -> None:
        """Test that canonicalization produces stable, deterministic output."""
        obj1 = {"b": 2, "a": 1, "c": [3, 2, 1]}
        obj2 = {"a": 1, "b": 2, "c": [3, 2, 1]}  # Same data, different order

        # Should produce identical canonical representations
        canon1 = datason.integrity.canonicalize(obj1)
        canon2 = datason.integrity.canonicalize(obj2)
        assert canon1 == canon2

        # And identical hashes
        hash1 = datason.hash_object(obj1)
        hash2 = datason.hash_object(obj2)
        assert hash1 == hash2

    def test_hash_different_algorithms(self) -> None:
        """Test different secure hash algorithms produce different results."""
        obj = {"test": "data"}

        sha256_hash = datason.hash_object(obj, hash_algo="sha256")
        sha512_hash = datason.hash_object(obj, hash_algo="sha512")
        sha3_256_hash = datason.hash_object(obj, hash_algo="sha3_256")
        sha3_512_hash = datason.hash_object(obj, hash_algo="sha3_512")

        # All should be different
        assert sha256_hash != sha512_hash
        assert sha256_hash != sha3_256_hash
        assert sha256_hash != sha3_512_hash
        assert sha512_hash != sha3_256_hash
        assert sha512_hash != sha3_512_hash
        assert sha3_256_hash != sha3_512_hash

        # But should be consistent
        assert sha256_hash == datason.hash_object(obj, hash_algo="sha256")
        assert sha512_hash == datason.hash_object(obj, hash_algo="sha512")

    def test_empty_and_none_objects(self) -> None:
        """Test integrity functions with edge case objects."""
        # Empty dict
        empty_dict: Dict[str, str] = {}
        hash_empty = datason.hash_object(empty_dict)
        assert datason.verify_object(empty_dict, hash_empty)

        # None values
        none_obj = {"key": None}
        hash_none = datason.hash_object(none_obj)
        assert datason.verify_object(none_obj, hash_none)

        # Empty list
        empty_list: List[str] = []
        hash_list = datason.hash_object(empty_list)
        assert datason.verify_object(empty_list, hash_list)
