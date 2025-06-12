"""Tests for datason.integrity module."""

import datason
import pytest


class TestIntegrity:
    """Verify hashing and verification utilities."""

    def test_hash_and_verify(self) -> None:
        obj = {"a": 1, "b": [2, 3]}
        h1 = datason.hash_object(obj)
        h2 = datason.hash_object(obj)
        assert h1 == h2
        assert datason.verify_object(obj, h1)

    def test_redaction_hashing(self) -> None:
        obj = {"id": 1, "ssn": "123-45-6789"}
        original = datason.hash_object(obj)
        redacted, redacted_hash = datason.hash_and_redact(obj, redact={"fields": ["ssn"]})
        assert datason.verify_object(redacted, redacted_hash)
        assert original != redacted_hash
        redacted2, h2 = datason.hash_and_redact(obj, redact={"fields": ["ssn"]})
        assert h2 == redacted_hash

    def test_sign_and_verify(self) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            from cryptography.hazmat.primitives import serialization
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

