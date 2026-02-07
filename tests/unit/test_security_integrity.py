"""Tests for the security integrity module."""

from __future__ import annotations

import json

import datason
from datason.security.integrity import (
    compute_hash,
    compute_hmac,
    verify_hmac,
    verify_integrity,
    wrap_with_integrity,
)


class TestComputeHash:
    def test_sha256_default(self) -> None:
        result = compute_hash('{"key": "value"}')
        assert isinstance(result, str)
        assert len(result) == 64  # sha256 hex digest length

    def test_deterministic(self) -> None:
        assert compute_hash("test") == compute_hash("test")

    def test_different_inputs(self) -> None:
        assert compute_hash("a") != compute_hash("b")

    def test_sha512(self) -> None:
        result = compute_hash("test", algorithm="sha512")
        assert len(result) == 128  # sha512 hex digest length


class TestComputeHmac:
    def test_produces_hex_string(self) -> None:
        result = compute_hmac("data", "secret")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_different_keys(self) -> None:
        assert compute_hmac("data", "key1") != compute_hmac("data", "key2")

    def test_deterministic(self) -> None:
        assert compute_hmac("data", "key") == compute_hmac("data", "key")


class TestVerifyHmac:
    def test_valid_signature(self) -> None:
        sig = compute_hmac("data", "key")
        assert verify_hmac("data", "key", sig) is True

    def test_invalid_signature(self) -> None:
        assert verify_hmac("data", "key", "bad_sig") is False

    def test_wrong_key(self) -> None:
        sig = compute_hmac("data", "key1")
        assert verify_hmac("data", "key2", sig) is False

    def test_tampered_data(self) -> None:
        sig = compute_hmac("original", "key")
        assert verify_hmac("tampered", "key", sig) is False


class TestWrapWithIntegrity:
    def test_hash_envelope(self) -> None:
        data = datason.dumps({"test": 42})
        wrapped = wrap_with_integrity(data)
        parsed = json.loads(wrapped)
        assert "__datason_payload__" in parsed
        assert "__datason_hash__" in parsed
        assert parsed["__datason_payload__"] == {"test": 42}

    def test_hmac_envelope(self) -> None:
        data = datason.dumps({"test": 42})
        wrapped = wrap_with_integrity(data, key="my-secret")
        parsed = json.loads(wrapped)
        assert "__datason_payload__" in parsed
        assert "__datason_hmac__" in parsed
        assert "__datason_hash__" not in parsed


class TestVerifyIntegrity:
    def test_valid_hash(self) -> None:
        data = datason.dumps({"x": 1})
        wrapped = wrap_with_integrity(data)
        is_valid, payload = verify_integrity(wrapped)
        assert is_valid is True
        assert json.loads(payload) == {"x": 1}

    def test_valid_hmac(self) -> None:
        data = datason.dumps({"x": 1})
        wrapped = wrap_with_integrity(data, key="secret")
        is_valid, payload = verify_integrity(wrapped, key="secret")
        assert is_valid is True

    def test_tampered_payload(self) -> None:
        data = datason.dumps({"x": 1})
        wrapped = wrap_with_integrity(data)
        # Tamper with the payload
        envelope = json.loads(wrapped)
        envelope["__datason_payload__"]["x"] = 999
        tampered = json.dumps(envelope)
        is_valid, _payload = verify_integrity(tampered)
        assert is_valid is False

    def test_wrong_hmac_key(self) -> None:
        data = datason.dumps({"x": 1})
        wrapped = wrap_with_integrity(data, key="correct-key")
        is_valid, _payload = verify_integrity(wrapped, key="wrong-key")
        assert is_valid is False

    def test_no_envelope(self) -> None:
        is_valid, payload = verify_integrity('{"plain": "data"}')
        assert is_valid is False

    def test_roundtrip_hash(self) -> None:
        """Full round-trip: serialize → wrap → verify → deserialize."""
        original = {"name": "test", "values": [1, 2, 3]}
        serialized = datason.dumps(original)
        wrapped = wrap_with_integrity(serialized)
        is_valid, payload = verify_integrity(wrapped)
        assert is_valid is True
        restored = datason.loads(payload)
        assert restored == original

    def test_roundtrip_hmac(self) -> None:
        original = {"secret": "data"}
        serialized = datason.dumps(original)
        wrapped = wrap_with_integrity(serialized, key="my-key")
        is_valid, payload = verify_integrity(wrapped, key="my-key")
        assert is_valid is True
        restored = datason.loads(payload)
        assert restored == original
