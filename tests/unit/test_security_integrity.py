"""Tests for the security integrity module."""

from __future__ import annotations

import json

import pytest

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


@pytest.mark.parametrize("data", ['{"a":1}', '{ "z": 2, "a": "é" }', "[1,2,3]", "null"])
@pytest.mark.parametrize("key", [None, "review-secret"])
def test_formatting_independent_envelope(data: str, key: str | None) -> None:
    wrapped = wrap_with_integrity(data, key=key)
    valid, payload = verify_integrity(wrapped, key=key)
    assert valid
    assert json.loads(payload) == json.loads(data)


def test_hmac_cannot_downgrade_to_hash() -> None:
    envelope = json.loads(wrap_with_integrity('{"authorized": false}', key="secret"))
    envelope.pop("__datason_hmac__")
    envelope["__datason_payload__"]["authorized"] = True
    envelope["__datason_hash__"] = compute_hash('{"authorized":true}')
    assert verify_integrity(json.dumps(envelope), key="secret")[0] is False


@pytest.mark.parametrize("payload", ["[]", "null", "{", '{"__datason_payload__": 1, "__datason_hash__": 42}'])
def test_malformed_envelopes_fail_closed(payload: str) -> None:
    assert verify_integrity(payload)[0] is False


def test_signature_cannot_change_authentication_mode() -> None:
    wrapped = wrap_with_integrity('{"a":1}', key="secret")
    assert verify_integrity(wrapped)[0] is False
    assert verify_integrity(wrap_with_integrity('{"a":1}'), key="secret")[0] is False


def test_legacy_default_formatted_envelope_still_verifies() -> None:
    data = '{"a": 1}'
    envelope = {"__datason_payload__": {"a": 1}, "__datason_hmac__": compute_hmac(data, "secret")}
    assert verify_integrity(json.dumps(envelope), key="secret")[0]


@pytest.mark.parametrize("data", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_ambiguous_or_nonfinite_payload_cannot_be_signed(data: str) -> None:
    with pytest.raises(ValueError):
        wrap_with_integrity(data, key="secret")


def test_empty_hmac_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        wrap_with_integrity('{"a":1}', key="")
    assert verify_integrity(wrap_with_integrity('{"a":1}'), key="")[0] is False
