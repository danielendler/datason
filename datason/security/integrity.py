"""Data integrity verification for datason.

Provides hash-based integrity checking for serialized data.
Use to detect tampering or corruption after serialization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def compute_hash(data: str, algorithm: str = "sha256") -> str:
    """Compute a hash of serialized JSON data.

    Args:
        data: JSON string to hash.
        algorithm: Hash algorithm (sha256, sha384, sha512, md5).

    Returns:
        Hex digest of the hash.
    """
    h = hashlib.new(algorithm)
    h.update(data.encode("utf-8"))
    return h.hexdigest()


def compute_hmac(data: str, key: str, algorithm: str = "sha256") -> str:
    """Compute an HMAC signature of serialized JSON data.

    Args:
        data: JSON string to sign.
        key: Secret key for HMAC.
        algorithm: Hash algorithm for HMAC.

    Returns:
        Hex digest of the HMAC.
    """
    if not key:
        raise ValueError("HMAC key must not be empty")
    return hmac.new(
        key.encode("utf-8"),
        data.encode("utf-8"),
        algorithm,
    ).hexdigest()


def verify_hmac(data: str, key: str, expected: str, algorithm: str = "sha256") -> bool:
    """Verify an HMAC signature using constant-time comparison.

    Args:
        data: JSON string to verify.
        key: Secret key used for signing.
        expected: Expected HMAC hex digest.
        algorithm: Hash algorithm used for signing.

    Returns:
        True if the signature is valid.
    """
    actual = compute_hmac(data, key, algorithm)
    return hmac.compare_digest(actual, expected)


def wrap_with_integrity(data: str, key: str | None = None) -> str:
    """Wrap serialized data with integrity metadata.

    Adds a hash (or HMAC if key provided) as an envelope around
    the original data, enabling verification on deserialization.

    Args:
        data: JSON string to protect.
        key: Optional secret key for HMAC (uses plain hash if None).

    Returns:
        JSON string with integrity envelope.
    """
    payload = json.loads(data, object_pairs_hook=_unique_object)
    canonical = _canonical_payload(payload)
    envelope: dict[str, Any] = {"__datason_payload__": payload, "__datason_integrity_version__": 1}
    if key is not None:
        envelope["__datason_hmac__"] = compute_hmac(canonical, key)
    else:
        envelope["__datason_hash__"] = compute_hash(canonical)
    return json.dumps(envelope, ensure_ascii=False)


def verify_integrity(envelope_str: str, key: str | None = None) -> tuple[bool, str]:
    """Verify and unwrap an integrity envelope.

    Args:
        envelope_str: JSON string with integrity metadata.
        key: Secret key if HMAC was used.

    Returns:
        Tuple of (is_valid, original_json_string).
        If verification fails, original data is still returned
        but is_valid is False.
    """
    try:
        envelope = json.loads(envelope_str, object_pairs_hook=_unique_object)
        if not isinstance(envelope, dict) or "__datason_payload__" not in envelope:
            return False, envelope_str
        version = envelope.get("__datason_integrity_version__")
        if version is not None and (type(version) is not int or version != 1):
            return False, envelope_str
        payload_str = _canonical_payload(envelope["__datason_payload__"])
        if version is None:
            payload_str = json.dumps(envelope["__datason_payload__"], ensure_ascii=False, allow_nan=False)
        signature = "__datason_hmac__" if key is not None else "__datason_hash__"
        other = "__datason_hash__" if key is not None else "__datason_hmac__"
        expected = envelope.get(signature)
        if other in envelope or not isinstance(expected, str) or len(expected) != 64:
            return False, payload_str
        actual = compute_hmac(payload_str, key) if key is not None else compute_hash(payload_str)
        return hmac.compare_digest(actual, expected), payload_str
    except (ValueError, TypeError, RecursionError):
        return False, envelope_str


def _canonical_payload(payload: Any) -> str:
    """Stable version-1 representation; this is not RFC 8785 canonical JSON."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys rather than authenticate an ambiguous object."""
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result
