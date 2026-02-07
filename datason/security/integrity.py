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
    envelope: dict[str, Any] = {"__datason_payload__": json.loads(data)}
    if key:
        envelope["__datason_hmac__"] = compute_hmac(data, key)
    else:
        envelope["__datason_hash__"] = compute_hash(data)
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
    envelope = json.loads(envelope_str)

    if "__datason_payload__" not in envelope:
        return False, envelope_str

    payload_str = json.dumps(envelope["__datason_payload__"], ensure_ascii=False)

    if key and "__datason_hmac__" in envelope:
        is_valid = verify_hmac(payload_str, key, envelope["__datason_hmac__"])
        return is_valid, payload_str

    if "__datason_hash__" in envelope:
        expected = envelope["__datason_hash__"]
        actual = compute_hash(payload_str)
        return hmac.compare_digest(actual, expected), payload_str

    return False, payload_str
