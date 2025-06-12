"""Data integrity utilities for datason.

This module provides reproducible hashing and verification
for Python objects with optional redaction support.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from .core import serialize

# Allowed secure hash algorithms (cryptographically strong)
ALLOWED_HASH_ALGORITHMS = {"sha256", "sha3_256", "sha3_512", "sha512"}


def validate_hash_algorithm(hash_algo: str) -> None:
    """Validate that the hash algorithm is strong and supported.

    Args:
        hash_algo: Hash algorithm to validate

    Raises:
        ValueError: If the hash algorithm is not supported or insecure
    """
    if hash_algo not in ALLOWED_HASH_ALGORITHMS:
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {hash_algo}. "
            f"Must be one of: {', '.join(sorted(ALLOWED_HASH_ALGORITHMS))}"
        )


def _apply_redaction(obj: Any, config: dict[str, Any]) -> Any:
    """Apply redaction to an object using :class:`RedactionEngine`."""
    try:
        from .redaction import RedactionEngine
    except ImportError as exc:
        raise RuntimeError(
            "Redaction module is not available. Install datason with redaction support "
            "or use integrity functions without the 'redact' parameter."
        ) from exc

    engine = RedactionEngine(
        redact_fields=config.get("fields"),
        redact_patterns=config.get("patterns"),
        redact_large_objects=config.get("large_objects", False),
        large_object_threshold=config.get("large_object_threshold", 10 * 1024 * 1024),
        redaction_replacement=config.get("replacement", "<REDACTED>"),
        include_redaction_summary=config.get("include_summary", False),
        audit_trail=config.get("audit_trail", False),
    )
    return engine.process_object(obj)


def canonicalize(obj: Any, *, redact: dict[str, Any] | None = None) -> str:
    """Return a canonical JSON representation of ``obj``.

    If ``redact`` is provided, redaction is applied before serialization.
    The output uses sorted keys and compact separators to ensure stable
    ordering for hashing.

    Args:
        obj: Object to canonicalize
        redact: Optional redaction configuration. If provided but redaction
               module is not available, raises RuntimeError.

    Returns:
        Canonical JSON string representation

    Raises:
        RuntimeError: If redact is specified but redaction module is unavailable
    """
    if redact:
        obj = _apply_redaction(obj, redact)

    serialized = serialize(obj)
    return json.dumps(serialized, sort_keys=True, separators=(",", ":"))


def hash_object(obj: Any, *, redact: dict[str, Any] | None = None, hash_algo: str = "sha256") -> str:
    """Compute a deterministic hash of ``obj``.

    Args:
        obj: Object to hash
        redact: Optional redaction configuration. If provided but redaction
               module is not available, raises RuntimeError.
        hash_algo: Hash algorithm to use (default: sha256). Must be a strong
                  algorithm.

    Returns:
        Hexadecimal hash string

    Raises:
        RuntimeError: If redact is specified but redaction module is unavailable
        ValueError: If an unsupported hash algorithm is specified
    """
    validate_hash_algorithm(hash_algo)
    canon = canonicalize(obj, redact=redact)
    h = hashlib.new(hash_algo)
    h.update(canon.encode("utf-8"))
    return h.hexdigest()


def hash_json(json_data: Any, hash_algo: str = "sha256") -> str:
    """Compute a deterministic hash for a JSON-compatible structure.

    Args:
        json_data: JSON-compatible structure to hash
        hash_algo: Hash algorithm to use (default: sha256). Must be a strong
                  algorithm.

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If an unsupported hash algorithm is specified
    """
    validate_hash_algorithm(hash_algo)
    canon = json.dumps(json_data, sort_keys=True, separators=(",", ":"))
    h = hashlib.new(hash_algo)
    h.update(canon.encode("utf-8"))
    return h.hexdigest()


def verify_object(
    obj: Any,
    expected_hash: str,
    *,
    redact: dict[str, Any] | None = None,
    hash_algo: str = "sha256",
) -> bool:
    """Verify that ``obj`` hashes to ``expected_hash``.

    Args:
        obj: Object to verify
        expected_hash: Expected hash value
        redact: Optional redaction configuration. If provided but redaction
               module is not available, raises RuntimeError.
        hash_algo: Hash algorithm to use (default: sha256). Must be a strong
                  algorithm.

    Returns:
        True if hash matches, False otherwise

    Raises:
        RuntimeError: If redact is specified but redaction module is unavailable
        ValueError: If an unsupported hash algorithm is specified
    """
    actual = hash_object(obj, redact=redact, hash_algo=hash_algo)
    return actual == expected_hash


def verify_json(json_data: Any, expected_hash: str, hash_algo: str = "sha256") -> bool:
    """Verify a JSON-compatible structure against ``expected_hash``.

    Args:
        json_data: JSON-compatible structure to verify
        expected_hash: Expected hash value
        hash_algo: Hash algorithm to use (default: sha256). Must be a strong
                  algorithm.

    Returns:
        True if hash matches, False otherwise

    Raises:
        ValueError: If an unsupported hash algorithm is specified
    """
    actual = hash_json(json_data, hash_algo=hash_algo)
    return actual == expected_hash


def hash_and_redact(obj: Any, *, redact: dict[str, Any] | None = None, hash_algo: str = "sha256") -> tuple[Any, str]:
    """Redact ``obj``, hash the result, and return ``(redacted_obj, hash)``.

    Args:
        obj: Object to redact and hash
        redact: Redaction configuration. If None, no redaction is applied.
               If provided but redaction module is not available, raises RuntimeError.
        hash_algo: Hash algorithm to use (default: sha256). Must be a strong
                  algorithm.

    Returns:
        Tuple of (redacted_object, hash_string)

    Raises:
        RuntimeError: If redact is specified but redaction module is unavailable
        ValueError: If an unsupported hash algorithm is specified
    """
    redacted = _apply_redaction(obj, redact or {}) if redact else obj
    hash_val = hash_object(redacted, hash_algo=hash_algo)
    return redacted, hash_val


def sign_object(
    obj: Any,
    private_key_pem: str,
    *,
    redact: dict[str, Any] | None = None,
) -> str:
    """Sign ``obj`` with an Ed25519 private key.

    The signature is returned as base64-encoded text. This function lazily
    imports :mod:`cryptography` so the package is only required when used.

    Args:
        obj: Object to sign
        private_key_pem: Ed25519 private key in PEM format
        redact: Optional redaction configuration. If provided but redaction
               module is not available, raises RuntimeError.

    Returns:
        Base64-encoded signature string

    Raises:
        RuntimeError: If cryptography package is not available or if redact
                     is specified but redaction module is unavailable
        TypeError: If key is not an Ed25519 private key
    """

    try:  # Lazy import
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("cryptography is required for signing") from exc

    canon = canonicalize(obj, redact=redact)
    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("Only Ed25519 keys are supported for signing")

    signature = private_key.sign(canon.encode("utf-8"))
    return base64.b64encode(signature).decode("ascii")


def verify_signature(
    obj: Any,
    signature: str,
    public_key_pem: str,
    *,
    redact: dict[str, Any] | None = None,
) -> bool:
    """Verify ``signature`` for ``obj`` using the given Ed25519 public key.

    Args:
        obj: Object to verify signature for
        signature: Base64-encoded signature to verify
        public_key_pem: Ed25519 public key in PEM format
        redact: Optional redaction configuration. If provided but redaction
               module is not available, raises RuntimeError.

    Returns:
        True if signature is valid, False otherwise

    Raises:
        RuntimeError: If cryptography package is not available or if redact
                     is specified but redaction module is unavailable
        TypeError: If key is not an Ed25519 public key
    """

    try:  # Lazy import
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("cryptography is required for signature verification") from exc

    canon = canonicalize(obj, redact=redact)
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Only Ed25519 keys are supported for verification")

    try:
        public_key.verify(base64.b64decode(signature), canon.encode("utf-8"))
        return True
    except Exception:  # pragma: no cover - verification failure path
        return False
