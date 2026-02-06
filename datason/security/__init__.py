"""Security modules for datason.

Includes PII redaction, data integrity (hash/sign/verify),
and safe pickle-to-JSON conversion.
"""

from .integrity import (
    compute_hash,
    compute_hmac,
    verify_hmac,
    verify_integrity,
    wrap_with_integrity,
)
from .redaction import (
    BUILTIN_PATTERNS,
    redact_string,
    redact_value,
    should_redact_field,
)

__all__ = [
    "BUILTIN_PATTERNS",
    "compute_hash",
    "compute_hmac",
    "redact_string",
    "redact_value",
    "should_redact_field",
    "verify_hmac",
    "verify_integrity",
    "wrap_with_integrity",
]
