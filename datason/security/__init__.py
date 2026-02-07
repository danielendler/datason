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
from .pickle_bridge import (  # nosec B403
    DEFAULT_ALLOWED_MODULES,
    json_to_pickle,
    pickle_file_to_json,
    pickle_to_json,
    scan_pickle_modules,
    validate_pickle_safety,
)
from .redaction import (
    BUILTIN_PATTERNS,
    redact_string,
    redact_value,
    should_redact_field,
)

__all__ = [
    "BUILTIN_PATTERNS",
    "DEFAULT_ALLOWED_MODULES",
    "compute_hash",
    "compute_hmac",
    "json_to_pickle",
    "pickle_file_to_json",
    "pickle_to_json",
    "redact_string",
    "redact_value",
    "scan_pickle_modules",
    "should_redact_field",
    "validate_pickle_safety",
    "verify_hmac",
    "verify_integrity",
    "wrap_with_integrity",
]
