"""PII redaction for datason serialization.

Provides field-name and regex-pattern based redaction that integrates
with the serialization pipeline. Redaction happens DURING serialization,
not as a post-processing step, so sensitive data never enters the
JSON output.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"

# Built-in patterns for common PII types
BUILTIN_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ipv4": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def should_redact_field(key: str, redact_fields: tuple[str, ...]) -> bool:
    """Check if a dict key should be redacted based on field names.

    Uses case-insensitive substring matching:
    - redact_fields=("password",) matches "password", "user_password", "PASSWORD"
    """
    if not redact_fields:
        return False
    key_lower = key.lower()
    return any(field.lower() in key_lower for field in redact_fields)


def redact_string(value: str, patterns: tuple[str, ...]) -> str:
    """Redact matching patterns from a string value.

    Args:
        value: The string to scan for sensitive data.
        patterns: Regex patterns to match and redact. Can be regex strings
                 or names from BUILTIN_PATTERNS (e.g., "email", "ssn").

    Returns:
        String with matched patterns replaced by [REDACTED].
    """
    if not patterns:
        return value

    result = value
    for pattern in patterns:
        # Resolve built-in pattern names
        regex = BUILTIN_PATTERNS.get(pattern, pattern)
        result = re.sub(regex, _REDACTED, result)
    return result


def redact_value(value: Any, patterns: tuple[str, ...]) -> Any:
    """Apply pattern redaction to a value (only affects strings)."""
    if isinstance(value, str) and patterns:
        return redact_string(value, patterns)
    return value
