"""Security features: redaction, integrity verification, and limits."""

from __future__ import annotations

import datason
from datason._errors import SecurityError
from datason.security.integrity import verify_integrity, wrap_with_integrity

# =========================================================================
# 1. Field-based redaction (hide sensitive fields by name)
# =========================================================================

user_data = {
    "username": "alice",
    "password": "s3cret!",
    "api_key": "sk-abc123def456",
    "email": "alice@company.com",
    "role": "admin",
}

# Redact fields containing "password" or "key" (case-insensitive, substring match)
safe = datason.dumps(user_data, redact_fields=("password", "key"))
print(f"Redacted fields: {safe}")
assert "s3cret!" not in safe
assert "sk-abc123" not in safe
assert "alice" in safe  # non-sensitive fields preserved

# =========================================================================
# 2. Pattern-based redaction (regex on string values)
# =========================================================================

message = {
    "sender": "system",
    "body": "Contact alice@example.com or call 555-123-4567 for help.",
}

# Built-in patterns: email, ssn, credit_card, phone_us, ipv4
safe = datason.dumps(message, redact_patterns=("email", "phone_us"))
print(f"Redacted patterns: {safe}")
assert "alice@example.com" not in safe
assert "555-123-4567" not in safe

# =========================================================================
# 3. Combined redaction with context manager
# =========================================================================

records = [
    {"name": "Alice", "ssn": "123-45-6789", "notes": "Email: a@b.com"},
    {"name": "Bob", "ssn": "987-65-4321", "notes": "IP: 192.168.1.1"},
]

with datason.config(redact_fields=("ssn",), redact_patterns=("email", "ipv4")):
    safe = datason.dumps(records)
    print(f"Combined: {safe}")
    assert "123-45-6789" not in safe
    assert "a@b.com" not in safe
    assert "192.168.1.1" not in safe

# =========================================================================
# 4. Integrity verification (hash-based)
# =========================================================================

data = datason.dumps({"account": "checking", "balance": 1000.00})
wrapped = wrap_with_integrity(data)
print(f"\nHash envelope: {wrapped[:80]}...")

# Verify integrity
is_valid, payload = verify_integrity(wrapped)
assert is_valid
print(f"Hash valid: {is_valid}")

# =========================================================================
# 5. Integrity verification (HMAC-based, with secret key)
# =========================================================================

wrapped = wrap_with_integrity(data, key="my-secret-key")
print(f"HMAC envelope: {wrapped[:80]}...")

# Verify with correct key
is_valid, payload = verify_integrity(wrapped, key="my-secret-key")
assert is_valid
print(f"HMAC valid (correct key): {is_valid}")

# Verify with wrong key
is_valid, _ = verify_integrity(wrapped, key="wrong-key")
assert not is_valid
print(f"HMAC valid (wrong key): {is_valid}")

# =========================================================================
# 6. Security limits (depth and size)
# =========================================================================

# Depth limit prevents stack overflow from deeply nested data
try:
    deep = {"a": None}
    current = deep
    for _ in range(200):
        inner = {"a": None}
        current["a"] = inner
        current = inner
    datason.dumps(deep)
except SecurityError as e:
    print(f"\nDepth limit caught: {e}")

# Size limit prevents memory exhaustion from huge dicts
try:
    datason.dumps({f"k{i}": i for i in range(200_000)})
except SecurityError as e:
    print(f"Size limit caught: {e}")

# Circular reference detection
try:
    circular: dict = {"a": 1}
    circular["self"] = circular
    datason.dumps(circular)
except SecurityError as e:
    print(f"Circular ref caught: {e}")

print("\nAll security examples passed!")
