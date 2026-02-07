# Security Features

datason includes security features designed for production environments.

## Built-in Limits

All limits are enforced by default and raise `SecurityError`:

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_depth` | 50 | Prevents stack overflow from deeply nested data |
| `max_size` | 100,000 | Prevents memory exhaustion from huge dicts/lists |
| Circular references | Always on | Prevents infinite loops via `id()` tracking |

```python
from datason._errors import SecurityError

# Override limits inline
datason.dumps(data, max_depth=10, max_size=1000)

# Circular references are always detected
d = {}
d["self"] = d
datason.dumps(d)  # raises SecurityError
```

## PII Redaction

Redact sensitive data during serialization (not as post-processing).

### By field name

Case-insensitive substring match:

```python
datason.dumps(
    {"username": "alice", "password": "secret123", "api_key": "sk-xxx"},
    redact_fields=("password", "key", "secret"),
)
# {"username": "alice", "password": "[REDACTED]", "api_key": "[REDACTED]"}
```

### By pattern

Built-in patterns: `email`, `ssn`, `credit_card`, `phone_us`, `ipv4`:

```python
datason.dumps(
    {"msg": "Contact alice@example.com or 555-123-4567"},
    redact_patterns=("email", "phone_us"),
)
# {"msg": "Contact [REDACTED] or [REDACTED]"}
```

Custom regex patterns are also supported:

```python
datason.dumps(data, redact_patterns=(r"secret-\w+",))
```

## Integrity Verification

Detect tampering or corruption with hash-based envelopes:

```python
from datason.security.integrity import wrap_with_integrity, verify_integrity

# Hash-based (no secret)
json_str = datason.dumps(data)
wrapped = wrap_with_integrity(json_str)
is_valid, payload = verify_integrity(wrapped)

# HMAC with secret key (tamper-proof)
wrapped = wrap_with_integrity(json_str, key="my-secret")
is_valid, payload = verify_integrity(wrapped, key="my-secret")
```

The envelope format:

```json
{
    "__datason_payload__": { ... },
    "__datason_hash__": "sha256hex..."
}
```
