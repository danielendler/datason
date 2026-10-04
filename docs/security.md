# Security Features

datason includes security features designed for production environments.

## Built-in Limits

All limits are enforced by default and raise `SecurityError`:

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_depth` | 50 | Prevents stack overflow from deeply nested data |
| `max_size` | 100,000 | Bounds dict/list container size |
| `max_string_length` | 1,000,000 | Bounds individual strings |
| `max_input_bytes` | 16 MiB | Bounds input and supported reconstruction buffer estimates |
| `max_nodes` | 1,000,000 | Bounds representation traversal |
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

# HMAC with secret key (authenticated payload)
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

New envelopes include `__datason_integrity_version__: 1` and sign a stable,
sorted, compact JSON representation of the payload. Whitespace and input key
order do not affect verification. Duplicate keys and non-finite numbers are
rejected when wrapping. This representation is specific to Datason, not RFC 8785.

Supplying a key requires an HMAC envelope: verification never falls back to an
unsigned hash. Empty keys are rejected. Legacy envelopes written with the old
default JSON formatting remain readable; old signatures of other formatting
cannot be reconstructed. Hash-only envelopes detect accidental corruption and
provide no authentication. HMAC does not provide encryption, ownership checks,
or replay prevention; applications must enforce those separately.


## Reconstruction trust

Built-in typed loaders call installed library constructors; sklearn state
restoration also imports estimator classes and calls their state hooks. Keep
model snapshots application-owned and plugins reviewed. For ordinary incoming
JSON, `allow_plugin_deserialization=False` rejects plugin reconstruction before
dispatch, including nested tags and non-strict loading. Parser callbacks and
custom decoder classes remain trusted code.

See the [built-in reconstruction review](ml-reconstruction-review.md) for all
constructor/import paths, allocation checks and fidelity limits. Bounds on
individual buffers do not provide a process-wide memory or execution sandbox.
