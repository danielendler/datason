# Security and data handling

Start with the data you are reading or exporting: plain incoming JSON, trusted
typed stored data, or redacted diagnostics. datason provides representation
limits, redaction, and integrity helpers. Plugins, model serializers, and JSON
callbacks remain trusted Python code. v2 is an alpha; see the
[hardening roadmap](hardening-roadmap.md) for review scope.

## Representation budgets

These defaults apply to serialization and parsed representations, including
metadata. Incoming JSON also has a byte budget before parsing.

| Option | Default | Purpose |
| --- | --- | --- |
| `max_depth` | 50 | Bound traversal depth |
| `max_size` | 100,000 | Bound entries per container |
| `max_string_length` | 1,000,000 | Bound string/key length |
| `max_nodes` | 1,000,000 | Bound traversal work |
| `max_input_bytes` | 16,777,216 | Bound encoded input and supported NumPy allocation estimates |
| Circular-reference detection | Always enabled by datason | Reject cycles during serialization |

Budget violations raise `SecurityError`. These are representation budgets, not a
process-wide memory limit or a sandbox for callbacks. A custom parser hook can
execute before full-tree validation. See [Serialization boundaries](serialization-boundaries.md).

```python
import datason
from datason._errors import SecurityError

circular = {}
circular["self"] = circular
try:
    datason.dumps(circular)
except SecurityError:
    pass
else:
    raise AssertionError("Expected circular-reference rejection")
```

For ordinary incoming JSON, use `allow_plugin_deserialization=False` to reject
typed plugin reconstruction. Built-in tuple/set/frozenset tags can still restore
collections. This control does not disable application parser callbacks.

## Redact diagnostics

Field matching is a case-insensitive substring match. Pattern matching applies
to string values; built-in names are `email`, `ssn`, `credit_card`, `phone_us`,
and `ipv4`. Custom regexes are also accepted.

```python
import json

import datason

record = {"username": "alice", "password": "private",
          "api_key": "private-key", "message": "Contact alice@example.com"}
text = datason.dumps(record, redact_fields=("password", "key"),
                     redact_patterns=("email",), include_type_hints=False)
assert json.loads(text) == {
    "username": "alice", "password": "[REDACTED]", "api_key": "[REDACTED]",
    "message": "Contact [REDACTED]",
}
assert record["password"] == "private"
```

Choose patterns from your actual data and test representative records. A rule
such as `key` also matches `monkey`; matching is not a complete classification
of sensitive data. Custom regex execution is trusted and not time-limited here.
Policies traverse plugin output too, and redaction can prevent reconstruction.
Use a separate diagnostic copy rather than redacting resumable state.
See the [diagnostics recipe](recipes.md#redacted-diagnostics).

## Verify integrity before reconstructing

A hash envelope detects accidental corruption. HMAC uses a shared secret to
authenticate the payload. Verify first and stop on failure; the helper returns
payload text even when verification fails.

```python
import secrets
from decimal import Decimal

import datason
from datason.security.integrity import wrap_with_integrity, verify_integrity

# Demonstration key; in an application, retrieve a persistent secret securely.
key = secrets.token_hex(32)
text = datason.dumps({"balance": Decimal("19.99")})
wrapped = wrap_with_integrity(text, key=key)
valid, payload = verify_integrity(wrapped, key=key)
if not valid:
    raise ValueError("Integrity verification failed")
restored = datason.loads(payload)
assert restored["balance"] == Decimal("19.99")
assert verify_integrity(wrapped, key="wrong-key")[0] is False
```

Hash-only mode uses `wrap_with_integrity(text)` and `verify_integrity(wrapped)`
without a key; it provides no authentication. Integrity helpers parse JSON
separately from `loads`: enforce an input-size budget in your application before
verifying externally supplied envelopes.

New envelopes include `__datason_integrity_version__: 1` and sign a stable,
sorted, compact JSON representation of the payload. Whitespace and input key
order do not affect verification. Duplicate keys and non-finite numbers are
rejected when wrapping. This representation is specific to datason, not RFC 8785.

Supplying a key requires an HMAC envelope: verification never falls back to an
unsigned hash. Empty keys are rejected. Legacy envelopes written with the old
default JSON formatting remain readable; old signatures of other formatting
cannot be reconstructed. HMAC does not provide encryption, ownership checks,
or replay prevention; applications must enforce those separately.

## Pickle migration

Pickle conversion requires `trusted=True`, because loading pickle can execute
Python code. Module scanning is diagnostic and does not establish safety.
See [Trusted pickle migration](pickle-migration.md) for an example and the
[serialization boundaries](serialization-boundaries.md) for reconstruction controls.
