# Data Integrity & Verification

Datason provides built‑in utilities for reproducible hashing and verification of objects.
These helpers make it easy to confirm that data has not been modified during
serialization or redaction workflows.

## Canonical Hashing

Use `hash_object` to compute a deterministic hash of any Python object. Complex
structures are serialized with `datason.serialize` before hashing and keys are
sorted for stability.

```python
from datason import hash_object, verify_object

obj = {"id": 1, "values": [1, 2, 3]}
obj_hash = hash_object(obj)
assert verify_object(obj, obj_hash)
```

## Redaction‑Aware Workflows

Hashes can be computed on redacted data to ensure sensitive fields are removed
consistently.

```python
from datason import hash_and_redact

user = {"id": 1, "ssn": "123-45-6789"}
redacted, h = hash_and_redact(user, redact={"fields": ["ssn"]})
```

Applying the same redaction again will produce the same hash, enabling reliable
comparisons in tests or compliance audits.

## Verification Utilities

`verify_object` and `verify_json` compare data against an expected hash value.
This supports audit logging and tamper‑evident storage of serialized objects.

## Digital Signatures

For stronger guarantees you can sign serialized objects with an Ed25519 key.
The cryptography package is only required when calling these helpers.

```python
from datason import sign_object, verify_signature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()

public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()

sig = sign_object({"hello": "world"}, private_pem)
assert verify_signature({"hello": "world"}, sig, public_pem)
```

