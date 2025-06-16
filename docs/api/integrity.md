# 🔐 Integrity Functions

Datason includes a set of helpers for verifying data integrity and authenticity. These utilities provide deterministic hashing, optional redaction, and Ed25519 signatures.

## 🎯 Function Overview

| Function | Purpose | Best For |
|----------|---------|---------|
| `canonicalize()` | Deterministic JSON output | Stable hashing |
| `hash_object()` | Hash Python objects | Audit trails |
| `hash_json()` | Hash JSON structures | API responses |
| `verify_object()` | Verify object integrity | Validation |
| `verify_json()` | Verify JSON integrity | API testing |
| `hash_and_redact()` | Redact then hash | Compliance |
| `sign_object()` | Create digital signature | Authenticity |
| `verify_signature()` | Validate signature | Document integrity |

## 📦 Detailed Function Documentation

### canonicalize()

::: datason.integrity.canonicalize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### hash_object()

::: datason.integrity.hash_object
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### hash_json()

::: datason.integrity.hash_json
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### verify_object()

::: datason.integrity.verify_object
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### verify_json()

::: datason.integrity.verify_json
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### hash_and_redact()

::: datason.integrity.hash_and_redact
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### sign_object()

::: datason.integrity.sign_object
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### verify_signature()

::: datason.integrity.verify_signature
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

**Basic Verification Example:**
```python
import datason
from datason import integrity

payload = {"id": 1, "value": 42}
hash_val = integrity.hash_object(payload)
assert integrity.verify_object(payload, hash_val)
```

**Signing Example:**
```python
from datason import integrity
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode("utf-8")

document = {"msg": "hello"}

signature = integrity.sign_object(document, private_pem)
assert integrity.verify_signature(document, signature, public_pem)
```
