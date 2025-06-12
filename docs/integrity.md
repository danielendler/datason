# Data Integrity & Verification

Datason provides built‑in utilities for reproducible hashing and verification of objects.
These helpers make it easy to confirm that data has not been modified during
serialization or redaction workflows.

## Quick Start

```python
import datason

# Basic integrity verification
data = {"user_id": 123, "action": "purchase", "amount": 99.99}
data_hash = datason.hash_object(data)
is_valid = datason.verify_object(data, data_hash)  # True
```

## Core Functions

| Function | Purpose | Example Use Case |
|----------|---------|------------------|
| `hash_object()` | Generate deterministic hash | Audit trails, change detection |
| `verify_object()` | Verify data integrity | Data validation, tamper detection |
| `hash_and_redact()` | Hash after redaction | GDPR compliance, PII protection |
| `sign_object()` | Digital signatures | Legal documents, high-security data |
| `verify_signature()` | Signature verification | Document authenticity |

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

### Hash Algorithm Options

Choose the appropriate hash algorithm for your security requirements:

```python
# Default: SHA256 (recommended for most use cases)
hash_sha256 = hash_object(data)

# High security: SHA512
hash_sha512 = hash_object(data, hash_algo="sha512")

# SHA-3 family (newest standards)
hash_sha3_256 = hash_object(data, hash_algo="sha3_256")
hash_sha3_512 = hash_object(data, hash_algo="sha3_512")

# These will raise ValueError (blocked for security):
# hash_object(data, hash_algo="md5")    # ❌ ValueError
# hash_object(data, hash_algo="sha1")   # ❌ ValueError
```

**Recommendation**: Use SHA256 (default) or SHA512 for production systems. SHA-3 algorithms provide the highest security.

## Redaction‑Aware Workflows

Perfect for GDPR, HIPAA, and other privacy compliance requirements. Hashes can be computed on redacted data to ensure sensitive fields are removed consistently.

```python
from datason import hash_and_redact

# Example: Healthcare data (HIPAA compliance)
patient_data = {
    "patient_id": "P123456",
    "name": "John Doe",
    "ssn": "123-45-6789",
    "diagnosis": "Type 2 Diabetes",
    "doctor": "Dr. Smith"
}

# Create HIPAA-compliant version
redacted, redacted_hash = hash_and_redact(
    patient_data,
    redact={
        "fields": ["ssn", "name"],  # Remove PII
        "replacement": "[REDACTED-HIPAA]"
    }
)

# Verify redacted data integrity
assert verify_object(redacted, redacted_hash)
```

Applying the same redaction again will produce the same hash, enabling reliable
comparisons in tests or compliance audits.

### Advanced Redaction Options

```python
# Pattern-based redaction
user_data = {
    "email": "user@example.com",
    "phone": "555-123-4567",
    "notes": "Call customer at 555-987-6543 or email support@company.com"
}

redacted, hash_val = hash_and_redact(
    user_data,
    redact={
        "patterns": [r'\b\d{3}-\d{3}-\d{4}\b', r'\b\S+@\S+\.\S+\b'],  # Phone & email patterns
        "replacement": "[REDACTED]",
        "include_summary": True  # Include redaction metadata
    }
)
```

## Verification Utilities

`verify_object` and `verify_json` compare data against an expected hash value.
This supports audit logging and tamper‑evident storage of serialized objects.

### Basic Verification

```python
# Transaction verification
transaction = {
    "id": "TXN-2024-001",
    "amount": 1500.00,
    "timestamp": "2024-01-15T10:30:00Z"
}

# Store hash for later verification
original_hash = hash_object(transaction)

# Later: verify data hasn't been tampered with
is_valid = verify_object(transaction, original_hash)
if not is_valid:
    raise SecurityError("Transaction data has been tampered with!")
```

### JSON Data Verification

```python
# Direct JSON verification (for external data)
json_data = {"status": "completed", "results": [1, 2, 3]}
json_hash = hash_json(json_data)
assert verify_json(json_data, json_hash)
```

## ML Model Verification

Ensure model integrity throughout your MLOps pipeline:

```python
# Model metadata verification
model_info = {
    "model_name": "fraud_detection_v2.1",
    "accuracy": 0.94,
    "training_date": "2024-01-15",
    "data_scientist": "alice@company.com"
}

# Development: create hash
dev_hash = hash_object(model_info)

# QA: verify unchanged
qa_check = verify_object(model_info, dev_hash)
if not qa_check:
    raise ValueError("Model metadata changed since development!")

# Production: final verification
prod_check = verify_object(model_info, dev_hash)
print(f"Safe to deploy: {prod_check}")
```

## Digital Signatures

For stronger guarantees you can sign serialized objects with an Ed25519 key.
The cryptography package is only required when calling these helpers.

### Installation

```bash
# Install with cryptography support
pip install datason[crypto]
```

### Basic Signing

```python
from datason import sign_object, verify_signature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

# Generate key pair (in practice, load from secure storage)
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Serialize keys for storage/transmission
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()

public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()

# Sign important document
contract = {
    "contract_id": "CONTRACT-2024-001",
    "amount": 10000000.00,
    "parties": ["Company A", "Company B"],
    "effective_date": "2024-01-15"
}

signature = sign_object(contract, private_pem)
is_authentic = verify_signature(contract, signature, public_pem)
assert is_authentic
```

### Redaction-Aware Signing

Sign documents while protecting sensitive information:

```python
# Sign contract with confidential amount redacted
signature = sign_object(
    contract,
    private_pem,
    redact={"fields": ["amount"], "replacement": "[CONFIDENTIAL]"}
)

# Verify with same redaction
is_valid = verify_signature(
    contract,
    signature,
    public_pem,
    redact={"fields": ["amount"], "replacement": "[CONFIDENTIAL]"}
)
```

## Production Use Cases

### 1. Financial Audit Trails

```python
# SOX compliance: immutable transaction records
def create_audit_entry(transaction_data):
    entry = {
        "transaction": transaction_data,
        "timestamp": datetime.now(),
        "hash": hash_object(transaction_data),
        "compliance_officer": "auditor@company.com"
    }

    # Store for regulatory review
    audit_hash = hash_object(entry)
    store_audit_record(entry, audit_hash)
    return audit_hash

# Later verification
def verify_audit_trail(stored_entry, expected_hash):
    return verify_object(stored_entry, expected_hash)
```

### 2. Data Pipeline Integrity

```python
# Track data through processing pipeline
def process_data_with_integrity(raw_data):
    pipeline_steps = []

    # Step 1: Ingestion
    ingestion_hash = hash_object(raw_data)
    pipeline_steps.append({"step": "ingestion", "hash": ingestion_hash})

    # Step 2: Cleaning
    cleaned_data = clean_data(raw_data)
    cleaning_hash = hash_object(cleaned_data)
    pipeline_steps.append({"step": "cleaning", "hash": cleaning_hash})

    # Step 3: Feature engineering
    featured_data = engineer_features(cleaned_data)
    feature_hash = hash_object(featured_data)
    pipeline_steps.append({"step": "features", "hash": feature_hash})

    # Create complete audit trail
    audit_trail = {
        "pipeline_id": str(uuid.uuid4()),
        "steps": pipeline_steps,
        "final_data_hash": feature_hash
    }

    return featured_data, hash_object(audit_trail)
```

### 3. Document Management

```python
# Legal document integrity with signatures
def sign_legal_document(document, private_key_pem):
    # Add timestamp and metadata
    signed_doc = {
        "document": document,
        "signed_at": datetime.now(),
        "signature_version": "1.0"
    }

    # Create digital signature
    signature = sign_object(signed_doc, private_key_pem)

    return {
        "signed_document": signed_doc,
        "signature": signature,
        "document_hash": hash_object(signed_doc)
    }

def verify_legal_document(signed_package, public_key_pem):
    document = signed_package["signed_document"]
    signature = signed_package["signature"]

    return verify_signature(document, signature, public_key_pem)
```

## Best Practices

### Security Guidelines

1. **Hash Algorithm Selection**
   - Use SHA256 (default) for most applications
   - Use SHA512 for high-security environments
   - Avoid MD5 and SHA1 in production (deprecated)

2. **Key Management**
   - Store private keys securely (HSM, key vault)
   - Rotate keys regularly
   - Use different keys for different purposes

3. **Redaction Strategy**
   - Define clear PII/sensitive data policies
   - Test redaction patterns thoroughly
   - Document compliance requirements

### Performance Considerations

- Hashing is fast (~10ms for <1MB objects)
- Digital signatures add ~1-2ms overhead
- Use `hash_json()` for pre-serialized data
- Consider caching hashes for large, static objects

### Error Handling

```python
try:
    # Integrity verification
    if not verify_object(data, expected_hash):
        logger.error("Data integrity check failed")
        raise SecurityError("Data has been tampered with")

    # Digital signature verification
    if not verify_signature(doc, sig, public_key):
        logger.error("Document signature verification failed")
        raise SecurityError("Document signature is invalid")

except RuntimeError as e:
    if "cryptography is required" in str(e):
        logger.error("Cryptography package not installed")
        # Fall back to hash-only verification
        return verify_object(data, expected_hash)
    raise
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
import datason

app = FastAPI()

@app.post("/secure-submit")
async def secure_submit(data: dict, expected_hash: str):
    # Verify data integrity
    if not datason.verify_object(data, expected_hash):
        raise HTTPException(
            status_code=400,
            detail="Data integrity check failed"
        )

    # Process verified data
    result = process_secure_data(data)

    # Return with integrity hash
    return {
        "result": result,
        "result_hash": datason.hash_object(result)
    }
```

### BentoML Integration

```python
import bentoml
import datason

@bentoml.service
class SecureMLService:
    def __init__(self):
        self.model = load_model()

    @bentoml.api
    def predict_with_verification(self, input_data: dict, data_hash: str):
        # Verify input integrity
        if not datason.verify_object(input_data, data_hash):
            raise ValueError("Input data integrity check failed")

        # Make prediction
        prediction = self.model.predict(input_data)

        # Return with integrity verification
        result = {"prediction": prediction, "model_version": "v1.0"}
        return {
            "result": result,
            "result_hash": datason.hash_object(result)
        }
```

## Troubleshooting

### Common Issues

1. **"Redaction module unavailable"**
   ```bash
   pip install datason[all]  # Install with redaction support
   ```

2. **"cryptography is required for signing"**
   ```bash
   pip install datason[crypto]  # Install with cryptography
   ```

3. **"Unsupported or insecure hash algorithm" error**
   ```bash
   ValueError: Unsupported or insecure hash algorithm: md5. Must be one of: sha256, sha3_256, sha3_512, sha512
   ```
   - Use only secure algorithms: `sha256`, `sha512`, `sha3_256`, `sha3_512`
   - This error prevents security vulnerabilities from weak hash algorithms

4. **Hash mismatch after serialization**
   - Ensure consistent redaction configuration
   - Check for floating-point precision issues
   - Verify datetime timezone handling

### Debug Mode

```python
# Enable detailed hashing information
import datason.integrity as integrity

# Check canonical representation
canonical = integrity.canonicalize(your_data)
print(f"Canonical JSON: {canonical}")

# Test hash consistency
hash1 = datason.hash_object(your_data)
hash2 = datason.hash_object(your_data)
print(f"Hashes match: {hash1 == hash2}")
```

## API Reference

### Core Functions

- `hash_object(obj, *, redact=None, hash_algo='sha256')` → `str`
- `verify_object(obj, expected_hash, *, redact=None, hash_algo='sha256')` → `bool`
- `hash_and_redact(obj, *, redact=None, hash_algo='sha256')` → `tuple[Any, str]`
- `hash_json(json_data, hash_algo='sha256')` → `str`
- `verify_json(json_data, expected_hash, hash_algo='sha256')` → `bool`

### Digital Signatures (requires cryptography)

- `sign_object(obj, private_key_pem, *, redact=None)` → `str`
- `verify_signature(obj, signature, public_key_pem, *, redact=None)` → `bool`

### Supported Hash Algorithms

For security reasons, only cryptographically strong hash algorithms are supported:

- `sha256` (default, recommended)
- `sha512` (high security)
- `sha3_256` (SHA-3 family, high security)
- `sha3_512` (SHA-3 family, highest security)

**🚨 Security Note**: Weak algorithms like MD5 and SHA1 are **blocked** to prevent security vulnerabilities. Attempting to use them will raise a `ValueError`.

For complete examples, see `examples/integrity_verification_demo.py`.
