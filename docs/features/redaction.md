# 🔐 Data Privacy & Redaction

The datason redaction engine provides comprehensive data privacy protection for sensitive information in ML workflows, including field-level redaction, pattern-based redaction, and audit trail logging for compliance requirements.

## Overview

Data privacy is crucial when working with sensitive information in machine learning and data science workflows. The redaction engine helps you:

- **Protect PII**: Automatically detect and redact personally identifiable information
- **Ensure Compliance**: Meet GDPR, HIPAA, PCI-DSS, and other regulatory requirements
- **Audit Trails**: Maintain complete logs of redaction operations for compliance
- **Domain-Specific**: Pre-built configurations for financial, healthcare, and general use

## Quick Start

```python
import datason as ds

# Simple redaction
sensitive_data = {
    "customer_email": "john.doe@example.com",
    "credit_card": "4532-1234-5678-9012",
    "password": "secret123",
    "data": [1, 2, 3, 4, 5]
}

# Create redaction engine
engine = ds.create_minimal_redaction_engine()
redacted = engine.process_object(sensitive_data)

print(redacted)
# {
#     "customer_email": "<REDACTED>",
#     "credit_card": "<REDACTED>",
#     "password": "<REDACTED>",
#     "data": [1, 2, 3, 4, 5]
# }
```

## Redaction Engines

### Pre-built Engines

datason provides three pre-configured redaction engines for common use cases:

=== "Minimal Protection"

    ```python
    # Basic privacy protection
    engine = ds.create_minimal_redaction_engine()
    ```

    **Protects:**
    - Passwords, secrets, keys, tokens
    - Email addresses

    **Use Cases:**
    - Development environments
    - Basic privacy requirements
    - Lightweight applications

=== "Financial Data"

    ```python
    # Financial industry compliance
    engine = ds.create_financial_redaction_engine()
    ```

    **Protects:**
    - Credit card numbers
    - Social Security Numbers (SSN)
    - Tax IDs
    - Account numbers
    - Routing numbers
    - CVV codes
    - PINs

    **Features:**
    - Audit trail enabled
    - Redaction summary
    - Large object detection (5MB threshold)

    **Use Cases:**
    - Banking applications
    - Payment processing
    - Financial analytics

=== "Healthcare Data"

    ```python
    # Healthcare compliance (HIPAA)
    engine = ds.create_healthcare_redaction_engine()
    ```

    **Protects:**
    - Patient IDs
    - Medical record numbers
    - Personal information (names, addresses, phone)
    - Dates of birth
    - Diagnosis information

    **Features:**
    - Full audit trail
    - Redaction summary
    - Large object protection

    **Use Cases:**
    - Medical research
    - Healthcare analytics
    - Patient data processing

## Custom Redaction Engine

For specific requirements, create a custom redaction engine:

```python
from datason import RedactionEngine

# Create custom engine
engine = RedactionEngine(
    # Field patterns to redact
    redact_fields=[
        "*.password",           # Any field named 'password'
        "*.secret",             # Any field named 'secret'  
        "user.*.email",         # Email in user objects
        "*.ssn",                # Social Security Numbers
        "config.api_key",       # Specific field path
    ],

    # Regex patterns for content redaction
    redact_patterns=[
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit cards
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN format
    ],

    # Large object redaction
    redact_large_objects=True,
    large_object_threshold=1024 * 1024,  # 1MB

    # Customization
    redaction_replacement="[CONFIDENTIAL]",

    # Compliance features
    include_redaction_summary=True,
    audit_trail=True,
)
```

## Field Pattern Matching

Field patterns support wildcards for flexible matching:

```python
patterns = [
    "password",           # Exact match: field named 'password'
    "*.password",         # Any field ending with 'password'
    "user.*.email",       # Email field in any user object
    "config.api.*",       # Any API-related config field
    "*.secret*",          # Any field containing 'secret'
]
```

## Pattern-based Redaction

Automatically detect sensitive content using regex patterns:

```python
# Common sensitive patterns
patterns = [
    # Credit card numbers (various formats)
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",

    # US Social Security Numbers
    r"\b\d{3}-\d{2}-\d{4}\b",

    # Email addresses
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",

    # Phone numbers (US format)
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",

    # IPv4 addresses
    r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
]

engine = RedactionEngine(redact_patterns=patterns)

text = "Contact John at john.doe@company.com or call 555-123-4567"
redacted_text, was_redacted = engine.redact_text(text)
print(redacted_text)
# "Contact John at <REDACTED> or call <REDACTED>"
```

## Large Object Protection

Protect against accidentally serializing large data objects:

```python
import numpy as np

engine = RedactionEngine(
    redact_large_objects=True,
    large_object_threshold=1024 * 1024,  # 1MB threshold
)

data = {
    "large_array": np.random.random((1000, 1000)),  # ~8MB array
    "small_data": [1, 2, 3, 4, 5],
}

redacted = engine.process_object(data)
print(redacted["large_array"])
# "<LARGE_OBJECT_REDACTED: ndarray, ~8,000,000 bytes>"
```

## Audit Trail & Compliance

Enable comprehensive logging for compliance requirements:

```python
engine = RedactionEngine(
    redact_fields=["*.password", "*.ssn"],
    audit_trail=True,
    include_redaction_summary=True,
)

data = {
    "user": {
        "name": "John Doe",
        "password": "secret123",
        "ssn": "123-45-6789"
    }
}

redacted = engine.process_object(data)

# Get redaction summary
summary = engine.get_redaction_summary()
print(summary)
# {
#     "redaction_summary": {
#         "fields_redacted": ["user.password", "user.ssn"],
#         "patterns_matched": [],
#         "large_objects_redacted": [],
#         "total_redactions": 2,
#         "redaction_timestamp": "2024-01-15T10:30:00.000000+00:00"
#     }
# }

# Get audit trail
audit_trail = engine.get_audit_trail()
for entry in audit_trail:
    print(f"{entry['timestamp']}: Redacted {entry['target']} ({entry['redaction_type']})")
```

## Integration with Serialization

Combine redaction with datason serialization:

```python
import datason as ds
import pandas as pd

# Sensitive data
sensitive_data = {
    "customers": pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "ssn": ["123-45-6789", "987-65-4321", "555-12-3456"]
    }),
    "api_config": {
        "api_key": "secret-key-12345",
        "endpoint": "https://api.example.com"
    }
}

# Create redaction engine
engine = ds.create_financial_redaction_engine()

# Redact sensitive information
redacted_data = engine.process_object(sensitive_data)

# Serialize the redacted data
config = ds.get_api_config()
json_safe = ds.serialize(redacted_data, config=config)
```

## Advanced Examples

### Custom Domain Patterns

```python
# Redaction for specific domains
class CompanyRedactionEngine(RedactionEngine):
    def __init__(self):
        super().__init__(
            redact_fields=[
                "*.employee_id",
                "*.salary",
                "*.performance_rating",
                "hr.*.personal_info",
                "finance.*.budget",
            ],
            redact_patterns=[
                r"\b[Ee]mp-\d{6}\b",  # Employee IDs
                r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b",  # Currency amounts
            ],
            audit_trail=True,
        )

engine = CompanyRedactionEngine()
```

### Conditional Redaction

```python
class ConditionalRedactionEngine(RedactionEngine):
    def __init__(self, environment="production"):
        # Only redact in production
        if environment == "production":
            super().__init__(
                redact_fields=["*.password", "*.api_key"],
                audit_trail=True,
            )
        else:
            # Development - minimal redaction
            super().__init__(
                redact_fields=["*.password"],
                audit_trail=False,
            )

# Usage
prod_engine = ConditionalRedactionEngine("production")
dev_engine = ConditionalRedactionEngine("development")
```

## Best Practices

### 1. **Start with Pre-built Engines**
Use domain-specific engines as starting points:

```python
# Good: Start with proven configuration
engine = ds.create_financial_redaction_engine()

# Then customize if needed
engine.redact_fields.extend(["*.internal_id", "*.customer_segment"])
```

### 2. **Test Redaction Patterns**
Always test your patterns with sample data:

```python
# Test patterns before deployment
test_data = {
    "test_email": "test@example.com",
    "test_ssn": "123-45-6789",
    "test_credit": "4532-1234-5678-9012"
}

redacted = engine.process_object(test_data)
print("Redaction test:", redacted)
```

### 3. **Monitor Redaction Performance**
Large objects can impact performance:

```python
import time

start_time = time.time()
redacted = engine.process_object(large_data)
redaction_time = time.time() - start_time

summary = engine.get_redaction_summary()
print(f"Redacted {summary['redaction_summary']['total_redactions']} items in {redaction_time:.2f}s")
```

### 4. **Compliance Documentation**
Maintain documentation for compliance:

```python
# Document your redaction strategy
redaction_policy = {
    "description": "Customer data redaction for ML training",
    "fields_protected": ["email", "ssn", "credit_card"],
    "compliance_standards": ["GDPR", "CCPA"],
    "retention_policy": "Audit logs kept for 7 years",
    "engine_config": "create_financial_redaction_engine()"
}
```

## Compliance Standards

The redaction engine helps meet various compliance requirements:

| Standard | Focus Area | Supported Features |
|----------|------------|-------------------|
| **GDPR** | Personal data protection | Field redaction, audit trails, data minimization |
| **HIPAA** | Healthcare data | PHI detection, audit logging, access controls |
| **PCI-DSS** | Payment card data | Credit card detection, secure handling |
| **CCPA** | California privacy | Personal information redaction |
| **SOX** | Financial reporting | Data integrity, audit trails |

## Performance Considerations

- **Pattern Complexity**: Complex regex patterns can impact performance
- **Large Objects**: Enable large object detection for memory protection  
- **Audit Trails**: Add minimal overhead but provide compliance value
- **Field Patterns**: Wildcard patterns are efficient for nested structures

## Error Handling

```python
try:
    redacted = engine.process_object(data)
except Exception as e:
    print(f"Redaction error: {e}")
    # Fallback: redact everything or fail safe
    redacted = {"error": "Data redacted due to processing error"}
```

## API Reference

See the [API Overview](../api/index.md) for complete function documentation and parameters.
