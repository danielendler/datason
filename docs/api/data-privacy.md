# 🛡️ Data Privacy & Redaction

Privacy protection and sensitive data redaction functions.

## 🎯 Overview

Data privacy features provide automatic PII detection and redaction for secure data handling.

## 📦 Functions

### RedactionEngine

::: datason.RedactionEngine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### Pre-built Redaction Engines

::: datason.create_financial_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.create_healthcare_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

::: datason.create_minimal_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## 🔗 Related Documentation

- **[Modern API](modern-api.md)** - dump_secure() function
- **[Core Functions](core-functions.md)** - Using with traditional API
