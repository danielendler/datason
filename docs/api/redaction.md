## Redaction & Privacy

Privacy protection and sensitive data redaction.

### RedactionEngine

::: datason.RedactionEngine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
      members:
        - __init__
        - process_object
        - redact_text
        - get_redaction_summary
        - get_audit_trail

### Pre-built Redaction Engines

::: datason.create_minimal_redaction_engine
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

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
