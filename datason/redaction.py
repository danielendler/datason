"""Redaction and Privacy Control Module for datason v0.5.5.

This module provides comprehensive redaction capabilities for sensitive data
in ML workflows, including field-level redaction, pattern-based redaction,
and audit trail logging for compliance requirements.
"""

import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union


@dataclass
class RedactionSummary:
    """Summary of redaction operations performed."""

    fields_redacted: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)
    large_objects_redacted: List[str] = field(default_factory=list)
    total_redactions: int = 0
    redaction_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RedactionAuditEntry:
    """Audit trail entry for compliance tracking."""

    timestamp: str
    redaction_type: str  # 'field', 'pattern', 'size'
    target: str  # field path or pattern description
    original_type: str
    replacement: str
    context: Optional[str] = None


class RedactionEngine:
    """Core redaction engine for sensitive data protection."""

    # Common sensitive field patterns
    DEFAULT_SENSITIVE_PATTERNS = [
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card numbers
        r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone numbers
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",  # Credit card patterns
    ]

    def __init__(
        self,
        redact_fields: Optional[List[str]] = None,
        redact_patterns: Optional[List[str]] = None,
        redact_large_objects: bool = False,
        large_object_threshold: int = 10 * 1024 * 1024,  # 10MB
        redaction_replacement: str = "<REDACTED>",
        include_redaction_summary: bool = False,
        audit_trail: bool = False,
    ):
        """Initialize the redaction engine.

        Args:
            redact_fields: List of field patterns to redact (e.g., ["password", "*.secret"])
            redact_patterns: List of regex patterns to redact
            redact_large_objects: Whether to redact objects larger than threshold
            large_object_threshold: Size threshold for large object redaction (bytes)
            redaction_replacement: Replacement text for redacted content
            include_redaction_summary: Whether to include redaction summary
            audit_trail: Whether to maintain audit trail for compliance
        """
        self.redact_fields = redact_fields or []
        self.redact_patterns = redact_patterns or []
        self.redact_large_objects = redact_large_objects
        self.large_object_threshold = large_object_threshold
        self.redaction_replacement = redaction_replacement
        self.include_redaction_summary = include_redaction_summary
        self.audit_trail = audit_trail

        # Compile regex patterns for performance
        self._compiled_patterns: List[Pattern] = []
        for pattern in self.redact_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                warnings.warn(f"Invalid regex pattern '{pattern}': {e}", stacklevel=2)

        # Audit trail storage
        self._audit_entries: List[RedactionAuditEntry] = []

        # Redaction summary
        self._summary = RedactionSummary()

    def _should_redact_field(self, field_path: str) -> bool:
        """Check if a field should be redacted based on patterns.

        Args:
            field_path: Dot-separated path to the field

        Returns:
            True if field should be redacted
        """
        return any(self._match_field_pattern(field_path, pattern) for pattern in self.redact_fields)

    def _match_field_pattern(self, field_path: str, pattern: str) -> bool:
        """Match field path against pattern with wildcard support.

        Args:
            field_path: The field path to check
            pattern: Pattern with optional wildcards (*, ?)

        Returns:
            True if pattern matches field path
        """
        # Convert glob pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        regex_pattern = f"^{regex_pattern}$"

        try:
            return bool(re.match(regex_pattern, field_path, re.IGNORECASE))
        except re.error:
            # Fallback to simple string matching
            return pattern.lower() in field_path.lower()

    def redact_text(self, text: str, context: str = "") -> Tuple[str, bool]:
        """Redact sensitive patterns from text.

        Args:
            text: Text to redact
            context: Context for audit trail

        Returns:
            Tuple of (redacted_text, was_redacted)
        """
        if not isinstance(text, str) or not self._compiled_patterns:
            return text, False

        redacted_text = text
        was_redacted = False

        for pattern in self._compiled_patterns:
            matches = pattern.findall(redacted_text)
            if matches:
                redacted_text = pattern.sub(self.redaction_replacement, redacted_text)
                was_redacted = True

                if self.audit_trail:
                    for match in matches:
                        self._add_audit_entry(
                            redaction_type="pattern",
                            target=pattern.pattern,
                            original_type="string",
                            replacement=self.redaction_replacement,
                            context=context,
                        )

        if was_redacted:
            self._summary.patterns_matched.extend([p.pattern for p in self._compiled_patterns])
            self._summary.total_redactions += 1

        return redacted_text, was_redacted

    def should_redact_large_object(self, obj: Any) -> bool:
        """Check if object should be redacted due to size.

        Args:
            obj: Object to check

        Returns:
            True if object should be redacted due to size
        """
        if not self.redact_large_objects:
            return False

        try:
            # Estimate object size
            size = sys.getsizeof(obj)

            # For collections, estimate recursively (with limit)
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                # Sample first few items
                sample_size = min(10, len(obj))
                avg_item_size = sum(sys.getsizeof(obj[i]) for i in range(sample_size)) / sample_size
                size += int(avg_item_size * (len(obj) - sample_size))
            elif isinstance(obj, dict) and len(obj) > 0:
                # Sample first few items
                items = list(obj.items())
                sample_size = min(10, len(items))
                avg_item_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in items[:sample_size]) / sample_size
                size += int(avg_item_size * (len(items) - sample_size))

            return size > self.large_object_threshold

        except (TypeError, AttributeError):
            # Can't determine size, assume it's not large
            return False

    def redact_large_object(self, obj: Any, field_path: str = "") -> Any:
        """Redact a large object.

        Args:
            obj: Object to redact
            field_path: Field path for audit trail

        Returns:
            Redacted representation
        """
        size = sys.getsizeof(obj)
        obj_type = type(obj).__name__

        redacted = f"<LARGE_OBJECT_REDACTED: {obj_type}, ~{size:,} bytes>"

        if self.audit_trail:
            self._add_audit_entry(
                redaction_type="size",
                target=field_path or f"large_{obj_type}",
                original_type=obj_type,
                replacement=redacted,
                context=f"Object size: {size:,} bytes",
            )

        self._summary.large_objects_redacted.append(field_path or f"<{obj_type}>")
        self._summary.total_redactions += 1

        return redacted

    def redact_field_value(self, value: Any, field_path: str) -> Any:
        """Redact a field value.

        Args:
            value: Value to redact
            field_path: Field path for audit trail

        Returns:
            Redacted value
        """
        if self.audit_trail:
            self._add_audit_entry(
                redaction_type="field",
                target=field_path,
                original_type=type(value).__name__,
                replacement=self.redaction_replacement,
                context="Field pattern match",
            )

        self._summary.fields_redacted.append(field_path)
        self._summary.total_redactions += 1

        return self.redaction_replacement

    def process_object(self, obj: Any, field_path: str = "", _visited: Optional[Set[int]] = None) -> Any:
        """Process an object for redaction.

        Args:
            obj: Object to process
            field_path: Current field path
            _visited: Set of visited object IDs (for circular reference detection)

        Returns:
            Processed object with redactions applied
        """
        if _visited is None:
            _visited = set()

        # Circular reference detection
        obj_id = id(obj)
        if obj_id in _visited:
            return "<CIRCULAR_REFERENCE>"

        # Check for large object redaction first
        if self.should_redact_large_object(obj):
            return self.redact_large_object(obj, field_path)

        # For mutable objects, track in visited set
        if isinstance(obj, (dict, list, set)):
            _visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                return self._process_dict(obj, field_path, _visited)
            elif isinstance(obj, (list, tuple)):
                return self._process_list(obj, field_path, _visited)
            elif isinstance(obj, str):
                # Apply pattern redaction to strings
                redacted_text, _ = self.redact_text(obj, field_path)
                return redacted_text
            else:
                # For other types, return as-is
                return obj
        finally:
            # Clean up visited set
            if isinstance(obj, (dict, list, set)):
                _visited.discard(obj_id)

    def _process_dict(self, obj: dict, field_path: str, _visited: Set[int]) -> dict:
        """Process a dictionary for redaction."""
        result = {}

        for key, value in obj.items():
            current_path = f"{field_path}.{key}" if field_path else str(key)

            # Check if this field should be redacted
            if self._should_redact_field(current_path):
                result[key] = self.redact_field_value(value, current_path)
            else:
                # Recursively process the value
                result[key] = self.process_object(value, current_path, _visited)

        return result

    def _process_list(self, obj: Union[list, tuple], field_path: str, _visited: Set[int]) -> Union[list, tuple]:
        """Process a list or tuple for redaction."""
        result = []

        for i, item in enumerate(obj):
            current_path = f"{field_path}[{i}]" if field_path else f"[{i}]"
            processed_item = self.process_object(item, current_path, _visited)
            result.append(processed_item)

        # Return same type as input
        return type(obj)(result)

    def _add_audit_entry(
        self,
        redaction_type: str,
        target: str,
        original_type: str,
        replacement: str,
        context: Optional[str] = None,
    ) -> None:
        """Add an entry to the audit trail."""
        entry = RedactionAuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            redaction_type=redaction_type,
            target=target,
            original_type=original_type,
            replacement=replacement,
            context=context,
        )
        self._audit_entries.append(entry)

    def get_redaction_summary(self) -> Optional[Dict[str, Any]]:
        """Get redaction summary if enabled."""
        if not self.include_redaction_summary:
            return None

        return {
            "redaction_summary": {
                "fields_redacted": list(set(self._summary.fields_redacted)),
                "patterns_matched": list(set(self._summary.patterns_matched)),
                "large_objects_redacted": list(set(self._summary.large_objects_redacted)),
                "total_redactions": self._summary.total_redactions,
                "redaction_timestamp": self._summary.redaction_timestamp,
            }
        }

    def get_audit_trail(self) -> Optional[List[Dict[str, Any]]]:
        """Get audit trail if enabled."""
        if not self.audit_trail:
            return None

        return [
            {
                "timestamp": entry.timestamp,
                "redaction_type": entry.redaction_type,
                "target": entry.target,
                "original_type": entry.original_type,
                "replacement": entry.replacement,
                "context": entry.context,
            }
            for entry in self._audit_entries
        ]


def create_financial_redaction_engine() -> RedactionEngine:
    """Create a redaction engine optimized for financial data."""
    return RedactionEngine(
        redact_fields=[
            "*.password",
            "*.secret",
            "*.key",
            "*.token",
            "*.ssn",
            "*.social_security",
            "*.tax_id",
            "*.account_number",
            "*.routing_number",
            "*.credit_card",
            "*.card_number",
            "*.cvv",
            "*.pin",
        ],
        redact_patterns=[
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit cards
            r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
            r"\b\d{9}\b",  # US Tax ID
            r"\b\d{10,12}\b",  # Account numbers
        ],
        redact_large_objects=True,
        large_object_threshold=5 * 1024 * 1024,  # 5MB for financial data
        include_redaction_summary=True,
        audit_trail=True,
    )


def create_healthcare_redaction_engine() -> RedactionEngine:
    """Create a redaction engine optimized for healthcare data."""
    return RedactionEngine(
        redact_fields=[
            "*.patient_id",
            "*.medical_record",
            "*.ssn",
            "*.phone",
            "*.email",
            "*.address",
            "*.name",
            "*.dob",
            "*.birth_date",
            "*.diagnosis",
        ],
        redact_patterns=[
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        ],
        redact_large_objects=True,
        include_redaction_summary=True,
        audit_trail=True,
    )


def create_minimal_redaction_engine() -> RedactionEngine:
    """Create a minimal redaction engine for basic privacy protection."""
    return RedactionEngine(
        redact_fields=["*.password", "*.secret", "*.key", "*.token"],
        redact_patterns=[
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
        ],
        redact_large_objects=False,
        include_redaction_summary=False,
        audit_trail=False,
    )
