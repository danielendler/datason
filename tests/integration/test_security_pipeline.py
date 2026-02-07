"""Integration tests for security features with plugin types.

Tests redaction, integrity, and security limits when used
together with typed plugin data.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

np = pytest.importorskip("numpy")
import datason
from datason._errors import SecurityError
from datason.security.integrity import verify_integrity, wrap_with_integrity


class TestRedactionWithPluginTypes:
    """Redaction applied to dicts containing plugin-typed values."""

    def test_redact_field_containing_numpy_array(self) -> None:
        data = {"weights": np.array([0.1, 0.9]), "name": "model"}
        s = datason.dumps(data, redact_fields=("weights",))
        parsed = json.loads(s)
        assert parsed["weights"] == "[REDACTED]"
        assert parsed["name"] == "model"

    def test_redact_field_containing_datetime(self) -> None:
        data = {"secret_date": dt.datetime(2024, 1, 1), "public": "ok"}  # noqa: DTZ001
        s = datason.dumps(data, redact_fields=("secret",))
        parsed = json.loads(s)
        # "secret_date" contains "secret" as substring → redacted
        assert parsed["secret_date"] == "[REDACTED]"  # noqa: S105
        assert parsed["public"] == "ok"

    def test_pattern_redaction_in_nested_dict(self) -> None:
        data = {
            "user": {
                "name": "Alice",
                "email": "alice@example.com",
                "score": np.float64(95.5),
            }
        }
        s = datason.dumps(data, redact_patterns=(r"\S+@\S+\.\S+",))
        assert "alice@example.com" not in s
        assert "Alice" in s

    def test_redaction_preserves_non_redacted_plugin_types(self) -> None:
        data = {
            "password": "secret123",
            "timestamp": dt.datetime(2024, 6, 15),  # noqa: DTZ001
            "values": np.array([1.0, 2.0]),
        }
        s = datason.dumps(data, redact_fields=("password",), include_type_hints=True)
        result = datason.loads(s)
        assert result["password"] == "[REDACTED]"  # noqa: S105
        assert isinstance(result["timestamp"], dt.datetime)
        assert isinstance(result["values"], np.ndarray)


class TestIntegrityWithPluginTypes:
    """Integrity wrapping/verification with typed data."""

    def test_wrap_verify_with_datetime_data(self) -> None:
        data = {"ts": dt.datetime(2024, 1, 1), "value": 42}  # noqa: DTZ001
        serialized = datason.dumps(data, include_type_hints=True)
        envelope = wrap_with_integrity(serialized)

        is_valid, payload = verify_integrity(envelope)
        assert is_valid
        result = datason.loads(payload)
        assert isinstance(result["ts"], dt.datetime)

    def test_wrap_verify_with_numpy_data(self) -> None:
        data = {"weights": np.array([0.1, 0.9, 0.5])}
        serialized = datason.dumps(data, include_type_hints=True)
        envelope = wrap_with_integrity(serialized)

        is_valid, payload = verify_integrity(envelope)
        assert is_valid
        result = datason.loads(payload)
        assert isinstance(result["weights"], np.ndarray)

    def test_wrap_verify_with_mixed_types(self) -> None:
        import uuid

        data = {
            "id": uuid.uuid4(),
            "ts": dt.datetime(2024, 6, 15),  # noqa: DTZ001
            "arr": np.array([1, 2, 3]),
            "name": "test",
        }
        serialized = datason.dumps(data, include_type_hints=True)
        envelope = wrap_with_integrity(serialized)

        is_valid, payload = verify_integrity(envelope)
        assert is_valid
        result = datason.loads(payload)
        assert isinstance(result["id"], uuid.UUID)
        assert isinstance(result["ts"], dt.datetime)
        assert isinstance(result["arr"], np.ndarray)

    def test_hmac_with_mixed_types(self) -> None:
        data = {"ts": dt.datetime(2024, 1, 1), "arr": np.array([1.0])}  # noqa: DTZ001
        serialized = datason.dumps(data, include_type_hints=True)
        key = "my-secret-key"

        envelope = wrap_with_integrity(serialized, key=key)
        is_valid, payload = verify_integrity(envelope, key=key)
        assert is_valid
        _ = payload  # accessed to avoid unused-variable warning

        # Wrong key should fail
        is_valid_wrong, _ = verify_integrity(envelope, key="wrong-key")
        assert not is_valid_wrong

    def test_tampered_payload_detected(self) -> None:
        data = {"value": 42}
        serialized = datason.dumps(data)
        envelope = wrap_with_integrity(serialized)

        # Tamper with the envelope
        parsed = json.loads(envelope)
        parsed["__datason_payload__"]["value"] = 999
        tampered = json.dumps(parsed)

        is_valid, _ = verify_integrity(tampered)
        assert not is_valid


class TestSecurityLimitsWithPlugins:
    """Security limits enforced during recursive plugin processing."""

    def test_depth_limit_with_nested_plugin_types(self) -> None:
        # Build deeply nested structure with datetime leaves
        data: dict[str, Any] = {"level": 0}
        current: dict[str, Any] = data
        for i in range(1, 55):
            current["child"] = {"level": i}
            current = current["child"]
        current["ts"] = dt.datetime(2024, 1, 1)  # noqa: DTZ001

        with pytest.raises(SecurityError, match="depth"):
            datason.dumps(data, max_depth=50)

    def test_size_limit_with_large_dict_of_timestamps(self) -> None:
        data = {f"ts_{i}": dt.datetime(2024, 1, 1) for i in range(101)}  # noqa: DTZ001
        with pytest.raises(SecurityError, match="size"):
            datason.dumps(data, max_size=100)

    def test_circular_ref_in_dict_with_plugin_values(self) -> None:
        data: dict[str, Any] = {"arr": np.array([1.0, 2.0])}
        data["self"] = data  # Circular reference

        with pytest.raises(SecurityError, match="[Cc]ircular"):
            datason.dumps(data)


class TestRedactionAndIntegrityTogether:
    """Full pipeline: serialize → redact → wrap → verify → deserialize."""

    def test_full_pipeline(self) -> None:
        data = {
            "username": "alice",
            "password": "secret123",
            "email": "alice@example.com",
            "scores": np.array([95.0, 88.0, 92.0]),
        }

        # Serialize with redaction
        s = datason.dumps(
            data,
            redact_fields=("password",),
            redact_patterns=(r"\S+@\S+\.\S+",),
            include_type_hints=True,
        )

        # Verify redaction applied
        assert "secret123" not in s
        assert "alice@example.com" not in s

        # Wrap with integrity
        envelope = wrap_with_integrity(s, key="my-key")

        # Verify and unwrap
        is_valid, payload = verify_integrity(envelope, key="my-key")
        assert is_valid

        # Deserialize
        result = datason.loads(payload)
        assert result["password"] == "[REDACTED]"  # noqa: S105
        assert isinstance(result["scores"], np.ndarray)
        assert result["username"] == "alice"
