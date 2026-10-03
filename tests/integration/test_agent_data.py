"""Common agent data should normalize predictably at a JSON boundary."""

import base64
import dataclasses
import datetime as dt
import json
from enum import Enum, IntEnum

import pytest

import datason
from datason._errors import DeserializationError, SecurityError


class Status(Enum):
    READY = "ready"


class Count(IntEnum):
    ONE = 1


class TextStatus(str, Enum):
    READY = "ready"


@dataclasses.dataclass
class ToolResult:
    status: Status
    created: dt.datetime
    payload: bytes
    password: str = "secret"  # noqa: S105 — fixture for redaction


def test_dataclass_normalizes_fields_without_reconstructing_class():
    original = ToolResult(Status.READY, dt.datetime(2026, 10, 3), b"\x00\xff")
    restored = datason.loads(datason.dumps(original))
    assert type(restored) is dict
    assert restored == {
        "status": "ready",
        "created": original.created,
        "payload": original.payload,
        "password": "secret",
    }


@pytest.mark.parametrize("enum", [Status.READY, TextStatus.READY, Count.ONE])
def test_enums_normalize_to_values(enum):
    assert datason.loads(datason.dumps(enum)) == enum.value


@pytest.mark.parametrize("original", [b"", bytes(range(256)), bytearray(b"binary")])
def test_binary_type_and_contents_round_trip(original):
    restored = datason.loads(datason.dumps(original))
    assert type(restored) is type(original)
    assert restored == original


@pytest.mark.parametrize("value", ["not base64!", "é", 1])
def test_binary_payload_validation(value):
    payload = {"__datason_type__": "bytes", "__datason_value__": value}
    with pytest.raises(DeserializationError):
        datason.loads(json.dumps(payload))


def test_api_normalization_and_redaction():
    original = ToolResult(Status.READY, dt.datetime(2026, 10, 3), b"binary")
    encoded = datason.dumps(original, include_type_hints=False, redact_fields=("password",))
    assert json.loads(encoded) == {
        "status": "ready",
        "created": "2026-10-03T00:00:00",
        "payload": base64.b64encode(b"binary").decode(),
        "password": "[REDACTED]",
    }


def test_dataclass_cycle_uses_shared_security_check():
    @dataclasses.dataclass
    class Node:
        next: object = None

    node = Node()
    node.next = node
    with pytest.raises(SecurityError, match="Circular"):
        datason.dumps(node)


def test_pydantic_aliases_nested_values_and_explicit_validation():
    pydantic = pytest.importorskip("pydantic")
    if not hasattr(pydantic.BaseModel, "model_validate"):
        pytest.skip("Pydantic v2 validation example")

    class Result(pydantic.BaseModel):
        happened: dt.datetime = pydantic.Field(alias="timestamp")
        payload: bytes
        password: str

    original = Result(timestamp=dt.datetime(2026, 10, 3), payload=b"\xff", password="secret")  # noqa: S106
    restored = datason.loads(datason.dumps(original))
    assert type(restored) is dict
    assert Result.model_validate(restored) == original
    redacted = json.loads(datason.dumps(original, include_type_hints=False, redact_fields=("password",)))
    assert redacted["password"] == "[REDACTED]"  # noqa: S105
    assert redacted["timestamp"] == "2026-10-03T00:00:00"
