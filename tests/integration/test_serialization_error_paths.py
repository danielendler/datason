"""Serialization failures identify the field that caused them."""

import io
from dataclasses import dataclass

import pytest

import datason
from datason._errors import SerializationError


class Unsupported:
    def __repr__(self):
        return "sensitive-value-must-not-be-in-the-error"


@pytest.mark.parametrize("use_file", [False, True])
@pytest.mark.parametrize(
    ("value", "path"),
    [
        (Unsupported(), "$"),
        ({"diagnostics": {"unhandled": Unsupported()}}, "$.diagnostics.unhandled"),
        ({"tools": [{"result": Unsupported()}]}, "$.tools[0].result"),
        ({"a.b": {'quoted"key': Unsupported()}}, '$["a.b"]["quoted\\"key"]'),
        ({"values": (1, Unsupported())}, "$.values[1]"),
    ],
)
def test_unsupported_value_path(value, path, use_file):
    with pytest.raises(SerializationError) as caught:
        if use_file:
            datason.dump(value, io.StringIO())
        else:
            datason.dumps(value)
    assert caught.value.path == path
    assert f"(at {path})" in str(caught.value)
    assert str(caught.value).count("(at ") == 1
    assert "sensitive-value" not in str(caught.value)


def test_collision_identifies_the_containing_mapping():
    with pytest.raises(SerializationError, match="collide") as caught:
        datason.dumps({"tools": [{"metadata": {1: "integer", "1": "string"}}]})
    assert caught.value.path == "$.tools[0].metadata"


def test_normalized_dataclass_fields_retain_the_application_path():
    @dataclass
    class Diagnostic:
        detail: object

    with pytest.raises(SerializationError) as caught:
        datason.dumps({"checks": [Diagnostic(Unsupported())]})
    assert caught.value.path == "$.checks[0].detail"


def test_callback_output_is_located_at_its_source_field():
    class Custom:
        pass

    with pytest.raises(SerializationError, match="Reserved") as caught:
        datason.dumps({"result": Custom()}, default=lambda obj: {"__datason_type__": "custom"})
    assert caught.value.path == "$.result"


def test_a_failed_sibling_does_not_leave_an_incorrect_path():
    with pytest.raises(SerializationError) as caught:
        datason.dumps({"valid": [{"value": 1}], "invalid": Unsupported()})
    assert caught.value.path == "$.invalid"
    assert datason.loads(datason.dumps({"valid": [1, 2]})) == {"valid": [1, 2]}


def test_original_plugin_error_and_cause_are_preserved(monkeypatch):
    cause = ValueError("normalization failed")
    error = SerializationError("Cannot normalize custom data")

    def fail(*args):
        raise error from cause

    monkeypatch.setattr("datason._core.default_registry.find_serializer", fail)
    with pytest.raises(SerializationError) as caught:
        datason.dumps({"result": Unsupported()})
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert caught.value.path == "$.result"
