"""JSON callback compatibility without bypassing serialization policies."""

from __future__ import annotations

import datetime as dt
import io
import json
from typing import Any

import pytest

import datason
from datason._errors import SecurityError


class Unknown:
    """An object without a registered type plugin."""


class UnknownEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Unknown):
            return {"custom": True}
        return super().default(o)


def test_nested_default_matches_json() -> None:
    value = {"items": [Unknown()]}
    handler = lambda obj: {"custom": True}  # noqa: E731
    assert datason.dumps(value, default=handler) == json.dumps(value, default=handler)


def test_dump_uses_default() -> None:
    stream = io.StringIO()
    datason.dump(Unknown(), stream, default=lambda obj: "custom")
    assert stream.getvalue() == '"custom"'


def test_custom_encoder_default_matches_json() -> None:
    value = {"item": Unknown()}
    assert datason.dumps(value, cls=UnknownEncoder) == json.dumps(value, cls=UnknownEncoder)


def test_dump_uses_custom_encoder() -> None:
    stream = io.StringIO()
    datason.dump(Unknown(), stream, cls=UnknownEncoder)
    assert json.loads(stream.getvalue()) == {"custom": True}


def test_explicit_default_overrides_encoder_default() -> None:
    assert datason.dumps(Unknown(), cls=UnknownEncoder, default=lambda obj: "override") == '"override"'


@pytest.mark.parametrize("write_file", [False, True])
def test_stateful_encoder_is_instantiated_once(write_file: bool) -> None:
    instances: list[UnknownEncoder] = []

    class StatefulEncoder(UnknownEncoder):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            instances.append(self)

    if write_file:
        stream = io.StringIO()
        datason.dump(Unknown(), stream, cls=StatefulEncoder)
        result = stream.getvalue()
    else:
        result = datason.dumps(Unknown(), cls=StatefulEncoder)
    assert json.loads(result) == {"custom": True}
    assert len(instances) == 1


def test_default_is_used_before_string_fallback() -> None:
    assert datason.dumps(Unknown(), default=lambda obj: "custom", fallback_to_string=True) == '"custom"'


def test_known_types_keep_their_plugin() -> None:
    def unexpected(obj: Any) -> Any:
        raise AssertionError("Known types must use their registered plugin")

    value = dt.datetime(2024, 6, 15)
    assert datason.loads(datason.dumps(value, default=unexpected)) == value


@pytest.mark.parametrize("options", [{"default": lambda obj: {"password": "secret"}}, {"cls": UnknownEncoder}])
def test_callback_output_is_redacted(options: dict[str, Any]) -> None:
    value = json.loads(datason.dumps(Unknown(), redact_fields=("password", "custom"), **options))
    assert set(value.values()) == {"[REDACTED]"}


def test_default_output_obeys_string_limit() -> None:
    with pytest.raises(SecurityError, match="String length"):
        datason.dumps(Unknown(), default=lambda obj: "too long", max_string_length=3)


def test_default_output_obeys_collection_limit() -> None:
    with pytest.raises(SecurityError, match="Sequence size"):
        datason.dumps(Unknown(), default=lambda obj: [1, 2], max_size=1)


def test_circular_default_output_is_rejected() -> None:
    with pytest.raises(SecurityError, match="Circular"):
        datason.dumps(Unknown(), default=lambda obj: obj)


def test_default_replacement_chain_is_bounded() -> None:
    with pytest.raises(SecurityError, match="depth"):
        datason.dumps(Unknown(), default=lambda obj: Unknown(), max_depth=3)


def test_check_circular_false_preserves_datason_limits() -> None:
    value: list[Any] = []
    value.append(value)
    with pytest.raises(SecurityError, match="Circular"):
        datason.dumps(value, check_circular=False)
    assert datason.dumps({"a": [1]}, check_circular=False) == json.dumps({"a": [1]}, check_circular=False)


def test_skipkeys_does_not_serialize_discarded_values() -> None:
    value = {"keep": 1, (1, 2): Unknown()}
    assert datason.dumps(value, skipkeys=True) == json.dumps(value, skipkeys=True)


def test_skipkeys_propagates_into_nested_dicts() -> None:
    value = {"nested": {"keep": 1, (1, 2): Unknown()}}
    assert datason.dumps(value, skipkeys=True) == json.dumps(value, skipkeys=True)


@pytest.mark.parametrize("name", ["dumps", "dump", "loads", "load"])
def test_unknown_kwargs_name_the_public_function(name: str) -> None:
    args: tuple[Any, ...] = {
        "dumps": ({},),
        "dump": ({}, io.StringIO()),
        "loads": ("{}",),
        "load": (io.StringIO("{}"),),
    }[name]
    with pytest.raises(TypeError, match=rf"{name}\(\) got an unexpected keyword argument 'nope'"):
        getattr(datason, name)(*args, nope=True)


def test_load_parser_hooks_still_work() -> None:
    result = datason.load(io.StringIO('{"value": 1.5}'), parse_float=str, object_hook=lambda value: list(value.items()))
    assert result == [("value", "1.5")]
