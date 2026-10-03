"""Policies must hold at actual wire boundaries, including plugin output."""

import io
import json

import pytest

import datason
from datason._errors import DeserializationError, SecurityError, SerializationError


@pytest.mark.parametrize("value", ["alice@example.com", ["alice@example.com"], {"contacts": ["alice@example.com"]}])
def test_every_string_leaf_is_redacted(value) -> None:
    assert "alice@example.com" not in datason.dumps(value, redact_patterns=("email",))


def test_dataframe_output_uses_field_and_pattern_policies() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"password": ["example-secret"], "email": ["alice@example.com"]})
    output = datason.dumps(frame, redact_fields=("password",), redact_patterns=("email",))
    assert "example-secret" not in output
    assert "alice@example.com" not in output
    assert json.loads(output)["__datason_type__"] == "pandas.DataFrame"


def test_numpy_output_uses_nonfinite_policy_and_size_budget() -> None:
    np = pytest.importorskip("numpy")
    output = datason.dumps(np.array([float("nan")]), include_type_hints=False)
    assert json.loads(output) == [None]
    with pytest.raises(SecurityError, match="size"):
        datason.dumps(np.arange(4), max_size=2)


def test_datetime_in_dataframe_crosses_type_conversion_boundary() -> None:
    pd = pytest.importorskip("pandas")
    output = datason.dumps(pd.DataFrame({"when": [pd.Timestamp("2026-01-01")]}))
    assert json.loads(output)["__datason_type__"] == "pandas.DataFrame"


@pytest.mark.parametrize("payload", ["[1,2,3]", '{"a":1,"b":2,"c":3}'])
def test_load_size_budget(payload: str) -> None:
    with pytest.raises(SecurityError, match="size"):
        datason.loads(payload, max_size=2)


@pytest.mark.parametrize("method", [datason.dumps, datason.loads])
def test_string_budget(method) -> None:
    with pytest.raises(SecurityError, match="String"):
        method('"abcdefgh"', max_string_length=3)


def test_parse_and_file_input_are_bounded() -> None:
    with pytest.raises(SecurityError, match="bytes"):
        datason.loads("[1,2,3]", max_input_bytes=3)
    with pytest.raises(SecurityError, match="bytes"):
        datason.load(io.StringIO("[1,2,3]"), max_input_bytes=3)


def test_plugin_is_not_executed_before_input_validation(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("plugin ran before validation")

    monkeypatch.setattr("datason._deserialize.default_registry.find_deserializer", forbidden)
    payload = '{"__datason_type__":"datetime","__datason_value__":"abcdefgh"}'
    with pytest.raises(SecurityError):
        datason.loads(payload, max_string_length=3)
    with pytest.raises(DeserializationError, match="allow_plugin_deserialization"):
        datason.loads(payload, allow_plugin_deserialization=False)


def test_reserved_keys_and_collisions_do_not_silently_change_data() -> None:
    with pytest.raises(SerializationError, match="Reserved"):
        datason.dumps({"__datason_type__": "datetime", "__datason_value__": "2026-01-01"})
    with pytest.raises(SerializationError, match="collide"):
        datason.dumps({1: "integer", "1": "string"})


def test_total_traversal_budget() -> None:
    with pytest.raises(SecurityError, match="node"):
        datason.dumps([[1], [2]], max_nodes=3)
    with pytest.raises(SecurityError, match="node"):
        datason.loads("[[1],[2]]", max_nodes=3)
