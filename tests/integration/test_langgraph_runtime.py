"""Actual framework runtime records, fan-out, dynamic interrupts and hydration."""

import datetime as dt
import json
import operator
from typing import Annotated, TypedDict

import pytest

from datason._errors import DeserializationError, SerializationError
from datason._registry import default_registry
from datason.integrations.langgraph import DatasonSerializer

pytest.importorskip("langgraph")
sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, Send, interrupt


class ApprovalState(TypedDict):
    approved: bool


def approval_builder():
    def approve(state):
        return {"approved": interrupt({"question": "Continue?"})}

    builder = StateGraph(ApprovalState)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    return builder


def test_dynamic_interrupt_survives_reopen(tmp_path):
    db, config = str(tmp_path / "approval.sqlite"), {"configurable": {"thread_id": "approval"}}
    builder = approval_builder()
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        result = builder.compile(checkpointer=saver).invoke({"approved": False}, config)
        identifier = result["__interrupt__"][0].id
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        graph = builder.compile(checkpointer=saver)
        pending = graph.get_state(config).tasks[0].interrupts[0]
        assert isinstance(pending, Interrupt)
        assert pending.id == identifier
        assert pending.value == {"question": "Continue?"}
        assert graph.invoke(Command(resume={identifier: True}), config)["approved"] is True


class FanoutState(TypedDict):
    numbers: list[int]
    results: Annotated[list[int], operator.add]


def test_send_fanout_survives_reopen(tmp_path):
    builder = StateGraph(FanoutState)
    builder.add_node("double", lambda state: {"results": [state["number"] * 2]})
    builder.add_conditional_edges(START, lambda state: [Send("double", {"number": n}) for n in state["numbers"]])
    builder.add_edge("double", END)
    db, config = str(tmp_path / "fanout.sqlite"), {"configurable": {"thread_id": "fanout"}}
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        paused = builder.compile(checkpointer=saver, interrupt_before=["double"])
        paused.invoke({"numbers": [1, 2], "results": []}, config)
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        resumed = builder.compile(checkpointer=saver)
        assert sorted(resumed.invoke(None, config)["results"]) == [2, 4]


class VersionedState(TypedDict):
    schema_version: int
    record: dict
    result: int


def test_application_schema_upgrade_and_explicit_hydration(tmp_path):
    pydantic = pytest.importorskip("pydantic")

    class Reading(pydantic.BaseModel):
        observed: dt.datetime
        value: int

    def consume(state):
        version = state["schema_version"]
        if version != 1:
            raise ValueError("Unsupported application schema version")
        old = state["record"]
        hydrated = Reading.model_validate({"observed": old["observed"], "value": old["score"]})
        return {"schema_version": 2, "record": hydrated.model_dump(), "result": hydrated.value * 2}

    builder = StateGraph(VersionedState)
    builder.add_node("consume", consume)
    builder.add_edge(START, "consume")
    builder.add_edge("consume", END)
    initial = {"schema_version": 1, "record": {"observed": dt.datetime(2026, 10, 4), "score": 3}, "result": 0}
    db, config = str(tmp_path / "schema.sqlite"), {"configurable": {"thread_id": "schema"}}
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        builder.compile(checkpointer=saver, interrupt_before=["consume"]).invoke(initial, config)
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        result = builder.compile(checkpointer=saver).invoke(None, config)
        assert result["schema_version"] == 2
        assert result["result"] == 6
        assert Reading.model_validate(result["record"]).observed == initial["record"]["observed"]


def test_adapter_rejects_actual_native_checkpoint():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    native = JsonPlusSerializer().dumps_typed({"data": [1, 2]})
    with pytest.raises(DeserializationError, match="Unsupported checkpoint format"):
        DatasonSerializer().loads_typed(native)


def test_adapter_registration_is_idempotent():
    DatasonSerializer()
    count = default_registry.plugin_count
    DatasonSerializer()
    assert default_registry.plugin_count == count


@pytest.mark.parametrize(
    "tag,value",
    [
        ("langgraph.Interrupt.v1", {"id": 1, "value": True}),
        ("langgraph.Send.v1", {"node": 1, "arg": {}}),
        ("langgraph.Interrupt.v1", []),
        ("langgraph.Interrupt.v1", {"id": "valid", "value": True, "response_schema": "invalid"}),
        ("langgraph.Interrupt.v1", {"id": "valid"}),
        ("langgraph.Send.v1", {"node": "target"}),
    ],
)
def test_runtime_record_validation(tag, value):
    payload = json.dumps({"__datason_type__": tag, "__datason_value__": value}).encode()
    with pytest.raises(DeserializationError):
        DatasonSerializer().loads_typed(("datason-json-v1", payload))


def test_checkpoint_written_by_older_framework_resumes(tmp_path):
    import sqlite3
    from pathlib import Path

    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "langgraph-1.0.0-checkpoint.json").read_text())
    db = str(tmp_path / "older.sqlite")
    with sqlite3.connect(db) as connection:
        connection.executescript(fixture["sqlite_dump"])  # Owned repository fixture, not external SQL.
    config = {"configurable": {"thread_id": fixture["thread_id"]}}
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        graph = approval_builder().compile(checkpointer=saver)
        pending = graph.get_state(config).tasks[0].interrupts[0]
        assert pending.id == fixture["interrupt_id"]
        assert graph.invoke(Command(resume={pending.id: True}), config)["approved"] is True


def test_response_schema_survives_runtime_record_when_supported():
    import inspect

    if "response_schema" not in inspect.signature(Interrupt).parameters:
        payload = json.dumps(
            {
                "__datason_type__": "langgraph.Interrupt.v1",
                "__datason_value__": {
                    "id": "schema-interrupt",
                    "value": True,
                    "response_schema": {"type": "boolean"},
                },
            }
        ).encode()
        with pytest.raises(DeserializationError, match="cannot restore.*response schema"):
            DatasonSerializer().loads_typed(("datason-json-v1", payload))
        return
    schema = {"type": "boolean"}
    original = Interrupt(value={"question": "Continue?"}, id="schema-interrupt", response_schema=schema)
    serializer = DatasonSerializer()
    restored = serializer.loads_typed(serializer.dumps_typed((original,)))[0]
    assert restored.response_schema == schema


def test_python_schema_class_is_rejected_on_write():
    import inspect

    if "response_schema" not in inspect.signature(Interrupt).parameters:
        pytest.skip("Older SDK cannot construct an interrupt with a response schema")
    serializer = DatasonSerializer()
    with pytest.raises(SerializationError, match="JSON Schema data"):
        serializer.dumps_typed(Interrupt(value=True, id="invalid-schema", response_schema=int))


def test_send_timeout_policy_is_rejected_on_write():
    import inspect

    if "timeout" not in inspect.signature(Send).parameters:
        pytest.skip("Older SDK cannot construct a Send with a timeout")
    serializer = DatasonSerializer()
    with pytest.raises(SerializationError, match="timeout policies"):
        serializer.dumps_typed(Send("target", {}, timeout=1))


@pytest.mark.parametrize("kind", ["interrupt", "send"])
def test_untagged_runtime_export_is_explicit_normalization(kind):
    import datason

    DatasonSerializer()  # Enables the reviewed codec without changing core defaults.
    obj = Interrupt(value={"approved": True}, id="export") if kind == "interrupt" else Send("target", {"number": 1})
    plain = json.loads(datason.dumps(obj, include_type_hints=False))
    expected = (
        {"id": "export", "value": {"approved": True}, "response_schema": None}
        if kind == "interrupt"
        else {
            "node": "target",
            "arg": {"number": 1},
        }
    )
    assert plain == expected
