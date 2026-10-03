"""Exercise the actual framework's SQLite persistence and graph resumption."""

import datetime as dt
from typing import Any, TypedDict

import pytest

from datason._config import SerializationConfig
from datason._errors import DeserializationError
from datason.integrations.langgraph import DatasonSerializer


def test_adapter_rejects_other_wire_formats():
    with pytest.raises(DeserializationError, match="Unsupported checkpoint format"):
        DatasonSerializer().loads_typed(("pickle", b"untrusted"))


@pytest.mark.parametrize(
    "options",
    [{"include_type_hints": False}, {"fallback_to_string": True}, {"strict": False}, {"redact_fields": ("password",)}],
)
def test_adapter_rejects_lossy_checkpoint_options(options):
    with pytest.raises(ValueError):
        DatasonSerializer(SerializationConfig(**options))


def test_sqlite_checkpoint_survives_connection_reopen(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("langgraph")
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    from langgraph.checkpoint.serde.base import SerializerProtocol
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        vector: Any
        observed: dt.datetime
        payload: bytes
        steps: int

    def advance(state):
        return {"vector": state["vector"] + 1, "steps": state["steps"] + 1}

    builder = StateGraph(State)
    builder.add_node("advance", advance)
    builder.add_edge(START, "advance")
    builder.add_edge("advance", END)
    serializer = DatasonSerializer()
    assert isinstance(serializer, SerializerProtocol)
    db = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "fidelity-test"}}
    initial = {
        "vector": np.array([1, 2], dtype=np.float32),
        "observed": dt.datetime(2026, 10, 3),
        "payload": b"\x00\xff",
        "steps": 0,
    }
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = serializer
        graph = builder.compile(checkpointer=saver, interrupt_before=["advance"])
        graph.invoke(initial, config)
        assert graph.get_state(config).next == ("advance",)
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        saver.serde = DatasonSerializer()
        resumed = builder.compile(checkpointer=saver)
        result = resumed.invoke(None, config)
        assert result["steps"] == 1
        assert result["vector"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(result["vector"], initial["vector"] + 1)
        assert result["observed"] == initial["observed"]
        assert result["payload"] == initial["payload"]
