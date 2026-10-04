"""Pause a LangGraph workflow, reopen SQLite, and resume typed stored state."""

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from datason.integrations.langgraph import DatasonSerializer


class State(TypedDict):
    observed: dt.datetime
    payload: bytes
    steps: int


def advance(state: State) -> dict[str, int]:
    return {"steps": state["steps"] + 1}


builder = StateGraph(State)
builder.add_node("advance", advance)
builder.add_edge(START, "advance")
builder.add_edge("advance", END)
config = {"configurable": {"thread_id": "job-1"}}
initial: State = {
    "observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
    "payload": b"hello",
    "steps": 0,
}

with TemporaryDirectory() as directory:
    database = str(Path(directory) / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(database) as saver:
        saver.serde = DatasonSerializer()
        graph = builder.compile(checkpointer=saver, interrupt_before=["advance"])
        graph.invoke(initial, config)
        assert graph.get_state(config).next == ("advance",)

    # A fresh connection demonstrates restoration from stored JSON.
    with SqliteSaver.from_conn_string(database) as saver:
        saver.serde = DatasonSerializer()
        graph = builder.compile(checkpointer=saver)
        restored = graph.invoke(None, config)
        assert restored["steps"] == 1
        assert restored["observed"] == initial["observed"]
        assert restored["payload"] == b"hello"

print("Checkpoint reopened and resumed successfully.")
