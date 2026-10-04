# LangGraph checkpoint adapter

Use the development source installation from [Getting started](getting-started.md#installation),
then install the optional framework and SQLite dependencies. These versions match
the existing compatibility example:

```bash
python -m pip install 'langgraph==1.2.12' 'langgraph-checkpoint-sqlite==3.1.1'
```

## Pause, reopen, and resume

This complete example stores a datetime and bytes, closes the SQLite connection,
then resumes from a fresh connection. Its temporary directory is cleaned up on
exit. For persistent application state, choose a durable database path.

```python
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
```

The same example is available as
[`examples/langgraph_checkpoint.py`](https://github.com/danielendler/datason/blob/main/examples/langgraph_checkpoint.py).
Run it with `python examples/langgraph_checkpoint.py` after installation.

## Storage and compatibility contract

The adapter implements LangGraph's typed serializer protocol and stores UTF-8 JSON
under the `datason-json-v1` format label. It rejects other formats and does not
fall back to pickle. Type hints, strict loading, and no string fallback are required.
Redaction belongs in a separate diagnostic export because it changes state.
Datason does not install or import LangGraph through its core package.

Compatibility tests exercise pause-before-node, dynamic interrupts, pending
Send fan-out, explicit model hydration and application-schema upgrades after a
SQLite close/reopen. The [framework matrix](framework-compatibility.md) pins older
and current SDK versions and replays a checkpoint captured by the older SDK.

The adapter opts into closed, reviewed Interrupt/Send codecs when LangGraph is
installed. Those codecs register once in the shared plugin registry. Other
runtime/message object families and Send timeout policies need explicit codecs.
This is a bounded integration contract, not general framework-object hydration.

Dataclasses, Pydantic models, and Enums normalize to fields/values. If a node needs
an application-class instance, validate or hydrate it explicitly in application
code or provide a reviewed type plugin. Existing checkpoints produced by another
serializer need a deliberate migration; changing `saver.serde` does not migrate
them. Plugins and framework callbacks remain trusted code. This adapter provides
serialization, not encryption, authentication, schema migration, or a durable
workflow service.

Run the compatibility example after installing the optional packages:

```bash
pytest tests/integration/test_langgraph_checkpoint.py tests/integration/test_langgraph_runtime.py
```
