# LangGraph checkpoint adapter

Install the optional framework and persistence dependencies:

```bash
pip install datason numpy langgraph langgraph-checkpoint-sqlite
```

Pass an explicit datason serializer to the checkpointer:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from datason.integrations.langgraph import DatasonSerializer

with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    saver.serde = DatasonSerializer()
    graph = builder.compile(checkpointer=saver)
    graph.invoke(initial_state, {"configurable": {"thread_id": "job-1"}})
```

The adapter implements LangGraph's typed serializer protocol and stores UTF-8 JSON
under the `datason-json-v1` format label. It rejects other formats and does not
fall back to pickle. Type hints, strict loading, and no string fallback are required.
Redaction belongs in a separate diagnostic export because it changes state.
Datason does not install or import LangGraph through its core package.

The integration test pauses a graph before a node, closes SQLite, opens a fresh
connection, and resumes from stored state. It checks a NumPy float32 array,
datetime, and binary payload. Locally verified with LangGraph 1.2.12 and
langgraph-checkpoint-sqlite 3.1.1. This establishes a bounded compatibility example,
not support for every LangGraph object or historical checkpoint format.

Dataclasses, Pydantic models, and Enums normalize to fields/values. If a node needs
an application-class instance, validate or hydrate it explicitly in application
code or provide a reviewed type plugin. Existing checkpoints produced by another
serializer need a deliberate migration; changing `saver.serde` does not migrate
them. Plugins and framework callbacks remain trusted code. This adapter provides
serialization, not encryption, authentication, schema migration, or a durable
workflow service.

Run the compatibility example after installing the optional packages:

```bash
pytest tests/integration/test_langgraph_checkpoint.py
```
