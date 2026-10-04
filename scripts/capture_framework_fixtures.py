"""Capture owned offline checkpoints under the documented older SDK pins."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sqlite3
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests/fixtures"


def load_test_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_langgraph() -> None:
    if version("langgraph") != "1.0.0" or version("langgraph-checkpoint-sqlite") != "2.0.11":
        raise ValueError("Use the documented older LangGraph/SQLite pins for capture")
    test = load_test_module("langgraph_fixture", "tests/integration/test_langgraph_runtime.py")
    config = {"configurable": {"thread_id": "legacy-approval"}}
    with TemporaryDirectory() as directory:
        db = str(Path(directory) / "checkpoint.sqlite")
        with test.sqlite.SqliteSaver.from_conn_string(db) as saver:
            saver.serde = test.DatasonSerializer()
            result = test.approval_builder().compile(checkpointer=saver).invoke({"approved": False}, config)
            identifier = result["__interrupt__"][0].id
        with sqlite3.connect(db) as connection:
            sql = "\n".join(connection.iterdump()) + "\n"
    record = {
        "langgraph": version("langgraph"),
        "checkpoint": version("langgraph-checkpoint"),
        "sqlite": version("langgraph-checkpoint-sqlite"),
        "thread_id": "legacy-approval",
        "interrupt_id": identifier,
        "wire_format": "datason-json-v1",
        "codec_sha256": hashlib.sha256((ROOT / "datason/integrations/_langgraph_types.py").read_bytes()).hexdigest(),
        "sqlite_dump": sql,
    }
    (OUTPUT / "langgraph-1.0.0-checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")


async def capture_agents() -> None:
    if version("openai-agents") != "0.22.0":
        raise ValueError("Use openai-agents 0.22.0 for capture")
    test = load_test_module("agents_fixture", "tests/integration/test_agents_snapshot.py")
    context = test.Context(
        test.dt.datetime(2026, 10, 4, tzinfo=test.dt.timezone.utc),
        test.np.array([1, 2], dtype=test.np.float32),
        b"\x00\xff",
    )
    calls = []
    paused = await test.Runner.run(
        test.make_agent(calls), "Record a value", context=context, run_config=test.RunConfig(tracing_disabled=True)
    )
    assert not calls and len(paused.interruptions) == 1
    payload = paused.to_state().to_json(context_serializer=test.serialize_context, strict_context=True)
    record = {"openai_agents": version("openai-agents"), "state": payload}
    (OUTPUT / "agents-0.22.0-snapshot.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    capture_langgraph()
    asyncio.run(capture_agents())
