"""Compare native LangGraph and Datason on the original scientific boundary cases.

No provider calls or pickle fallback. A native fix changes the report to success;
it does not make Datason's fidelity regression tests fail.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import datason
from datason.integrations.langgraph import DatasonSerializer


def cases():
    return {
        "runtime_source_sha256": {
            str(path.relative_to(Path(datason.__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(datason.__file__).parent.rglob("*.py"))
        },
        "int32": np.int32(7),
        "float32": np.float32(0.75),
        "datetime64_array": np.array(["2026-10-04", "2026-10-05"], dtype="datetime64[D]"),
        "timedelta64_array": np.array([3, 7], dtype="timedelta64[ms]"),
        "empty_multidimensional": np.empty((0, 3), dtype="float32"),
    }


def check(serializer, value):
    try:
        restored = serializer.loads_typed(serializer.dumps_typed(value))
        np.testing.assert_array_equal(restored, value)
        assert restored.dtype == value.dtype
        assert restored.shape == value.shape
        return {"serialization": "success", "value_dtype_shape": "preserved"}
    except Exception as exc:  # report native errors without replacing them with repr values
        return {"serialization": "failed", "error_type": type(exc).__name__, "error": str(exc)}


def report():
    native = JsonPlusSerializer()
    adapter = DatasonSerializer()
    return {
        "installed_distribution_versions": {
            name: importlib.metadata.version(name) for name in ("datason", "numpy", "langgraph", "langgraph-checkpoint")
        },
        "scope": "local scientific values, not arbitrary graph state or independent adoption",
        "cases": {
            name: {"native": check(native, value), "datason": check(adapter, value)} for name, value in cases().items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
