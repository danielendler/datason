import importlib
import sys
import time
from pathlib import Path

# Ensure package root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datason


def test_profile_run_disabled(monkeypatch):
    monkeypatch.delenv("DATASON_PROFILE", raising=False)
    profiling = importlib.reload(importlib.import_module("datason._profiling"))
    with profiling.profile_run() as timings:
        with profiling.stage("demo"):
            pass
    assert timings == {}


def test_profile_run_enabled_and_sink(monkeypatch):
    monkeypatch.setenv("DATASON_PROFILE", "1")
    profiling = importlib.reload(importlib.import_module("datason._profiling"))
    captured = {}

    def sink(d):
        captured.update(d)

    profiling.set_profile_sink(sink)
    with profiling.profile_run() as timings:
        with profiling.stage("stage_one"):
            time.sleep(0.001)
    assert "stage_one" in timings
    assert captured == timings


def test_profiling_overhead_smoke(monkeypatch):
    data = {"text": "x" * 100_000}

    def measure():
        start = time.perf_counter()
        for _ in range(5):
            datason.serialize(data)
        return (time.perf_counter() - start) / 5

    monkeypatch.delenv("DATASON_PROFILE", raising=False)
    importlib.reload(importlib.import_module("datason._profiling"))
    baseline = measure()

    monkeypatch.setenv("DATASON_PROFILE", "1")
    importlib.reload(importlib.import_module("datason._profiling"))
    datason.set_profile_sink(lambda d: None)
    profiled = measure()

    overhead = (profiled - baseline) / baseline
    assert overhead <= 0.03 + 0.02  # allow small variance
