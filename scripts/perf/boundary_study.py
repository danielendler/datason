"""Bounded JSON/typed contract study; run with python -m scripts.perf.boundary_study.

Timing comparisons share a wire contract only on already-JSON data. Snapshot
reconstruction does more work. Fresh-process imports do not clear OS caches.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import datason
from scripts.perf.replay_benchmark import _measure_peak_memory_bytes, _metrics, load_records

POLICY = datason.api_config().__dict__
PACKAGES = ("datason", "numpy", "pandas", "orjson", "pydantic", "torch", "tensorflow", "jax", "scikit-learn")


def codecs():
    """Keep each codec's native output; Pydantic retains insertion order."""
    result = {
        "datason_api": (lambda value: datason.dumps(value, **POLICY), datason.loads),
        "stdlib_json": (lambda value: json.dumps(value, sort_keys=True, allow_nan=False), json.loads),
    }
    try:
        import orjson

        result["orjson"] = (lambda value: orjson.dumps(value, option=orjson.OPT_SORT_KEYS), orjson.loads)
    except ImportError:
        pass
    try:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(object)
        result["pydantic_any"] = (adapter.dump_json, adapter.validate_json)
    except ImportError:
        pass
    return result


def operations(payload, selected):
    result = {}
    sizes = {}
    for name, (dump, load) in selected.items():
        encoded = dump(payload)
        assert load(encoded) == payload, f"{name} changed the JSON contract"
        sizes[name] = len(encoded.encode("utf-8") if isinstance(encoded, str) else encoded)
        result[name + ":dumps"] = lambda dump=dump: dump(payload)
        result[name + ":loads"] = lambda load=load, encoded=encoded: load(encoded)
    return result, sizes


def measure(ops, rounds, iterations, seed):
    """Interleave operations in shuffled order, retaining each round separately."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible timing order, not cryptography
    pooled = {name: [] for name in ops}
    results = {name: [] for name in ops}
    for fn in ops.values():
        for _ in range(5):
            fn()
    for _ in range(rounds):
        samples = {name: [] for name in ops}
        names = list(ops)
        for _ in range(iterations):
            rng.shuffle(names)
            for name in names:
                start = time.perf_counter_ns()
                ops[name]()
                samples[name].append((time.perf_counter_ns() - start) / 1_000_000)
        for name, values in samples.items():
            pooled[name].extend(values)
            results[name].append(_metrics(values, 0))
    return {
        name: {
            "pooled": _metrics(pooled[name], _measure_peak_memory_bytes(fn)),
            "rounds": results[name],
        }
        for name, fn in ops.items()
    }


def typed_payload(count):
    import numpy as np

    return {
        "observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
        "request_id": UUID(int=42),
        "score": np.float32(0.75),
        "weights": np.arange(count, dtype=np.float32).reshape(-1, 4),
        "preview": b"\x00\xffbinary",
    }


def verify_snapshot(payload, restored):
    import numpy as np

    assert restored["observed"] == payload["observed"]
    assert restored["request_id"] == payload["request_id"]
    assert restored["preview"] == payload["preview"]
    assert restored["score"].dtype == payload["score"].dtype
    assert restored["weights"].dtype == payload["weights"].dtype
    assert restored["weights"].shape == payload["weights"].shape
    np.testing.assert_array_equal(restored["weights"], payload["weights"])


def typed_case(count, rounds, iterations, seed):
    payload = typed_payload(count)
    normalized = json.loads(datason.dumps(payload, **POLICY))
    ops, sizes = operations(normalized, codecs())
    # JSON baselines receive the projection; conversion is deliberately excluded.
    ops["datason_api:dumps"] = lambda: datason.dumps(payload, **POLICY)
    snapshot = datason.dumps(payload)
    verify_snapshot(payload, datason.loads(snapshot))
    ops["datason_snapshot:dumps"] = lambda: datason.dumps(payload)
    ops["datason_snapshot:loads"] = lambda: datason.loads(snapshot)
    sizes["datason_snapshot"] = len(snapshot.encode("utf-8"))
    return {
        "id": f"typed-{count}",
        "contract": "API projection vs pre-normalized JSON codecs; snapshot verified separately",
        "array_shape": list(payload["weights"].shape),
        "encoded_bytes": sizes,
        "metrics": measure(ops, rounds, iterations, seed),
    }


IMPORT_WORKER = """
import importlib.metadata, json, sys, time
from pathlib import Path
if sys.platform != 'linux':
    raise RuntimeError('RSS measurement requires Linux /proc/self/status')
def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return {key: int(fields[key].split()[0]) * 1024 for key in ('VmRSS', 'VmHWM')}
before = memory()
start = time.perf_counter_ns()
import datason
elapsed = (time.perf_counter_ns() - start) / 1e6
after = memory()
packages = {}
for name in ('datason', 'numpy', 'pandas', 'torch', 'tensorflow', 'jax', 'scikit-learn'):
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
print(json.dumps({'import_ms': elapsed, 'rss_increase_bytes': after['VmRSS']-before['VmRSS'],
                 'peak_rss_bytes': after['VmHWM'], 'rss_method': 'Linux /proc/self/status',
                 'loaded_optional': [name for name in ('numpy','pandas','torch','tensorflow','jax','sklearn') if name in sys.modules],
                 'packages': packages}))
"""


def fresh_imports(interpreters, source, repeats):
    result = {}
    env = {**os.environ, "PYTHONPATH": str(source)}
    for label, executable in interpreters.items():
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            proc = subprocess.run(  # noqa: S603 -- local interpreter selected by operator
                [executable, "-c", IMPORT_WORKER], env=env, check=True, capture_output=True, text=True, timeout=60
            )
            sample = json.loads(proc.stdout)
            sample["process_ms"] = (time.perf_counter_ns() - start) / 1_000_000
            samples.append(sample)
        result[label] = {
            "samples": samples,
            "median_import_ms": statistics.median(item["import_ms"] for item in samples),
            "median_rss_increase_bytes": statistics.median(item["rss_increase_bytes"] for item in samples),
        }
    return result


def profile_typed():
    payload = typed_payload(4096)
    encoded = datason.dumps(payload)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        datason.dumps(payload)
        datason.loads(encoded)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(20)
    return stream.getvalue()


def provenance():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = Path(datason.__file__).resolve().parent.parent
    revision = subprocess.check_output(  # noqa: S603 -- fixed git arguments
        [shutil.which("git"), "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "base_commit": revision,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "datason").rglob("*.py"))
        },
    }


def run_study(paths, rounds=5, iterations=100, seed=42):
    cases = []
    for record in load_records(paths):
        ops, sizes = operations(record.payload, codecs())
        cases.append(
            {
                "id": record.payload_id,
                "workload_class": record.workload_class,
                "source_tag": record.source_tag,
                "contract": "already-JSON synthetic replay sample",
                "encoded_bytes": sizes,
                "metrics": measure(ops, rounds, iterations, seed),
            }
        )
    cases.extend(typed_case(count, rounds, iterations, seed) for count in (64, 4096))
    return {
        "schema_version": "1.0",
        "provenance": provenance(),
        "settings": {"rounds": rounds, "iterations_per_round": iterations, "seed": seed},
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)},
        "memory_scope": "tracemalloc Python-managed peak, measured outside latency samples",
        "cases": cases,
        "profile_snapshot_dump_load": profile_typed(),
    }


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("perf/workloads/sample"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=100)
    parser.add_argument("--import-python", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--import-repeats", type=positive, default=3)
    args = parser.parse_args()
    report = run_study(list(args.input.glob("*.ndjson")), args.rounds, args.iterations)
    interpreters = dict(item.split("=", 1) for item in args.import_python)
    source = Path(datason.__file__).resolve().parent.parent
    report["fresh_process_imports"] = fresh_imports(interpreters, source, args.import_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(report['cases'])} bounded cases to {args.output}")


if __name__ == "__main__":
    main()
