"""Compare Python core vs Rust core using profiling.

Run locally after building the Rust extension:
  DATASON_PROFILE=1 python examples/rust_vs_python_benchmark.py

To force Python-only, set DATASON_RUST=off. To enable Rust auto, set DATASON_RUST=auto (default).
"""

import json
import os
import time
from statistics import mean

import datason
from datason import _rustcore


def bench_serialize(payload, iterations=200):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        datason.save_string(payload)
        times.append((time.perf_counter() - t0) * 1000)
    return mean(times)


def bench_deserialize(text, iterations=200):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        datason.load_basic(text)
        times.append((time.perf_counter() - t0) * 1000)
    return mean(times)


def main():
    payload = {
        "ints": list(range(100)),
        "floats": [i / 10 for i in range(100)],
        "strings": ["item" + str(i) for i in range(100)],
        "nested": {"a": [1, 2, 3], "b": None, "c": True},
    }
    text = json.dumps(payload)

    accel = os.getenv("DATASON_RUST", "auto")
    print(f"DATASON_RUST={accel} | Rust available={_rustcore.AVAILABLE}")

    s_ms = bench_serialize(payload)
    l_ms = bench_deserialize(text)
    print(f"Average save_string: {s_ms:.3f} ms")
    print(f"Average load_basic:  {l_ms:.3f} ms")


if __name__ == "__main__":
    main()
