"""Bounded, interleaved comparison of warmed optional-plugin dispatch.

Run in an environment containing the optional libraries. The baseline directory
must contain an application-owned checkout of the earlier lazy implementation.
Its LazyPlugin and miscellaneous ML dispatcher are imported. Conversion code
is unchanged; libraries and stdlib handlers are shared.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import datason
from datason._registry import default_registry
from datason.plugins import _OPTIONAL
from datason.plugins._lazy import LazyPlugin


def cases():
    import catboost
    import jax.numpy as jnp
    import numpy as np
    import optuna
    import pandas as pd
    import plotly.graph_objects as go
    import polars as pl
    import scipy.sparse as sp
    import tensorflow as tf
    import torch
    from pydantic import BaseModel
    from sklearn.linear_model import LinearRegression

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    class Model(BaseModel):
        values: list[float]

    @dataclasses.dataclass
    class Record:
        values: list[float]

    def arrays(original, restored):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))

    def torch_values(original, restored):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        assert torch.equal(original, restored)

    def tf_values(original, restored):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        np.testing.assert_array_equal(original.numpy(), restored.numpy())

    def sparse_values(original, restored):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        assert original.format == restored.format and (original != restored).nnz == 0

    out = []
    for size in (10, 1000):
        values = np.arange(size, dtype=np.float32)
        rows = values.tolist()
        study = optuna.create_study(direction="minimize")
        for i in range(size):
            study.add_trial(
                optuna.trial.create_trial(
                    value=float(i),
                    params={"x": i / size},
                    distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
                )
            )
        features = 2 if size == 10 else 64
        matrix = np.random.default_rng(136).normal(size=(features + 16, features))
        estimator = LinearRegression().fit(matrix, matrix.sum(axis=1))
        entries = [
            ("numpy", values, arrays),
            ("pandas", pd.DataFrame({"x": values}), pd.testing.assert_frame_equal),
            ("scipy", sp.csr_matrix(values.reshape(1, -1)), sparse_values),
            ("torch", torch.arange(size, dtype=torch.float32), torch_values),
            ("tensorflow", tf.constant(rows, dtype=tf.float32), tf_values),
            (
                "sklearn",
                estimator,
                lambda a, b: np.testing.assert_allclose(
                    a.predict(np.ones((2, a.n_features_in_))), b.predict(np.ones((2, a.n_features_in_)))
                ),
            ),
            (
                "polars",
                pl.DataFrame({"x": pl.Series(list(range(size)), dtype=pl.Int64)}),
                lambda a, b: a.schema == b.schema and a.equals(b),
            ),
            ("jax", jnp.asarray(values), arrays),
            (
                "catboost",
                catboost.CatBoostRegressor(iterations=5, thread_count=1, monotone_constraints=[0] * size),
                None,
            ),
            ("optuna", study, None),
            (
                "plotly",
                go.Figure(go.Scatter(x=list(range(size)), y=rows), layout={"template": None}),
                lambda a, b: a.to_dict() == b.to_dict(),
            ),
            ("pydantic", Model(values=rows), lambda a, b: Model.model_validate(b) == a),
            ("json", {"values": rows}, lambda a, b: a == b),
            ("datetime", [dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc)] * size, lambda a, b: a == b),
            ("uuid", [UUID(int=i) for i in range(size)], lambda a, b: a == b),
            ("decimal", [Decimal(i) / 10 for i in range(size)], lambda a, b: a == b),
            ("path", [Path(f"artifact/{i}") for i in range(size)], lambda a, b: a == b),
            ("structured", Record(rows), lambda a, b: b == {"values": a.values}),
        ]
        for family, obj, verify in entries:
            out.append(
                {
                    "family": family,
                    "size_label": "small" if size == 10 else "medium",
                    "fixture_scale": size,
                    "value": obj,
                    "verify": verify,
                }
            )
    return out


def handlers(cls, original):
    definitions = {row[0]: row for row in _OPTIONAL}
    return [cls(*definitions[p.name]) if isinstance(p, LazyPlugin) else p for p in original]


def use(plugins):
    with default_registry._lock:
        default_registry._plugins[:] = plugins


def timed(fn, iterations):
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return {"median_us": statistics.median(elapsed), "mean_us": statistics.mean(elapsed)}


def dispatch_probes(corpus, modes, rounds, rng):
    records = []
    misc = {"polars", "jax", "catboost", "optuna", "plotly"}
    for entry in corpus:
        family = entry["family"]
        if entry["size_label"] != "small" or family in {"json", "datetime", "uuid", "decimal", "path", "structured"}:
            continue
        name = "ml_misc" if family in misc else ("scipy_sparse" if family == "scipy" else family)
        plugins = {mode: next(p for p in handlers if p.name == name) for mode, handlers in modes.items()}
        samples = []
        for round_no in range(rounds):
            order = list(modes)
            rng.shuffle(order)
            sample = {"round": round_no, "order": order}
            for mode in order:
                plugin = plugins[mode]
                assert plugin.can_handle(entry["value"])
                sample[mode] = timed(lambda p=plugin, e=entry: p.can_handle(e["value"]), 2000)
            samples.append(sample)
        medians = {mode: statistics.median(s[mode]["median_us"] for s in samples) for mode in modes}
        records.append(
            {
                "family": family,
                "samples": samples,
                "iterations_per_round": 2000,
                "median_of_round_medians_us": medians,
                "paired_faster_rounds": sum(s["after"]["median_us"] < s["before"]["median_us"] for s in samples),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", default=5, type=int)
    parser.add_argument("--max-iterations", default=150, type=int)
    parser.add_argument(
        "--dispatch-only", action="store_true", help="Validate all fixtures, then time only warmed can_handle checks"
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.max_iterations < 1:
        parser.error("rounds and iterations must be positive")
    baseline_file = args.baseline_dir / "datason/plugins/_lazy.py"
    spec = importlib.util.spec_from_file_location("datason.plugins._study_baseline", baseline_file)
    assert spec and spec.loader
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    misc_file = args.baseline_dir / "datason/plugins/ml_misc.py"
    misc_spec = importlib.util.spec_from_file_location("datason.plugins._study_baseline_misc", misc_file)
    assert misc_spec and misc_spec.loader
    baseline_misc = importlib.util.module_from_spec(misc_spec)
    misc_spec.loader.exec_module(baseline_misc)
    original = list(default_registry._plugins)
    modes = {"before": handlers(baseline.LazyPlugin, original), "after": handlers(LazyPlugin, original)}
    old_misc = next(p for p in modes["before"] if p.name == "ml_misc")
    old_misc._load()
    old_misc._plugin = baseline_misc.MlMiscPlugin()
    corpus = cases()
    rng = random.Random(136)  # noqa: S311 -- reproducible measurement order, no security use
    records = []
    try:
        for entry in corpus:
            wires = []
            for plugins in modes.values():
                use(plugins)
                wire = datason.dumps(entry["value"])
                restored = datason.loads(wire)
                if entry["verify"] is None:
                    assert restored == json.loads(wire)["__datason_value__"]
                else:
                    assert entry["verify"](entry["value"], restored) is not False, (
                        entry["family"],
                        entry["size_label"],
                    )
                wires.append(wire)
            assert wires[0] == wires[1]
            entry["wire"] = wires[0]
        for entry in [] if args.dispatch_only else corpus:
            record = {k: entry[k] for k in ("family", "size_label", "fixture_scale")}
            record.update({"output_bytes": len(entry["wire"].encode()), "operations": {}})
            for op in ("dumps", "loads"):
                fn = (
                    (lambda e=entry: datason.dumps(e["value"]))
                    if op == "dumps"
                    else (lambda e=entry: datason.loads(e["wire"]))
                )
                use(modes["before"])
                start = time.perf_counter_ns()
                fn()
                duration = max((time.perf_counter_ns() - start) / 1e9, 1e-9)
                iterations = min(args.max_iterations, max(5, int(0.025 / duration)))
                samples = []
                for round_no in range(args.rounds):
                    order = list(modes)
                    rng.shuffle(order)
                    sample = {"round": round_no, "order": order}
                    for mode in order:
                        use(modes[mode])
                        for _ in range(5):
                            fn()
                        sample[mode] = timed(fn, iterations)
                    samples.append(sample)
                medians = {mode: statistics.median(s[mode]["median_us"] for s in samples) for mode in modes}
                record["operations"][op] = {
                    "iterations_per_round": iterations,
                    "samples": samples,
                    "median_of_round_medians_us": medians,
                    "reduction_pct": (1 - medians["after"] / medians["before"]) * 100,
                    "paired_faster_rounds": sum(s["after"]["median_us"] < s["before"]["median_us"] for s in samples),
                }
            records.append(record)
        probes = dispatch_probes(corpus, modes, args.rounds, rng)
    finally:
        use(original)
    source_file = Path(datason.__file__).resolve().parent / "plugins/_lazy.py"
    report = {
        "scope": "Warmed end-to-end dumps/loads in one process with shuffled paired modes. All optional descriptors and miscellaneous ML families are activated. The earlier LazyPlugin and ml_misc dispatcher differ; conversion algorithms are unchanged, and libraries and stdlib handlers are shared. Small/medium labels describe fixture scale, not equal workload sizes. CatBoost/Optuna loads restore metadata; Pydantic/structured loads restore normalized fields, with explicit model validation in the fidelity check. Not a universal speedup or CI reproduction.",
        "rounds": args.rounds,
        "max_iterations": args.max_iterations,
        "seed": 136,
        "python": sys.version,
        "dispatch_only": args.dispatch_only,
        "source": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "baseline_source": str(baseline_file),
        "baseline_sha256": hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
        "baseline_misc_sha256": hashlib.sha256(misc_file.read_bytes()).hexdigest(),
        "candidate_misc_sha256": hashlib.sha256((source_file.parent / "ml_misc.py").read_bytes()).hexdigest(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in (
                "numpy",
                "pandas",
                "scipy",
                "torch",
                "tensorflow",
                "scikit-learn",
                "polars",
                "jax",
                "catboost",
                "optuna",
                "plotly",
                "pydantic",
            )
        },
        "cases": records,
        "dispatch_probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in records:
        print(
            row["family"],
            row["size_label"],
            " ".join(
                f"{op}: {v['median_of_round_medians_us']['before']:.2f} -> {v['median_of_round_medians_us']['after']:.2f} us ({v['reduction_pct']:+.1f}%)"
                for op, v in row["operations"].items()
            ),
        )
    for row in probes:
        before, after = row["median_of_round_medians_us"].values()
        print(
            row["family"],
            f"can_handle: {before:.3f} -> {after:.3f} us; faster rounds {row['paired_faster_rounds']}/{args.rounds}",
        )


if __name__ == "__main__":
    main()
