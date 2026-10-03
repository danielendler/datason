#!/usr/bin/env python3
"""Replay-style real-data benchmark runner for datason.

This script runs latency/throughput/memory measurements on NDJSON workload
records and optionally compares against a baseline summary.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datason


@dataclass(frozen=True)
class WorkloadRecord:
    """A single replay workload input record."""

    workload_class: str
    payload_id: str
    size_bytes: int
    has_type_hints: bool
    source_tag: str
    payload: Any


def _load_ndjson_records(path: Path) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} invalid JSON: {exc}") from exc

        required = {"workload_class", "payload_id", "has_type_hints", "source_tag", "payload"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"{path}:{i} missing required keys: {sorted(missing)}")

        payload = raw["payload"]
        if "size_bytes" in raw and isinstance(raw["size_bytes"], int):
            size_bytes = raw["size_bytes"]
        else:
            size_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

        records.append(
            WorkloadRecord(
                workload_class=str(raw["workload_class"]),
                payload_id=str(raw["payload_id"]),
                size_bytes=int(size_bytes),
                has_type_hints=bool(raw["has_type_hints"]),
                source_tag=str(raw["source_tag"]),
                payload=payload,
            )
        )
    return records


def load_records(paths: list[Path]) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    for path in sorted(paths):
        if path.suffix == ".ndjson":
            records.extend(_load_ndjson_records(path))
    if not records:
        raise ValueError("No workload records loaded; check --input path/glob")
    return records


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def iterations_for_size(size_bytes: int) -> int:
    if size_bytes < 2_000:
        return 200
    if size_bytes < 20_000:
        return 80
    return 20


def _time_operation(fn, iterations: int) -> list[float]:
    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        latencies_ms.append((end - start) / 1_000_000.0)
    return latencies_ms


def _measure_peak_memory_bytes(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def _metrics(latencies_ms: list[float], peak_memory_bytes: int) -> dict[str, float | int]:
    sorted_lat = sorted(latencies_ms)
    mean_ms = statistics.fmean(sorted_lat)
    throughput_ops_per_sec = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "samples": len(sorted_lat),
        "p50_ms": percentile(sorted_lat, 50),
        "p95_ms": percentile(sorted_lat, 95),
        "p99_ms": percentile(sorted_lat, 99),
        "mean_ms": mean_ms,
        "min_ms": sorted_lat[0],
        "max_ms": sorted_lat[-1],
        "throughput_ops_per_sec": throughput_ops_per_sec,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _scenario_key(record: WorkloadRecord, operation: str) -> str:
    return f"{record.workload_class}|{record.payload_id}|{operation}|hints={int(record.has_type_hints)}"


def build_operations(payload: Any, include_type_hints: bool) -> dict[str, Any]:
    """Create bound operation callables for a single record payload."""
    serialized = datason.dumps(payload, include_type_hints=include_type_hints)

    def run_dumps() -> str:
        return datason.dumps(payload, include_type_hints=include_type_hints)

    def run_loads() -> Any:
        return datason.loads(serialized)

    def run_roundtrip() -> Any:
        return datason.loads(datason.dumps(payload, include_type_hints=include_type_hints))

    return {
        "dumps": run_dumps,
        "loads": run_loads,
        "roundtrip": run_roundtrip,
    }


def run_benchmark(records: list[WorkloadRecord]) -> dict[str, Any]:
    scenario_results: list[dict[str, Any]] = []
    aggregate_latencies: dict[tuple[str, str], list[float]] = {}
    aggregate_memory: dict[tuple[str, str], int] = {}

    for record in records:
        operations = build_operations(record.payload, include_type_hints=record.has_type_hints)

        iterations = iterations_for_size(record.size_bytes)
        for operation, fn in operations.items():
            latencies_ms = _time_operation(fn, iterations=iterations)
            peak_memory_bytes = _measure_peak_memory_bytes(fn)
            result = {
                "scenario_key": _scenario_key(record, operation),
                "workload_class": record.workload_class,
                "payload_id": record.payload_id,
                "source_tag": record.source_tag,
                "has_type_hints": record.has_type_hints,
                "size_bytes": record.size_bytes,
                "operation": operation,
                "metrics": _metrics(latencies_ms, peak_memory_bytes),
            }
            scenario_results.append(result)

            agg_key = (record.workload_class, operation)
            aggregate_latencies.setdefault(agg_key, []).extend(latencies_ms)
            aggregate_memory[agg_key] = max(aggregate_memory.get(agg_key, 0), peak_memory_bytes)

    aggregate_results: dict[str, dict[str, Any]] = {}
    for (workload_class, operation), latencies in sorted(aggregate_latencies.items()):
        key = f"{workload_class}|{operation}"
        aggregate_results[key] = {
            "workload_class": workload_class,
            "operation": operation,
            "metrics": _metrics(latencies, aggregate_memory[(workload_class, operation)]),
        }

    return {
        "schema_version": "1.0",
        "tool": "replay_benchmark.py",
        "scenario_results": scenario_results,
        "aggregate_results": aggregate_results,
    }


def compare_against_baseline(
    baseline: dict[str, Any], current: dict[str, Any], threshold_pct: float
) -> tuple[list[dict[str, Any]], str]:
    warnings: list[dict[str, Any]] = []
    lines = [
        "## Real-Data Replay Comparison (head vs base)",
        "",
        f"- p95 regression warning threshold: `{threshold_pct:.1f}%`",
        "",
        "| Workload | Operation | Base p95 (ms) | Head p95 (ms) | Delta | Status |",
        "|---|---|---:|---:|---:|---|",
    ]

    base_agg = baseline.get("aggregate_results", {})
    head_agg = current.get("aggregate_results", {})
    for key in sorted(set(base_agg) & set(head_agg)):
        b = float(base_agg[key]["metrics"]["p95_ms"])
        h = float(head_agg[key]["metrics"]["p95_ms"])
        delta_pct = ((h - b) / b) * 100.0 if b > 0 else 0.0
        status = "ok"
        if delta_pct > threshold_pct:
            status = "regression"
            warnings.append(
                {
                    "aggregate_key": key,
                    "base_p95_ms": b,
                    "head_p95_ms": h,
                    "delta_pct": delta_pct,
                }
            )
        lines.append(
            f"| `{head_agg[key]['workload_class']}` | `{head_agg[key]['operation']}` | "
            f"{b:.3f} | {h:.3f} | {delta_pct:+.2f}% | {status} |"
        )

    if warnings:
        lines.extend(["", f"⚠️ Regressions detected: `{len(warnings)}`"])
    else:
        lines.extend(["", "✅ No p95 regressions above threshold detected."])
    return warnings, "\n".join(lines) + "\n"


def profile_top_regressions(
    current: dict[str, Any],
    records: list[WorkloadRecord],
    warnings: list[dict[str, Any]],
    profile_top_n: int,
    profile_dir: Path,
) -> list[str]:
    if profile_top_n <= 0 or not warnings:
        return []
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiled: list[str] = []

    warning_keys = [w["aggregate_key"] for w in sorted(warnings, key=lambda x: x["delta_pct"], reverse=True)]
    targets = warning_keys[:profile_top_n]
    scenario_map = {s["scenario_key"]: s for s in current["scenario_results"]}

    for target in targets:
        workload_class, operation = target.split("|", 1)
        scenario_key = None
        for k, scenario in scenario_map.items():
            if scenario["workload_class"] == workload_class and scenario["operation"] == operation:
                scenario_key = k
                break
        if not scenario_key:
            continue

        # Reconstruct function from record context for profiling.
        scenario = scenario_map[scenario_key]
        rec = next(
            r
            for r in records
            if r.workload_class == scenario["workload_class"] and r.payload_id == scenario["payload_id"]
        )
        operations = build_operations(rec.payload, include_type_hints=rec.has_type_hints)
        fn = operations[scenario["operation"]]

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(200):
            fn()
        pr.disable()

        safe_name = scenario_key.replace("|", "_").replace("=", "-")
        prof_path = profile_dir / f"{safe_name}.prof"
        pr.dump_stats(str(prof_path))
        profiled.append(str(prof_path))

    return profiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run replay-style real-data benchmark suite.")
    parser.add_argument(
        "--input",
        default="perf/workloads/sample",
        help="Input file, directory, or glob with NDJSON workload records",
    )
    parser.add_argument(
        "--output",
        default="perf/results/replay-summary.json",
        help="Output path for benchmark summary JSON",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline summary JSON for p95 comparison",
    )
    parser.add_argument(
        "--p95-threshold-pct",
        type=float,
        default=10.0,
        help="Warn/fail threshold for p95 regression percentage",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when baseline comparison finds regressions",
    )
    parser.add_argument(
        "--report-markdown",
        default=None,
        help="Optional markdown report output path",
    )
    parser.add_argument(
        "--profile-top-regressions",
        type=int,
        default=0,
        help="Capture cProfile traces for top N regressed aggregate scenarios",
    )
    parser.add_argument(
        "--profile-dir",
        default="perf/results/profiles",
        help="Directory for optional cProfile output files",
    )
    return parser.parse_args()


def resolve_input_paths(input_arg: str) -> list[Path]:
    p = Path(input_arg)
    if p.exists() and p.is_file():
        return [p]
    if p.exists() and p.is_dir():
        return sorted(p.glob("*.ndjson"))
    return [Path(x) for x in sorted(Path(".").glob(input_arg))]


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args.input)
    records = load_records(input_paths)

    summary = run_benchmark(records)
    summary["inputs"] = [str(x) for x in input_paths]
    summary["record_count"] = len(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    regressions: list[dict[str, Any]] = []
    markdown_report = ""
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        regressions, markdown_report = compare_against_baseline(
            baseline=baseline,
            current=summary,
            threshold_pct=args.p95_threshold_pct,
        )
        summary["comparison"] = {
            "baseline_path": str(baseline_path),
            "p95_threshold_pct": args.p95_threshold_pct,
            "regressions": regressions,
        }

    profiled_files = profile_top_regressions(
        current=summary,
        records=records,
        warnings=regressions,
        profile_top_n=args.profile_top_regressions,
        profile_dir=Path(args.profile_dir),
    )
    if profiled_files:
        summary["profiles"] = profiled_files

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote replay benchmark summary: {output_path}")

    if args.report_markdown and markdown_report:
        report_path = Path(args.report_markdown)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown_report, encoding="utf-8")
        print(f"Wrote markdown report: {report_path}")

    if markdown_report:
        print(markdown_report)

    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
