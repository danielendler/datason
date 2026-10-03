"""Tests for replay benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/perf/replay_benchmark.py")
SAMPLE_DIR = Path("perf/workloads/sample")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_replay_benchmark_generates_summary(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    result = _run(["--input", str(SAMPLE_DIR), "--output", str(out)])
    assert result.returncode == 0, result.stderr
    assert out.exists()

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.0"
    assert summary["record_count"] >= 4
    assert "aggregate_results" in summary
    assert any(key.startswith("api|") for key in summary["aggregate_results"])


def test_replay_benchmark_compare_no_regression(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    report = tmp_path / "report.md"

    first = _run(["--input", str(SAMPLE_DIR), "--output", str(base)])
    assert first.returncode == 0, first.stderr

    second = _run(
        [
            "--input",
            str(SAMPLE_DIR),
            "--output",
            str(head),
            "--baseline",
            str(base),
            "--p95-threshold-pct",
            "1000",
            "--report-markdown",
            str(report),
            "--fail-on-regression",
        ]
    )
    assert second.returncode == 0, second.stderr
    assert report.exists()
    assert "No p95 regressions" in report.read_text(encoding="utf-8")


def test_replay_benchmark_compare_detects_regression(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"

    first = _run(["--input", str(SAMPLE_DIR), "--output", str(base)])
    assert first.returncode == 0, first.stderr

    # Force unrealistically strict baseline values to trigger regression.
    base_data = json.loads(base.read_text(encoding="utf-8"))
    for _, record in base_data["aggregate_results"].items():
        record["metrics"]["p95_ms"] = 0.00001
    base.write_text(json.dumps(base_data, indent=2), encoding="utf-8")

    second = _run(
        [
            "--input",
            str(SAMPLE_DIR),
            "--output",
            str(head),
            "--baseline",
            str(base),
            "--p95-threshold-pct",
            "10",
            "--fail-on-regression",
        ]
    )
    assert second.returncode != 0
