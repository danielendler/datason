"""Contract checks for the study, without speed assertions on shared runners."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import datason
from scripts.perf.boundary_study import codecs, fresh_imports, operations, typed_case, typed_payload, verify_snapshot


def test_codecs_preserve_json_values_and_report_utf8_bytes():
    payload = {"unicode": "€", "nested": [False, None, 42, 0.5], "empty": {}}
    ops, sizes = operations(payload, codecs())
    for name, (dump, load) in codecs().items():
        encoded = dump(payload)
        raw_bytes = encoded.encode("utf-8") if isinstance(encoded, str) else encoded
        assert sizes[name] == len(raw_bytes)
        assert load(ops[name + ":dumps"]()) == payload
        assert ops[name + ":loads"]() == payload


def test_typed_snapshot_preserves_fidelity_and_study_marks_projection():
    payload = typed_payload(64)
    verify_snapshot(payload, datason.loads(datason.dumps(payload)))
    result = typed_case(64, rounds=2, iterations=2, seed=7)
    assert "pre-normalized" in result["contract"]
    assert result["array_shape"] == [16, 4]
    assert result["encoded_bytes"]["datason_snapshot"] > result["encoded_bytes"]["datason_api"]
    for metric in result["metrics"].values():
        assert metric["pooled"]["samples"] == 4
        assert len(metric["rounds"]) == 2


def test_mismatched_json_codec_is_rejected_before_timing():
    with pytest.raises(AssertionError, match="changed the JSON contract"):
        operations({"value": 3}, {"lossy": (json.dumps, lambda _: {})})


@pytest.mark.skipif(sys.platform != "linux", reason="RSS evidence uses Linux /proc")
def test_fresh_process_measures_actual_import_and_records_environment():
    root = Path(datason.__file__).resolve().parent.parent
    report = fresh_imports({"current": sys.executable}, root, repeats=1)["current"]
    sample = report["samples"][0]
    assert sample["process_ms"] >= sample["import_ms"] > 0
    assert "datason" in sample["packages"]
    assert "numpy" in sample["packages"]
    assert sample["loaded_optional"] == []
