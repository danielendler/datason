"""Tests for supported types matrix generation."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.gen_supported_types import generate_supported_types_table


@pytest.mark.skipif(
    not os.environ.get("RUN_SUPPORTED_TYPES"),
    reason="Run only in full/ML CI environments to avoid noise",
)
def test_generate_supported_types_table(tmp_path):
    """Table generation runs without regressions and creates file."""
    doc_path = Path("docs/supported-types.md")
    # Ensure function runs and writes the file; will raise on regression
    generate_supported_types_table(doc_path=doc_path)
    assert doc_path.exists()
