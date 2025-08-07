"""MkDocs hooks for datason documentation build."""

import sys
from pathlib import Path

# Ensure repository root is on path when MkDocs loads this file
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gen_supported_types import generate_supported_types_table  # noqa: E402


def on_pre_build(config):
    """Generate supported types table before building docs."""
    generate_supported_types_table()
