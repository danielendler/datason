#!/usr/bin/env python3
"""Run each Python docs block independently, with optional LangGraph coverage."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent


def check_blocks(include_langgraph: bool) -> int:
    """Use fresh processes and temporary directories to isolate example state."""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    count = 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for path in paths:
        if path.name == "langgraph-checkpoints.md" and not include_langgraph:
            print("Skipping optional LangGraph block; use --with-langgraph to include it.")
            continue
        blocks = re.finditer(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)
        for block in blocks:
            line = path.read_text(encoding="utf-8")[: block.start()].count("\n") + 1
            with TemporaryDirectory() as directory:
                result = subprocess.run(  # noqa: S603 -- Execute reviewed repository examples without a shell.
                    [sys.executable, "-c", block.group(1)],
                    cwd=directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            if result.returncode:
                raise SystemExit(f"{path.relative_to(ROOT)}:{line}\n{result.stdout}{result.stderr}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-langgraph", action="store_true", help="Requires optional framework/SQLite packages")
    args = parser.parse_args()
    print(f"Passed {check_blocks(args.with_langgraph)} independent Python documentation blocks.")


if __name__ == "__main__":
    main()
