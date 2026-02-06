#!/usr/bin/env python3
"""Pre-commit hook: enforce 500-line max per module."""

import sys


def main() -> int:
    status = 0
    for filepath in sys.argv[1:]:
        try:
            with open(filepath) as f:
                lines = sum(1 for _ in f)
            if lines > 500:  # noqa: PLR2004
                print(f"FAIL: {filepath} has {lines} lines (limit 500)")
                status = 1
        except OSError as e:
            print(f"ERROR: could not read {filepath}: {e}")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
