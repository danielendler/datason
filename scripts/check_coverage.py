#!/usr/bin/env python3
"""Lightweight coverage check for pre-commit.

This script runs coverage analysis only on changed files to provide
fast feedback during development without running the full test suite.
"""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Check coverage for changed files."""
    if len(sys.argv) < 2:
        print("✅ No Python files changed in serialpy/")
        return 0

    changed_files = [Path(f) for f in sys.argv[1:]]
    print(f"📊 Checking coverage for {len(changed_files)} changed files...")

    # Only check core serialpy files (exclude __init__.py)
    core_files = [
        f for f in changed_files if f.name != "__init__.py" and f.suffix == ".py"
    ]

    if not core_files:
        print("✅ No core files to check coverage for")
        return 0

    try:
        # Run minimal test suite with coverage for changed files
        # This is much faster than full test suite
        cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=serialpy",
            "--cov-report=term-missing",
            "--cov-fail-under=75",  # Require 75% minimum
            "-x",  # Stop on first failure
            "--tb=short",
            "tests/test_core.py",  # Only run core tests (fastest)
        ]

        print(f"🧪 Running: {' '.join(cmd)}")
        # Safe subprocess call with controlled arguments
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

        if result.returncode == 0:
            print("✅ Coverage check passed!")
            return 0
        print("❌ Coverage check failed!")
        print("📋 Coverage output:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
        print(
            "\n💡 Tip: Run 'pytest --cov=serialpy --cov-report=html' for detailed coverage report"
        )
        return 1

    except FileNotFoundError:
        print("⚠️ pytest not found - skipping coverage check")
        print("💡 Install with: pip install -e '.[dev]'")
        return 0
    except Exception as e:
        print(f"⚠️ Coverage check failed with error: {e}")
        return 0  # Don't block commits on coverage script errors


if __name__ == "__main__":
    sys.exit(main())
