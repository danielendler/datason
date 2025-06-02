#!/usr/bin/env python3
"""
Coverage check script for datason.
Ensures coverage standards are maintained for modified files.
"""

import subprocess
import sys
from typing import List

# Only run coverage for datason package files
PACKAGE_DIR = "datason"


def get_changed_python_files() -> List[str]:
    """Get list of changed Python files in the datason package."""
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--name-only", "HEAD~1"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print("✅ No Python files changed in datason/")
        return []

    files = result.stdout.strip().split("\n")
    # Filter for Python files in the datason package
    py_files = [f for f in files if f.startswith(f"{PACKAGE_DIR}/") and f.endswith(".py")]

    # Only check core datason files (exclude __init__.py)
    return [f for f in py_files if not f.endswith("__init__.py")]


def run_coverage_check() -> int:
    """Run pytest with coverage for the entire package."""
    cmd = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        f"--cov={PACKAGE_DIR}",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",  # Fail if coverage below 80%
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

    print("📊 Coverage Report:")
    print(result.stdout)

    if result.stderr:
        print("⚠️  Stderr:")
        print(result.stderr)

    return result.returncode


def main() -> int:
    changed_files = get_changed_python_files()

    if not changed_files:
        print("✅ No datason package files changed - skipping coverage check")
        return 0

    print(f"🔍 Changed files: {', '.join(changed_files)}")
    print("🧪 Running coverage check...")

    exit_code = run_coverage_check()

    if exit_code == 0:
        print("✅ Coverage check passed!")
    else:
        print("❌ Coverage check failed!")
        print("\n💡 Tip: Run 'pytest --cov=datason --cov-report=html' for detailed coverage report")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
