#!/usr/bin/env python3
"""
Test execution script for datason with organized test categories.

This script provides easy commands for running different test suites:
- Fast tests (core functionality)
- Full tests (everything except benchmarks)
- Benchmark tests (performance tests)
- Coverage tests (coverage boost tests)
- Integration tests (end-to-end scenarios)

Usage:
    python scripts/run_tests.py fast           # Fast core tests (~10-20 seconds)
    python scripts/run_tests.py full           # All tests except benchmarks (~30-60 seconds)
    python scripts/run_tests.py benchmarks     # Benchmark tests only (~60-120 seconds)
    python scripts/run_tests.py coverage       # Coverage boost tests only
    python scripts/run_tests.py integration    # Integration tests only
    python scripts/run_tests.py all            # Everything including benchmarks
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"🏃‍♂️ {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run datason tests with different categories")
    parser.add_argument(
        "category",
        choices=["fast", "full", "benchmarks", "coverage", "integration", "all"],
        help="Test category to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (number of workers)")

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]

    # Add common options
    if args.verbose:
        base_cmd.append("-v")
    if args.failfast:
        base_cmd.append("-x")
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])

    # Configure test category
    if args.category == "fast":
        cmd = base_cmd + ["tests/core", "-m", "core and not slow", "--maxfail=5", "--tb=short"]
        description = "Fast Core Tests (core functionality only)"

    elif args.category == "full":
        cmd = base_cmd + ["tests/core", "tests/features", "tests/integration", "-m", "not benchmark and not slow"]
        description = "Full Test Suite (excluding benchmarks and coverage)"

    elif args.category == "benchmarks":
        cmd = base_cmd + ["tests/benchmarks", "--benchmark-only", "--benchmark-disable-gc", "--benchmark-warmup=on"]
        description = "Benchmark Tests (performance measurements)"

    elif args.category == "coverage":
        cmd = base_cmd + ["tests/coverage", "-m", "coverage"]
        description = "Coverage Boost Tests (increase coverage metrics)"

    elif args.category == "integration":
        cmd = base_cmd + ["tests/integration", "-m", "integration"]
        description = "Integration Tests (end-to-end scenarios)"

    elif args.category == "all":
        cmd = base_cmd + [
            "tests/",
            "tests/benchmarks",
            "--benchmark-skip",  # Include benchmarks but don't run them with benchmark plugin
        ]
        description = "All Tests (complete test suite)"

    # Run the tests
    exit_code = run_command(cmd, description)

    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print(f"🎉 {description} completed successfully!")

        if args.category == "fast":
            print("\n💡 Tips:")
            print("   • Run 'python scripts/run_tests.py full' for complete testing")
            print("   • Run 'python scripts/run_tests.py benchmarks' for performance tests")

    else:
        print(f"💥 {description} failed!")
        print("\n🔧 Troubleshooting:")
        print("   • Check the error messages above")
        print("   • Try running with --verbose for more details")
        print("   • Use --failfast to stop on first failure")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
