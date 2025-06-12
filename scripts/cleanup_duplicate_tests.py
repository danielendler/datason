#!/usr/bin/env python3
"""
Script to analyze and cleanup duplicate test functions in the datason test suite.
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def extract_test_functions(file_path):
    """Extract all test function names from a Python test file."""
    test_functions = []
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            # Find all test function definitions
            pattern = r"def (test_\w+)\("
            matches = re.findall(pattern, content)
            test_functions = matches
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return test_functions


def analyze_test_duplicates():
    """Analyze test suite for duplicate function names."""
    test_root = Path("tests")

    # Function name -> list of files containing it
    function_locations = defaultdict(list)

    # Walk through all Python test files
    for test_file in test_root.rglob("test_*.py"):
        if "__pycache__" in str(test_file):
            continue

        functions = extract_test_functions(test_file)
        for func in functions:
            function_locations[func].append(str(test_file))

    # Find duplicates
    duplicates = {func: files for func, files in function_locations.items() if len(files) > 1}

    return duplicates, function_locations


def print_duplicate_analysis():
    """Print analysis of duplicate test functions."""
    duplicates, all_functions = analyze_test_duplicates()

    print("=== DUPLICATE TEST FUNCTION ANALYSIS ===\n")

    # Group by category
    core_duplicates = {}
    deserializer_duplicates = {}
    api_duplicates = {}
    other_duplicates = {}

    for func, files in duplicates.items():
        if any("core" in f for f in files):
            core_duplicates[func] = files
        elif any("deserializer" in f for f in files):
            deserializer_duplicates[func] = files
        elif any("api" in f for f in files):
            api_duplicates[func] = files
        else:
            other_duplicates[func] = files

    print(f"CORE MODULE DUPLICATES ({len(core_duplicates)}):")
    for func, files in sorted(core_duplicates.items()):
        print(f"  {func}:")
        for file in files:
            print(f"    - {file}")
        print()

    print(f"DESERIALIZER MODULE DUPLICATES ({len(deserializer_duplicates)}):")
    for func, files in sorted(deserializer_duplicates.items()):
        print(f"  {func}:")
        for file in files:
            print(f"    - {file}")
        print()

    print(f"API MODULE DUPLICATES ({len(api_duplicates)}):")
    for func, files in sorted(api_duplicates.items()):
        print(f"  {func}:")
        for file in files:
            print(f"    - {file}")
        print()

    print(f"OTHER DUPLICATES ({len(other_duplicates)}):")
    for func, files in sorted(other_duplicates.items()):
        print(f"  {func}:")
        for file in files:
            print(f"    - {file}")
        print()

    print("=== SUMMARY ===")
    print(f"Total unique test functions: {len(all_functions)}")
    print(f"Functions with duplicates: {len(duplicates)}")
    print(f"Total duplicate instances: {sum(len(files) - 1 for files in duplicates.values())}")

    # Files with most duplicates
    file_duplicate_count = defaultdict(int)
    for files in duplicates.values():
        for file in files:
            file_duplicate_count[file] += 1

    print("\nFILES WITH MOST DUPLICATES:")
    for file, count in sorted(file_duplicate_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {file}: {count} duplicate functions")


def identify_files_to_remove():
    """Identify files that should be removed/consolidated."""
    # Based on our analysis, these are the files we want to keep
    keep_files = {
        "tests/unit/test_core_comprehensive.py",
        "tests/unit/test_deserializers_comprehensive.py",
        "tests/unit/test_api_comprehensive.py",
        "tests/unit/test_config_comprehensive.py",
        "tests/unit/test_serializers_comprehensive.py",
        "tests/unit/test_validation_comprehensive.py",
        "tests/unit/test_converters_comprehensive.py",
        "tests/unit/test_data_utils_comprehensive.py",
    }

    # Files that are likely redundant
    remove_candidates = [
        "tests/core/test_core.py",
        "tests/core/test_deserializers.py",
        "tests/unit/test_deserializer_enhancements.py",
        "tests/test_deserializer_hot_path.py",
    ]

    # Coverage directory files (most are likely redundant)
    coverage_dir = Path("tests/coverage")
    if coverage_dir.exists():
        coverage_files = list(coverage_dir.glob("test_*.py"))
        remove_candidates.extend([str(f) for f in coverage_files])

    return keep_files, remove_candidates


if __name__ == "__main__":
    print_duplicate_analysis()
    print("\n" + "=" * 60 + "\n")

    keep_files, remove_candidates = identify_files_to_remove()

    print("PROPOSED CLEANUP PLAN:")
    print(f"\nFILES TO KEEP ({len(keep_files)}):")
    for file in sorted(keep_files):
        print(f"  ✅ {file}")

    print(f"\nFILES TO REMOVE/CONSOLIDATE ({len(remove_candidates)}):")
    for file in sorted(remove_candidates):
        if os.path.exists(file):
            print(f"  🗑️  {file}")
        else:
            print(f"  ❌ {file} (doesn't exist)")
