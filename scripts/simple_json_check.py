#!/usr/bin/env python3
"""Simple script to find inappropriate json imports in DataSON codebase."""

import re
import sys
from pathlib import Path

# Files where stdlib json import is legitimate
ALLOWED_FILES = {
    "datason/json.py",  # Drop-in compatibility module
    "datason/integrity.py",  # Canonical JSON output needed
    "tests/unit/test_json_compatibility_requirement.py",
    "tests/unit/test_json_drop_in_compatibility.py",
    "tests/unit/test_enhanced_api_strategy.py",
    "scripts/setup_github_labels.py",  # Infrastructure
}


def check_file(file_path: Path) -> bool:
    """Check if file has inappropriate json imports."""
    file_str = str(file_path)

    # Skip allowed files
    if file_str in ALLOWED_FILES:
        return True

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return True  # Skip problematic files

    # Look for json imports
    import_json = re.search(r"^import json\b", content, re.MULTILINE)
    from_json = re.search(r"^from json import", content, re.MULTILINE)

    if import_json or from_json:
        # Check if this is appropriate usage
        if file_str.startswith("datason/"):
            print(f"❌ CORE LIBRARY: {file_str} - Uses stdlib json inappropriately")
            return False
        elif any(x in file_str.lower() for x in ["test", "example", "docs"]):
            print(f"⚠️  EXAMPLE/TEST: {file_str} - Could showcase DataSON instead")
            return True  # Warning, not error
        else:
            print(f"⚠️  OTHER: {file_str} - Has json import")
            return True

    return True


def main() -> int:
    """Check all Python files for inappropriate json imports."""
    print("🔍 Checking for inappropriate stdlib json imports...\n")

    failed = False
    core_issues = []
    other_issues = []

    # Find all Python files
    for py_file in Path(".").rglob("*.py"):
        if ".git/" in str(py_file) or "venv/" in str(py_file):
            continue

        file_str = str(py_file)

        # Skip if checking would cause issues
        if not check_file(py_file):
            if file_str.startswith("datason/"):
                core_issues.append(file_str)
                failed = True
            else:
                other_issues.append(file_str)

    print("\n📊 SUMMARY:")
    print(f"❌ Core library files with issues: {len(core_issues)}")
    print(f"⚠️  Other files with json imports: {len(other_issues)}")

    if core_issues:
        print("\n🚨 CRITICAL: These core library files need fixing:")
        for file in core_issues:
            print(f"   - {file}")
        print("\n💡 Replace with DataSON functions:")
        print("   - json.dumps() → datason.dumps_json()")
        print("   - json.loads() → datason.loads()")
        print("   - json.dump() → datason.dump_json()")
        print("   - json.load() → datason.load_json()")

    if other_issues:
        print("\n⚠️  Consider updating these to showcase DataSON:")
        for file in other_issues[:10]:  # Show first 10
            print(f"   - {file}")
        if len(other_issues) > 10:
            print(f"   ... and {len(other_issues) - 10} more")

    if failed:
        print("\n❌ FAIL: Core library uses stdlib json inappropriately")
        return 1
    else:
        print("\n✅ PASS: No inappropriate json usage in core library")
        return 0


if __name__ == "__main__":
    sys.exit(main())
