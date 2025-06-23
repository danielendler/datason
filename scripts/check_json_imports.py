#!/usr/bin/env python3
"""Pre-commit hook to detect inappropriate stdlib json imports.

This script identifies places where 'import json' is used inappropriately,
helping maintain DataSON's philosophy of eating its own dog food.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

# Files where stdlib json import is architecturally justified
ALLOWED_JSON_IMPORTS: Set[str] = {
    # Core architecture - compatibility module
    "datason/json.py",
    # Core architecture - integrity verification needs deterministic output
    "datason/integrity.py",
    # Tests that specifically test stdlib compatibility
    "tests/unit/test_json_compatibility_requirement.py",
    "tests/unit/test_json_drop_in_compatibility.py",
    "tests/unit/test_enhanced_api_strategy.py",
    # CI/infrastructure scripts
    ".github/workflows/ci.yml",
    "scripts/setup_github_labels.py",
    # External benchmarking (needs stdlib for comparison)
    "docs/EXTERNAL_BENCHMARK_SETUP.md",
}

# Patterns that might be legitimate in documentation/examples for comparison
DOC_PATTERNS = re.compile(r"(docs/|examples/|README|CHANGELOG|\.md$)", re.IGNORECASE)


class JsonImportAnalyzer(ast.NodeVisitor):
    """AST visitor to find json imports and analyze their usage."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.json_imports: List[Dict] = []
        self.json_usage: List[Dict] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check for 'import json' statements."""
        for alias in node.names:
            if alias.name == "json":
                self.json_imports.append({"type": "import", "line": node.lineno, "alias": alias.asname or "json"})
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for 'from json import ...' statements."""
        if node.module == "json":
            imports = [alias.name for alias in node.names]
            self.json_imports.append({"type": "from_import", "line": node.lineno, "imports": imports})
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for json function calls."""
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "json":
                self.json_usage.append(
                    {"line": node.lineno, "function": node.func.attr, "context": self._get_context(node)}
                )
        self.generic_visit(node)

    def _get_context(self, node: ast.AST) -> str:
        """Get surrounding context for a node."""
        # This is simplified - in a real implementation you'd want more context
        return f"json.{node.func.attr}()" if hasattr(node, "func") else "unknown"


def analyze_file(file_path: Path) -> Dict:
    """Analyze a Python file for json imports and usage."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (UnicodeDecodeError, SyntaxError):
        return {"error": "Could not parse file"}

    analyzer = JsonImportAnalyzer(str(file_path))
    analyzer.visit(tree)

    return {"imports": analyzer.json_imports, "usage": analyzer.json_usage, "file_path": str(file_path)}


def is_legitimate_usage(file_path: str, analysis: Dict) -> bool:
    """Determine if json usage in a file is legitimate."""

    # Explicitly allowed files
    if file_path in ALLOWED_JSON_IMPORTS:
        return True

    # Check if it's documentation/examples that might need json for comparison
    if DOC_PATTERNS.search(file_path):
        # For docs/examples, we're more lenient but still check patterns
        return _analyze_doc_usage(file_path, analysis)

    # Core library files should not use stdlib json (except allowed ones)
    if file_path.startswith("datason/"):
        return False

    # Test files - check if they're testing compatibility
    if "test" in file_path.lower():
        return _analyze_test_usage(file_path, analysis)

    return False


def _analyze_doc_usage(file_path: str, analysis: Dict) -> bool:
    """Analyze if documentation usage is appropriate."""
    # For now, allow docs/examples but flag for review
    # In practice, you'd want more sophisticated analysis
    return True


def _analyze_test_usage(file_path: str, analysis: Dict) -> bool:
    """Analyze if test usage is appropriate."""
    # Allow tests that are explicitly testing compatibility
    compatibility_indicators = ["compatibility", "json_replacement", "drop_in", "stdlib", "comparison", "benchmark"]

    file_name = Path(file_path).name.lower()
    return any(indicator in file_name for indicator in compatibility_indicators)


def suggest_alternatives(usage_info: Dict) -> List[str]:
    """Suggest DataSON alternatives for json usage."""
    suggestions = []

    for usage in usage_info.get("usage", []):
        func = usage["function"]
        if func == "dumps":
            suggestions.append(f"Line {usage['line']}: Use datason.dumps_json() instead of json.dumps()")
        elif func == "loads":
            suggestions.append(f"Line {usage['line']}: Use datason.loads() instead of json.loads()")
        elif func == "dump":
            suggestions.append(f"Line {usage['line']}: Use datason.dump_json() instead of json.dump()")
        elif func == "load":
            suggestions.append(f"Line {usage['line']}: Use datason.load_json() instead of json.load()")

    return suggestions


def main() -> int:
    """Main function to check all Python files."""
    print("🔍 Checking for inappropriate stdlib json imports...")

    failed_files = []
    warning_files = []

    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    python_files = [f for f in python_files if ".git/" not in str(f)]

    for file_path in python_files:
        analysis = analyze_file(file_path)

        if "error" in analysis:
            continue

        if not analysis["imports"]:
            continue  # No json imports

        file_str = str(file_path)

        if is_legitimate_usage(file_str, analysis):
            print(f"✅ {file_str} - Legitimate json usage")
            continue

        # Check severity
        if file_str.startswith("datason/"):
            # Core library - this is an error
            failed_files.append((file_str, analysis))
            print(f"❌ {file_str} - Inappropriate json import in core library")
        else:
            # Examples/tests - this is a warning
            warning_files.append((file_str, analysis))
            print(f"⚠️  {file_str} - Consider using DataSON instead of stdlib json")

    # Print detailed suggestions
    all_issues = failed_files + warning_files

    if all_issues:
        print("\n📋 Analysis Summary:")
        print(f"   ❌ Core library issues: {len(failed_files)}")
        print(f"   ⚠️  Example/test issues: {len(warning_files)}")

        for file_path, analysis in all_issues:
            suggestions = suggest_alternatives(analysis)
            if suggestions:
                print(f"\n💡 Suggestions for {file_path}:")
                for suggestion in suggestions:
                    print(f"   {suggestion}")

    if failed_files:
        print(f"\n🚨 CRITICAL: Found {len(failed_files)} core library files with inappropriate json imports!")
        print("These must be fixed before committing.")
        print("\n💡 Quick fixes:")
        print("   - Use datason.dumps_json() instead of json.dumps()")
        print("   - Use datason.loads() instead of json.loads()")
        print("   - Use datason.dump_json() instead of json.dump()")
        print("   - Use datason.load_json() instead of json.load()")
        print("\n📚 For more details, see: docs/development/json-usage-guidelines.md")
        return 1

    if warning_files:
        print(f"\n⚠️  Found {len(warning_files)} files that could showcase DataSON better.")
        print("Consider updating examples to demonstrate DataSON's capabilities.")

    print("\n✅ JSON import check complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
