#!/usr/bin/env python3
"""Pre-commit hook: Check that no function exceeds 50 lines."""
import ast
import sys


def check_file(path: str) -> list[str]:
    """Return list of violation messages for functions > 50 lines."""
    violations = []
    try:
        source = open(path).read()  # noqa: SIM115
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or 0) - node.lineno + 1
                if length > 50:
                    violations.append(
                        f"FAIL: {path}:{node.lineno} "
                        f"{node.name}() is {length} lines (limit 50)"
                    )
    except SyntaxError:
        pass
    return violations


def main() -> int:
    """Check all files passed as arguments."""
    status = 0
    for path in sys.argv[1:]:
        for violation in check_file(path):
            print(violation)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
