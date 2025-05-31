#!/usr/bin/env python3
"""Setup GitHub labels for datason repository.

This script creates consistent labels for PR management, issue tracking,
and automation workflows.
"""

import json
import subprocess
import sys
from typing import Dict, List


def run_gh_command(command: List[str]) -> Dict:
    """Run a GitHub CLI command and return JSON response."""
    try:
        result = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return {}


def create_label(name: str, color: str, description: str) -> bool:
    """Create a GitHub label."""
    command = [
        "gh",
        "label",
        "create",
        name,
        "--color",
        color,
        "--description",
        description,
        "--force",  # Update if exists
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode == 0:
        print(f"✅ Created/updated label: {name}")
        return True
    print(f"❌ Failed to create label {name}: {result.stderr}")
    return False


def setup_labels() -> bool:
    """Set up all datason labels."""

    labels = [
        # Type labels
        ("bug", "d73a4a", "Something isn't working"),
        ("enhancement", "a2eeef", "New feature or request"),
        ("documentation", "0075ca", "Improvements or additions to documentation"),
        ("security", "ee0701", "Security-related issues or improvements"),
        ("performance", "fbca04", "Performance improvements"),
        ("refactoring", "b60205", "Code refactoring without functional changes"),
        # Priority labels
        (
            "priority:critical",
            "b60205",
            "Critical priority - needs immediate attention",
        ),
        ("priority:high", "d93f0b", "High priority"),
        ("priority:medium", "fbca04", "Medium priority"),
        ("priority:low", "0e8a16", "Low priority"),
        # Status labels
        ("status:needs-review", "fbca04", "PR needs code review"),
        ("status:needs-testing", "f9d0c4", "Needs testing"),
        ("status:blocked", "d73a4a", "Blocked by another issue/PR"),
        ("status:in-progress", "1d76db", "Currently being worked on"),
        ("status:ready-to-merge", "0e8a16", "Ready to be merged"),
        # Automation labels
        ("auto-merge", "0e8a16", "Automatically merge when checks pass"),
        ("dependencies:core", "0366d6", "Core dependency updates"),
        ("dependencies:dev", "1d76db", "Development dependency updates"),
        ("dependencies:optional", "5319e7", "Optional dependency updates"),
        ("dependencies:github-actions", "0075ca", "GitHub Actions dependency updates"),
        # Component labels
        ("component:core", "c2e0c6", "Core serialization functionality"),
        ("component:ml", "7057ff", "Machine learning integration"),
        ("component:docs", "0075ca", "Documentation"),
        ("component:ci", "1d76db", "Continuous integration"),
        ("component:tests", "f9d0c4", "Test-related changes"),
        # Size labels (for PRs)
        ("size:xs", "0e8a16", "Extra small PR (1-10 lines)"),
        ("size:s", "7cfc00", "Small PR (11-50 lines)"),
        ("size:m", "fbca04", "Medium PR (51-200 lines)"),
        ("size:l", "ff9500", "Large PR (201-500 lines)"),
        ("size:xl", "ff0000", "Extra large PR (500+ lines)"),
        # Good first issue labels
        ("good first issue", "7057ff", "Good for newcomers"),
        ("help wanted", "008672", "Extra attention is needed"),
        # Release labels
        ("release:major", "b60205", "Major version release"),
        ("release:minor", "fbca04", "Minor version release"),
        ("release:patch", "0e8a16", "Patch version release"),
        ("release:prerelease", "c2e0c6", "Pre-release version"),
        # Special labels
        ("breaking-change", "d73a4a", "Introduces breaking changes"),
        ("backwards-compatible", "0e8a16", "Maintains backward compatibility"),
        ("experimental", "f9d0c4", "Experimental feature"),
        ("duplicate", "cfd3d7", "This issue or pull request already exists"),
        ("invalid", "e4e669", "This doesn't seem right"),
        ("question", "d876e3", "Further information is requested"),
        ("wontfix", "ffffff", "This will not be worked on"),
    ]

    print("🏷️ Setting up GitHub labels for datason...")

    success_count = 0
    total_count = len(labels)

    for name, color, description in labels:
        if create_label(name, color, description):
            success_count += 1

    print(f"\n📊 Results: {success_count}/{total_count} labels created/updated")

    if success_count == total_count:
        print("🎉 All labels set up successfully!")
        return True
    print("⚠️ Some labels failed to create. Check the errors above.")
    return False


def main() -> None:
    """Main function."""
    # Check if GitHub CLI is installed
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)  # noqa: S603, S607
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ GitHub CLI not found. Please install it first:")
        print("   https://cli.github.com/")
        sys.exit(1)

    # Check if authenticated
    try:
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)  # noqa: S603, S607
    except subprocess.CalledProcessError:
        print("❌ Not authenticated with GitHub CLI. Please run:")
        print("   gh auth login")
        sys.exit(1)

    success = setup_labels()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
