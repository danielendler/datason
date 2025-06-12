#!/usr/bin/env python3
"""Documentation synchronization script for datason.

This script copies key documentation files from the root directory
to the docs/community/ directory to ensure they are available for mkdocs.
"""

import shutil
from pathlib import Path

# Files to sync from root to docs/community/
FILES_TO_SYNC = ["CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"]


def sync_docs() -> None:
    """Copy documentation files from root to docs/ directory."""
    root_dir = Path(__file__).parent.parent
    docs_dir = root_dir / "docs" / "community"

    # Ensure docs directory exists
    docs_dir.mkdir(exist_ok=True)

    files_copied = []
    files_skipped = []

    for filename in FILES_TO_SYNC:
        source_file = root_dir / filename
        dest_file = docs_dir / filename.lower()

        if source_file.exists():
            # Copy the file
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest_file)
            files_copied.append(str(dest_file))
            print(f"✅ Copied {filename} to {dest_file}")
        else:
            files_skipped.append(filename)
            print(f"⚠️  Skipped {filename} (not found in root)")

    print("\n📋 Summary:")
    print(f"   • Copied: {len(files_copied)} files")
    print(f"   • Skipped: {len(files_skipped)} files")

    if files_copied:
        print(f"   • Files copied: {', '.join(files_copied)}")

    if files_skipped:
        print(f"   • Files skipped: {', '.join(files_skipped)}")


if __name__ == "__main__":
    sync_docs()
