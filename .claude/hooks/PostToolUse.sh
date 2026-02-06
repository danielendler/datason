#!/bin/bash
# Post-tool-use hook: Enforce architecture limits after file writes
# This runs after Claude Code writes or edits files

# Only check on Write/Edit operations to Python files
if [[ "$TOOL_NAME" != "Write" && "$TOOL_NAME" != "Edit" ]]; then
    exit 0
fi

# Extract file path from the tool input (if available)
FILE_PATH="${TOOL_INPUT_FILE_PATH:-}"

# Only check Python files in datason/
if [[ "$FILE_PATH" != *.py ]] || [[ "$FILE_PATH" != *datason/* ]]; then
    exit 0
fi

# Check module line count (hard limit: 500)
if [[ -f "$FILE_PATH" ]]; then
    LINE_COUNT=$(wc -l < "$FILE_PATH" | tr -d ' ')
    if [[ "$LINE_COUNT" -gt 500 ]]; then
        echo "WARNING: $FILE_PATH is $LINE_COUNT lines (limit: 500). Split this module."
    fi
fi

exit 0
