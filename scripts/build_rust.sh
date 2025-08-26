#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="$HERE/.."

if ! command -v maturin >/dev/null 2>&1; then
  echo "maturin not found. Install with: pip install maturin" >&2
  exit 1
fi

echo "Building Rust extension with maturin (develop mode)..."
cd "$ROOT/rust"
maturin develop --release
echo "Done. Try: DATASON_RUST=auto python examples/rust_vs_python_benchmark.py"
