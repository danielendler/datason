#!/bin/bash
# Test execution script for datason with organized test categories

case "$1" in
    "fast")
        echo "🏃‍♂️ Running Fast Core Tests..."
        python -m pytest tests/core --maxfail=5 --tb=short
        ;;
    "full")
        echo "🔄 Running Full Test Suite (excluding benchmarks)..."
        python -m pytest tests/core tests/features tests/integration
        ;;
    "benchmarks")
        echo "📊 Running Benchmark Tests..."
        python -m pytest tests/benchmarks --benchmark-only
        ;;
    "coverage")
        echo "📈 Running Coverage Boost Tests..."
        python -m pytest tests/coverage
        ;;
    "all")
        echo "🚀 Running All Tests..."
        python -m pytest tests/ tests/benchmarks --benchmark-skip
        ;;
    *)
        echo "Usage: $0 {fast|full|benchmarks|coverage|all}"
        echo ""
        echo "Test Categories:"
        echo "  fast       - Fast core tests (~7-10 seconds)"
        echo "  full       - All tests except benchmarks (~30-60 seconds)"
        echo "  benchmarks - Performance benchmark tests (~60-120 seconds)"
        echo "  coverage   - Coverage boost tests"
        echo "  all        - Complete test suite"
        exit 1
        ;;
esac
