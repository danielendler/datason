#!/bin/bash
# Test execution script for datason with organized test categories
# Note: Major performance benchmarking is handled by external datason-benchmarks repo

case "$1" in
    "fast")
        echo "🏃‍♂️ Running Fast Unit Tests..."
        python -m pytest tests/unit --maxfail=5 --tb=short
        ;;
    "full")
        echo "🔄 Running Full Test Suite..."
        python -m pytest tests/unit tests/integration tests/edge_cases
        ;;
    "performance")
        echo "📊 Running Local Performance Tests..."
        python -m pytest tests/performance/
        ;;
    "all")
        echo "🚀 Running All Tests..."
        python -m pytest tests/
        ;;
    *)
        echo "Usage: $0 {fast|full|performance|all}"
        echo ""
        echo "Test Categories:"
        echo "  fast        - Fast unit tests (~20-30 seconds)"
        echo "  full        - All tests except performance (~45-60 seconds)"
        echo "  performance - Local performance tests (~5-10 seconds)"
        echo "  all         - Complete local test suite (~60-90 seconds)"
        echo ""
        echo "📊 Major performance benchmarking is handled by external datason-benchmarks repo"
        echo "   and runs automatically on PRs via .github/workflows/pr-performance-check.yml"
        exit 1
        ;;
esac
