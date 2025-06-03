#!/usr/bin/env python3
"""
Implementation Guide for Step 1.1: Type Detection Caching
=========================================================

This script demonstrates how to implement the first performance optimization
and measure its impact using our CI performance tracking system.

Usage:
    python implement_step_1_1.py --measure-before
    # Edit datason/core.py to add caching
    python implement_step_1_1.py --measure-after
    python implement_step_1_1.py --compare
"""

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

import datason


def create_type_heavy_test_data() -> Dict[str, Any]:
    """Create test data that will benefit from type caching."""
    # This data has many repeated types that will benefit from caching
    base_objects = [{"id": i, "name": f"Item {i}", "active": i % 2 == 0} for i in range(100)]

    mixed_objects = [{"timestamp": datetime.now(timezone.utc), "value": i * 1.5} for i in range(50)]

    return {
        "homogeneous_list": base_objects,
        "mixed_types": mixed_objects,
        "nested_repeated": {
            "section_a": base_objects[:25],
            "section_b": base_objects[25:50],
            "section_c": base_objects[50:75],
            "section_d": base_objects[75:],
        },
    }


def benchmark_type_caching_impact():
    """Benchmark the impact of type caching on our specific test case."""
    test_data = create_type_heavy_test_data()

    results = {}

    for name, data in test_data.items():
        print(f"🔬 Benchmarking: {name}")

        # Run multiple iterations for stable measurement
        times = []
        for _ in range(10):
            start = time.perf_counter()
            datason.serialize(data)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        results[name] = {
            "mean_time_ms": avg_time * 1000,
            "iterations": len(times),
            "all_times": [t * 1000 for t in times],
        }

        print(f"  Average: {avg_time * 1000:.2f}ms")

    return results


def save_benchmark_results(results: Dict[str, Any], label: str):
    """Save benchmark results with timestamp."""
    filename = f"step_1_1_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "results": results,
        "python_version": datason.__version__ if hasattr(datason, "__version__") else "unknown",
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Results saved to: {filename}")
    return filename


def compare_results(before_file: str, after_file: str):
    """Compare before and after results."""
    with open(before_file) as f:
        before = json.load(f)

    with open(after_file) as f:
        after = json.load(f)

    print("\n" + "=" * 60)
    print("TYPE CACHING OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"Before: {before['timestamp']}")
    print(f"After:  {after['timestamp']}")

    print("\nPerformance Changes:")
    print("-" * 40)

    improvements = []

    for test_name in before["results"]:
        if test_name in after["results"]:
            before_time = before["results"][test_name]["mean_time_ms"]
            after_time = after["results"][test_name]["mean_time_ms"]

            improvement = ((before_time - after_time) / before_time) * 100
            improvements.append(improvement)

            print(f"{test_name:20}: {before_time:6.2f}ms → {after_time:6.2f}ms ({improvement:+5.1f}%)")

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage improvement: {avg_improvement:+.1f}%")

        if avg_improvement > 5:
            print("✅ Significant improvement detected!")
        elif avg_improvement > 0:
            print("🟡 Minor improvement detected")
        else:
            print("❌ No improvement or regression detected")
    else:
        print("❌ No comparable results found")


def show_implementation_guide():
    """Show the implementation guide for Step 1.1."""
    guide = """
📋 IMPLEMENTATION GUIDE: Step 1.1 - Type Detection Caching
==========================================================

The goal is to cache type handlers to reduce repeated isinstance() calls
for objects of the same type.

IMPLEMENTATION STEPS:

1. Find the type detection code in datason/core.py or similar
2. Add a module-level cache:

   ```python
   # Add at module level
   _TYPE_HANDLER_CACHE = {}

   def get_cached_type_handler(obj_type):
       '''Get type handler with caching.'''
       if obj_type not in _TYPE_HANDLER_CACHE:
           _TYPE_HANDLER_CACHE[obj_type] = _compute_type_handler(obj_type)
       return _TYPE_HANDLER_CACHE[obj_type]
   ```

3. Replace direct type handler calls with cached versions:

   ```python
   # Instead of:
   handler = get_type_handler(type(obj))

   # Use:
   handler = get_cached_type_handler(type(obj))
   ```

4. Optional: Add cache size limit for memory safety:

   ```python
   MAX_CACHE_SIZE = 1000

   def get_cached_type_handler(obj_type):
       if obj_type not in _TYPE_HANDLER_CACHE:
           if len(_TYPE_HANDLER_CACHE) >= MAX_CACHE_SIZE:
               # Clear oldest entries or use LRU
               _TYPE_HANDLER_CACHE.clear()
           _TYPE_HANDLER_CACHE[obj_type] = _compute_type_handler(obj_type)
       return _TYPE_HANDLER_CACHE[obj_type]
   ```

TESTING PROCESS:

1. Run: python implement_step_1_1.py --measure-before
2. Implement the caching as described above
3. Run: python implement_step_1_1.py --measure-after
4. Run: python implement_step_1_1.py --compare

EXPECTED RESULTS:
- 15-25% improvement for homogeneous_list (repeated object types)
- 10-20% improvement for nested_repeated (repeated structures)
- Minimal impact on mixed_types (diverse object types)

ROLLBACK PLAN:
If performance degrades, simply remove the caching and revert to
original type handler calls.
"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(description="Implement and measure Step 1.1 optimization")
    parser.add_argument("--measure-before", action="store_true", help="Measure performance before optimization")
    parser.add_argument("--measure-after", action="store_true", help="Measure performance after optimization")
    parser.add_argument("--compare", action="store_true", help="Compare before/after results")
    parser.add_argument("--before-file", type=str, help="Before results file for comparison")
    parser.add_argument("--after-file", type=str, help="After results file for comparison")
    parser.add_argument("--guide", action="store_true", help="Show implementation guide")

    args = parser.parse_args()

    if args.guide:
        show_implementation_guide()
        return

    if args.measure_before:
        print("🔍 Measuring performance BEFORE type caching optimization...")
        results = benchmark_type_caching_impact()
        save_benchmark_results(results, "before")
        print("\n✅ Baseline measurements complete!")
        print("💡 Next: Implement type caching, then run with --measure-after")

    elif args.measure_after:
        print("🔍 Measuring performance AFTER type caching optimization...")
        results = benchmark_type_caching_impact()
        save_benchmark_results(results, "after")
        print("\n✅ Post-optimization measurements complete!")
        print("💡 Next: Run with --compare to see the improvement")

    elif args.compare:
        if args.before_file and args.after_file:
            compare_results(args.before_file, args.after_file)
        else:
            # Try to find the most recent before/after files
            import glob

            before_files = sorted(glob.glob("step_1_1_before_*.json"))
            after_files = sorted(glob.glob("step_1_1_after_*.json"))

            if before_files and after_files:
                print(f"📊 Comparing {before_files[-1]} vs {after_files[-1]}")
                compare_results(before_files[-1], after_files[-1])
            else:
                print("❌ No before/after files found. Run --measure-before and --measure-after first.")

    else:
        print("📋 Step 1.1 Implementation Helper")
        print("=" * 40)
        print("Usage:")
        print("  --guide           Show implementation guide")
        print("  --measure-before  Measure current performance")
        print("  --measure-after   Measure optimized performance")
        print("  --compare         Compare before/after results")


if __name__ == "__main__":
    main()
