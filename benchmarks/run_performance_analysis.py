#!/usr/bin/env python3
"""
On-Demand Performance Analysis Script
====================================

Run this script after each improvement to measure performance impact.
Tracks versions, compares with previous results, and provides clear feedback.

Usage:
    python run_performance_analysis.py                    # Run all tests
    python run_performance_analysis.py --quick            # Run basic tests only
    python run_performance_analysis.py --competitive      # Include competitive analysis
    python run_performance_analysis.py --compare          # Compare with previous run
    python run_performance_analysis.py --save-baseline    # Save results as new baseline
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path to import datason
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import our test suites
from ci_performance_tracker import StableBenchmarkSuite
from comprehensive_performance_suite import ComprehensivePerformanceSuite

import datason


class OnDemandPerformanceAnalyzer:
    """On-demand performance analysis with version tracking and comparison."""

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.version_info = self._get_version_info()

    def _get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information for tracking."""
        import datason

        version_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "datason_version": getattr(datason, "__version__", "unknown"),
            "python_version": sys.version,
            "git_info": self._get_git_info(),
            "dependencies": self._get_dependency_versions(),
        }

        return version_info

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit and branch information."""
        try:
            # Get current commit hash
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="..", text=True).strip()

            # Get current branch
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd="..", text=True).strip()

            # Check if there are uncommitted changes
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd="..", text=True).strip()

            return {
                "commit": commit,
                "branch": branch,
                "has_uncommitted_changes": bool(status),
                "status": status[:200] if status else "",  # Truncate long status
            }
        except subprocess.CalledProcessError:
            return {
                "commit": "unknown",
                "branch": "unknown",
                "has_uncommitted_changes": False,
                "status": "git not available",
            }

    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        versions = {}

        # Core dependencies
        try:
            import numpy

            versions["numpy"] = numpy.__version__
        except ImportError:
            versions["numpy"] = "not installed"

        try:
            import pandas

            versions["pandas"] = pandas.__version__
        except ImportError:
            versions["pandas"] = "not installed"

        # Competitive libraries
        try:
            import orjson

            versions["orjson"] = orjson.__version__
        except ImportError:
            versions["orjson"] = "not installed"

        try:
            import ujson

            versions["ujson"] = ujson.__version__
        except ImportError:
            versions["ujson"] = "not installed"

        try:
            import msgpack

            versions["msgpack"] = msgpack.version
        except ImportError:
            versions["msgpack"] = "not installed"

        return versions

    def run_quick_analysis(self) -> Dict[str, Any]:
        """Run quick performance analysis (CI-style tests)."""
        print("🏃‍♂️ Running Quick Performance Analysis")
        print("=" * 50)

        suite = StableBenchmarkSuite()
        results = suite.run_all_benchmarks()

        # Add version tracking
        results["version_info"] = self.version_info
        results["analysis_type"] = "quick"

        return results

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis with ML and competitive testing."""
        print("🔬 Running Comprehensive Performance Analysis")
        print("=" * 50)

        suite = ComprehensivePerformanceSuite()
        results = suite.run_comprehensive_suite()

        # Add version tracking
        results["version_info"] = self.version_info
        results["analysis_type"] = "comprehensive"

        return results

    def save_results(self, results: Dict[str, Any], label: str = "") -> str:
        """Save results with timestamp and version info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.version_info["datason_version"]
        commit = self.version_info["git_info"]["commit"][:8]

        if label:
            filename = f"performance_{label}_{version}_{commit}_{timestamp}.json"
        else:
            filename = f"performance_{version}_{commit}_{timestamp}.json"

        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: {filepath}")

        # Also save as latest for easy comparison
        latest_path = self.results_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return str(filepath)

    def compare_with_previous(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with previous run."""
        print("\n📊 Comparing with Previous Results")
        print("-" * 40)

        # Find the most recent previous result
        result_files = sorted(self.results_dir.glob("performance_*.json"))
        if len(result_files) < 2:
            print("⚠️  No previous results found for comparison")
            return {"status": "no_previous_results"}

        # Load previous result (second most recent)
        previous_file = result_files[-2]
        with open(previous_file) as f:
            previous_results = json.load(f)

        print(f"📋 Comparing with: {previous_file.name}")
        print(f"   Previous version: {previous_results.get('version_info', {}).get('datason_version', 'unknown')}")
        print(f"   Current version:  {current_results['version_info']['datason_version']}")

        comparison = self._perform_comparison(previous_results, current_results)

        # Print comparison summary
        self._print_comparison_summary(comparison)

        return comparison

    def _perform_comparison(self, previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed comparison between two result sets."""
        comparison = {
            "previous_version": previous.get("version_info", {}).get("datason_version", "unknown"),
            "current_version": current["version_info"]["datason_version"],
            "previous_commit": previous.get("version_info", {}).get("git_info", {}).get("commit", "unknown")[:8],
            "current_commit": current["version_info"]["git_info"]["commit"][:8],
            "performance_changes": {},
            "improvements": [],
            "regressions": [],
            "competitive_changes": {},
        }

        # Compare core benchmarks
        if "benchmarks" in previous and "benchmarks" in current:
            for category in ["serialization", "deserialization", "type_detection"]:
                self._compare_category(comparison, previous, current, category)

        # Compare competitive results if available
        if "competitive_analysis" in current:
            current_competitive = current["competitive_analysis"].get("datason_vs_competitors", {})
            if "competitive_analysis" in previous:
                previous_competitive = previous["competitive_analysis"].get("datason_vs_competitors", {})
                self._compare_competitive(comparison, previous_competitive, current_competitive)

        return comparison

    def _compare_category(self, comparison: Dict, previous: Dict, current: Dict, category: str) -> None:
        """Compare a specific benchmark category."""
        if category not in previous.get("benchmarks", {}) or category not in current.get("benchmarks", {}):
            return

        prev_cat = previous["benchmarks"][category]
        curr_cat = current["benchmarks"][category]

        for test_name in prev_cat:
            if test_name not in curr_cat:
                continue

            # Handle nested structure (serialization/deserialization)
            if isinstance(prev_cat[test_name], dict) and "standard" in prev_cat[test_name]:
                for config in ["standard", "performance_config"]:
                    if config in prev_cat[test_name] and config in curr_cat[test_name]:
                        self._compare_metric(
                            comparison,
                            f"{category}.{test_name}.{config}",
                            prev_cat[test_name][config]["mean"],
                            curr_cat[test_name][config]["mean"],
                        )
            else:
                # Handle flat structure (type_detection)
                if "mean" in prev_cat[test_name] and "mean" in curr_cat[test_name]:
                    self._compare_metric(
                        comparison, f"{category}.{test_name}", prev_cat[test_name]["mean"], curr_cat[test_name]["mean"]
                    )

    def _compare_metric(self, comparison: Dict, test_name: str, prev_time: float, curr_time: float) -> None:
        """Compare a specific metric and categorize the change."""
        change_pct = ((curr_time - prev_time) / prev_time) * 100

        comparison["performance_changes"][test_name] = {
            "previous_ms": prev_time * 1000,
            "current_ms": curr_time * 1000,
            "change_pct": change_pct,
        }

        # Categorize significant changes (>2% threshold for on-demand analysis)
        if change_pct < -2:  # Improvement
            comparison["improvements"].append(
                {
                    "test": test_name,
                    "change_pct": change_pct,
                    "previous_ms": prev_time * 1000,
                    "current_ms": curr_time * 1000,
                }
            )
        elif change_pct > 2:  # Regression
            comparison["regressions"].append(
                {
                    "test": test_name,
                    "change_pct": change_pct,
                    "previous_ms": prev_time * 1000,
                    "current_ms": curr_time * 1000,
                }
            )

    def _compare_competitive(self, comparison: Dict, previous: Dict, current: Dict) -> None:
        """Compare competitive positioning changes."""
        for competitor in current:
            if competitor in previous:
                prev_slowdown = previous[competitor]["average_slowdown_factor"]
                curr_slowdown = current[competitor]["average_slowdown_factor"]
                change_pct = ((curr_slowdown - prev_slowdown) / prev_slowdown) * 100

                comparison["competitive_changes"][competitor] = {
                    "previous_slowdown": prev_slowdown,
                    "current_slowdown": curr_slowdown,
                    "change_pct": change_pct,
                }

    def _print_comparison_summary(self, comparison: Dict) -> None:
        """Print a formatted comparison summary."""
        if comparison["improvements"]:
            print(f"\n🟢 Performance Improvements ({len(comparison['improvements'])})")
            for imp in comparison["improvements"]:
                print(
                    f"  {imp['test']}: {imp['change_pct']:+.1f}% ({imp['current_ms']:.2f}ms ← {imp['previous_ms']:.2f}ms)"
                )

        if comparison["regressions"]:
            print(f"\n🔴 Performance Regressions ({len(comparison['regressions'])})")
            for reg in comparison["regressions"]:
                print(
                    f"  {reg['test']}: {reg['change_pct']:+.1f}% ({reg['current_ms']:.2f}ms ← {reg['previous_ms']:.2f}ms)"
                )

        if comparison["competitive_changes"]:
            print("\n⚔️ Competitive Position Changes")
            for competitor, change in comparison["competitive_changes"].items():
                print(
                    f"  vs {competitor}: {change['change_pct']:+.1f}% ({change['current_slowdown']:.1f}x ← {change['previous_slowdown']:.1f}x)"
                )

        if not comparison["improvements"] and not comparison["regressions"]:
            print("\n🟡 No significant performance changes detected")

    def check_dependencies(self) -> bool:
        """Check if benchmarking dependencies are available."""
        missing = []

        if importlib.util.find_spec("numpy") is None:
            missing.append("numpy")

        if importlib.util.find_spec("pandas") is None:
            missing.append("pandas")

        if missing:
            print(f"⚠️  Missing dependencies for full analysis: {', '.join(missing)}")
            print("   Install with: pip install -r requirements-benchmarking.txt")
            return False

        return True

    def run_analysis(self, quick: bool = False, competitive: bool = False, compare: bool = False) -> Dict[str, Any]:
        """Run the requested analysis type."""
        print("🎯 Datason Performance Analysis")
        print(f"   Version: {self.version_info['datason_version']}")
        print(f"   Commit:  {self.version_info['git_info']['commit'][:8]}")
        print(f"   Branch:  {self.version_info['git_info']['branch']}")

        if self.version_info["git_info"]["has_uncommitted_changes"]:
            print("   ⚠️  WARNING: Uncommitted changes detected!")

        print()

        # Check dependencies
        if not quick and not self.check_dependencies():
            print("🔄 Falling back to quick analysis...")
            quick = True

        # Run analysis
        results = self.run_quick_analysis() if quick else self.run_comprehensive_analysis()

        # Save results
        label = "quick" if quick else "comprehensive"
        self.save_results(results, label)

        # Compare with previous if requested
        if compare:
            comparison = self.compare_with_previous(results)
            results["comparison"] = comparison

        # Add memory optimization benchmark
        print("\n🧠 Memory Allocation Optimization Benchmarks:")
        print("-" * 50)

        # Test memory efficiency with different data patterns
        memory_test_cases = {
            "large_homogeneous_ints": list(range(2000)),
            "large_homogeneous_strings": ["test_string"] * 1000,
            "repeated_small_dicts": [{"id": i, "active": True} for i in range(500)],
            "deep_nested_structure": {"level1": {"level2": {"level3": {"data": [{"item": i} for i in range(100)]}}}},
            "mixed_collection_types": {
                "lists": [[i] * 10 for i in range(50)],
                "dicts": [{"key": f"value_{i}"} for i in range(50)],
                "primitives": [1, "a", True, None] * 25,
            },
        }

        memory_results = {}
        for test_name, test_data in memory_test_cases.items():
            print(f"  🧠 Testing {test_name}...")

            # Measure multiple runs for consistency
            times = []
            for _ in range(5):
                start_time = time.time()
                _ = datason.serialize(test_data)  # Use underscore for unused result
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

            avg_time = sum(times) / len(times)
            memory_results[test_name] = {"avg_time_ms": avg_time, "min_time_ms": min(times), "max_time_ms": max(times)}
            print(f"    Datason: {avg_time:.2f}ms (avg)")

        # Store memory optimization results
        results["memory_optimization"] = memory_results

        return results


def main():
    """Main function for on-demand performance analysis."""
    parser = argparse.ArgumentParser(description="On-demand performance analysis for datason")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis only (CI-style tests)")
    parser.add_argument(
        "--competitive", action="store_true", help="Include competitive analysis (requires competitive libraries)"
    )
    parser.add_argument("--compare", action="store_true", help="Compare with previous results")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as new baseline for CI")

    args = parser.parse_args()

    analyzer = OnDemandPerformanceAnalyzer()

    # Run analysis
    results = analyzer.run_analysis(quick=args.quick, competitive=args.competitive, compare=args.compare)

    # Save as baseline if requested
    if args.save_baseline:
        print("\n📋 Saving as new baseline for CI...")
        baseline_path = "results/baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Baseline saved: {baseline_path}")

    print("\n✅ Analysis complete!")

    # Print summary recommendations
    if "competitive_analysis" in results:
        recommendations = results["competitive_analysis"].get("recommendations", [])
        if recommendations:
            print("\n💡 Key Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")

    return 0


if __name__ == "__main__":
    exit(main())
