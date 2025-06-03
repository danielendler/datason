#!/usr/bin/env python3
"""
Comprehensive Performance Suite for datason
===========================================

This script provides comprehensive performance testing including:
1. ML library integration benchmarks
2. Competitive comparisons with other serialization tools
3. Real-world complexity scenarios
4. Plugin performance impact analysis

This complements the basic CI tracker with deeper analysis.
"""

import json
import pickle  # nosec B403 - Safe usage for benchmarking only, controlled data
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List

import datason

# Optional ML library imports
ML_LIBRARIES = {}
try:
    import numpy as np

    ML_LIBRARIES["numpy"] = np.__version__
except ImportError:
    np = None

try:
    import pandas as pd

    ML_LIBRARIES["pandas"] = pd.__version__
except ImportError:
    pd = None

# Competitive serialization libraries
COMPETITIVE_LIBS = {}
try:
    import orjson

    COMPETITIVE_LIBS["orjson"] = orjson.__version__
except ImportError:
    orjson = None

try:
    import ujson

    COMPETITIVE_LIBS["ujson"] = ujson.__version__
except ImportError:
    ujson = None


class ComprehensivePerformanceSuite:
    """Comprehensive performance testing including ML libraries and competitive analysis."""

    def __init__(self):
        self.results = {}
        self.metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "datason_version": getattr(datason, "__version__", "unknown"),
            "ml_libraries": ML_LIBRARIES,
            "competitive_libs": COMPETITIVE_LIBS,
        }

    def benchmark_function(self, func, iterations: int = 5, warmup: int = 1) -> Dict[str, float]:
        """Benchmark a function with error handling."""
        try:
            # Warmup
            for _ in range(warmup):
                func()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = func()
                end = time.perf_counter()
                times.append(end - start)

            return {
                "mean": mean(times),
                "min": min(times),
                "max": max(times),
                "std": stdev(times) if len(times) > 1 else 0.0,
                "iterations": iterations,
                "result_size": len(str(result)) if result is not None else 0,
            }
        except Exception as e:
            return {
                "error": str(e),
                "mean": float("inf"),
                "iterations": 0,
            }

    def create_ml_datasets(self) -> Dict[str, Any]:
        """Create realistic ML datasets for benchmarking."""
        datasets = {}

        # NumPy arrays (common in ML)
        if np is not None:
            datasets["numpy_arrays"] = {
                "small_array": np.random.randn(100, 10).tolist(),  # Convert to list for JSON compatibility
                "medium_array": np.random.randn(500, 20).tolist(),
                "mixed_types": {
                    "floats": np.random.randn(100, 5).tolist(),
                    "ints": np.random.randint(0, 100, (100, 3)).tolist(),
                    "bools": np.random.choice([True, False], (100, 2)).tolist(),
                },
            }

        # Pandas DataFrames (very common in data science)
        if pd is not None:
            df_small = pd.DataFrame(
                {
                    "id": range(50),
                    "name": [f"item_{i}" for i in range(50)],
                    "value": np.random.randn(50).tolist() if np is not None else [float(i) for i in range(50)],
                    "category": ["A", "B", "C"] * 16 + ["A", "B"],
                }
            )

            datasets["pandas_dataframes"] = {
                "small_df": df_small.to_dict("records"),
                "mixed_df": pd.DataFrame(
                    {
                        "strings": ["text_" + str(i) for i in range(100)],
                        "numbers": range(100),
                        "decimals": [Decimal(str(i * 0.01)) for i in range(100)],
                    }
                ).to_dict("records"),
            }

        return datasets

    def create_complex_real_world_data(self) -> Dict[str, Any]:
        """Create complex, real-world-like data structures."""
        return {
            "enterprise_api_response": {
                "metadata": {
                    "api_version": "v2.1",
                    "timestamp": datetime.now(timezone.utc),
                    "request_id": str(uuid.uuid4()),
                    "pagination": {
                        "page": 1,
                        "per_page": 50,  # Reduced size for faster testing
                        "total": 1000,
                        "has_next": True,
                    },
                    "performance_metrics": {
                        "query_time_ms": 45.67,
                        "cache_hit_ratio": 0.89,
                        "memory_usage_mb": 234.5,
                    },
                },
                "data": [
                    {
                        "user_id": str(uuid.uuid4()),
                        "profile": {
                            "personal": {
                                "name": f"User {i}",
                                "email": f"user{i}@example.com",
                                "created_at": datetime.now(timezone.utc),
                            },
                            "preferences": {
                                "notifications": {
                                    "email": i % 2 == 0,
                                    "push": i % 3 == 0,
                                },
                                "privacy": {
                                    "profile_visibility": ["public", "friends", "private"][i % 3],
                                    "data_sharing": i % 4 == 0,
                                },
                            },
                        },
                        "activity": {
                            "posts": [
                                {
                                    "post_id": str(uuid.uuid4()),
                                    "content": f"Post content {j} from user {i}",
                                    "timestamp": datetime.now(timezone.utc),
                                    "engagement": {
                                        "likes": j * 3,
                                        "comments": j * 2,
                                        "shares": j,
                                    },
                                }
                                for j in range(min(i + 1, 3))  # Variable number of posts
                            ],
                        },
                    }
                    for i in range(50)  # Reduced dataset size
                ],
            },
        }

    def benchmark_competitive_serialization(self, data: Any, name: str) -> Dict[str, Any]:
        """Benchmark datason against competitive serialization libraries."""
        results = {"test_name": name}

        # Datason (our library)
        results["datason_standard"] = self.benchmark_function(lambda: datason.serialize(data))

        try:
            perf_config = datason.get_performance_config()
            results["datason_performance"] = self.benchmark_function(
                lambda: datason.serialize(data, config=perf_config)
            )
        except Exception as e:
            results["datason_performance"] = {"error": str(e)}

        # Standard JSON (baseline)
        results["json_standard"] = self.benchmark_function(lambda: json.dumps(data, default=str, ensure_ascii=False))

        # Competitive libraries
        if orjson is not None:
            results["orjson"] = self.benchmark_function(lambda: orjson.dumps(data, default=str).decode())

        if ujson is not None:
            results["ujson"] = self.benchmark_function(lambda: ujson.dumps(data, ensure_ascii=False, default=str))

        # Pickle (for complex Python objects)
        results["pickle"] = self.benchmark_function(lambda: pickle.dumps(data))

        return results

    def run_ml_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks on ML library data."""
        print("\n" + "=" * 60)
        print("ML LIBRARY PERFORMANCE BENCHMARKS")
        print("=" * 60)

        ml_datasets = self.create_ml_datasets()
        results = {}

        for library, datasets in ml_datasets.items():
            print(f"\n🧠 Testing {library} integration:")
            results[library] = {}

            for dataset_name, data in datasets.items():
                print(f"  📊 {dataset_name}...")
                results[library][dataset_name] = self.benchmark_competitive_serialization(
                    data, f"{library}.{dataset_name}"
                )

                # Print quick summary
                datason_time = results[library][dataset_name]["datason_standard"]["mean"]
                print(f"    Datason: {datason_time * 1000:.2f}ms")

        return results

    def run_complex_data_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks on complex, real-world data."""
        print("\n" + "=" * 60)
        print("COMPLEX REAL-WORLD DATA BENCHMARKS")
        print("=" * 60)

        complex_data = self.create_complex_real_world_data()
        results = {}

        for dataset_name, data in complex_data.items():
            print(f"\n🌍 Testing {dataset_name}...")
            results[dataset_name] = self.benchmark_competitive_serialization(data, dataset_name)

            # Print competitive analysis
            res = results[dataset_name]
            datason_time = res["datason_standard"]["mean"]
            json_time = res["json_standard"]["mean"]

            print(f"  Datason Standard: {datason_time * 1000:.2f}ms")
            print(f"  JSON Standard: {json_time * 1000:.2f}ms")

            if "orjson" in res and "mean" in res["orjson"]:
                orjson_time = res["orjson"]["mean"]
                print(f"  OrJSON: {orjson_time * 1000:.2f}ms")
                print(f"  Datason vs OrJSON: {(datason_time / orjson_time):.1f}x slower")

        return results

    def analyze_competitive_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance relative to competitive libraries."""
        analysis = {
            "competitive_summary": {},
            "datason_vs_competitors": {},
            "recommendations": [],
        }

        all_comparisons = []

        for category, benchmarks in results.items():
            if category in ["ml_benchmarks", "complex_data_benchmarks"]:
                for test_name, test_results in benchmarks.items():
                    if isinstance(test_results, dict):
                        datason_time = test_results.get("datason_standard", {}).get("mean", float("inf"))

                        # Compare with each competitor
                        for competitor, competitor_results in test_results.items():
                            if competitor.startswith("datason"):
                                continue

                            if isinstance(competitor_results, dict) and "mean" in competitor_results:
                                competitor_time = competitor_results["mean"]
                                if competitor_time > 0:
                                    ratio = datason_time / competitor_time
                                    all_comparisons.append(
                                        {
                                            "test": f"{category}.{test_name}",
                                            "competitor": competitor,
                                            "datason_time_ms": datason_time * 1000,
                                            "competitor_time_ms": competitor_time * 1000,
                                            "slowdown_factor": ratio,
                                        }
                                    )

        # Summarize competitive position
        if all_comparisons:
            by_competitor = {}
            for comp in all_comparisons:
                competitor = comp["competitor"]
                if competitor not in by_competitor:
                    by_competitor[competitor] = []
                by_competitor[competitor].append(comp["slowdown_factor"])

            for competitor, ratios in by_competitor.items():
                avg_slowdown = mean(ratios)
                analysis["datason_vs_competitors"][competitor] = {
                    "average_slowdown_factor": avg_slowdown,
                    "best_case": min(ratios),
                    "worst_case": max(ratios),
                    "test_count": len(ratios),
                }

        # Generate recommendations
        analysis["recommendations"] = self._generate_competitive_recommendations(analysis["datason_vs_competitors"])

        return analysis

    def _generate_competitive_recommendations(self, competitive_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on competitive analysis."""
        recommendations = []

        for competitor, metrics in competitive_data.items():
            avg_slowdown = metrics["average_slowdown_factor"]

            if competitor == "orjson" and avg_slowdown > 5:
                recommendations.append(
                    f"🔥 CRITICAL: {avg_slowdown:.1f}x slower than OrJSON. "
                    "Consider Rust core for JSON operations (Phase 4)."
                )
            elif competitor == "json_standard" and avg_slowdown > 10:
                recommendations.append(
                    f"⚠️  HIGH: {avg_slowdown:.1f}x slower than standard JSON. "
                    "Prioritize fast-path optimizations (Phase 1)."
                )
            elif competitor == "pickle" and avg_slowdown > 3:
                recommendations.append(
                    f"📦 MEDIUM: {avg_slowdown:.1f}x slower than pickle for complex objects. "
                    "Consider binary serialization options."
                )

        if not recommendations:
            recommendations.append("✅ Performance is competitive across all tested libraries.")

        return recommendations

    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive performance suite."""
        print("🚀 Starting Comprehensive Performance Suite")
        print("=" * 80)

        results = {
            "metadata": self.metadata,
            "ml_benchmarks": {},
            "complex_data_benchmarks": {},
            "competitive_analysis": {},
        }

        # ML library benchmarks
        if ML_LIBRARIES:
            results["ml_benchmarks"] = self.run_ml_benchmarks()
        else:
            print("\n⚠️  No ML libraries available. Install numpy, pandas for full testing.")

        # Complex real-world data benchmarks
        results["complex_data_benchmarks"] = self.run_complex_data_benchmarks()

        # Competitive analysis
        results["competitive_analysis"] = self.analyze_competitive_performance(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_performance_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive performance summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("=" * 80)

        competitive_analysis = results.get("competitive_analysis", {})
        competitors = competitive_analysis.get("datason_vs_competitors", {})

        if competitors:
            print("\n📊 Competitive Position:")
            for competitor, metrics in competitors.items():
                avg_slowdown = metrics["average_slowdown_factor"]
                print(f"  vs {competitor:12}: {avg_slowdown:6.1f}x slower (avg)")

        recommendations = competitive_analysis.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")

        print("\n🔬 Test Coverage:")
        print(f"  ML Libraries: {len(ML_LIBRARIES)} available")
        print(f"  Competitive: {len(COMPETITIVE_LIBS)} available")
        print(f"  Test Categories: {len([k for k in results if 'benchmarks' in k])}")


def main():
    """Main function for comprehensive performance testing."""
    suite = ComprehensivePerformanceSuite()
    suite.run_comprehensive_suite()


if __name__ == "__main__":
    main()
