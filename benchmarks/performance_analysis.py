#!/usr/bin/env python3
"""
Performance Analysis Script for datason
=======================================

Analyzes historical performance data to identify trends, regressions, and improvements.
Uses datason itself for robust datetime parsing!
"""

import glob
import os
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import datason


def load_performance_history() -> List[Dict[str, Any]]:
    """Load all historical performance data."""
    history_files = sorted(glob.glob("performance-history/performance_*.json"))
    results_files = sorted(glob.glob("results/performance_results_*.json"))

    all_files = history_files + results_files
    data = []

    for file_path in all_files:
        try:
            with open(file_path) as f:
                # Use datason to deserialize - handles all the datetime parsing!
                result = datason.deserialize(f.read())
                result["source_file"] = file_path
                data.append(result)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {file_path}: {e}")

    # Sort by timestamp (datason handles the datetime objects)
    data.sort(key=lambda x: x["metadata"]["timestamp"])
    return data


def analyze_performance_trends(data: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
    """Analyze performance trends over the specified time period."""
    if len(data) < 2:
        return {"error": "Not enough data for trend analysis"}

    # Filter data by time period - no timezone issues thanks to datason!
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_data = []

    for result in data:
        timestamp = result["metadata"]["timestamp"]
        # datason already parsed this as a proper datetime object
        if isinstance(timestamp, datetime):
            # Ensure timezone awareness
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if timestamp >= cutoff_date:
                recent_data.append(result)

    if len(recent_data) < 2:
        return {"error": f"Not enough data in the last {days} days"}

    # Extract key metrics over time
    metrics: Dict[str, List[Dict[str, Any]]] = {}

    for result in recent_data:
        timestamp = result["metadata"]["timestamp"]
        timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

        # Skip if benchmarks key doesn't exist (old format files)
        if "benchmarks" not in result:
            continue

        # Extract serialization metrics
        if "serialization" in result["benchmarks"]:
            for test_name, test_data in result["benchmarks"]["serialization"].items():
                for config, perf_data in test_data.items():
                    if isinstance(perf_data, dict) and "mean" in perf_data:
                        key = f"serialization.{test_name}.{config}"
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(
                            {
                                "timestamp": timestamp_str,
                                "value": perf_data["mean"] * 1000,  # Convert to ms
                            }
                        )

        # Extract deserialization metrics
        if "deserialization" in result["benchmarks"]:
            for test_name, test_data in result["benchmarks"]["deserialization"].items():
                if isinstance(test_data, dict) and "standard" in test_data:
                    key = f"deserialization.{test_name}.standard"
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append({"timestamp": timestamp_str, "value": test_data["standard"]["mean"] * 1000})

    # Calculate trends
    trends = {}
    for metric_name, values in metrics.items():
        if len(values) >= 2:
            times = [v["value"] for v in values]

            # Calculate trend statistics
            first_half = times[: len(times) // 2]
            second_half = times[len(times) // 2 :]

            trend_direction = "stable"
            trend_magnitude = 0.0

            if len(first_half) > 0 and len(second_half) > 0:
                avg_first = statistics.mean(first_half)
                avg_second = statistics.mean(second_half)

                if avg_first > 0:
                    trend_magnitude = ((avg_second - avg_first) / avg_first) * 100

                    if trend_magnitude > 10:
                        trend_direction = "regression"
                    elif trend_magnitude < -10:
                        trend_direction = "improvement"

            trends[metric_name] = {
                "direction": trend_direction,
                "magnitude": trend_magnitude,
                "current": times[-1],
                "baseline": times[0],
                "samples": len(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            }

    return {
        "period_days": days,
        "total_samples": len(recent_data),
        "date_range": {
            "start": recent_data[0]["metadata"]["timestamp"],
            "end": recent_data[-1]["metadata"]["timestamp"],
        },
        "trends": trends,
    }


def generate_performance_report(data: List[Dict[str, Any]]) -> None:
    """Generate a comprehensive performance report."""
    print("📊 Datason Performance Analysis Report")
    print("=" * 50)
    print("🎯 Using datason for robust datetime parsing!")

    if not data:
        print("❌ No performance data found")
        return

    print("\n📈 Data Summary:")
    print(f"   • Total samples: {len(data)}")

    # Handle datetime display properly
    start_time = data[0]["metadata"]["timestamp"]
    end_time = data[-1]["metadata"]["timestamp"]
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(start_time, datetime) else str(start_time)
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(end_time, datetime) else str(end_time)

    print(f"   • Date range: {start_str} → {end_str}")
    print(f"   • Latest version: {data[-1]['metadata'].get('datason_version', 'unknown')}")

    # Analyze recent trends (30 days)
    print("\n📊 Performance Trends (Last 30 Days):")
    print("-" * 40)

    trends = analyze_performance_trends(data, days=30)

    if "error" in trends:
        print(f"⚠️  {trends['error']}")
        return

    print(f"   • Analysis period: {trends['period_days']} days")
    print(f"   • Samples analyzed: {trends['total_samples']}")

    # Group trends by category
    regressions = []
    improvements = []
    stable = []

    for metric, trend in trends["trends"].items():
        if trend["direction"] == "regression":
            regressions.append((metric, trend))
        elif trend["direction"] == "improvement":
            improvements.append((metric, trend))
        else:
            stable.append((metric, trend))

    # Report regressions
    if regressions:
        print(f"\n🔴 Performance Regressions ({len(regressions)}):")
        for metric, trend in sorted(regressions, key=lambda x: x[1]["magnitude"], reverse=True):
            print(f"   • {metric}: {trend['magnitude']:+.1f}% ({trend['baseline']:.2f}ms → {trend['current']:.2f}ms)")

    # Report improvements
    if improvements:
        print(f"\n🟢 Performance Improvements ({len(improvements)}):")
        for metric, trend in sorted(improvements, key=lambda x: abs(x[1]["magnitude"]), reverse=True):
            print(f"   • {metric}: {trend['magnitude']:+.1f}% ({trend['baseline']:.2f}ms → {trend['current']:.2f}ms)")

    # Report stable metrics
    print(f"\n🟡 Stable Metrics ({len(stable)}):")
    print("   • Performance remains consistent (within ±10%)")

    # Show most variable metrics
    variable_metrics = sorted(trends["trends"].items(), key=lambda x: x[1]["std_dev"], reverse=True)[:5]
    print("\n📊 Most Variable Metrics (Top 5):")
    for metric, trend in variable_metrics:
        cv = (trend["std_dev"] / trend["current"]) * 100 if trend["current"] > 0 else 0
        print(f"   • {metric}: CV={cv:.1f}% (std={trend['std_dev']:.2f}ms)")

    print("\n💡 Recommendations:")
    if regressions:
        print("   • Investigate performance regressions")
        print("   • Consider profiling the affected code paths")
    if len([t for _, t in trends["trends"].items() if t["std_dev"] > 5]) > 3:
        print("   • High variability detected - consider environment stability")
    else:
        print("   • Performance appears stable overall")

    print("\n📅 Next Analysis: Run this script after significant changes")
    print("✨ Powered by datason's robust datetime handling!")


def main() -> None:
    """Main function."""
    print("🎯 Loading performance history...")

    # Change to benchmarks directory if not already there
    if os.path.basename(os.getcwd()) != "benchmarks":
        if os.path.exists("benchmarks"):
            os.chdir("benchmarks")
        else:
            print("❌ Error: Run this script from the project root or benchmarks directory")
            return

    # Load data
    data = load_performance_history()

    if not data:
        print("❌ No performance data found. Run some benchmarks first:")
        print("   python ci_performance_tracker.py")
        return

    # Generate report
    generate_performance_report(data)

    # Save summary
    if len(data) >= 2:
        trends = analyze_performance_trends(data)
        if "error" not in trends:
            summary_path = "performance-history/latest_analysis.json"
            os.makedirs("performance-history", exist_ok=True)
            with open(summary_path, "w") as f:
                # Use datason to serialize the analysis results too!
                json_str = datason.serialize(trends)
                f.write(json_str)
            print(f"\n💾 Analysis saved to: {summary_path}")


if __name__ == "__main__":
    main()
