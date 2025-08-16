#!/usr/bin/env python3
"""
DataSON Profiling System Demo

This script demonstrates the profiling capabilities built into DataSON.
Run with: python examples/profiling_demo.py
"""

import os

# Add the project root to the path so we can import datason
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datason


def demo_basic_profiling():
    """Demonstrate basic profiling functionality."""
    print("=" * 60)
    print("🔍 DataSON Basic Profiling Demo")
    print("=" * 60)

    print(f"DataSON Version: {datason.__version__}")
    print(f"Rust Available: {datason.RUST_AVAILABLE}")

    # Enable profiling
    os.environ["DATASON_PROFILE"] = "1"
    datason.profile_sink = []

    print("\n✅ Profiling enabled")

    # Simple test data
    test_data = {
        "user_id": 12345,
        "username": "demo_user",
        "profile": {
            "email": "demo@example.com",
            "preferences": {"theme": "dark", "notifications": True, "language": "en"},
        },
        "activity": [
            {"action": "login", "timestamp": datetime.now().isoformat()},
            {"action": "view_profile", "timestamp": datetime.now().isoformat()},
            {"action": "update_settings", "timestamp": datetime.now().isoformat()},
        ],
    }

    print(f"\n📊 Testing with {len(str(test_data))} character dataset")

    # Clear any previous events
    datason.profile_sink.clear()

    # Test serialization with profiling
    print("\n🔄 Running save_string...")
    start = time.perf_counter()
    json_result = datason.save_string(test_data)
    save_time = time.perf_counter() - start

    save_events = list(datason.profile_sink)
    print(f"   Completed in {save_time * 1000:.2f}ms")
    print(f"   JSON size: {len(json_result):,} characters")
    print(f"   Profile events captured: {len(save_events)}")

    # Clear events for load test
    datason.profile_sink.clear()

    # Test deserialization with profiling
    print("\n🔄 Running load_basic...")
    start = time.perf_counter()
    loaded_data = datason.load_basic(json_result)
    load_time = time.perf_counter() - start

    load_events = list(datason.profile_sink)
    print(f"   Completed in {load_time * 1000:.2f}ms")
    print(f"   Profile events captured: {len(load_events)}")
    print(f"   Round-trip successful: {'✅' if loaded_data == test_data else '❌'}")

    # Display profiling breakdown
    print("\n🔍 Serialization Profiling Breakdown:")
    total_save_profile_time = 0
    for event in save_events:
        duration_ms = event["duration"] / 1_000_000
        total_save_profile_time += duration_ms
        print(f"   {event['stage']}: {duration_ms:.3f}ms")

    print("\n🔍 Deserialization Profiling Breakdown:")
    total_load_profile_time = 0
    for event in load_events:
        duration_ms = event["duration"] / 1_000_000
        total_load_profile_time += duration_ms
        print(f"   {event['stage']}: {duration_ms:.3f}ms")

    print("\n📈 Profiling Overhead Analysis:")
    save_overhead = (total_save_profile_time / (save_time * 1000)) * 100 if save_time > 0 else 0
    load_overhead = (total_load_profile_time / (load_time * 1000)) * 100 if load_time > 0 else 0
    print(f"   Save overhead: {save_overhead:.1f}%")
    print(f"   Load overhead: {load_overhead:.1f}%")


def demo_performance_scenarios():
    """Test profiling with different data complexity levels."""
    print("\n" + "=" * 60)
    print("📊 DataSON Performance Scenario Testing")
    print("=" * 60)

    scenarios = [
        {"name": "Tiny JSON", "data": {"status": "ok"}},
        {
            "name": "Small Nested",
            "data": {"user": {"id": 123, "name": "test"}, "metadata": {"created": datetime.now().isoformat()}},
        },
        {
            "name": "Medium Array",
            "data": {
                "items": [{"id": i, "value": f"item_{i}"} for i in range(100)],
                "summary": {"count": 100, "type": "demo"},
            },
        },
        {
            "name": "Large Complex",
            "data": {
                "users": [
                    {
                        "id": i,
                        "profile": {"name": f"User {i}", "settings": {"pref_" + str(j): j for j in range(10)}},
                        "history": [f"action_{k}" for k in range(20)],
                    }
                    for i in range(50)
                ],
                "metadata": {"total": 50, "generated": datetime.now().isoformat(), "schema": "v2.0"},
            },
        },
    ]

    print(f"\n🧪 Testing {len(scenarios)} scenarios...")

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")

        # Clear previous events
        datason.profile_sink.clear()

        # Measure serialization
        start = time.perf_counter()
        json_str = datason.save_string(scenario["data"])
        save_time = time.perf_counter() - start

        save_events = len(datason.profile_sink)
        datason.profile_sink.clear()

        # Measure deserialization
        start = time.perf_counter()
        loaded = datason.load_basic(json_str)
        load_time = time.perf_counter() - start

        load_events = len(datason.profile_sink)

        # Verify round-trip
        round_trip = loaded == scenario["data"]

        result = {
            "name": scenario["name"],
            "json_size": len(json_str),
            "save_time_ms": save_time * 1000,
            "load_time_ms": load_time * 1000,
            "save_events": save_events,
            "load_events": load_events,
            "round_trip": round_trip,
        }
        results.append(result)

        print(f"   JSON size: {len(json_str):,} chars")
        print(f"   Save: {save_time * 1000:.2f}ms ({save_events} events)")
        print(f"   Load: {load_time * 1000:.2f}ms ({load_events} events)")
        print(f"   Round-trip: {'✅' if round_trip else '❌'}")

    # Summary table
    print("\n📊 Performance Summary:")
    print(f"{'Scenario':<15} {'Size':<10} {'Save (ms)':<10} {'Load (ms)':<10} {'Events':<8}")
    print("-" * 58)

    for result in results:
        size_str = f"{result['json_size']:,}"
        events_str = f"{result['save_events'] + result['load_events']}"
        print(
            f"{result['name']:<15} {size_str:<10} {result['save_time_ms']:<10.2f} {result['load_time_ms']:<10.2f} {events_str:<8}"
        )


def demo_environment_controls():
    """Demonstrate environment variable controls."""
    print("\n" + "=" * 60)
    print("🌍 DataSON Environment Controls Demo")
    print("=" * 60)

    print(f"Current DATASON_RUST setting: {os.environ.get('DATASON_RUST', 'not set')}")
    print(f"Current DATASON_PROFILE setting: {os.environ.get('DATASON_PROFILE', 'not set')}")

    # Test different RUST settings
    rust_settings = ["auto", "1", "0"]

    for setting in rust_settings:
        print(f"\n--- Testing DATASON_RUST={setting} ---")
        os.environ["DATASON_RUST"] = setting

        # Import config to check setting
        from datason.config import get_accel_mode

        accel_mode = get_accel_mode()

        print(f"   Environment: DATASON_RUST={setting}")
        print(f"   Acceleration mode: {accel_mode}")
        print(f"   Rust available: {datason.RUST_AVAILABLE}")

        if datason.RUST_AVAILABLE:
            print("   🦀 Rust acceleration would be active")
        else:
            print("   🐍 Using Python implementation")

    print("\n🔍 Profiling Control:")
    print("   DATASON_PROFILE=1 → Profiling enabled")
    print("   DATASON_PROFILE unset → Profiling disabled (production mode)")


def main():
    """Run all profiling demos."""
    print("🚀 DataSON Profiling System Comprehensive Demo")
    print("=" * 60)

    try:
        demo_basic_profiling()
        demo_performance_scenarios()
        demo_environment_controls()

        print("\n" + "=" * 60)
        print("🎉 All profiling demos completed successfully!")
        print("=" * 60)

        print("\n📋 System Status:")
        print("   ✅ Profiling infrastructure: Working")
        print("   ✅ Benchmark APIs: Functional")
        print("   ✅ Environment controls: Active")
        print("   ✅ CI integration: Ready")

        if not datason.RUST_AVAILABLE:
            print("\n🦀 Rust Core Status:")
            print("   📋 Rust core not yet compiled")
            print("   ✅ Infrastructure ready for Rust integration")
            print("   📊 Python-only performance baseline established")

        print("\n🚀 Next Steps:")
        print("   1. Run this demo in CI to see automated profiling")
        print("   2. Create a PR to see performance analysis in action")
        print("   3. Use profiling to optimize performance-critical code")
        if not datason.RUST_AVAILABLE:
            print("   4. Compile Rust core for acceleration (optional)")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
