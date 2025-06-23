"""Advanced Datason Serialization Features Demo

This script demonstrates the enhanced features of datason including:
- Simple & Direct API for common use cases
- Configurable serialization options
- Advanced type handling
- ML/AI workflow optimizations
- API response formatting
- Custom type serializers
"""

import decimal

# import json
import uuid
from collections import namedtuple
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict

# Import datason with new features
import datason
import datason as ds
from datason.config import (
    DataFrameOrient,
    DateFormat,
    NanHandling,
    SerializationConfig,
    TypeCoercion,
    get_api_config,
    get_ml_config,
    get_performance_config,
    get_strict_config,
)

# Try to import optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available - skipping DataFrame examples")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not available - skipping numpy examples")


# Sample data types for demonstration
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


Task = namedtuple("Task", ["id", "title", "priority", "due_date"])


def create_sample_data():
    """Create complex sample data with various types."""
    data = {
        # Basic types
        "string": "Hello, World!",
        "integer": 42,
        "float": 3.14159,
        "boolean": True,
        "null_value": None,
        "nan_value": float("nan"),
        # Advanced Python types
        "uuid": uuid.uuid4(),
        "decimal": decimal.Decimal("123.456789"),
        "complex_number": 3 + 4j,
        "path": Path("./test.txt"),
        "bytes_data": b"Hello bytes!",
        "set_data": {1, 2, 3, 4, 5},
        "range_data": range(10, 20, 2),
        # Date/time data
        "datetime_naive": datetime(2023, 12, 25, 10, 30, 45),
        "datetime_utc": datetime(2023, 12, 25, 10, 30, 45, tzinfo=timezone.utc),
        # Enum and namedtuple
        "priority": Priority.HIGH,
        "task": Task(
            id="task-001",
            title="Complete project documentation",
            priority=Priority.HIGH,
            due_date=datetime(2024, 1, 15),
        ),
        # Nested structures
        "nested_dict": {"level1": {"level2": {"data": [1, 2, 3], "meta": {"created": datetime.now()}}}},
        "mixed_list": ["string", 42, {"nested": True}, uuid.uuid4(), datetime.now()],
    }

    # Add pandas DataFrame if available
    if HAS_PANDAS:
        data["dataframe"] = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [85.5, 92.0, 78.5],
                "active": [True, False, True],
                "last_login": [
                    pd.Timestamp("2023-12-01"),
                    pd.Timestamp("2023-11-28"),
                    pd.NaT,  # Pandas NaT (Not a Time)
                ],
            }
        )

        data["series"] = pd.Series([1, 2, 3, 4, 5], name="values")
        data["categorical"] = pd.Categorical(["A", "B", "A", "C", "B"])

    # Add numpy arrays if available
    if HAS_NUMPY:
        data["numpy_array"] = np.array([1, 2, 3, 4, 5])
        data["numpy_2d"] = np.array([[1, 2], [3, 4]])
        data["numpy_float"] = np.float64(3.14159)
        data["numpy_bool"] = np.bool_(True)
        data["numpy_nan"] = np.float64("nan")

    return data


def demo_simple_direct_api():
    """Demonstrate the simple & direct API - no configuration needed!"""
    print("🚀 Simple & Direct API Demo")
    print("=" * 50)

    data = create_sample_data()

    print("\n✨ Simple API - Just Pick the Right Function:")

    # API-optimized serialization
    print("\n1️⃣ dump_api() - Clean JSON for web APIs:")
    api_result = datason.dump_api(data)
    print("   Perfect for FastAPI/Flask responses")
    print(f"   UUID format: {type(api_result.get('uuid', 'N/A'))}")
    print("   Clean, predictable JSON structure")

    # ML-optimized serialization
    print("\n2️⃣ dump_ml() - Optimized for ML/AI objects:")
    datason.dump_ml(data)  # Auto-handled ML objects
    print("   Handles NumPy arrays, tensors, DataFrames")
    print("   Preserves ML-specific data types")

    # Fast serialization
    print("\n3️⃣ dump_fast() - Performance optimized:")
    datason.dump_fast(data)  # Performance optimized
    print("   Minimal processing for speed")
    print("   Best for large datasets")

    # Secure serialization
    sensitive_data = {
        "user": "john_doe",
        "email": "john@example.com",
        "ssn": "123-45-6789",
        "api_key": "sk_abc123xyz789",
        "notes": "Regular customer",
    }

    print("\n4️⃣ dump_secure() - Automatic PII redaction:")
    secure_result = datason.dump_secure(sensitive_data)
    print(f"   Input: {sensitive_data}")
    print(f"   Output: {secure_result}")
    print("   ✅ Sensitive data automatically redacted!")

    print("\n💡 Simple API Benefits:")
    print("   • No configuration needed")
    print("   • Intention-revealing function names")
    print("   • Automatic optimizations")
    print("   • Just pick the right function for your use case!")

    return api_result


def demo_basic_serialization():
    """Demonstrate basic serialization with different configurations."""
    print("\n🔧 Traditional API with Configuration")
    print("=" * 50)

    data = create_sample_data()

    # Default serialization
    print("\n📋 Default Configuration:")
    result = datason.serialize(data)
    print(f"✅ Serialized {len(result)} top-level keys")
    print(f"📅 DateTime format: {result.get('datetime_naive', 'N/A')}")
    print(f"🆔 UUID format: {type(result.get('uuid', 'N/A'))}")
    print(f"🔢 Decimal preserved: {type(result.get('decimal', 'N/A'))}")

    return result


def demo_date_formats():
    """Demonstrate different date/time serialization formats."""
    print("\n⏰ Date/Time Format Demo")
    print("=" * 50)

    dt = datetime(2023, 12, 25, 10, 30, 45, tzinfo=timezone.utc)

    formats = [
        (DateFormat.ISO, "ISO 8601"),
        (DateFormat.UNIX, "Unix timestamp"),
        (DateFormat.UNIX_MS, "Unix milliseconds"),
        (DateFormat.STRING, "Human readable"),
    ]

    for date_format, description in formats:
        config = SerializationConfig(date_format=date_format)
        result = datason.serialize(dt, config=config)
        print(f"📅 {description}: {result}")

    # Custom format
    custom_config = SerializationConfig(date_format=DateFormat.CUSTOM, custom_date_format="%B %d, %Y at %I:%M %p")
    result = datason.serialize(dt, config=custom_config)
    print(f"📅 Custom format: {result}")


def demo_nan_handling():
    """Demonstrate different NaN handling strategies."""
    print("\n🤷 NaN Handling Demo")
    print("=" * 50)

    data = [1, 2, float("nan"), 4, None]

    strategies = [
        (NanHandling.NULL, "Convert to null"),
        (NanHandling.STRING, "Convert to string"),
        (NanHandling.KEEP, "Keep as-is"),
        (NanHandling.DROP, "Drop from collections"),
    ]

    for strategy, description in strategies:
        config = SerializationConfig(nan_handling=strategy)
        result = datason.serialize(data, config=config)
        print(f"🎯 {description}: {result}")


def demo_dataframe_orientations():
    """Demonstrate pandas DataFrame serialization orientations."""
    if not HAS_PANDAS:
        print("⚠️  Skipping DataFrame demo - pandas not available")
        return

    print("\n📊 DataFrame Orientation Demo")
    print("=" * 50)

    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "London"]})

    orientations = [
        (DataFrameOrient.RECORDS, "Records (default)"),
        (DataFrameOrient.SPLIT, "Split format"),
        (DataFrameOrient.COLUMNS, "Column-oriented"),
        (DataFrameOrient.VALUES, "Values only"),
    ]

    for orient, description in orientations:
        config = SerializationConfig(dataframe_orient=orient)
        result = datason.serialize(df, config=config)
        print(f"\n📋 {description}:")
        print(
            ds.dumps_json(result, indent=2)[:200] + "..." if len(str(result)) > 200 else ds.dumps_json(result, indent=2)
        )


def demo_type_coercion():
    """Demonstrate different type coercion strategies."""
    print("\n🔄 Type Coercion Demo")
    print("=" * 50)

    data = {
        "uuid": uuid.uuid4(),
        "complex": 3 + 4j,
        "decimal": decimal.Decimal("123.456"),
        "path": Path("./test.txt"),
        "enum": Priority.HIGH,
    }

    strategies = [
        (TypeCoercion.SAFE, "Safe (default)"),
        (TypeCoercion.STRICT, "Strict (preserve details)"),
        (TypeCoercion.AGGRESSIVE, "Aggressive (simplify)"),
    ]

    for strategy, description in strategies:
        config = SerializationConfig(type_coercion=strategy)
        result = datason.serialize(data, config=config)
        print(f"\n🎯 {description}:")
        print(f"  UUID: {type(result['uuid'])} - {str(result['uuid'])[:40]}...")
        print(f"  Complex: {result['complex']}")
        print(f"  Enum: {result['enum']}")


def demo_preset_configurations():
    """Demonstrate preset configurations for different use cases."""
    print("\n⚙️  Preset Configuration Demo")
    print("=" * 50)

    data = create_sample_data()

    presets = [
        (get_ml_config(), "ML/AI Workflow"),
        (get_api_config(), "API Response"),
        (get_performance_config(), "Performance Optimized"),
        (get_strict_config(), "Strict Type Checking"),
    ]

    for config, description in presets:
        result = datason.serialize(data, config=config)
        print(f"\n🎛️  {description}:")
        print(f"  📅 Date format: {result.get('datetime_naive', 'N/A')}")
        print(f"  🔢 Decimal: {type(result.get('decimal', 'N/A'))}")
        print(f"  📊 Keys sorted: {list(result.keys())[:3] == sorted(list(result.keys())[:3])}")

        if HAS_PANDAS and "dataframe" in result:
            df_result = result["dataframe"]
            if isinstance(df_result, list):
                print(f"  📊 DataFrame: records format ({len(df_result)} rows)")
            elif isinstance(df_result, dict) and "data" in df_result:
                print("  📊 DataFrame: split format")
            else:
                print(f"  📊 DataFrame: {type(df_result)}")


def demo_custom_serializers():
    """Demonstrate custom type serializers."""
    print("\n🛠️  Custom Serializer Demo")
    print("=" * 50)

    # Define a custom class
    class Person:
        def __init__(self, name: str, age: int, skills: list):
            self.name = name
            self.age = age
            self.skills = skills

    # Define custom serializer
    def serialize_person(person: Person) -> Dict[str, Any]:
        return {
            "_type": "Person",
            "full_name": person.name.title(),
            "age_group": "adult" if person.age >= 18 else "minor",
            "skill_count": len(person.skills),
            "skills": sorted(person.skills),
        }

    # Create test data
    person = Person("john doe", 28, ["python", "machine learning", "data analysis"])

    # Serialize without custom handler
    print("🔸 Default serialization:")
    default_result = datason.serialize(person)
    print(ds.dumps_json(default_result, indent=2))

    # Serialize with custom handler
    print("\n🔸 Custom serializer:")
    config = SerializationConfig(custom_serializers={Person: serialize_person})
    custom_result = datason.serialize(person, config=config)
    print(ds.dumps_json(custom_result, indent=2))


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🚀 Convenience Functions Demo")
    print("=" * 50)

    dt = datetime(2023, 12, 25, 10, 30, 45, tzinfo=timezone.utc)

    # Quick configuration
    print("🔸 serialize_with_config():")
    result = datason.serialize_with_config(dt, date_format="unix_ms", sort_keys=True)
    print(f"  Result: {result}")

    # Global configuration
    print("\n🔸 Global configuration:")
    original = datason.get_default_config()

    try:
        # Set ML config globally
        datason.configure(get_ml_config())
        print("  Set ML config as default")

        # Now all serializations use ML config
        result = datason.serialize(dt)
        print(f"  Serialized with ML config: {result}")

    finally:
        # Restore original
        datason.set_default_config(original)
        print("  Restored original config")


def demo_performance_comparison():
    """Demonstrate performance differences between configurations."""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 50)

    import time

    # Create larger dataset for timing
    large_data = {
        "items": [
            {
                "id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "value": decimal.Decimal(str(i * 3.14159)),
                "metadata": {"index": i, "processed": i % 2 == 0},
            }
            for i in range(100)
        ]
    }

    configs = [
        (SerializationConfig(), "Default"),
        (get_performance_config(), "Performance Optimized"),
        (get_strict_config(), "Strict (detailed)"),
    ]

    for config, name in configs:
        start_time = time.time()
        _ = datason.serialize(large_data, config=config)
        end_time = time.time()

        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"⏱️  {name}: {duration:.2f}ms")


def main():
    """Run all demonstrations."""
    print("🎯 Datason Features Demo - Simple API & Advanced Configuration")
    print("=" * 70)
    print("This demo showcases the simple & direct API alongside advanced")
    print("configuration options for specialized use cases.")
    print()

    try:
        # Simple API first - no configuration needed!
        demo_simple_direct_api()

        print("\n" + "=" * 70)
        print("📚 Advanced Configuration Examples:")
        print()

        # Advanced configuration examples
        demo_basic_serialization()
        demo_date_formats()
        demo_nan_handling()
        demo_dataframe_orientations()
        demo_type_coercion()
        demo_preset_configurations()
        demo_custom_serializers()
        demo_convenience_functions()
        demo_performance_comparison()

        print("\n✅ All demos completed successfully!")
        print("\n🚀 Quick Start Tips:")
        print("  • Use dump_api() for web APIs (clean JSON)")
        print("  • Use dump_ml() for ML/AI objects")
        print("  • Use dump_secure() for sensitive data")
        print("  • Use dump_fast() for performance")
        print("\n⚙️ Advanced Configuration:")
        print("  • Use get_ml_config() for ML/AI workflows")
        print("  • Use get_api_config() for web API responses")
        print("  • Use get_performance_config() for large datasets")
        print("  • Create custom configurations for specialized needs")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
