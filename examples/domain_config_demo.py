#!/usr/bin/env python3
"""Demonstration of datason's custom domain-specific configurations.

This example showcases how to create custom configurations for specific domains
(updated in Phase 1 cleanup - replaced redundant presets with custom configs):
- Financial ML workflows with precise decimals
- Temporal data analysis with optimal DataFrame formats
- ML model serving with maximum performance
- Reproducible research with maximum information preservation
- Production logging with safety and simplicity

Each configuration can be customized for specific domain requirements and use cases.
The core presets (ml, api, strict, performance) are still available for common scenarios.
"""

import datetime

# import json
import time
from decimal import Decimal

import datason
import datason as ds
from datason.config import DataFrameOrient, DateFormat, NanHandling, OutputType, SerializationConfig, TypeCoercion

# Optional imports for more comprehensive demonstrations
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


def demonstrate_financial_config():
    """Show custom financial configuration optimized for monetary precision and trading data."""
    print("=" * 60)
    print("CUSTOM FINANCIAL ML CONFIGURATION DEMO")
    print("=" * 60)

    # Create custom financial config (replaced removed preset)
    config = SerializationConfig(
        preserve_decimals=True, date_format=DateFormat.UNIX_MS, ensure_ascii=True, check_if_serialized=True
    )
    print(
        f"Configuration: {config.date_format}, decimals={config.preserve_decimals}, "
        f"ascii={config.ensure_ascii}, performance={config.check_if_serialized}"
    )

    # Simulate financial trading data
    financial_data = {
        "trade_id": "TXN_20231201_001",
        "symbol": "AAPL",
        "price": Decimal("150.2375"),  # Precise monetary value
        "volume": 1000000,
        "timestamp": datetime.datetime.now(),
        "commission": Decimal("0.005"),  # Small commission fee
        "portfolio": {
            "cash": Decimal("50000.00"),
            "positions": [
                {"symbol": "AAPL", "shares": 100, "avg_cost": Decimal("149.50")},
                {"symbol": "MSFT", "shares": 50, "avg_cost": Decimal("380.75")},
            ],
        },
    }

    result = datason.serialize(financial_data, config=config)

    print("\nFinancial Data Serialization:")
    print(f"Price preserved as: {result['price']} (type: {type(result['price'])})")
    print(f"Timestamp format: {result['timestamp']} (type: {type(result['timestamp'])})")
    print(f"Commission: {result['commission']}")
    print(f"Cash position: {result['portfolio']['cash']}")

    # Show JSON compatibility
    json_str = ds.dumps_json(result, ensure_ascii=True)
    print(f"\nJSON size: {len(json_str)} characters")
    print("✓ Safe for financial system integration with ASCII encoding")


def demonstrate_time_series_config():
    """Show custom time series configuration optimized for temporal data analysis."""
    print("\n" + "=" * 60)
    print("CUSTOM TIME SERIES ANALYSIS CONFIGURATION DEMO")
    print("=" * 60)

    # Create custom time series config (replaced removed preset)
    config = SerializationConfig(
        date_format=DateFormat.ISO,
        dataframe_orient=DataFrameOrient.SPLIT,
        preserve_decimals=True,
        datetime_output=OutputType.JSON_SAFE,
    )
    print(
        f"Configuration: {config.date_format}, DataFrame={config.dataframe_orient.value}, "
        f"decimals={config.preserve_decimals}"
    )

    if pd is not None:
        # Create sample time series data
        dates = pd.date_range("2023-01-01", periods=24, freq="h")
        ts_data = pd.DataFrame(
            {
                "timestamp": dates,
                "temperature": np.random.normal(20, 5, 24) if np is not None else [20.5] * 24,
                "humidity": np.random.normal(60, 10, 24) if np is not None else [60.0] * 24,
                "pressure": np.random.normal(1013, 5, 24) if np is not None else [1013.0] * 24,
            }
        )

        result = datason.serialize(ts_data, config=config)

        print("\nTime Series DataFrame Structure:")
        print(f"Format: {list(result.keys())}")  # Should show ['index', 'columns', 'data']
        print(f"Columns: {result['columns']}")
        print(f"Data points: {len(result['data'])}")
        print(f"Sample data: {result['data'][0]}")

        print("✓ Efficient split format optimized for time series analysis")
    else:
        print("Pandas not available - showing basic temporal data")
        temporal_data = {
            "measurements": [
                {"time": "2023-01-01T00:00:00", "value": 25.5},
                {"time": "2023-01-01T01:00:00", "value": 26.1},
                {"time": "2023-01-01T02:00:00", "value": 24.8},
            ]
        }
        result = datason.serialize(temporal_data, config=config)
        print(f"Temporal data: {result}")


def demonstrate_inference_config():
    """Show custom inference configuration optimized for ML model serving performance."""
    print("\n" + "=" * 60)
    print("CUSTOM ML INFERENCE CONFIGURATION DEMO")
    print("=" * 60)

    # Create custom inference config (replaced removed preset)
    config = SerializationConfig(
        date_format=DateFormat.UNIX,
        dataframe_orient=DataFrameOrient.VALUES,
        type_coercion=TypeCoercion.AGGRESSIVE,
        preserve_decimals=False,
        sort_keys=False,
        check_if_serialized=True,
        include_type_hints=False,
    )
    print(
        f"Configuration: performance={config.check_if_serialized}, "
        f"coercion={config.type_coercion.value}, sort={config.sort_keys}"
    )

    # Simulate ML inference input/output
    inference_data = {
        "model_id": "bert-base-v1.2",
        "input_features": [0.1, 0.5, 0.8, 0.3, 0.9],
        "predictions": [0.85, 0.12, 0.03],
        "confidence": 0.85,
        "inference_time_ms": 42,
        "model_version": "1.2.3",
        "batch_size": 1,
    }

    # Time the serialization for performance measurement
    start_time = time.perf_counter()
    result = datason.serialize(inference_data, config=config)
    end_time = time.perf_counter()

    print("\nInference Data Serialization:")
    print(f"Serialization time: {(end_time - start_time) * 1000:.3f} ms")
    print(f"Result keys: {list(result.keys())}")
    print(f"Predictions: {result['predictions']}")
    print(f"Confidence: {result['confidence']}")

    print("✓ Optimized for minimal latency in production ML serving")


def demonstrate_research_config():
    """Show custom research configuration optimized for reproducible experiments."""
    print("\n" + "=" * 60)
    print("CUSTOM RESEARCH CONFIGURATION DEMO")
    print("=" * 60)

    # Create custom research config (replaced removed preset)
    config = SerializationConfig(
        date_format=DateFormat.ISO,
        preserve_decimals=True,
        preserve_complex=True,
        sort_keys=True,
        include_type_hints=True,
    )
    print(
        f"Configuration: type_hints={config.include_type_hints}, "
        f"complex={config.preserve_complex}, sort={config.sort_keys}"
    )

    # Simulate research experiment data
    research_data = {
        "experiment_id": "exp_neural_arch_search_001",
        "hypothesis": "Transformer attention patterns improve classification",
        "parameters": {"learning_rate": 0.001, "batch_size": 32, "hidden_dims": [512, 256, 128], "dropout": 0.1},
        "results": {
            "accuracy": 0.924,
            "f1_score": 0.891,
            "complex_eigenvalue": complex(2.5, 1.3),  # Complex research calculation
            "training_time": datetime.timedelta(hours=4, minutes=23),
            "best_epoch": 47,
        },
        "reproducibility": {
            "random_seed": 42,
            "framework_version": "pytorch_2.1.0",
            "data_hash": "sha256:abc123...",
            "code_commit": "git:1a2b3c4d",
        },
    }

    result = datason.serialize(research_data, config=config)

    print("\nResearch Data Serialization:")
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Complex eigenvalue preserved: {result['results']['complex_eigenvalue']}")
    print(f"Keys are sorted: {list(result.keys()) == sorted(result.keys())}")

    # Check if type hints are included (when available)
    if "__datason_metadata__" in str(result) or any("__type__" in str(v) for v in result.values()):
        print("✓ Type metadata included for perfect reproducibility")
    else:
        print("✓ Configuration supports type metadata when needed")

    print("✓ Maximum information preservation for research reproducibility")


def demonstrate_logging_config():
    """Show custom logging configuration optimized for production safety."""
    print("\n" + "=" * 60)
    print("CUSTOM PRODUCTION LOGGING CONFIGURATION DEMO")
    print("=" * 60)

    # Create custom logging config (replaced removed preset)
    config = SerializationConfig(
        date_format=DateFormat.ISO,
        nan_handling=NanHandling.STRING,
        ensure_ascii=True,
        max_string_length=1000,
        preserve_decimals=False,
        preserve_complex=False,
    )
    print(
        f"Configuration: nan_handling={config.nan_handling.value}, "
        f"max_string={config.max_string_length}, ascii={config.ensure_ascii}"
    )

    # Simulate application logging data
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "level": "ERROR",
        "service": "ml-pipeline",
        "message": "Model inference failed for batch processing",
        "context": {
            "user_id": "user_12345",
            "request_id": "req_abcdef",
            "model_id": "production_v2.1",
            "error_code": "INFERENCE_TIMEOUT",
        },
        "metrics": {
            "processing_time_ms": 5000,
            "queue_depth": 150,
            "memory_usage_mb": 2048,
            "nan_value": float("nan"),  # This should be handled safely
        },
        "stack_trace": "Very long error message that goes on and on..." + "x" * 2000,  # Long string
        "tags": ["ml", "production", "error", "timeout"],
    }

    result = datason.serialize(log_data, config=config)

    print("\nLogging Data Serialization:")
    print(f"Log level: {result['level']}")
    print(f"Service: {result['service']}")
    print(f"NaN handling: {result['metrics']['nan_value']} (type: {type(result['metrics']['nan_value'])})")
    print(f"Long string truncated: {len(result['stack_trace'])} chars (original: ~2050)")
    print(f"Tags preserved: {result['tags']}")

    # Verify JSON safety
    json_str = ds.dumps_json(result, ensure_ascii=True)
    print(f"JSON serializable: {len(json_str)} chars")
    print("✓ Safe for production logging systems with proper truncation")


def demonstrate_performance_comparison():
    """Compare performance across different configurations."""
    print("\n" + "=" * 60)
    print("CONFIGURATION PERFORMANCE COMPARISON")
    print("=" * 60)

    # Sample data for performance testing
    test_data = {
        "id": "test_001",
        "values": list(range(1000)),
        "timestamp": datetime.datetime.now(),
        "metadata": {"version": "1.0", "source": "benchmark"},
    }

    # Core presets and custom configs for comparison
    configs = [
        ("Default", datason.SerializationConfig()),
        ("Performance", datason.get_performance_config()),  # Core preset
        ("ML", datason.get_ml_config()),  # Core preset
        ("API", datason.get_api_config()),  # Core preset
        ("Custom Financial", SerializationConfig(preserve_decimals=True, date_format=DateFormat.UNIX_MS)),
    ]

    print(f"{'Configuration':<12} {'Time (ms)':<10} {'JSON Size':<10} {'Features'}")
    print("-" * 60)

    for name, config in configs:
        # Time the serialization
        start_time = time.perf_counter()
        result = datason.serialize(test_data, config=config)
        end_time = time.perf_counter()

        # Measure output size
        json_str = ds.dumps_json(result, default=str)

        # Key features
        features = []
        if hasattr(config, "check_if_serialized") and config.check_if_serialized:
            features.append("perf")
        if hasattr(config, "preserve_decimals") and config.preserve_decimals:
            features.append("decimals")
        if hasattr(config, "sort_keys") and config.sort_keys:
            features.append("sorted")

        print(f"{name:<12} {(end_time - start_time) * 1000:<10.3f} {len(json_str):<10} {', '.join(features)}")


def main():
    """Run all configuration demonstrations."""
    print("DATASON CUSTOM DOMAIN-SPECIFIC CONFIGURATION DEMONSTRATIONS")
    print("============================================================")
    print("Showcasing custom configurations for different ML/data workflows\n")
    print("(Updated for Phase 1 cleanup - core presets: ml, api, strict, performance)\n")

    demonstrate_financial_config()
    demonstrate_time_series_config()
    demonstrate_inference_config()
    demonstrate_research_config()
    demonstrate_logging_config()
    demonstrate_performance_comparison()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Custom Financial Config: Precise decimals, high-frequency performance")
    print("✓ Custom Time Series Config: Efficient temporal data handling with split format")
    print("✓ Custom Inference Config: Maximum performance for ML model serving")
    print("✓ Custom Research Config: Maximum information preservation and reproducibility")
    print("✓ Custom Logging Config: Production-safe with truncation and ASCII encoding")
    print("\nCustom configurations provide flexibility while core presets (ml, api, strict, performance)")
    print("cover the most common scenarios. Phase 1 cleanup removed redundant presets for a cleaner API!")


if __name__ == "__main__":
    main()
