"""
Comprehensive Guide to Datason File Operations

This guide demonstrates all file operation capabilities with real-world examples.
Covers both JSON and JSONL formats, ML workflows, security features, and more.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import datason

# Optional ML imports for examples
try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def basic_file_operations():
    """Basic JSON and JSONL file operations."""
    print("📂 Basic File Operations")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Sample data
        experiments = [
            {"experiment_id": 1, "accuracy": 0.95, "loss": 0.05},
            {"experiment_id": 2, "accuracy": 0.97, "loss": 0.03},
            {"experiment_id": 3, "accuracy": 0.93, "loss": 0.07},
        ]

        # Save as JSONL (one experiment per line)
        jsonl_path = temp_path / "experiments.jsonl"
        datason.save_ml(experiments, jsonl_path)
        print(f"✓ Saved {len(experiments)} experiments to JSONL")

        # Save as JSON (single array)
        json_path = temp_path / "experiments.json"
        datason.save_ml(experiments, json_path)
        print("✓ Saved experiments to JSON")

        # Load back
        jsonl_data = list(datason.load_smart_file(jsonl_path))
        json_data = list(datason.load_smart_file(json_path))

        print(f"✓ Loaded {len(jsonl_data)} records from JSONL")
        print(f"✓ Loaded {len(json_data)} records from JSON")


def ml_workflow_example():
    """Complete ML workflow with model training and persistence."""
    print("\n🤖 ML Workflow Example")

    if not HAS_SKLEARN:
        print("⚠️ Sklearn not available, skipping ML example")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Generate and prepare data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

        # Create feature metadata
        feature_info = pd.DataFrame(
            {
                "feature_name": [f"feature_{i}" for i in range(20)],
                "importance": np.random.random(20),
                "data_type": np.random.choice(["numerical", "categorical"], 20),
            }
        )

        # 2. Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # 3. Package everything for persistence
        ml_package = {
            "model": model,
            "training_data": {"X": X, "y": y, "feature_info": feature_info},
            "results": {
                "predictions": predictions,
                "probabilities": probabilities,
                "accuracy": (predictions == y).mean(),
            },
            "metadata": {
                "trained_at": datetime.now(),
                "model_type": "RandomForestClassifier",
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            },
        }

        # 4. Save with ML optimization
        model_path = temp_path / "model_package.jsonl.gz"  # Compressed
        datason.save_ml(ml_package, model_path)
        print("✓ Saved ML package with model and data")

        # 5. Load back with smart reconstruction
        loaded_package = list(datason.load_smart_file(model_path))[0]

        # 6. Verify everything is preserved
        assert isinstance(loaded_package["training_data"]["X"], np.ndarray)
        assert isinstance(loaded_package["training_data"]["feature_info"], pd.DataFrame)
        assert loaded_package["metadata"]["accuracy"] == ml_package["metadata"]["accuracy"]

        print("✓ Model package loaded successfully")
        print(f"✓ Training accuracy: {loaded_package['metadata']['accuracy']:.3f}")


def streaming_workflow():
    """Streaming large datasets to JSONL files."""
    print("\n🌊 Streaming Workflow")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Simulate streaming training logs
        logs_path = temp_path / "training_logs.jsonl"

        with datason.stream_save_ml(logs_path) as stream:
            for epoch in range(100):
                # Simulate training metrics for each epoch
                log_entry = {
                    "epoch": epoch,
                    "timestamp": datetime.now(),
                    "train_loss": np.random.exponential(1.0),
                    "val_loss": np.random.exponential(1.0),
                    "train_accuracy": np.random.beta(8, 2),  # Usually high
                    "val_accuracy": np.random.beta(7, 3),  # Slightly lower
                    "learning_rate": 0.001 * (0.95**epoch),
                    "batch_metrics": {
                        "gradient_norm": np.random.exponential(0.1),
                        "weight_norm": np.random.lognormal(0, 0.1),
                    },
                    "model_weights_sample": np.random.randn(10),  # Sample of weights
                }
                stream.write(log_entry)

        print(f"✓ Streamed 100 training epochs to {logs_path}")

        # Load and analyze the logs
        all_logs = list(datason.load_smart_file(logs_path))

        # Extract metrics for analysis
        train_losses = [log["train_loss"] for log in all_logs]
        val_losses = [log["val_loss"] for log in all_logs]

        print(f"✓ Loaded {len(all_logs)} log entries")
        print(f"✓ Final train loss: {train_losses[-1]:.3f}")
        print(f"✓ Final val loss: {val_losses[-1]:.3f}")


def security_and_redaction():
    """Secure data handling with PII redaction."""
    print("\n🔒 Security and Redaction")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Sensitive customer data
        customer_data = {
            "customers": [
                {
                    "id": "cust_001",
                    "name": "Alice Johnson",
                    "email": "alice.johnson@email.com",
                    "ssn": "123-45-6789",
                    "credit_card": "4532-1234-5678-9012",
                    "api_key": "sk-1234567890abcdef",
                    "purchase_history": [{"item": "laptop", "amount": 1200.00}, {"item": "mouse", "amount": 25.00}],
                },
                {
                    "id": "cust_002",
                    "name": "Bob Smith",
                    "email": "bob.smith@email.com",
                    "ssn": "987-65-4321",
                    "credit_card": "5555-4444-3333-2222",
                    "api_key": "sk-abcdef1234567890",
                    "purchase_history": [{"item": "keyboard", "amount": 80.00}],
                },
            ],
            "internal_notes": {
                "password": "admin123",
                "database_url": "postgresql://user:pass@db.company.com/customers",
            },
        }

        # Save with comprehensive redaction
        secure_path = temp_path / "customer_data_secure.jsonl"
        datason.save_secure(
            customer_data,
            secure_path,
            redact_pii=True,  # Auto-detect PII patterns
            redact_fields=["password", "database_url", "api_key"],  # Explicit fields
        )
        print("✓ Saved customer data with PII redaction")

        # Load back and check redaction
        secure_data = list(datason.load_smart_file(secure_path))[0]

        print(f"✓ Redaction summary: {secure_data['redaction_summary']['total_redactions']} items redacted")
        print(f"✓ PII patterns detected: {secure_data['redaction_summary']['patterns_found']}")

        # Verify sensitive data is redacted
        first_customer = secure_data["customers"][0]
        assert "[REDACTED:SSN]" in str(first_customer["ssn"])
        assert "[REDACTED:CREDIT_CARD]" in str(first_customer["credit_card"])

        print("✓ Sensitive data properly redacted")


def complex_data_types():
    """Handling complex data types with perfect reconstruction."""
    print("\n🧬 Complex Data Types")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create complex nested data with various types
        complex_data = {
            "neural_network": {
                "architecture": "transformer",
                "layers": [
                    {"type": "embedding", "weights": np.random.randn(1000, 512), "bias": np.random.randn(512)},
                    {
                        "type": "attention",
                        "query_weights": np.random.randn(512, 512),
                        "key_weights": np.random.randn(512, 512),
                        "value_weights": np.random.randn(512, 512),
                    },
                ],
            },
            "training_data": {
                "sequences": np.random.randint(0, 1000, (10000, 128)),  # Token sequences
                "attention_masks": np.random.choice([0, 1], (10000, 128)),
                "labels": np.random.randint(0, 10, 10000),
            },
            "metrics_history": pd.DataFrame(
                {
                    "epoch": range(50),
                    "train_loss": np.random.exponential(1, 50),
                    "val_loss": np.random.exponential(1, 50),
                    "perplexity": np.random.lognormal(2, 0.5, 50),
                    "timestamp": pd.date_range("2024-01-01", periods=50, freq="H"),
                }
            ),
            "hyperparameters": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "dropout_rate": 0.1,
            },
        }

        # Create template for perfect type reconstruction
        template = {
            "neural_network": {
                "architecture": "",
                "layers": [{"type": "", "weights": np.array([[0.0]]), "bias": np.array([0.0])}],
            },
            "training_data": {
                "sequences": np.array([[0]]),
                "attention_masks": np.array([[0]]),
                "labels": np.array([0]),
            },
            "metrics_history": complex_data["metrics_history"].iloc[:1],  # DataFrame template
            "hyperparameters": {},
        }

        # Save with ML optimization
        model_path = temp_path / "complex_model.jsonl"
        datason.save_ml(complex_data, model_path)
        print("✓ Saved complex neural network data")

        # Load with perfect reconstruction
        loaded_data = list(datason.load_perfect_file(model_path, template))[0]

        # Verify all types are perfectly preserved
        assert isinstance(loaded_data["neural_network"]["layers"][0]["weights"], np.ndarray)
        assert loaded_data["neural_network"]["layers"][0]["weights"].shape == (1000, 512)

        assert isinstance(loaded_data["training_data"]["sequences"], np.ndarray)
        assert loaded_data["training_data"]["sequences"].shape == (10000, 128)

        assert isinstance(loaded_data["metrics_history"], pd.DataFrame)
        assert len(loaded_data["metrics_history"]) == 50

        print("✓ Perfect type reconstruction verified")
        print(f"✓ Embedding weights shape: {loaded_data['neural_network']['layers'][0]['weights'].shape}")
        print(f"✓ Training sequences shape: {loaded_data['training_data']['sequences'].shape}")


def format_conversion():
    """Converting between JSON and JSONL formats."""
    print("\n🔄 Format Conversion")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create experimental results
        experiments = [
            {
                "experiment_id": f"exp_{i:03d}",
                "config": {
                    "model": np.random.choice(["cnn", "rnn", "transformer"]),
                    "lr": 10 ** np.random.uniform(-5, -2),
                    "batch_size": int(2 ** np.random.uniform(4, 8)),
                },
                "results": {
                    "accuracy": np.random.beta(8, 2),
                    "f1_score": np.random.beta(7, 3),
                    "training_time": np.random.exponential(3600),  # seconds
                    "confusion_matrix": np.random.randint(0, 100, (5, 5)),
                },
            }
            for i in range(20)
        ]

        # 1. Save as JSONL (good for streaming/appending)
        jsonl_path = temp_path / "experiments.jsonl"
        datason.save_ml(experiments, jsonl_path)
        print(f"✓ Saved {len(experiments)} experiments as JSONL")

        # 2. Load from JSONL
        jsonl_loaded = list(datason.load_smart_file(jsonl_path))

        # 3. Save as JSON (good for single dataset)
        json_path = temp_path / "experiments.json"
        datason.save_ml(jsonl_loaded, json_path, format="json")
        print("✓ Converted to JSON format")

        # 4. Load from JSON
        json_loaded = list(datason.load_smart_file(json_path))

        # 5. Verify data integrity
        assert len(jsonl_loaded) == len(json_loaded) == len(experiments)

        # Compare first experiment
        original = experiments[0]
        jsonl_first = jsonl_loaded[0]
        json_first = json_loaded[0]

        assert original["experiment_id"] == jsonl_first["experiment_id"] == json_first["experiment_id"]
        assert np.array_equal(original["results"]["confusion_matrix"], jsonl_first["results"]["confusion_matrix"])
        assert np.array_equal(original["results"]["confusion_matrix"], json_first["results"]["confusion_matrix"])

        print("✓ Format conversion preserved all data integrity")


def compression_examples():
    """Working with compressed files."""
    print("\n📦 Compression Examples")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create large dataset
        large_dataset = {
            "high_res_images": np.random.randint(0, 255, (100, 256, 256, 3), dtype=np.uint8),
            "feature_maps": np.random.randn(100, 512, 512),
            "metadata": pd.DataFrame(
                {
                    "image_id": [f"img_{i:04d}" for i in range(100)],
                    "label": np.random.choice(["cat", "dog", "bird"], 100),
                    "confidence": np.random.random(100),
                    "processing_time": np.random.exponential(0.1, 100),
                }
            ),
        }

        # Save uncompressed
        uncompressed_path = temp_path / "large_dataset.jsonl"
        datason.save_ml(large_dataset, uncompressed_path)
        uncompressed_size = uncompressed_path.stat().st_size

        # Save compressed
        compressed_path = temp_path / "large_dataset.jsonl.gz"
        datason.save_ml(large_dataset, compressed_path)
        compressed_size = compressed_path.stat().st_size

        compression_ratio = uncompressed_size / compressed_size

        print(f"✓ Uncompressed size: {uncompressed_size / 1024 / 1024:.1f} MB")
        print(f"✓ Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
        print(f"✓ Compression ratio: {compression_ratio:.1f}x smaller")

        # Load compressed data
        loaded_compressed = list(datason.load_smart_file(compressed_path))[0]

        # Verify data integrity
        assert isinstance(loaded_compressed["high_res_images"], np.ndarray)
        assert loaded_compressed["high_res_images"].shape == (100, 256, 256, 3)
        assert isinstance(loaded_compressed["metadata"], pd.DataFrame)

        print("✓ Compressed data loaded successfully with full integrity")


def api_discovery():
    """Discovering available file operations."""
    print("\n🔍 API Discovery")

    # Get help for file operations
    help_info = datason.help_api()

    if "file_operations" in help_info:
        file_ops = help_info["file_operations"]
        print("📋 Available file operations:")

        for func_name, func_info in file_ops.items():
            print(f"  • {func_name}: {func_info['description']}")

            # Show example if available
            if "examples" in func_info and func_info["examples"]:
                print(f"    Example: {func_info['examples'][0]}")

    # Get overall API info
    api_info = datason.get_api_info()
    print("\n📊 API Summary:")
    print(f"  • Total functions: {len(api_info.get('all_functions', []))}")
    print(f"  • File operations: {'file_operations' in api_info.get('features', [])}")
    print("  • Formats supported: JSON, JSONL")
    print("  • Compression: Automatic (.gz detection)")


def main():
    """Run all examples."""
    print("🚀 Datason File Operations Guide")
    print("=" * 50)

    # Run all examples
    basic_file_operations()
    ml_workflow_example()
    streaming_workflow()
    security_and_redaction()
    complex_data_types()
    format_conversion()
    compression_examples()
    api_discovery()

    print("\n" + "=" * 50)
    print("🎉 All examples completed successfully!")
    print("\n📚 Key takeaways:")
    print("  • Use .jsonl for streaming/line-by-line data")
    print("  • Use .json for single objects/arrays")
    print("  • Add .gz for automatic compression")
    print("  • Use save_secure() for sensitive data")
    print("  • Use save_ml() for ML workflows")
    print("  • Use templates for perfect type reconstruction")


if __name__ == "__main__":
    main()
