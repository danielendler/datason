"""Advanced ML/AI Examples for datason.

This example demonstrates datason's capabilities with major ML/AI libraries
including PyTorch, TensorFlow, scikit-learn, scipy, and others. Shows both
serialization and deserialization in realistic ML workflows.
"""

import json
import uuid
from datetime import datetime

import datason as ds


def demonstrate_pytorch_workflow() -> None:
    """Demonstrate PyTorch tensor serialization in a realistic ML workflow."""
    print("🔥 PyTorch ML Workflow")
    print("=" * 50)

    try:
        import torch

        # Simulate a training experiment
        experiment_data = {
            "experiment_id": uuid.uuid4(),
            "created_at": datetime.now(),
            "model_config": {
                "architecture": "SimpleNet",
                "layers": [784, 128, 64, 10],
                "activation": "relu",
            },
            "training_data": {
                # Small sample tensors
                "input_batch": torch.randn(4, 784),
                "target_batch": torch.randint(0, 10, (4,)),
                "weights": torch.randn(784, 128) * 0.1,
                "gradients": torch.randn(784, 128) * 0.01,
            },
            "metrics": {
                "epoch": 5,
                "loss": torch.tensor(0.245),
                "accuracy": torch.tensor(0.923),
                "learning_rate": 0.001,
            },
            "device_info": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        }

        print("Original PyTorch experiment:")
        print(f"  Input shape: {experiment_data['training_data']['input_batch'].shape}")
        print(f"  Loss: {experiment_data['metrics']['loss'].item():.4f}")
        print(f"  Device: {experiment_data['device_info']}")
        print()

        # Serialize (without our ML serializer extension for now)
        serialized = ds.serialize(experiment_data)
        print("✅ Serialized PyTorch experiment")
        print(f"  Input batch type: {type(serialized['training_data']['input_batch'])}")
        print(f"  Loss value preserved: {serialized['metrics']['loss']}")
        print()

        # Full round trip through JSON
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Experiment ID restored: {type(deserialized['experiment_id']).__name__}")
        print(f"  Timestamp restored: {type(deserialized['created_at']).__name__}")
        print()

    except ImportError:
        print("⚠️  PyTorch not available - skipping PyTorch examples")
        print()


def demonstrate_sklearn_pipeline() -> None:
    """Demonstrate scikit-learn model metadata serialization."""
    print("🤖 Scikit-Learn ML Pipeline")
    print("=" * 50)

    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create sample data and train model
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

        # Create and train pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=10, random_state=42),
                ),
            ]
        )

        pipeline.fit(X, y)
        accuracy = pipeline.score(X, y)

        # ML experiment metadata
        ml_experiment = {
            "experiment_id": uuid.uuid4(),
            "timestamp": datetime.now(),
            "dataset": {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                "target_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            },
            "model_pipeline": {
                "steps": [step[0] for step in pipeline.steps],
                "scaler_params": pipeline.named_steps["scaler"].get_params(),
                "classifier_params": pipeline.named_steps["classifier"].get_params(),
            },
            "performance": {
                "training_accuracy": accuracy,
                "feature_importances": pipeline.named_steps["classifier"].feature_importances_.tolist(),
                "n_trees": len(pipeline.named_steps["classifier"].estimators_),
            },
            "arrays": {
                "sample_predictions": pipeline.predict(X[:5]).tolist(),
                "prediction_probabilities": pipeline.predict_proba(X[:5]).tolist(),
                "feature_means": pipeline.named_steps["scaler"].mean_.tolist(),
                "feature_scales": pipeline.named_steps["scaler"].scale_.tolist(),
            },
        }

        print("Original ML Pipeline:")
        print(f"  Dataset shape: {X.shape}")
        print(f"  Training accuracy: {accuracy:.4f}")
        print(f"  Feature importances shape: {len(ml_experiment['performance']['feature_importances'])}")
        print()

        # Serialize
        serialized = ds.serialize(ml_experiment)
        print("✅ Serialized ML pipeline metadata")
        print(f"  All numpy arrays converted: {isinstance(serialized['performance']['feature_importances'], list)}")
        print(f"  Model parameters preserved: {len(serialized['model_pipeline']['classifier_params'])}")
        print()

        # Round trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Experiment ID: {type(deserialized['experiment_id']).__name__}")
        print(f"  Accuracy preserved: {deserialized['performance']['training_accuracy']:.4f}")
        print()

    except ImportError:
        print("⚠️  Scikit-learn not available - skipping sklearn examples")
        print()


def demonstrate_computer_vision_workflow() -> None:
    """Demonstrate computer vision workflow with image data."""
    print("👁️  Computer Vision Workflow")
    print("=" * 50)

    try:
        import numpy as np
        from PIL import Image

        # Create a simple synthetic image
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)

        # CV experiment data
        cv_experiment = {
            "experiment_id": uuid.uuid4(),
            "timestamp": datetime.now(),
            "dataset_info": {
                "name": "synthetic_rgb_dataset",
                "image_count": 1000,
                "resolution": [64, 64],
                "channels": 3,
                "data_augmentation": ["rotation", "flip", "brightness"],
            },
            "sample_image": {
                "format": "RGB",
                "size": image.size,
                "mode": image.mode,
                # In practice, you'd store image paths or base64
                "pixel_stats": {
                    "mean": img_array.mean(axis=(0, 1)).tolist(),
                    "std": img_array.std(axis=(0, 1)).tolist(),
                    "min": img_array.min(),
                    "max": img_array.max(),
                },
            },
            "model_architecture": {
                "type": "CNN",
                "layers": [
                    {"type": "conv2d", "filters": 32, "kernel_size": [3, 3]},
                    {"type": "maxpool", "pool_size": [2, 2]},
                    {"type": "conv2d", "filters": 64, "kernel_size": [3, 3]},
                    {"type": "flatten"},
                    {"type": "dense", "units": 128},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
            },
            "training_metrics": {
                "epochs": 20,
                "batch_size": 32,
                "loss_history": np.random.exponential(0.5, 20).tolist(),
                "accuracy_history": (0.5 + 0.5 * (1 - np.exp(-np.linspace(0, 3, 20)))).tolist(),
                "validation_split": 0.2,
            },
        }

        print("Original CV Experiment:")
        print(f"  Image shape: {img_array.shape}")
        print(f"  Final accuracy: {cv_experiment['training_metrics']['accuracy_history'][-1]:.4f}")
        print(f"  Model layers: {len(cv_experiment['model_architecture']['layers'])}")
        print()

        # Serialize
        serialized = ds.serialize(cv_experiment)
        print("✅ Serialized CV experiment")
        print(f"  Numpy arrays converted: {isinstance(serialized['sample_image']['pixel_stats']['mean'], list)}")
        print(f"  Image metadata preserved: {serialized['sample_image']['size']}")
        print()

        # Round trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Experiment ID: {type(deserialized['experiment_id']).__name__}")
        print(f"  Loss history preserved: {len(deserialized['training_metrics']['loss_history'])}")
        print()

    except ImportError:
        print("⚠️  PIL not available - skipping computer vision examples")
        print()


def demonstrate_time_series_analysis() -> None:
    """Demonstrate time series analysis with pandas and datetime handling."""
    print("📈 Time Series Analysis")
    print("=" * 50)

    try:
        import numpy as np
        import pandas as pd

        # Create time series data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        values = np.cumsum(np.random.randn(100)) + 100
        ts_data = pd.Series(values, index=dates)

        # Time series analysis results
        ts_analysis = {
            "analysis_id": uuid.uuid4(),
            "created_at": datetime.now(),
            "data_info": {
                "start_date": ts_data.index.min(),
                "end_date": ts_data.index.max(),
                "frequency": "daily",
                "n_observations": len(ts_data),
                "missing_values": ts_data.isna().sum(),
            },
            "time_series": {
                "timestamps": ts_data.index.tolist(),
                "values": ts_data.values.tolist(),
                "rolling_mean_7d": ts_data.rolling(7).mean().tolist(),
                "rolling_std_7d": ts_data.rolling(7).std().tolist(),
            },
            "statistics": {
                "mean": ts_data.mean(),
                "std": ts_data.std(),
                "min": ts_data.min(),
                "max": ts_data.max(),
                "trend": "increasing" if ts_data.iloc[-1] > ts_data.iloc[0] else "decreasing",
                "volatility": ts_data.std() / ts_data.mean(),
            },
            "forecasting": {
                "model_type": "ARIMA",
                "horizon": 30,
                "confidence_intervals": [0.8, 0.95],
                "last_observation": ts_data.iloc[-1],
                "next_forecast": ts_data.iloc[-1] + np.random.randn() * ts_data.std(),
            },
        }

        print("Original Time Series:")
        print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
        print(f"  Mean value: {ts_data.mean():.2f}")
        print(f"  Volatility: {ts_analysis['statistics']['volatility']:.4f}")
        print()

        # Serialize
        serialized = ds.serialize(ts_analysis)
        print("✅ Serialized time series analysis")
        print(f"  Timestamps converted: {isinstance(serialized['time_series']['timestamps'][0], str)}")
        print(f"  Statistics preserved: {serialized['statistics']['mean']:.2f}")
        print()

        # Round trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Analysis ID: {type(deserialized['analysis_id']).__name__}")
        print(f"  Start date restored: {type(deserialized['data_info']['start_date']).__name__}")
        print(f"  Timestamps restored: {type(deserialized['time_series']['timestamps'][0]).__name__}")
        print()

    except ImportError:
        print("⚠️  Pandas not available - skipping time series examples")
        print()


def demonstrate_nlp_workflow() -> None:
    """Demonstrate NLP workflow with text processing and model metadata."""
    print("📝 Natural Language Processing")
    print("=" * 50)

    try:
        import numpy as np

        # Simulate NLP experiment (without heavy dependencies)
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming natural language processing.",
            "Deep neural networks can understand complex text patterns.",
            "datason makes it easy to serialize ML experiment data.",
        ]

        # Simulate text processing results
        vocab_size = 1000
        embedding_dim = 128

        nlp_experiment = {
            "experiment_id": uuid.uuid4(),
            "timestamp": datetime.now(),
            "dataset": {
                "name": "sample_text_corpus",
                "n_documents": len(documents),
                "total_tokens": sum(len(doc.split()) for doc in documents),
                "avg_doc_length": np.mean([len(doc.split()) for doc in documents]),
                "vocabulary_size": vocab_size,
                "sample_documents": documents[:2],  # Store subset for inspection
            },
            "preprocessing": {
                "steps": [
                    "tokenization",
                    "lowercasing",
                    "stop_word_removal",
                    "lemmatization",
                ],
                "vocabulary": {
                    "most_common": [
                        ("the", 45),
                        ("and", 32),
                        ("is", 28),
                        ("to", 24),
                        ("of", 22),
                    ],
                    "total_unique_tokens": vocab_size,
                    "min_frequency": 2,
                    "max_sequence_length": 512,
                },
            },
            "model_config": {
                "architecture": "Transformer",
                "parameters": {
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "n_heads": 8,
                    "n_layers": 6,
                    "dropout": 0.1,
                    "max_position_embeddings": 512,
                },
                "total_parameters": vocab_size * embedding_dim + 1_000_000,  # Rough estimate
            },
            "training_results": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "loss_curve": np.logspace(0, -1, 10).tolist(),  # Decreasing loss
                "metrics": {
                    "perplexity": 15.3,
                    "bleu_score": 0.76,
                    "rouge_l": 0.82,
                    "accuracy": 0.94,
                },
            },
            "embeddings_sample": {
                "word_vectors": np.random.randn(10, embedding_dim).tolist(),  # Sample embeddings
                "similarity_matrix": np.corrcoef(np.random.randn(5, embedding_dim)).tolist(),
                "nearest_neighbors": {
                    "king": ["queen", "monarch", "ruler", "royal"],
                    "good": ["great", "excellent", "fine", "nice"],
                    "fast": ["quick", "rapid", "speedy", "swift"],
                },
            },
        }

        print("Original NLP Experiment:")
        print(f"  Total documents: {nlp_experiment['dataset']['n_documents']}")
        print(f"  Vocabulary size: {nlp_experiment['dataset']['vocabulary_size']:,}")
        print(f"  Model parameters: {nlp_experiment['model_config']['total_parameters']:,}")
        print(f"  Final perplexity: {nlp_experiment['training_results']['metrics']['perplexity']}")
        print()

        # Serialize
        serialized = ds.serialize(nlp_experiment)
        print("✅ Serialized NLP experiment")
        print(f"  Numpy arrays converted: {isinstance(serialized['embeddings_sample']['word_vectors'], list)}")
        print(f"  Loss curve preserved: {len(serialized['training_results']['loss_curve'])}")
        print()

        # Round trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Experiment ID: {type(deserialized['experiment_id']).__name__}")
        print(f"  Timestamp: {type(deserialized['timestamp']).__name__}")
        print(f"  Embeddings shape preserved: {len(deserialized['embeddings_sample']['word_vectors'][0])}")
        print()

    except ImportError:
        print("⚠️  Required libraries not available - skipping NLP examples")
        print()


def demonstrate_experiment_tracking() -> None:
    """Demonstrate ML experiment tracking and hyperparameter optimization."""
    print("📊 ML Experiment Tracking")
    print("=" * 50)

    try:
        import numpy as np

        # Simulate hyperparameter optimization results
        experiments = []
        for i in range(5):
            exp = {
                "experiment_id": uuid.uuid4(),
                "timestamp": datetime.now(),
                "hyperparameters": {
                    "learning_rate": 10 ** np.random.uniform(-4, -2),
                    "batch_size": int(2 ** np.random.uniform(4, 7)),
                    "n_layers": np.random.randint(3, 8),
                    "hidden_size": int(2 ** np.random.uniform(6, 9)),
                    "dropout": np.random.uniform(0.1, 0.5),
                    "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                    "regularization": np.random.uniform(1e-5, 1e-3),
                },
                "results": {
                    "validation_accuracy": np.random.uniform(0.85, 0.95),
                    "training_time": np.random.uniform(120, 600),  # seconds
                    "convergence_epoch": np.random.randint(15, 50),
                    "final_loss": np.random.uniform(0.1, 0.8),
                    "memory_usage": np.random.uniform(2.5, 8.0),  # GB
                },
                "model_artifacts": {
                    "checkpoint_path": f"models/experiment_{i}/checkpoint.pt",
                    "config_hash": f"hash_{uuid.uuid4().hex[:8]}",
                    "model_size_mb": np.random.uniform(50, 200),
                },
            }
            experiments.append(exp)

        # Find best experiment
        best_exp = max(experiments, key=lambda x: x["results"]["validation_accuracy"])

        tracking_data = {
            "tracking_session_id": uuid.uuid4(),
            "created_at": datetime.now(),
            "project_info": {
                "name": "image_classification_optimization",
                "dataset": "CIFAR-10",
                "objective": "maximize_validation_accuracy",
                "budget": {"max_experiments": 50, "max_time_hours": 24},
            },
            "experiments": experiments,
            "optimization_summary": {
                "total_experiments": len(experiments),
                "best_experiment_id": best_exp["experiment_id"],
                "best_accuracy": best_exp["results"]["validation_accuracy"],
                "best_hyperparams": best_exp["hyperparameters"],
                "parameter_importance": {
                    "learning_rate": 0.35,
                    "n_layers": 0.22,
                    "hidden_size": 0.18,
                    "dropout": 0.15,
                    "batch_size": 0.10,
                },
            },
            "convergence_analysis": {
                "accuracy_progression": [exp["results"]["validation_accuracy"] for exp in experiments],
                "time_per_experiment": [exp["results"]["training_time"] for exp in experiments],
                "efficiency_score": best_exp["results"]["validation_accuracy"] / best_exp["results"]["training_time"],
            },
        }

        print("Original Experiment Tracking:")
        print(f"  Total experiments: {len(experiments)}")
        print(f"  Best accuracy: {best_exp['results']['validation_accuracy']:.4f}")
        print(f"  Best learning rate: {best_exp['hyperparameters']['learning_rate']:.2e}")
        print()

        # Serialize
        serialized = ds.serialize(tracking_data)
        print("✅ Serialized experiment tracking data")
        print(f"  All experiment IDs converted: {isinstance(serialized['experiments'][0]['experiment_id'], str)}")
        print(f"  Numpy arrays preserved: {len(serialized['convergence_analysis']['accuracy_progression'])}")
        print()

        # Round trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)
        deserialized = ds.deserialize(parsed)

        print("✅ Round-trip completed")
        print(f"  Session ID: {type(deserialized['tracking_session_id']).__name__}")
        print(f"  Best experiment ID: {type(deserialized['optimization_summary']['best_experiment_id']).__name__}")
        print(f"  Timestamps restored: {type(deserialized['created_at']).__name__}")
        print()

    except Exception as e:
        print(f"⚠️  Error in experiment tracking demo: {e}")
        print()


def main() -> None:
    """Run all advanced ML/AI demonstrations."""
    print("🚀 datason Advanced ML/AI Examples")
    print("=" * 70)
    print("Demonstrating datason's capabilities with real-world ML workflows")
    print()

    demonstrate_pytorch_workflow()
    print("\n" + "─" * 70 + "\n")

    demonstrate_sklearn_pipeline()
    print("\n" + "─" * 70 + "\n")

    demonstrate_computer_vision_workflow()
    print("\n" + "─" * 70 + "\n")

    demonstrate_time_series_analysis()
    print("\n" + "─" * 70 + "\n")

    demonstrate_nlp_workflow()
    print("\n" + "─" * 70 + "\n")

    demonstrate_experiment_tracking()

    print("✨ Advanced examples completed!")
    print("\n🎯 Key takeaways:")
    print("   • datason seamlessly handles complex ML data structures")
    print("   • Preserves experiment metadata and UUIDs through serialization")
    print("   • Converts numpy arrays and pandas objects intelligently")
    print("   • Enables reproducible ML experiment tracking")
    print("   • Supports round-trip serialization for data persistence")


if __name__ == "__main__":
    main()
