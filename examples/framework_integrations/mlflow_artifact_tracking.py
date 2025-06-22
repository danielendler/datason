#!/usr/bin/env python3
"""
MLflow Artifact Tracking with DataSON

A comprehensive example showing how to integrate DataSON with MLflow for
ML experiment tracking, model serialization, and artifact management.

Features:
- Model serialization with dump_ml()
- Experiment metrics and parameters logging
- Artifact tracking with DataSON format
- Cross-experiment comparison
- Model registry integration
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import mlflow

try:
    import mlflow.sklearn
except ImportError:
    mlflow.sklearn = None
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import datason as ds


class MLflowDataSONExperiment:
    """MLflow experiment with DataSON integration."""

    def __init__(self, experiment_name: str = "datason_ml_tracking"):
        """Initialize MLflow experiment."""
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def run_classification_experiment(self, n_estimators: int = 100, max_depth: int = 5) -> Dict[str, Any]:
        """Run a complete ML experiment with DataSON artifact tracking."""

        with mlflow.start_run() as run:
            # 1. Generate and prepare data
            print("🔄 Generating dataset...")
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 2. Train model
            print("🚀 Training model...")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # 3. Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # 4. Log parameters using DataSON
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "test_size": 0.2,
            }
            mlflow.log_params(params)

            # 5. Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

            # 6. Create comprehensive experiment metadata using DataSON
            experiment_data = {
                "run_info": {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "start_time": run.info.start_time,
                    "end_time": None,  # Will be updated
                },
                "model_config": {
                    "algorithm": "RandomForestClassifier",
                    "hyperparameters": params,
                    "feature_importance": dict(enumerate(model.feature_importances_)),
                },
                "performance": {
                    "accuracy": accuracy,
                    "detailed_metrics": report,
                    "confusion_matrix": None,  # Could add sklearn.metrics.confusion_matrix
                },
                "data_info": {
                    "train_shape": X_train.shape,
                    "test_shape": X_test.shape,
                    "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
                },
            }

            # 7. Save experiment metadata using DataSON's ML-optimized serialization
            print("💾 Saving experiment metadata with DataSON...")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                # Use dump_ml for optimized ML data serialization
                serialized_data = ds.dump_ml(experiment_data)
                json_string = ds.dumps_json(serialized_data, indent=2)
                f.write(json_string)
                temp_path = f.name

            # Log the DataSON artifact
            mlflow.log_artifact(temp_path, "experiment_metadata")
            os.unlink(temp_path)  # Clean up temp file

            # 8. Save model with DataSON metadata
            print("🤖 Saving model with DataSON metadata...")
            model_metadata = {
                "model_type": "sklearn.ensemble.RandomForestClassifier",
                "training_data_hash": hash(X_train.tobytes()),
                "performance_summary": {"accuracy": accuracy, "n_trees": n_estimators, "max_depth": max_depth},
                "datason_version": ds.__version__ if hasattr(ds, "__version__") else "unknown",
            }

            # Save model metadata using DataSON
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                serialized_metadata = ds.dump_api(model_metadata)  # Use dump_api for clean JSON
                json_string = ds.dumps_json(serialized_metadata, indent=2)
                f.write(json_string)
                temp_metadata_path = f.name

            # Log both the sklearn model and DataSON metadata
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(temp_metadata_path, "model_metadata")
            os.unlink(temp_metadata_path)

            print(f"✅ Experiment completed! Run ID: {run.info.run_id}")
            print(f"📊 Accuracy: {accuracy:.4f}")

            return {"run_id": run.info.run_id, "accuracy": accuracy, "model": model, "experiment_data": experiment_data}

    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs using DataSON."""
        from datetime import datetime, timezone

        comparison_data = {"comparison_timestamp": datetime.now(timezone.utc).isoformat(), "runs": {}}

        for run_id in run_ids:
            run = mlflow.get_run(run_id)

            # Extract metrics and parameters
            comparison_data["runs"][run_id] = {
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
            }

        # Use DataSON for intelligent comparison serialization
        print("🔍 Generating experiment comparison...")
        serialized_comparison = ds.dump_ml(comparison_data)

        # Save comparison results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_string = ds.dumps_json(serialized_comparison, indent=2)
            f.write(json_string)
            temp_path = f.name

        print(f"💾 Comparison saved to: {temp_path}")
        return comparison_data

    def load_experiment_artifacts(self, run_id: str) -> Dict[str, Any]:
        """Load and parse experiment artifacts using DataSON."""

        # Download artifacts to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_dir)

            artifacts = {}
            artifact_path = Path(temp_dir)

            # Load DataSON artifacts intelligently
            for json_file in artifact_path.rglob("*.json"):
                try:
                    # Use load_smart for intelligent parsing
                    with open(json_file) as f:
                        content = f.read()

                    parsed_data = ds.load_smart(content)
                    artifacts[json_file.stem] = parsed_data
                    print(f"✅ Loaded artifact: {json_file.name}")

                except Exception as e:
                    print(f"⚠️  Failed to load {json_file.name}: {e}")

            return artifacts


def run_comprehensive_mlflow_demo():
    """Run a comprehensive MLflow + DataSON demonstration."""
    print("🚀 MLflow + DataSON Integration Demo")
    print("=" * 50)

    # Initialize experiment
    experiment = MLflowDataSONExperiment("datason_comprehensive_demo")

    # Run multiple experiments with different parameters
    print("\n1️⃣ Running multiple experiments...")
    run_results = []

    for n_est, max_d in [(50, 3), (100, 5), (150, 7)]:
        print(f"\nExperiment: n_estimators={n_est}, max_depth={max_d}")
        result = experiment.run_classification_experiment(n_est, max_d)
        run_results.append(result)

    # Compare experiments
    print("\n2️⃣ Comparing experiments...")
    run_ids = [r["run_id"] for r in run_results]
    comparison = experiment.compare_experiments(run_ids)

    # Load artifacts from best run
    best_run = max(run_results, key=lambda x: x["accuracy"])
    print(f"\n3️⃣ Loading artifacts from best run (ID: {best_run['run_id']})...")
    artifacts = experiment.load_experiment_artifacts(best_run["run_id"])

    print(f"\n✅ Demo completed! Best accuracy: {best_run['accuracy']:.4f}")
    print("\nKey DataSON features demonstrated:")
    print("- 🤖 dump_ml() for ML-optimized serialization")
    print("- 🌐 dump_api() for clean JSON output")
    print("- 🧠 load_smart() for intelligent parsing")
    print("- 📊 Complex nested data structure handling")
    print("- 🔄 Cross-experiment comparison utilities")

    return {"best_run": best_run, "all_runs": run_results, "comparison": comparison, "artifacts": artifacts}


if __name__ == "__main__":
    # Check dependencies
    try:
        import mlflow

        print("✅ All dependencies available")

        # Run the comprehensive demo
        results = run_comprehensive_mlflow_demo()

        print("\n🎯 Results Summary:")
        print(f"- Total experiments: {len(results['all_runs'])}")
        print(f"- Best accuracy: {results['best_run']['accuracy']:.4f}")
        print("- MLflow UI: mlflow ui")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install mlflow scikit-learn")
        print("\nBasic example without sklearn:")

        # Fallback basic example
        with mlflow.start_run():
            # Simple example with mock data
            mock_results = {
                "accuracy": 0.95,
                "model_params": {"learning_rate": 0.01, "epochs": 100},
                "training_data": {"samples": 1000, "features": 10},
            }

            # Use DataSON for enhanced serialization
            mlflow.log_params(mock_results["model_params"])
            mlflow.log_metric("accuracy", mock_results["accuracy"])

            # Save comprehensive results with DataSON
            serialized = ds.dump_ml(mock_results)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json_string = ds.dumps_json(serialized, indent=2)
                f.write(json_string)
                temp_path = f.name

            mlflow.log_artifact(temp_path, "results")
            os.unlink(temp_path)

            print("✅ Basic MLflow + DataSON example completed!")
