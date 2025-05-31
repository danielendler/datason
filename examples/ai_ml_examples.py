#!/usr/bin/env python3
"""
🤖 AI/ML Examples for datason

This module demonstrates datason's capabilities with real-world AI/ML workflows.
Perfect for data scientists, ML engineers, and AI researchers who need to
serialize complex training results, model outputs, and experimental data.

Examples include:
- Training pipeline results
- Model comparison data
- Prediction API responses
- Time series forecasting
- Multi-modal AI experiments

Run: python examples/ai_ml_examples.py
"""

from datetime import datetime
import json

import numpy as np

import datason


# Example 1: Basic ML Model Training Results
def serialize_training_results():
    """Serialize ML model training results including metrics and metadata."""
    training_results = {
        "model_name": "neural_network_classifier",
        "version": "1.0.0",
        "training_accuracy": 0.95,
        "validation_accuracy": 0.87,
        "loss_history": [0.8, 0.5, 0.3, 0.2, 0.15],
        "training_time_seconds": 3600,
        "timestamp": datetime.now(),
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.2,
        },
        "feature_importance": np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
        "confusion_matrix": np.array([[85, 5], [3, 87]]),
    }

    # Serialize complex training results
    serialized = datason.serialize(training_results)
    print("Training Results JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Example 2: Data Pipeline State Serialization
def serialize_data_pipeline():
    """Serialize the state of a data processing pipeline."""
    pipeline_state = {
        "pipeline_id": "user_behavior_analysis_v2",
        "status": "completed",
        "input_data_shape": (100000, 25),  # Tuple serialization
        "processed_samples": 98547,
        "dropped_samples": 1453,
        "feature_columns": [
            "user_id",
            "session_duration",
            "page_views",
            "clicks",
            "time_on_site",
            "bounce_rate",
            "conversion_probability",
        ],
        "processing_steps": [
            {"step": "data_cleaning", "duration_ms": 1200, "records_affected": 1453},
            {"step": "feature_engineering", "duration_ms": 3400, "features_created": 8},
            {"step": "normalization", "duration_ms": 800, "method": "min_max"},
        ],
        "output_statistics": {
            "mean_values": np.array([45.2, 3.1, 12.8, 0.75]),
            "std_values": np.array([12.3, 1.2, 4.5, 0.25]),
            "null_counts": {"user_id": 0, "session_duration": 5, "page_views": 12},
        },
        "completed_at": datetime.now(),
    }

    serialized = datason.serialize(pipeline_state)
    print("\nData Pipeline State JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Example 3: Model Comparison and A/B Testing
def serialize_model_comparison():
    """Serialize A/B test results comparing multiple models."""
    model_comparison = {
        "experiment_id": "model_comparison_2024_01",
        "test_duration_days": 14,
        "total_users_tested": 50000,
        "models": {
            "baseline_logistic_regression": {
                "accuracy": 0.82,
                "precision": 0.79,
                "recall": 0.85,
                "f1_score": 0.82,
                "roc_auc": 0.88,
                "inference_time_ms": 2.3,
                "model_size_mb": 0.5,
                "feature_weights": np.array([0.4, -0.2, 0.8, -0.1, 0.6]),
            },
            "ensemble_random_forest": {
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.91,
                "f1_score": 0.89,
                "roc_auc": 0.94,
                "inference_time_ms": 15.7,
                "model_size_mb": 12.3,
                "feature_importance": np.array([0.25, 0.18, 0.35, 0.12, 0.10]),
            },
            "neural_network": {
                "accuracy": 0.91,
                "precision": 0.89,
                "recall": 0.93,
                "f1_score": 0.91,
                "roc_auc": 0.96,
                "inference_time_ms": 8.4,
                "model_size_mb": 25.7,
                "layer_activations": np.random.randn(100, 64),  # Sample activation data
            },
        },
        "statistical_significance": {
            "baseline_vs_ensemble": {"p_value": 0.001, "significant": True},
            "baseline_vs_neural": {"p_value": 0.0001, "significant": True},
            "ensemble_vs_neural": {"p_value": 0.045, "significant": True},
        },
        "recommendation": "neural_network",
        "evaluated_at": datetime.now(),
    }

    serialized = datason.serialize(model_comparison)
    print("\nModel Comparison JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Example 4: Real-time Prediction API Response
def serialize_prediction_response():
    """Serialize a real-time ML prediction API response."""
    prediction_response = {
        "request_id": "pred_20240115_143022_abc123",
        "model_version": "2.1.3",
        "prediction": {
            "class": "high_value_customer",
            "probability": 0.847,
            "confidence_interval": [0.82, 0.87],
            "class_probabilities": {
                "low_value": 0.053,
                "medium_value": 0.100,
                "high_value": 0.847,
            },
        },
        "feature_vector": np.array(
            [
                25.5,  # age
                45000,  # annual_income
                3.2,  # avg_order_value
                12,  # orders_per_year
                0.85,  # loyalty_score
            ]
        ),
        "feature_names": [
            "age",
            "annual_income",
            "avg_order_value",
            "orders_per_year",
            "loyalty_score",
        ],
        "processing_time_ms": 23.7,
        "model_metrics": {
            "last_training_accuracy": 0.91,
            "model_drift_score": 0.02,  # Low drift is good
            "prediction_uncertainty": 0.12,
        },
        "timestamp": datetime.now(),
    }

    serialized = datason.serialize(prediction_response)
    print("\nPrediction Response JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Example 5: Time Series Forecasting Results
def serialize_forecasting_results():
    """Serialize time series forecasting model results."""
    forecasting_results = {
        "model_type": "LSTM_forecaster",
        "target_variable": "daily_sales",
        "forecast_horizon_days": 30,
        "historical_data_points": 365,
        "training_period": {
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 12, 31),
        },
        "forecast_period": {
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 30),
        },
        "predictions": {
            "point_forecast": np.random.normal(1000, 100, 30),  # 30 days of predictions
            "lower_bound_80": np.random.normal(900, 80, 30),
            "upper_bound_80": np.random.normal(1100, 80, 30),
            "lower_bound_95": np.random.normal(850, 70, 30),
            "upper_bound_95": np.random.normal(1150, 70, 30),
        },
        "model_performance": {
            "mae": 45.6,  # Mean Absolute Error
            "mape": 4.2,  # Mean Absolute Percentage Error
            "rmse": 67.3,  # Root Mean Square Error
            "r_squared": 0.89,
        },
        "seasonality_components": {
            "weekly_pattern": np.sin(np.linspace(0, 2 * np.pi, 7)),
            "monthly_trend": np.array([0.95, 0.98, 1.05, 1.1, 1.15, 1.2, 1.1]),
            "holiday_effects": {
                "christmas": 1.8,
                "thanksgiving": 1.4,
                "black_friday": 2.3,
            },
        },
        "generated_at": datetime.now(),
    }

    serialized = datason.serialize(forecasting_results)
    print("\nForecasting Results JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Example 6: Experiment Tracking and MLOps
def serialize_experiment_tracking():
    """Serialize comprehensive ML experiment tracking data."""
    experiment_data = {
        "experiment_name": "customer_churn_prediction_v3",
        "experiment_id": "exp_20240115_001",
        "researcher": "data_science_team",
        "objective": "minimize_churn_rate",
        "dataset_info": {
            "name": "customer_behavior_2023",
            "version": "1.2.0",
            "total_samples": 125000,
            "features": 23,
            "target_distribution": {"churned": 0.23, "retained": 0.77},
            "data_splits": {"train": 0.7, "validation": 0.15, "test": 0.15},
        },
        "model_architecture": {
            "type": "gradient_boosting",
            "framework": "xgboost",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
        },
        "training_results": {
            "final_metrics": {
                "accuracy": 0.87,
                "precision": 0.84,
                "recall": 0.89,
                "f1_score": 0.865,
                "roc_auc": 0.92,
            },
            "training_time_minutes": 15.7,
            "convergence_epoch": 89,
            "learning_curves": {
                "train_loss": np.random.exponential(0.5, 100)[::-1],  # Decreasing loss
                "val_loss": np.random.exponential(0.6, 100)[::-1],
            },
        },
        "feature_analysis": {
            "top_features": [
                {"name": "days_since_last_login", "importance": 0.28},
                {"name": "total_spend_last_6m", "importance": 0.22},
                {"name": "support_tickets_count", "importance": 0.18},
                {"name": "feature_usage_score", "importance": 0.15},
                {"name": "subscription_tenure_months", "importance": 0.12},
            ],
            "correlation_matrix": np.random.correlation_matrix(
                5
            ),  # Would be actual correlation data
            "feature_distributions": {
                "days_since_last_login": {
                    "mean": 12.5,
                    "std": 8.3,
                    "min": 0,
                    "max": 90,
                    "quartiles": [3, 7, 15, 25],
                }
            },
        },
        "deployment_info": {
            "model_registry_id": "churn_model_v3_20240115",
            "deployment_ready": True,
            "performance_threshold_met": True,
            "bias_check_passed": True,
            "explainability_score": 0.78,
        },
        "next_steps": [
            "Deploy to staging environment",
            "A/B test against current production model",
            "Monitor for data drift",
            "Schedule weekly retraining",
        ],
        "created_at": datetime.now(),
    }

    serialized = datason.serialize(experiment_data)
    print("\nExperiment Tracking JSON:")
    print(json.dumps(serialized, indent=2))
    return serialized


# Utility function for correlation matrix generation (example)
def correlation_matrix(size):
    """Generate a random correlation matrix for demonstration."""
    A = np.random.randn(size, size)
    return np.corrcoef(A)


# Add the method to numpy for the example
np.random.correlation_matrix = correlation_matrix


if __name__ == "__main__":
    print("datason AI/ML Examples")
    print("=" * 50)

    # Run all examples
    serialize_training_results()
    serialize_data_pipeline()
    serialize_model_comparison()
    serialize_prediction_response()
    serialize_forecasting_results()
    serialize_experiment_tracking()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("datason handled all complex ML objects seamlessly.")
