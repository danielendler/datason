#!/usr/bin/env python3
"""MLflow artifact logging with datason."""

import mlflow

import datason


# Example training function
def train_model():
    return {"accuracy": 0.99, "params": {"lr": 0.001}}


if __name__ == "__main__":
    with mlflow.start_run():
        results = train_model()
        mlflow.log_dict(datason.serialize(results), "results.json")
        print("Logged results to MLflow")
