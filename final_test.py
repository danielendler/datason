#!/usr/bin/env python3
"""Final comprehensive test of all functionality."""

import datason
from datason.config import SerializationConfig


def main():
    print("Testing key ML functionality:")

    # Test PyTorch
    try:
        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = datason.dump_ml(tensor)
        pytorch_works = "__datason_type__" in result and result["__datason_type__"] == "torch.Tensor"
        print(f"PyTorch tensor: {pytorch_works}")
    except ImportError:
        print("PyTorch tensor: SKIPPED (not available)")

    # Test sklearn
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        result = datason.dump_ml(model)
        sklearn_works = "__datason_type__" in result and result["__datason_type__"] == "sklearn.model"
        print(f"Sklearn model: {sklearn_works}")
    except ImportError:
        print("Sklearn model: SKIPPED (not available)")

    # Test CatBoost
    try:
        import catboost

        model = catboost.CatBoostClassifier(n_estimators=2, random_state=42, verbose=False)
        result = datason.dump_ml(model)
        catboost_works = "__datason_type__" in result and result["__datason_type__"] == "catboost.model"
        print(f"CatBoost model: {catboost_works}")
    except ImportError:
        print("CatBoost model: SKIPPED (not available)")

    # Test security
    config = SerializationConfig(max_depth=0)
    try:
        datason.serialize({"level1": {"level2": "too deep"}}, config=config)
        print("Security limits: False (should have thrown SecurityError)")
    except datason.SecurityError:
        print("Security limits: True")

    # Test large object limits
    config = SerializationConfig(max_size=20)
    try:
        datason.serialize(list(range(25)), config=config)
        print("Size limits: False (should have thrown SecurityError)")
    except datason.SecurityError:
        print("Size limits: True")


if __name__ == "__main__":
    main()
