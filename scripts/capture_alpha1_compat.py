"""Capture compatibility payloads using the unmodified published a1 source.

Run with PYTHONPATH pointing to an a1 worktree and optional libraries installed.
Installed distribution metadata may name the candidate; provenance uses source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import datason
from datason._protocols import DeserializeContext, SerializeContext
from datason._registry import default_registry
from datason.security.integrity import wrap_with_integrity

SOURCE_COMMIT = "88b110f79922ceef7e4ae3bc92392c0bdbdfa2ba"
TEST_KEY = "public-test-fixture-key"


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y


class PointPlugin:
    name = "fixture_point"
    priority = 400

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, Point)

    def serialize(self, obj: Point, ctx: SerializeContext) -> Any:
        return {"__datason_type__": "example.point.v1", "__datason_value__": [obj.x, obj.y]}

    def can_deserialize(self, data: dict[str, Any]) -> bool:
        return data.get("__datason_type__") == "example.point.v1"

    def deserialize(self, data: dict[str, Any], ctx: DeserializeContext) -> Any:
        return Point(*data["__datason_value__"])


def sklearn_values() -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0]
    return {
        "sklearn_estimator": LinearRegression().fit(x, y),
        "sklearn_pipeline": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(x, y),
    }


def ml_values() -> dict[str, Any]:
    import jax.numpy as jnp
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import scipy.sparse as sp
    import tensorflow as tf
    import torch

    return {
        **sklearn_values(),
        "legacy_uint64_max": np.uint64(2**64 - 1),
        "scipy_csr": sp.csr_matrix([[1, 0], [0, 2]], dtype=np.float32),
        "torch_tensor": torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
        "tf_tensor": tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        "tf_sparse": tf.SparseTensor([[0, 0], [1, 1]], tf.constant([1.0, 2.0]), [2, 2]),
        "jax_array": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
        "polars_frame": pl.DataFrame({"count": [1, 2], "score": [1.25, 2.5]}),
        "plotly_figure": go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])]),
    }


def diagnostic_values() -> dict[str, Any]:
    import catboost
    import optuna
    from optuna.distributions import FloatDistribution

    model = catboost.CatBoostClassifier(iterations=2, verbose=False, allow_writing_files=False, thread_count=1)
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    study = optuna.create_study(study_name="fixture-study", direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(value=0.5, params={"x": 1.0}, distributions={"x": FloatDistribution(0, 2)})
    )
    return {"catboost_metadata": model, "optuna_metadata": study}


def capture() -> dict[str, Any]:
    source = Path(datason.__file__).resolve().parent.parent
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Capture requires an installed git executable")
    # Fixed read-only arguments against the local source checkout; no shell.
    actual = subprocess.check_output([git, "-C", str(source), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    if actual != SOURCE_COMMIT:
        raise ValueError("Capture must use the published a1 source commit")
    subprocess.run([git, "-C", str(source), "diff", "--exit-code", "HEAD", "--", "datason"], check=True)  # noqa: S603
    default_registry.register(PointPlugin())
    values = {**ml_values(), **diagnostic_values(), "custom_point": Point(3, 4)}
    payloads = {name: datason.dumps(value) for name, value in values.items()}
    for name, value, fmt in (
        ("unix_future", dt.datetime(2400, 1, 1, tzinfo=dt.timezone.utc), datason.DateFormat.UNIX),
        ("unix_ms_near_epoch", dt.datetime(1970, 1, 1, 1, tzinfo=dt.timezone.utc), datason.DateFormat.UNIX_MS),
    ):
        payloads[name] = datason.dumps(value, date_format=fmt)
    original = json.dumps({"message": "café", "count": 7}, ensure_ascii=False, separators=(",", ":"))
    payloads["compact_hmac"] = wrap_with_integrity(original, key=TEST_KEY)
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "tensorflow",
        "jax",
        "polars",
        "plotly",
        "catboost",
        "optuna",
    ]
    return {
        "source_tag": "v2.0.0a1",
        "source_commit": actual,
        "python_version": platform.python_version(),
        "library_versions": {name: version(name) for name in packages},
        "payloads": payloads,
        "original_signed_json": original,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(capture(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
