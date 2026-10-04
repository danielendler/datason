"""Replay immutable payloads from published a1, including known migration cases."""

import datetime as dt
import json
from pathlib import Path

import pytest

import datason
from datason._errors import DeserializationError
from datason._registry import PluginRegistry
from datason.security.integrity import verify_hmac, verify_integrity, wrap_with_integrity

_FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "v2.0.0a1-extended.json").read_text(encoding="utf-8"))
_PAYLOADS = _FIXTURE["payloads"]
_KEY = "public-test-fixture-key"


@pytest.mark.parametrize("name", ["sklearn_estimator", "sklearn_pipeline"])
def test_published_fitted_estimators_predict(name):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    restored = datason.loads(_PAYLOADS[name])
    np.testing.assert_allclose(restored.predict([[4.0], [5.0]]), [9.0, 11.0])


def test_published_sparse_matrix():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    restored = datason.loads(_PAYLOADS["scipy_csr"])
    assert restored.format == "csr"
    assert restored.dtype == np.dtype("float32")
    np.testing.assert_array_equal(restored.toarray(), [[1, 0], [0, 2]])


@pytest.mark.parametrize("name,library", [("torch_tensor", "torch"), ("tf_tensor", "tensorflow"), ("jax_array", "jax")])
def test_published_dense_ml_array(name, library):
    np = pytest.importorskip("numpy")
    pytest.importorskip(library)
    restored = datason.loads(_PAYLOADS[name])
    actual = np.asarray(restored)
    assert actual.dtype == np.dtype("float32")
    assert actual.shape == (2, 2)
    np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])


def test_published_tensorflow_sparse():
    tf = pytest.importorskip("tensorflow")
    restored = datason.loads(_PAYLOADS["tf_sparse"])
    assert restored.dtype == tf.float32
    assert restored.dense_shape.numpy().tolist() == [2, 2]
    assert tf.sparse.to_dense(restored).numpy().tolist() == [[1, 0], [0, 2]]


def test_published_polars_frame():
    pl = pytest.importorskip("polars")
    expected = pl.DataFrame({"count": [1, 2], "score": [1.25, 2.5]})
    assert datason.loads(_PAYLOADS["polars_frame"]).equals(expected)


def test_published_plotly_figure():
    go = pytest.importorskip("plotly.graph_objects")
    restored = datason.loads(_PAYLOADS["plotly_figure"])
    assert isinstance(restored, go.Figure)
    assert tuple(restored.data[0].x) == (1, 2)
    assert tuple(restored.data[0].y) == (3, 4)


@pytest.mark.parametrize("name", ["catboost_metadata", "optuna_metadata"])
def test_published_diagnostic_exports_remain_metadata(name):
    # These tags are handled without invoking CatBoost/Optuna constructors.
    restored = datason.loads(_PAYLOADS[name])
    assert type(restored) is dict
    if name == "catboost_metadata":
        assert restored["class"] == "CatBoostClassifier"
        assert restored["tree_count"] == 2
    else:
        assert restored["n_trials"] == 1
        assert restored["best_params"] == {"x": 1.0}


@pytest.mark.parametrize(
    "name,unit,expected",
    [
        ("unix_future", "seconds", dt.datetime(2400, 1, 1, tzinfo=dt.timezone.utc)),
        ("unix_ms_near_epoch", "milliseconds", dt.datetime(1970, 1, 1, 1, tzinfo=dt.timezone.utc)),
    ],
)
def test_ambiguous_legacy_dates_need_producer_configuration(name, unit, expected):
    wire = _PAYLOADS[name]
    assert datason.loads(wire) != expected  # Legacy magnitude heuristic cannot infer units.
    declared = json.loads(wire)
    declared["timestamp_unit"] = unit  # Supplied by the owner, never guessed.
    repaired = datason.loads(json.dumps(declared))
    assert repaired == expected
    assert datason.loads(datason.dumps(repaired)) == expected


def test_compact_legacy_signature_needs_original_signed_bytes():
    legacy = _PAYLOADS["compact_hmac"]
    assert verify_integrity(legacy, key=_KEY)[0] is False
    signature = json.loads(legacy)["__datason_hmac__"]
    original = _FIXTURE["original_signed_json"]
    assert verify_hmac(original, _KEY, signature)
    assert not verify_hmac(original.replace("café", "tampered"), _KEY, signature)
    # Rewrap the authenticated original, never a payload from a failed envelope.
    upgraded = wrap_with_integrity(original, key=_KEY)
    valid, raw = verify_integrity(upgraded, key=_KEY)
    assert valid
    assert json.loads(raw) == {"message": "café", "count": 7}


class PointPlugin:
    name = "fixture_point"
    priority = 400

    def can_deserialize(self, data):
        return data.get("__datason_type__") == "example.point.v1"

    def deserialize(self, data, ctx):
        x, y = data["__datason_value__"]
        return {"x": x, "y": y}  # Application-defined reconstruction contract.


def test_custom_payload_requires_an_explicit_reviewed_codec(monkeypatch):
    from datason import _deserialize

    registry = PluginRegistry()
    monkeypatch.setattr(_deserialize, "default_registry", registry)
    with pytest.raises(DeserializationError, match="No plugin registered"):
        datason.loads(_PAYLOADS["custom_point"])
    registry.register(PointPlugin())
    assert datason.loads(_PAYLOADS["custom_point"]) == {"x": 3, "y": 4}
    with pytest.raises(DeserializationError, match="No plugin registered"):
        datason.loads(_PAYLOADS["custom_point"].replace("point.v1", "point.v2"))
