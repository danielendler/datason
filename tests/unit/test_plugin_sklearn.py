"""Tests for the scikit-learn plugin."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datason
from datason._config import SerializationConfig
from datason._errors import PluginError
from datason._protocols import DeserializeContext, SerializeContext
from datason._types import TYPE_METADATA_KEY, VALUE_METADATA_KEY
from datason.plugins.sklearn import SklearnPlugin


@pytest.fixture()
def plugin() -> SklearnPlugin:
    return SklearnPlugin()


@pytest.fixture()
def ser_ctx() -> SerializeContext:
    return SerializeContext(config=SerializationConfig(include_type_hints=True))


@pytest.fixture()
def deser_ctx() -> DeserializeContext:
    return DeserializeContext(config=SerializationConfig())


@pytest.fixture()
def fitted_lr() -> LinearRegression:
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1.0, 2.0, 3.0])
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


@pytest.fixture()
def fitted_scaler() -> StandardScaler:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture()
def fitted_pipeline() -> Pipeline:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X, y)
    return pipe


class TestSklearnPluginProperties:
    def test_name(self, plugin: SklearnPlugin) -> None:
        assert plugin.name == "sklearn"

    def test_priority(self, plugin: SklearnPlugin) -> None:
        assert plugin.priority == 302


class TestCanHandle:
    def test_linear_regression(self, plugin: SklearnPlugin) -> None:
        assert plugin.can_handle(LinearRegression())

    def test_standard_scaler(self, plugin: SklearnPlugin) -> None:
        assert plugin.can_handle(StandardScaler())

    def test_pipeline(self, plugin: SklearnPlugin) -> None:
        pipe = Pipeline([("lr", LinearRegression())])
        assert plugin.can_handle(pipe)

    def test_rejects_dict(self, plugin: SklearnPlugin) -> None:
        assert not plugin.can_handle({"a": 1})

    def test_rejects_numpy_array(self, plugin: SklearnPlugin) -> None:
        assert not plugin.can_handle(np.array([1, 2]))


class TestSerializeEstimator:
    def test_class_name_recorded(
        self, plugin: SklearnPlugin, ser_ctx: SerializeContext, fitted_lr: LinearRegression
    ) -> None:
        result = plugin.serialize(fitted_lr, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "sklearn.estimator"
        class_path = result[VALUE_METADATA_KEY]["class"]
        assert "LinearRegression" in class_path
        assert class_path.startswith("sklearn.")

    def test_params_stored(self, plugin: SklearnPlugin, ser_ctx: SerializeContext, fitted_lr: LinearRegression) -> None:
        result = plugin.serialize(fitted_lr, ser_ctx)
        params = result[VALUE_METADATA_KEY]["params"]
        assert "fit_intercept" in params

    def test_state_stored(self, plugin: SklearnPlugin, ser_ctx: SerializeContext, fitted_lr: LinearRegression) -> None:
        result = plugin.serialize(fitted_lr, ser_ctx)
        state = result[VALUE_METADATA_KEY]["state"]
        # Fitted LR has coef_ and intercept_ in state
        assert "coef_" in state
        assert "intercept_" in state

    def test_unfitted_estimator(self, plugin: SklearnPlugin, ser_ctx: SerializeContext) -> None:
        lr = LinearRegression()
        result = plugin.serialize(lr, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "sklearn.estimator"

    def test_without_hints(self, plugin: SklearnPlugin) -> None:
        ctx = SerializeContext(config=SerializationConfig(include_type_hints=False))
        lr = LinearRegression()
        result = plugin.serialize(lr, ctx)
        assert isinstance(result, dict)
        assert "class" in result


class TestSerializePipeline:
    def test_pipeline_type(self, plugin: SklearnPlugin, ser_ctx: SerializeContext, fitted_pipeline: Pipeline) -> None:
        result = plugin.serialize(fitted_pipeline, ser_ctx)
        assert result[TYPE_METADATA_KEY] == "sklearn.Pipeline"

    def test_steps_serialized(
        self, plugin: SklearnPlugin, ser_ctx: SerializeContext, fitted_pipeline: Pipeline
    ) -> None:
        result = plugin.serialize(fitted_pipeline, ser_ctx)
        steps = result[VALUE_METADATA_KEY]["steps"]
        assert len(steps) == 2  # noqa: PLR2004
        assert steps[0]["name"] == "scaler"
        assert steps[1]["name"] == "lr"


class TestCanDeserialize:
    def test_estimator(self, plugin: SklearnPlugin) -> None:
        data = {TYPE_METADATA_KEY: "sklearn.estimator", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_pipeline(self, plugin: SklearnPlugin) -> None:
        data = {TYPE_METADATA_KEY: "sklearn.Pipeline", VALUE_METADATA_KEY: {}}
        assert plugin.can_deserialize(data)

    def test_rejects_non_sklearn(self, plugin: SklearnPlugin) -> None:
        data = {TYPE_METADATA_KEY: "torch.Tensor", VALUE_METADATA_KEY: {}}
        assert not plugin.can_deserialize(data)

    def test_rejects_missing_key(self, plugin: SklearnPlugin) -> None:
        assert not plugin.can_deserialize({"foo": "bar"})


class TestDeserialize:
    def test_security_rejects_non_sklearn(self, plugin: SklearnPlugin, deser_ctx: DeserializeContext) -> None:
        data = {
            TYPE_METADATA_KEY: "sklearn.estimator",
            VALUE_METADATA_KEY: {
                "class": "os.system",
                "params": {},
                "state": {},
            },
        }
        with pytest.raises(PluginError, match="Security"):
            plugin.deserialize(data, deser_ctx)

    def test_bad_value_raises(self, plugin: SklearnPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "sklearn.estimator", VALUE_METADATA_KEY: "not a dict"}
        with pytest.raises(PluginError, match="Expected dict"):
            plugin.deserialize(data, deser_ctx)

    def test_unknown_type_raises(self, plugin: SklearnPlugin, deser_ctx: DeserializeContext) -> None:
        data = {TYPE_METADATA_KEY: "sklearn.unknown", VALUE_METADATA_KEY: {}}
        with pytest.raises(PluginError, match="Unknown sklearn type"):
            plugin.deserialize(data, deser_ctx)


class TestRoundTrip:
    def test_linear_regression_prediction(self, fitted_lr: LinearRegression) -> None:
        X_test = np.array([[7, 8], [9, 10]])
        expected = fitted_lr.predict(X_test)

        s = datason.dumps(fitted_lr, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result, LinearRegression)
        np.testing.assert_array_almost_equal(result.predict(X_test), expected)

    def test_standard_scaler_transform(self, fitted_scaler: StandardScaler) -> None:
        X_test = np.array([[7.0, 8.0]])
        expected = fitted_scaler.transform(X_test)

        s = datason.dumps(fitted_scaler, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result, StandardScaler)
        np.testing.assert_array_almost_equal(result.transform(X_test), expected)

    def test_pipeline_roundtrip(self, fitted_pipeline: Pipeline) -> None:
        X_test = np.array([[7.0, 8.0]])
        expected = fitted_pipeline.predict(X_test)

        s = datason.dumps(fitted_pipeline, include_type_hints=True)
        result = datason.loads(s)

        assert isinstance(result, Pipeline)
        np.testing.assert_array_almost_equal(result.predict(X_test), expected)

    def test_estimator_in_dict(self, fitted_lr: LinearRegression) -> None:
        data = {"model": fitted_lr, "name": "linear"}
        s = datason.dumps(data, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result["model"], LinearRegression)
        assert result["name"] == "linear"

    def test_json_valid(self, fitted_lr: LinearRegression) -> None:
        s = datason.dumps(fitted_lr, include_type_hints=True)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_unfitted_lr_roundtrip(self) -> None:
        lr = LinearRegression(fit_intercept=False)
        s = datason.dumps(lr, include_type_hints=True)
        result = datason.loads(s)
        assert isinstance(result, LinearRegression)
        assert result.fit_intercept is False
