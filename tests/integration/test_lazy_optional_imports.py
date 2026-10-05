"""Use fresh interpreters so pytest's optional imports cannot hide eager loading."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import datason

_SOURCE = Path(datason.__file__).resolve().parent.parent
_OPTIONAL = (
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "tensorflow",
    "sklearn",
    "polars",
    "jax",
    "jaxlib",
    "catboost",
    "optuna",
    "plotly",
    "pydantic",
)


def run_fresh(code):
    proc = subprocess.run(  # noqa: S603 -- fixed current interpreter, owned test code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=_SOURCE,
        env={**os.environ, "PYTHONPATH": str(_SOURCE)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_import_and_non_scientific_boundaries_load_no_optional_library():
    run_fresh(f"""
import datetime as dt, json, sys
import datason
from datason._errors import DeserializationError, SecurityError, SerializationError
roots = {_OPTIONAL!r}
assert not set(roots).intersection(sys.modules)
data = {{"datetime": dt.datetime(2026, 10, 4), "binary": b"\\x00\\xff", "values": [1, True, None]}}
assert datason.loads(datason.dumps(data)) == data
assert json.loads(datason.dumps({{"x": 1}})) == {{"x": 1}}
class Unknown: pass
try: datason.dumps({{"nested": Unknown()}})
except SerializationError as exc: assert "nested" in str(exc)
else: raise AssertionError("unknown type accepted")
for tag in ("untrusted.module.Class", "torch.Tensor"):
    wire = json.dumps({{"__datason_type__": tag, "__datason_value__": {{}}}})
    try: datason.loads(wire, allow_plugin_deserialization=False)
    except DeserializationError: pass
    else: raise AssertionError("disabled dispatch occurred")
    try: datason.loads(wire, max_nodes=0)
    except SecurityError: pass
    else: raise AssertionError("budget accepted")
assert not set(roots).intersection(sys.modules)
""")


_CASES = {
    "numpy": (
        "numpy.ndarray",
        {"data": [[1.25, 2.5]], "dtype": "float32", "shape": [1, 2]},
        'assert value.dtype.name == "float32" and value.shape == (1, 2)',
    ),
    "pandas": (
        "pandas.Timestamp",
        "2026-10-04T00:00:00+00:00",
        'assert value.isoformat() == "2026-10-04T00:00:00+00:00"',
    ),
    "scipy": (
        "scipy.sparse.matrix",
        {"format": "coo", "data": [1.0], "row": [0], "col": [1], "shape": [2, 2], "dtype": "float64"},
        "assert value.shape == (2, 2) and value.toarray()[0, 1] == 1",
    ),
    "torch": (
        "torch.Tensor",
        {"data": [[1.25, 2.5]], "dtype": "float32", "shape": [1, 2], "device": "cuda:0"},
        'assert str(value.device) == "cpu" and list(value.shape) == [1, 2]',
    ),
    "tensorflow": (
        "tf.Tensor",
        {"data": [[1.25, 2.5]], "dtype": "float32", "shape": [1, 2]},
        'assert value.dtype.name == "float32" and list(value.shape) == [1, 2]',
    ),
    "sklearn": (
        "sklearn.estimator",
        {"class": "sklearn.preprocessing._data.StandardScaler", "params": {}, "state": {}},
        'assert type(value).__name__ == "StandardScaler"',
    ),
    "polars": (
        "polars.DataFrame",
        {"columns": ["x"], "data": {"x": [1, 2]}, "schema": {"x": "Int64"}},
        "assert value.shape == (2, 1)",
    ),
    "jax": (
        "jax.Array",
        {"data": [[1.25, 2.5]], "dtype": "float32", "shape": [1, 2]},
        'assert str(value.dtype) == "float32" and value.shape == (1, 2)',
    ),
    "plotly": ("plotly.Figure", {"data": [], "layout": {}}, 'assert type(value).__name__ == "Figure"'),
}


@pytest.mark.parametrize("family", list(_CASES))
def test_tagged_reconstruction_is_first_use_and_keeps_other_plugins_deferred(family):
    if importlib.util.find_spec(family) is None:
        pytest.skip(f"{family} is not installed")
    tag, payload, verify = _CASES[family]
    run_fresh(f"""
import json, sys
import datason
assert not set({_OPTIONAL!r}).intersection(sys.modules)
value = datason.loads(json.dumps({{"__datason_type__": {tag!r}, "__datason_value__": {payload!r}}}))
{verify}
# Libraries may import their own dependencies; Datason must not activate other codecs.
active = {{name.removeprefix("datason.plugins.") for name in sys.modules if name.startswith("datason.plugins.")}}
optional_plugins = {{"numpy", "pandas", "scipy_sparse", "torch", "tensorflow", "sklearn", "ml_misc", "pydantic"}}
expected = "ml_misc" if {family!r} in ("polars", "jax", "plotly") else ("scipy_sparse" if {family!r} == "scipy" else {family!r})
assert active.intersection(optional_plugins) == {{expected}}
from datason._registry import default_registry
proxy = next(item for item in default_registry._plugins if item.name == expected)
assert proxy._plugin.name == proxy.name and proxy._plugin.priority == proxy.priority
if {family!r} in ("polars", "jax", "plotly"):
    from datason.plugins.ml_misc import _loaded
    assert _loaded == {{{family!r}}}
""")


def test_numpy_application_subclass_does_not_initialize_ml_frameworks():
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy is not installed")
    run_fresh("""
import sys, datason, numpy as np
class ApplicationArray(np.ndarray): pass
value = np.arange(6, dtype=np.float32).reshape(2, 3).view(ApplicationArray)
restored = datason.loads(datason.dumps(value))
np.testing.assert_array_equal(restored, value)
assert restored.dtype == value.dtype and restored.shape == value.shape
assert not {"torch", "tensorflow", "sklearn", "jax", "catboost", "optuna", "plotly", "polars"}.intersection(sys.modules)
""")


def test_metadata_only_loads_need_no_installed_ml_library():
    run_fresh(f"""
import json, sys, datason
for tag in ("catboost.Model", "optuna.Study"):
    value = datason.loads(json.dumps({{"__datason_type__": tag, "__datason_value__": {{"params": {{}}}}}}))
    assert value == {{"params": {{}}}}
assert not set({_OPTIONAL!r}).intersection(sys.modules)
""")


@pytest.mark.parametrize("family", ["polars", "jax"])
def test_warmed_metadata_descriptor_still_loads_new_misc_families_on_demand(family):
    if importlib.util.find_spec(family) is None:
        pytest.skip(f"{family} is not installed")
    run_fresh(f"""
import json, sys, datason
wire = json.dumps({{"__datason_type__": "optuna.Study", "__datason_value__": {{"params": {{}}}}}})
assert datason.loads(wire) == {{"params": {{}}}}
assert not set({_OPTIONAL!r}).intersection(sys.modules)
from datason._registry import default_registry
proxy = next(p for p in default_registry._plugins if p.name == "ml_misc")
def forbidden(): raise AssertionError("warmed descriptor used loader")
proxy._load = forbidden
if {family!r} == "polars":
    import polars
    value = polars.DataFrame({{"x": [1, 2]}})
    restored = datason.loads(datason.dumps(value))
    assert restored.equals(value)
else:
    import jax.numpy as jnp
    value = jnp.array([1, 2], dtype=jnp.int32)
    restored = datason.loads(datason.dumps(value))
    assert restored.dtype == value.dtype and restored.tolist() == value.tolist()
from datason.plugins.ml_misc import _loaded
assert _loaded == {{{family!r}}}
""")
