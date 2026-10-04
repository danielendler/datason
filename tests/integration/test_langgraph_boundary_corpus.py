"""Keep the adapter's scientific corpus valid even when native SDKs improve."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("langgraph")

from datason.integrations.langgraph import DatasonSerializer
from scripts.validate_framework_boundaries import cases, check


@pytest.mark.parametrize(
    "case", ["int32", "float32", "datetime64_array", "timedelta64_array", "empty_multidimensional"]
)
def test_adapter_preserves_boundary_case(case):
    result = check(DatasonSerializer(), cases()[case])
    assert result == {"serialization": "success", "value_dtype_shape": "preserved"}
