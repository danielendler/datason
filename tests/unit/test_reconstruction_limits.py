"""Dense limits cover legacy shape inference and both allocation estimates."""

import pytest

from datason._config import SerializationConfig
from datason._errors import SecurityError
from datason._protocols import DeserializeContext
from datason._reconstruction import check_dense_allocation


@pytest.mark.parametrize("shape", [[3], [1, 1, 1]])
def test_dimension_and_rank_limits(shape):
    ctx = DeserializeContext(config=SerializationConfig(max_size=2))
    with pytest.raises(SecurityError, match="container limit"):
        check_dense_allocation([], shape, 4, ctx)


def test_dtype_width_cannot_exceed_buffer_budget():
    ctx = DeserializeContext(config=SerializationConfig(max_input_bytes=4))
    with pytest.raises(SecurityError, match="dtype item size"):
        check_dense_allocation([], [0], 8, ctx)


def test_legacy_inferred_shape_still_checks_actual_buffer_size():
    ctx = DeserializeContext(config=SerializationConfig(max_input_bytes=8))
    assert check_dense_allocation([[1, 2]], None, 4, ctx) is None
    with pytest.raises(SecurityError, match="byte budget"):
        check_dense_allocation([[1, 2], [3]], None, 4, ctx)
