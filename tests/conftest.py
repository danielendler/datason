"""Root conftest for datason v2 tests.

Single conftest with explicit fixtures. No global autouse —
prefer proper isolation in each test.

Includes hypothesis custom strategies and profile configuration.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings
from hypothesis import strategies as st

from datason._cache import type_cache
from datason._registry import default_registry

# =========================================================================
# Hypothesis profiles
# =========================================================================

settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=500)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# =========================================================================
# Pytest fixtures
# =========================================================================


@pytest.fixture()
def clean_state():
    """Reset all datason global state. Use explicitly in tests that need it."""
    type_cache.clear()
    default_registry.clear()
    yield
    type_cache.clear()
    default_registry.clear()


@pytest.fixture()
def sample_data() -> dict:
    """Provide a basic nested dict for serialization tests."""
    return {
        "name": "test",
        "count": 42,
        "ratio": 3.14,
        "active": True,
        "nothing": None,
        "tags": ["a", "b", "c"],
        "nested": {"x": 1, "y": 2},
    }


# =========================================================================
# Hypothesis strategies: stdlib types
# =========================================================================


@st.composite
def st_datetimes(draw):
    """Datetimes in a range safe for UNIX timestamp roundtrip."""
    return draw(
        st.datetimes(min_value=dt.datetime(1970, 1, 2), max_value=dt.datetime(9999, 12, 31))  # noqa: DTZ001
    )


@st.composite
def st_dates(draw):
    """Dates in the full valid range."""
    return draw(st.dates(min_value=dt.date(1, 1, 1), max_value=dt.date(9999, 12, 31)))


@st.composite
def st_times(draw):
    """Arbitrary time objects."""
    return draw(st.times())


@st.composite
def st_timedeltas(draw):
    """Timedeltas from integer seconds to avoid float precision loss."""
    seconds = draw(st.integers(min_value=-86400 * 999, max_value=86400 * 999))
    return dt.timedelta(seconds=seconds)


@st.composite
def st_decimals_finite(draw):
    """Finite decimals (no NaN/Inf/sNaN)."""
    return draw(st.decimals(allow_nan=False, allow_infinity=False, places=6))


@st.composite
def st_paths(draw):
    """PurePosixPath from random alphanumeric parts."""
    parts = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=5,
        )
    )
    return pathlib.PurePosixPath("/", *parts)


# =========================================================================
# Hypothesis strategies: numpy types
# =========================================================================

_NP_DTYPES = [np.float64, np.float32, np.int64, np.int32]


@st.composite
def st_numpy_arrays(draw, dtype=None, ndim=None):
    """Numpy arrays with controlled shape and dtype (no NaN)."""
    if dtype is None:
        dtype = draw(st.sampled_from(_NP_DTYPES))
    if ndim is None:
        ndim = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(draw(st.lists(st.integers(min_value=1, max_value=8), min_size=ndim, max_size=ndim)))

    if np.issubdtype(dtype, np.floating):
        elements = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    else:
        info = np.iinfo(dtype)
        elements = st.integers(min_value=int(info.min), max_value=int(info.max))

    flat = draw(st.lists(elements, min_size=int(np.prod(shape)), max_size=int(np.prod(shape))))
    return np.array(flat, dtype=dtype).reshape(shape)


# =========================================================================
# Hypothesis strategies: pandas types
# =========================================================================


@st.composite
def st_dataframes(draw):
    """Small DataFrames with 1-5 columns, 1-20 rows."""
    n_cols = draw(st.integers(min_value=1, max_value=5))
    n_rows = draw(st.integers(min_value=1, max_value=20))
    data = {}
    for i in range(n_cols):
        col_type = draw(st.sampled_from(["int", "float", "str"]))
        name = f"col_{i}"
        match col_type:
            case "int":
                data[name] = draw(
                    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=n_rows, max_size=n_rows)
                )
            case "float":
                data[name] = draw(
                    st.lists(
                        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                        min_size=n_rows,
                        max_size=n_rows,
                    )
                )
            case "str":
                data[name] = draw(st.lists(st.text(min_size=1, max_size=10), min_size=n_rows, max_size=n_rows))
    return pd.DataFrame(data)


@st.composite
def st_series(draw):
    """Pandas Series with 1-20 elements."""
    n = draw(st.integers(min_value=1, max_value=20))
    name = draw(st.text(min_size=1, max_size=10) | st.none())
    values = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return pd.Series(values, name=name)


@st.composite
def st_pandas_timestamps(draw):
    """Pandas Timestamps in a safe range."""
    d = draw(st.datetimes(min_value=dt.datetime(1970, 1, 2), max_value=dt.datetime(2262, 4, 11)))  # noqa: DTZ001
    return pd.Timestamp(d)


# =========================================================================
# Hypothesis strategies: mixed types
# =========================================================================


@st.composite
def st_mixed_serializable_data(draw):
    """Dict with a random mix of basic + plugin types."""
    data: dict = {
        "str_val": draw(st.text(max_size=50)),
        "int_val": draw(st.integers(min_value=-10000, max_value=10000)),
        "float_val": draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
    }
    if draw(st.booleans()):
        data["datetime_val"] = draw(st_datetimes())
    if draw(st.booleans()):
        data["uuid_val"] = draw(st.uuids())
    if draw(st.booleans()):
        data["decimal_val"] = Decimal(
            str(draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
        )
    return data
