"""Owned synthetic financial data: ordinary API values and typed local storage.

Inspired by financialModel02's response/storage split; this does not run that
application or migrate its old Datason payloads.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd

import datason


def sample_candidate() -> dict:
    return {
        "id": UUID(int=42),
        "observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
        "amount": Decimal("19.99"),
        "confidence": np.float32(0.75),
        "features": np.array([[19.99, 30], [19.99, 31]], dtype=np.float32),
        "transactions": pd.DataFrame(
            {
                "amount": pd.Series([19.99, 19.99], dtype="float64"),
                "days": pd.Series([30, None], dtype="Int64"),
                "observed": pd.date_range("2026-09-01", periods=2, tz="UTC"),
            }
        ),
    }


def api_result(candidate: dict) -> dict:
    """Choose ordinary JSON explicitly; do not guess types when reading it."""
    return json.loads(datason.dumps(candidate, **datason.api_config().__dict__))


def snapshot(candidate: dict) -> str:
    """Trusted, application-owned state, preserving supported type metadata."""
    policy = datason.strict_config(dataframe_orient=datason.DataFrameOrient.SPLIT)
    return datason.dumps(candidate, **policy.__dict__)


def demonstrate() -> dict:
    candidate = sample_candidate()
    stored = snapshot(candidate)
    restored = datason.loads(stored, **datason.strict_config().__dict__)
    assert restored["amount"] == candidate["amount"]
    assert restored["id"] == candidate["id"]
    assert restored["observed"] == candidate["observed"]
    assert restored["confidence"].dtype == candidate["confidence"].dtype
    assert restored["features"].dtype == candidate["features"].dtype
    assert restored["features"].shape == candidate["features"].shape
    np.testing.assert_array_equal(restored["features"], candidate["features"])
    pd.testing.assert_frame_equal(restored["transactions"], candidate["transactions"])
    return {"api": api_result(candidate), "snapshot_bytes": len(stored.encode("utf-8")), "fidelity_verified": True}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
