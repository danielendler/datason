"""Application-shaped data contract, independent of database or model services."""

import json
from decimal import Decimal
from uuid import UUID

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import datason
from examples.financial_snapshot import api_result, demonstrate, sample_candidate, snapshot


def test_financial_snapshot_preserves_supported_types():
    report = demonstrate()
    assert report["fidelity_verified"]
    candidate = datason.loads(snapshot(sample_candidate()))
    assert isinstance(candidate["amount"], Decimal)
    assert isinstance(candidate["id"], UUID)
    assert str(candidate["transactions"].dtypes["days"]) == "Int64"
    assert candidate["transactions"]["observed"].dt.tz is not None


def test_financial_api_uses_ordinary_values_and_explicit_projection():
    fields = api_result(sample_candidate())
    assert fields["confidence"] == 0.75
    assert isinstance(fields["id"], str)
    assert isinstance(fields["observed"], str)
    assert fields["transactions"][1]["days"] is None
    assert "__datason_type__" not in json.dumps(fields)
    assert json.loads(json.dumps(fields, allow_nan=False)) == fields


def test_financial_snapshot_isolated_from_active_diagnostic_redaction():
    with datason.config(redact_fields=("amount",), include_type_hints=False):
        fields = datason.loads(snapshot(sample_candidate()), **datason.strict_config().__dict__)
    assert fields["amount"] == Decimal("19.99")
