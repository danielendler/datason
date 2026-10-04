"""Validate an actual offline SDK tool call against its advertised output."""

import asyncio
import base64
import copy

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jsonschema")
pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as ModelValidationError

import datason
from examples.mcp_schema_boundary import BINARY, ToolResult, demonstrate, normalized_result


def test_actual_tool_output_matches_advertised_schema_and_binary_contract():
    report = asyncio.run(demonstrate())
    outputs = report["outputs"]
    assert outputs["finite"]["score"] == 0.75
    assert outputs["nonfinite"]["score"] is None
    for fields in outputs.values():
        assert fields["weights"] == [0.25, 0.75]
        assert base64.b64decode(fields["preview_base64"], validate=True) == BINARY
        assert fields["observed"] == "2026-10-04T00:00:00+00:00"
        assert "__datason_type__" not in str(fields)


def test_nullable_policy_needs_a_matching_schema():
    fields = normalized_result("nonfinite").model_dump()
    number_only = copy.deepcopy(ToolResult.model_json_schema(mode="serialization"))
    number_only["properties"]["score"] = {"type": "number"}
    with pytest.raises(ValidationError):
        Draft202012Validator(number_only).validate(fields)


def test_binary_encoding_is_validated_beyond_schema_annotations():
    fields = normalized_result("finite").model_dump()
    fields["preview_base64"] = "not valid base64!"
    with pytest.raises(ModelValidationError, match="valid base64"):
        ToolResult.model_validate(fields)


def test_tagged_persistence_is_not_the_wire_contract():
    fields = normalized_result("finite").model_dump()
    fields["observed"] = {"__datason_type__": "datetime", "__datason_value__": fields["observed"]}
    with pytest.raises(ValidationError):
        Draft202012Validator(ToolResult.model_json_schema(mode="serialization")).validate(fields)


def test_dedicated_wire_policy_ignores_active_diagnostic_configuration():
    with datason.config(
        include_type_hints=True, nan_handling=datason.NanHandling.STRING, redact_fields=("preview_base64",)
    ):
        fields = normalized_result("nonfinite").model_dump()
    assert fields["score"] is None
    assert base64.b64decode(fields["preview_base64"], validate=True) == BINARY
