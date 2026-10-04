"""An offline MCP tool with an explicit ordinary-JSON output contract."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
from typing import Literal
from uuid import UUID

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field, field_validator

import datason

BINARY = b"\x89PNG\r\n\x1a\n\x00\xff"


class ToolResult(BaseModel):
    """Application-owned wire model: validation and serialization shapes agree."""

    model_config = ConfigDict(extra="forbid")
    observed: str = Field(json_schema_extra={"format": "date-time"})
    request_id: str = Field(json_schema_extra={"format": "uuid"})
    score: float | None
    weights: list[float]
    preview_base64: str = Field(json_schema_extra={"contentEncoding": "base64"})

    @field_validator("preview_base64")
    @classmethod
    def validate_binary_encoding(cls, value: str) -> str:
        # contentEncoding is an annotation, not an automatic JSON Schema check.
        try:
            base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("preview_base64 must be valid base64") from exc
        return value


def normalized_result(mode: str) -> ToolResult:
    """Normalize scientific data, then explicitly validate the wire contract."""
    raw = {
        "observed": dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc),
        "request_id": UUID(int=42),
        "score": np.float32(0.75) if mode == "finite" else np.float32(np.inf),
        "weights": np.array([0.25, 0.75], dtype=np.float32),
        "preview_base64": BINARY,
    }
    # A dedicated policy keeps diagnostic redaction/tag/string settings separate.
    fields = json.loads(datason.dumps(raw, **datason.api_config().__dict__))
    schema = ToolResult.model_json_schema(mode="serialization")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(fields)
    return ToolResult.model_validate(fields)


def create_server() -> MCPServer:
    server = MCPServer("datason-schema-boundary")

    @server.tool()
    async def inspect_sample(mode: Literal["finite", "nonfinite"] = "finite") -> ToolResult:
        """Return an ordinary-JSON diagnostic sample; non-finite score is nullable."""
        return normalized_result(mode)

    return server


async def demonstrate() -> dict[str, object]:
    """Inspect the SDK-advertised schema and actual protocol-delivered result."""
    async with Client(create_server()) as client:
        tools = await client.list_tools()
        tool = next(item for item in tools.tools if item.name == "inspect_sample")
        outputs = {}
        for mode in ("finite", "nonfinite"):
            result = await client.call_tool("inspect_sample", {"mode": mode})
            if result.is_error:
                raise ValueError(f"MCP tool failed: {result.content}")
            Draft202012Validator(tool.output_schema, format_checker=FormatChecker()).validate(result.structured_content)
            outputs[mode] = result.structured_content
        return {"advertised_schema": tool.output_schema, "outputs": outputs}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(demonstrate()), indent=2))
