"""Exercise actual Agents SDK hooks and approval/resume with an offline model."""

import asyncio
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

import pytest

import datason

np = pytest.importorskip("numpy")
pytest.importorskip("agents")
from agents import Agent, RunConfig, RunContextWrapper, Runner, RunState, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText


@dataclass
class Context:
    observed: dt.datetime
    vector: Any
    payload: bytes


class OfflineModel(Model):
    async def get_response(self, system_instructions, input, *args, **kwargs):
        if isinstance(input, list) and any(
            isinstance(item, dict) and item.get("type") == "function_call_output" for item in input
        ):
            output = ResponseOutputMessage(
                id="message-1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="completed", annotations=[])],
            )
        else:
            output = ResponseFunctionToolCall(
                id="tool-1", type="function_call", name="record_value", call_id="call-1", arguments="{}"
            )
        return ModelResponse(output=[output], usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise AssertionError("Streaming was not requested")
        yield  # Make this an async iterator.


def serialize_context(value):
    return {"datason_context": datason.dumps(value, **datason.strict_config().__dict__)}


def deserialize_context(mapping):
    return Context(**datason.loads(mapping["datason_context"], **datason.strict_config().__dict__))


def make_agent(calls):
    @function_tool(needs_approval=True)
    async def record_value(ctx: RunContextWrapper[Context]) -> str:
        assert isinstance(ctx.context.observed, dt.datetime)
        assert ctx.context.vector.dtype == np.dtype("float32")
        assert ctx.context.payload == b"\x00\xff"
        ctx.context.vector += 1
        calls.append("executed")
        return "saved"

    return Agent(name="snapshot-fixture", model=OfflineModel(), tools=[record_value])


def test_sdk_snapshot_hooks_and_resume_without_provider_calls():
    async def scenario():
        calls = []
        agent = make_agent(calls)
        config = RunConfig(tracing_disabled=True)
        original = Context(
            dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc), np.array([1, 2], dtype=np.float32), b"\x00\xff"
        )
        paused = await Runner.run(agent, "Record a value", context=original, run_config=config)
        assert len(paused.interruptions) == 1
        assert calls == []
        state_json = paused.to_state().to_json(context_serializer=serialize_context, strict_context=True)
        stored = datason.dumps(state_json)
        rebound = make_agent(calls)  # Runtime model/tool dependencies are supplied by the application.
        restored = await RunState.from_json(
            rebound, datason.loads(stored), context_deserializer=deserialize_context, strict_context=True
        )
        restored.approve(restored.get_interruptions()[0])
        result = await Runner.run(rebound, restored, run_config=config)
        assert result.final_output == "completed"
        assert calls == ["executed"]
        context = result.context_wrapper.context
        assert context.observed == original.observed
        np.testing.assert_array_equal(context.vector, [2, 3])
        assert context.vector.dtype == np.dtype("float32")

    asyncio.run(scenario())


def test_older_sdk_snapshot_resumes_with_rebound_dependencies():
    from pathlib import Path

    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "agents-0.22.0-snapshot.json").read_text())

    async def scenario():
        calls = []
        rebound = make_agent(calls)
        state = await RunState.from_json(
            rebound, fixture["state"], context_deserializer=deserialize_context, strict_context=True
        )
        assert len(state.get_interruptions()) == 1
        assert calls == []
        state.approve(state.get_interruptions()[0])
        result = await Runner.run(rebound, state, run_config=RunConfig(tracing_disabled=True))
        assert result.final_output == "completed"
        assert calls == ["executed"]
        np.testing.assert_array_equal(result.context_wrapper.context.vector, [2, 3])

    asyncio.run(scenario())


def test_context_hooks_ignore_active_diagnostic_normalization():
    original = Context(dt.datetime(2026, 10, 4), np.array([1, 2], dtype=np.float32), b"\x00\xff")
    with datason.config(include_type_hints=False, fallback_to_string=True, redact_fields=("payload",)):
        restored = deserialize_context(serialize_context(original))
    assert restored.payload == original.payload
    assert restored.observed == original.observed
    assert restored.vector.dtype == original.vector.dtype
    np.testing.assert_array_equal(restored.vector, original.vector)
