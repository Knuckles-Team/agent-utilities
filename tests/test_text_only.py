"""Received constrained execution through a real provider with synthetic HTTP."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_utilities import host_ports
from agent_utilities.graph.state import HarnessConfig
from agent_utilities.runtime import (
    AgentRunExecutor,
    RunBudget,
    RunPlan,
    RunRequest,
    StopRule,
)

pytestmark = pytest.mark.spec("AU-BOUNDARY-R043.13.1")


def fixture(
    response=None,
    *,
    defaults=None,
    retries=0,
    responses=False,
    handler_override=None,
    timeout=2,
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response
            or {
                "id": "test-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler_override or handler))
    client = AsyncOpenAI(
        api_key="inert-test-only",  # Sanitizer: allow - inert synthetic provider fixture.
        base_url="http://test.invalid/v1",
        http_client=http,
        max_retries=retries,
        timeout=timeout,
    )
    provider = OpenAIProvider(openai_client=client)
    model = (OpenAIResponsesModel if responses else OpenAIChatModel)(
        "test-model", provider=provider, settings=defaults
    )
    plan = RunPlan(
        model=model,
        execution_constraint="text-only",
        steps=("hello",),
        budget=RunBudget(
            request_limit=1,
            tool_calls_limit=0,
            total_tokens_limit=20,
            timeout_seconds=2,
            max_attachments=0,
            max_attachment_bytes=0,
            max_concurrent_requests=1,
        ),
        stop=StopRule(max_steps=1),
        harness=HarnessConfig(retries=0, model_settings={"max_tokens": 10}),
    )
    return client, calls, plan


@pytest.mark.asyncio
async def test_real_text_loop_without_inherited_hooks(monkeypatch):
    client, calls, plan = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("inherited authority hook")

    for name in (
        "read_run_history",
        "authority_keepalive_scope",
        "scaffold_publication_scope",
        "emit_run_event",
        "resolve_model",
    ):
        monkeypatch.setattr(host_ports, name, forbidden)
    try:
        result = await AgentRunExecutor().execute_text_only(
            RunRequest(task="hello"), plan=plan
        )
        assert result.status == "completed" and result.output == "hello"
        assert result.usage["requests"] == 1 and result.usage["total_tokens"] == 3
        assert not client.is_closed()
        assert len(calls) == 1 and "tools" not in calls[0]
        assert calls[0].get("max_completion_tokens", calls[0].get("max_tokens")) == 10
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"defaults": {"extra_body": {"tools": [{"type": "web_search"}]}}},
        {"defaults": {"openai_native_tools": [{"type": "web_search_preview"}]}},
        {"retries": 2},
        {"responses": True},
        {"timeout": None},
        {"timeout": 0},
    ],
)
async def test_provider_effect_configuration_refuses(options):
    client, calls, plan = fixture(**options)
    try:
        if options.get("responses"):
            with pytest.raises(ValueError):
                await AgentRunExecutor().execute_text_only(
                    RunRequest(task="hello"), plan=plan
                )
        else:
            result = await AgentRunExecutor().execute_text_only(
                RunRequest(task="hello"), plan=plan
            )
            assert result.status == "failed"
        assert calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_reply_never_executes_or_retries():
    response = {
        "id": "test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool1",
                            "type": "function",
                            "function": {"name": "delete", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }
    client, calls, plan = fixture(response)
    try:
        result = await AgentRunExecutor().execute_text_only(
            RunRequest(task="hello"), plan=plan
        )
        assert result.status == "failed" and len(calls) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_names", ("delete",)),
        ("input_media_types", ("image/png",)),
        ("steps", ("a", "b")),
    ],
)
async def test_widened_plan_refuses_before_provider(field, value):
    client, calls, plan = fixture()
    try:
        with pytest.raises(ValueError):
            await AgentRunExecutor().execute_text_only(
                RunRequest(task="hello"), plan=replace(plan, **{field: value})
            )
        assert calls == []
    finally:
        await client.close()


def test_default_positional_run_plan_remains_compatible():
    plan = RunPlan("model", "instructions", ("prompt",))
    assert plan.instructions == "instructions" and plan.steps == ("prompt",)
    assert plan.execution_constraint is None


@pytest.mark.asyncio
async def test_over_budget_reply_retains_reported_consumption():
    client, calls, plan = fixture()
    plan = replace(
        plan,
        budget=replace(plan.budget, total_tokens_limit=2),
        harness=replace(plan.harness, model_settings={"max_tokens": 1}),
    )
    try:
        result = await AgentRunExecutor().execute_text_only(
            RunRequest(task="hello"), plan=plan
        )
        assert result.status == "stopped" and len(calls) == 1
        assert result.usage["requests"] == 1 and result.usage["total_tokens"] == 3
        assert result.output == ""
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retryable_http_error_is_one_attempt():
    calls = []

    def failure(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "synthetic failure"}})

    client, _, plan = fixture(handler_override=failure)
    try:
        result = await AgentRunExecutor().execute_text_only(
            RunRequest(task="hello"), plan=plan
        )
        assert result.status == "failed" and len(calls) == 1
        assert "synthetic failure" not in (result.error or "")
        assert result.usage["requests"] == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancel_drains_provider_before_finishing():
    entered, exited = asyncio.Event(), asyncio.Event()

    async def blocked(request):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            exited.set()

    client, _, plan = fixture(handler_override=blocked)
    task = asyncio.create_task(
        AgentRunExecutor().execute_text_only(RunRequest(task="hello"), plan=plan)
    )
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "received",
    [
        RunRequest(task="hello", context="private"),
        RunRequest(task="hello", metadata={"session": "forged"}),
        RunRequest(task="different"),
    ],
)
async def test_received_request_cannot_inherit_or_substitute(received):
    client, calls, plan = fixture()
    try:
        result = await AgentRunExecutor().execute_text_only(received, plan=plan)
        assert result.status == "failed" and calls == []
    finally:
        await client.close()
