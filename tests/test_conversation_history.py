"""Fresh-turn hydration through the real harness with a function model."""

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from agent_utilities import host_ports
from agent_utilities.graph import (
    DependencyPlan,
    RunBudget,
    RunPlan,
    RunRequest,
    RunTask,
    run,
)
from agent_utilities.harness.conversation_history import decode_completed_history

pytestmark = pytest.mark.spec("AU-BOUNDARY-R043.9.1")


def envelope():
    return {
        "version": 1,
        "phase": "completed",
        "run_id": "prior",
        "messages": [
            {"role": "user", "text": "earlier question"},
            {"role": "assistant", "text": "earlier answer"},
        ],
    }


def bind(value):
    async def provider(request):
        return value

    host_ports.register_run_history_provider(provider)


@pytest.mark.parametrize("bound", [False, True, None])
async def test_first_request_uses_only_bound_history_with_fresh_budget(bound):
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(
            parts=[TextPart("new answer")],
            usage=RequestUsage(input_tokens=2, output_tokens=3),
        )

    if bound is None:
        bind(None)
    elif bound:
        bind(json.dumps(envelope()))
    result = await run(
        RunRequest(task="new question", run_id="current"),
        plan=RunPlan(
            model=FunctionModel(respond),
            steps=("new question",),
            budget=RunBudget(request_limit=1, total_tokens_limit=10),
        ),
    )
    assert result.status == "completed"
    assert len(calls) == 1
    assert len(calls[0]) == (3 if bound else 1)
    if bound:
        assert calls[0][0].parts[0].content == "earlier question"
        assert calls[0][1].parts[0].content == "earlier answer"
    assert calls[0][-1].parts[0].content == "new question"
    assert result.usage["input_tokens"] == 2
    assert result.usage["output_tokens"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"version": True},
        {"phase": "inflight"},
        {"run_id": "current"},
        {"messages": []},
        {"messages": [{}]},
        {"budget": {"request_limit": 99}},
    ],
)
async def test_bad_history_refuses_before_model_effect(change):
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    value = envelope() | change
    bind(json.dumps(value))
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new",)),
    )
    assert result.status == "failed"
    assert calls == []
    assert "completed conversation history is invalid" in result.error


@pytest.mark.parametrize("content", ["private invalid JSON", "x" * 1_048_577])
def test_invalid_content_has_private_safe_error(content):
    with pytest.raises(ValueError, match="^completed conversation history is invalid$"):
        decode_completed_history(content, run_id="current")


async def test_dependency_history_refuses_before_model_effect():
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    bind(json.dumps(envelope()))
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(
            model=FunctionModel(respond),
            dependency_plan=DependencyPlan(
                tasks=(RunTask("one", "new"),),
                output_task_id="one",
                plan_id="plan",
            ),
        ),
    )
    assert result.status == "failed"
    assert "dependency-plan conversation history is unavailable" in result.error
    assert calls == []


async def test_provider_failure_is_not_empty_history_or_private_error():
    async def denied(request):
        raise PermissionError("private provider details")

    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    host_ports.register_run_history_provider(denied)
    result = await run(
        RunRequest(task="new"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new",)),
    )
    assert calls == []
    assert result.status == "failed"
    assert "conversation history provider failed" in result.error
    assert "private provider details" not in result.error


@pytest.mark.parametrize("kind", ["system-prompt", "tool-call", "tool-return"])
def test_non_text_execution_parts_are_not_replayed(kind):
    value = envelope()
    value["messages"][0]["role"] = kind
    with pytest.raises(ValueError, match="completed conversation history is invalid"):
        decode_completed_history(json.dumps(value), run_id="current")


def test_unfinished_user_turn_is_not_completed_history():
    value = envelope()
    value["messages"] = value["messages"][:1]
    with pytest.raises(ValueError, match="completed conversation history is invalid"):
        decode_completed_history(json.dumps(value), run_id="current")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instructions", "private instructions"),
        ("metadata", {"private": "secret"}),
        ("state", "incomplete"),
        ("state", "suspended"),
        ("state", "interrupted"),
        ("workspace_ref", "private workspace"),
        ("provider_details", {"secret": "private"}),
        ("provider_response_id", "private id"),
        ("usage", {"input_tokens": 9000}),
    ],
)
async def test_nested_model_authority_refuses_before_effects(field, value):
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    data = envelope()
    data["messages"][1][field] = value
    bind(json.dumps(data))
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new",)),
    )
    assert result.status == "failed"
    assert calls == []
    assert "private" not in result.error


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "text": "wrong"}, {"role": "user", "text": "wrong"}],
        [{"role": "user", "text": 1}, {"role": "assistant", "text": "answer"}],
        [{"role": "user", "text": "question"}, {"role": "assistant", "text": None}],
        [
            {"role": "user", "text": "question", "extra": 1},
            {"role": "assistant", "text": "answer"},
        ],
        [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}] * 513,
    ],
)
def test_message_shape_and_count_refuse(messages):
    value = envelope() | {"messages": messages}
    with pytest.raises(ValueError, match="completed conversation history is invalid"):
        decode_completed_history(json.dumps(value), run_id="current")


def test_maximum_message_count_is_supported():
    value = envelope()
    value["messages"] *= 512
    assert len(decode_completed_history(json.dumps(value), run_id="current")) == 1024
