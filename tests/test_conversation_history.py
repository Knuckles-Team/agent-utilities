"""Fresh-turn hydration through the real harness with a function model."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

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


@pytest.mark.spec("AU-BOUNDARY-R043.9.2.1")
@pytest.mark.parametrize("prior", [None, "completed"])
async def test_checkpoint_selection_is_one_read_without_legacy_fallback(prior):
    reads, legacy, calls = [], [], []
    value = json.dumps(envelope()) if prior else None

    async def checkpoints(request):
        reads.append(request.run_id)
        selected = host_ports.RunCheckpointSelection(request.run_id, None, value)
        # A later host change cannot cause a second, mixed selection read.
        host_ports.register_run_checkpoint_provider(None)
        return selected

    async def old_history(request):
        legacy.append(request.run_id)
        raise AssertionError("legacy fallback must not run")

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(
            parts=[TextPart("new answer")],
            usage=RequestUsage(input_tokens=2, output_tokens=3),
        )

    host_ports.register_run_checkpoint_provider(checkpoints)
    host_ports.register_run_history_provider(old_history)
    result = await run(
        RunRequest(task="new question", run_id="current"),
        plan=RunPlan(
            model=FunctionModel(respond),
            steps=("new question",),
            budget=RunBudget(request_limit=1, total_tokens_limit=10),
        ),
    )
    assert result.status == "completed"
    assert reads == ["current"] and legacy == []
    assert len(calls) == 1 and len(calls[0]) == (3 if prior else 1)
    assert calls[0][-1].parts[0].content == "new question"
    assert result.usage["input_tokens"] == 2
    assert result.usage["output_tokens"] == 3


@pytest.mark.spec("AU-BOUNDARY-R043.9.2.1")
@pytest.mark.parametrize("current", ["", "private malformed", '{"phase":"inflight"}'])
async def test_current_checkpoint_precedes_history_before_planning_or_model(current):
    effects = []

    async def checkpoints(request):
        return host_ports.RunCheckpointSelection(request.run_id, current, "bad prior")

    async def old_history(request):
        effects.append("legacy history")
        return json.dumps(envelope())

    def decide(request):
        effects.append("planning")
        return RunPlan(model="unused", steps=("new",))

    host_ports.register_run_checkpoint_provider(checkpoints)
    host_ports.register_run_history_provider(old_history)
    host_ports.register_run_decision_provider(decide)
    host_ports.register_model_provider(lambda name: effects.append("model"))
    result = await run(RunRequest(task="new", run_id="current"))
    assert result.status == "failed"
    assert effects == []
    assert "current-run checkpoint restore is unavailable" in result.error
    assert "private" not in result.error


@pytest.mark.spec("AU-BOUNDARY-R043.9.2.1")
@pytest.mark.parametrize(
    "selection",
    [
        None,
        {"run_id": "current", "completed_history": None},
        host_ports.RunCheckpointSelection("foreign", None, None),
        host_ports.RunCheckpointSelection("current", None, None, version=True),
        host_ports.RunCheckpointSelection("current", None, None, version=2),
        host_ports.RunCheckpointSelection("current", None, 12),
        host_ports.RunCheckpointSelection("current", None, "private invalid history"),
    ],
)
async def test_partial_or_malformed_selection_refuses_without_fallback(selection):
    effects = []

    async def checkpoints(request):
        return selection

    async def old_history(request):
        effects.append("legacy")
        return json.dumps(envelope())

    def respond(messages, info):
        effects.append("model")
        return ModelResponse(parts=[TextPart("unexpected")])

    def decide(request):
        effects.append("planning")
        return RunPlan(model=FunctionModel(respond), steps=("new",))

    host_ports.register_run_checkpoint_provider(checkpoints)
    host_ports.register_run_history_provider(old_history)
    host_ports.register_run_decision_provider(decide)
    result = await run(RunRequest(task="new", run_id="current"))
    assert result.status == "failed" and effects == []
    assert "private" not in result.error


@pytest.mark.spec("AU-BOUNDARY-R043.9.2.1")
@pytest.mark.parametrize("cancelled", [False, True])
async def test_checkpoint_provider_failure_is_private_and_does_not_plan(
    cancelled, events
):
    effects = []

    async def checkpoints(request):
        if cancelled:
            raise asyncio.CancelledError
        raise PermissionError("private storage failure")

    async def old_history(request):
        effects.append("legacy")
        return json.dumps(envelope())

    def decide(request):
        effects.append("planning")
        raise AssertionError("must not plan")

    host_ports.register_run_checkpoint_provider(checkpoints)
    host_ports.register_run_decision_provider(decide)
    host_ports.register_run_history_provider(old_history)
    host_ports.register_model_provider(lambda name: effects.append("model"))
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await run(RunRequest(task="new", run_id="current"))
        assert effects == []
        assert not any(
            event.get("type") in {"error", "graph_complete"} for event in events
        )
        return
    result = await run(RunRequest(task="new", run_id="current"))
    assert result.status == "failed" and effects == []
    assert "run checkpoint provider failed" in result.error
    assert "private" not in result.error


@pytest.mark.spec("AU-BOUNDARY-R043.9.2.1")
@pytest.mark.parametrize("case", ["current", "provider", "invalid", "success"])
async def test_supplied_plan_usage_survives_history_refusal_exactly_once(case, events):
    calls = []

    async def checkpoints(request):
        if case == "provider":
            raise PermissionError("private failure")
        return host_ports.RunCheckpointSelection(
            request.run_id,
            "checkpoint" if case == "current" else None,
            "invalid" if case == "invalid" else None,
        )

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(
            parts=[TextPart("answer")],
            usage=RequestUsage(input_tokens=2, output_tokens=3),
        )

    host_ports.register_run_checkpoint_provider(checkpoints)
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(
            model=FunctionModel(respond),
            steps=("new",),
            initial_usage=RunUsage(requests=1, input_tokens=3, output_tokens=4),
            budget=RunBudget(request_limit=2),
        ),
    )
    succeeded = case == "success"
    assert result.status == ("completed" if succeeded else "failed")
    assert len(calls) == int(succeeded)
    assert result.usage["requests"] == 1 + int(succeeded)
    assert result.usage["input_tokens"] == 3 + (2 if succeeded else 0)
    assert result.usage["output_tokens"] == 4 + (3 if succeeded else 0)
    assert events[-1]["result"]["usage"] == result.usage


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "child_id",
        "conversation_id",
        "predecessor_run_id",
        "history_revision",
        "phase",
    ],
)
async def test_selected_child_history_is_bound_before_model(mismatch):
    binding = host_ports.CompletedChildBinding(
        "child", "conversation", "prior", "revision"
    )
    value = {
        "version": 1,
        "phase": "completed",
        "child_id": "child",
        "conversation_id": "conversation",
        "predecessor_run_id": "prior",
        "history_revision": "revision",
        "messages": envelope()["messages"],
    }
    if mismatch:
        value[mismatch] = "foreign"

    async def select(request):
        return host_ports.RunCheckpointSelection(
            request.run_id, None, json.dumps(value), completed_child=binding
        )

    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    host_ports.register_run_checkpoint_provider(select)
    result = await run(
        RunRequest(task="new question", run_id="current"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new question",)),
    )
    if mismatch:
        assert result.status == "failed"
        assert not calls
    else:
        assert result.status == "completed"
        assert [message.parts[0].content for message in calls[0]] == [
            "earlier question",
            "earlier answer",
            "new question",
        ]


@pytest.mark.parametrize("current", [None, "checkpoint"])
async def test_missing_child_history_and_current_checkpoint_fail_closed(current):
    async def select(request):
        return host_ports.RunCheckpointSelection(
            request.run_id,
            current,
            None,
            completed_child=host_ports.CompletedChildBinding(
                "child", "conversation", "prior", "revision"
            ),
        )

    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    host_ports.register_run_checkpoint_provider(select)
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new",)),
    )
    assert result.status == "failed"
    assert not calls
    assert (
        "current-run checkpoint restore is unavailable"
        if current
        else "completed child history is invalid"
    ) in result.error


@pytest.mark.parametrize(
    "field", ["child_id", "conversation_id", "predecessor_run_id", "history_revision"]
)
async def test_oversized_child_binding_refuses_before_model(field):
    values = {
        "child_id": "child",
        "conversation_id": "conversation",
        "predecessor_run_id": "prior",
        "history_revision": "revision",
    }
    values[field] = "é" * 257
    binding = host_ports.CompletedChildBinding(**values)

    async def select(request):
        payload = {
            "version": 1,
            "phase": "completed",
            **values,
            "messages": envelope()["messages"],
        }
        return host_ports.RunCheckpointSelection(
            request.run_id, None, json.dumps(payload), completed_child=binding
        )

    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    host_ports.register_run_checkpoint_provider(select)
    result = await run(
        RunRequest(task="new", run_id="current"),
        plan=RunPlan(model=FunctionModel(respond), steps=("new",)),
    )
    assert result.status == "failed"
    assert not calls
