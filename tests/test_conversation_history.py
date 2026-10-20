"""Actual cross-turn model history using host-provided checkpoint content."""

import contextlib
import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run


class Checkpoint:
    admission_identity = "admitted"

    def __init__(self):
        self.content = None

    async def restore(self):
        return self.content

    async def save(self, content):
        self.content = content


def bind_checkpoint(port):
    @contextlib.asynccontextmanager
    async def scope(request, plan):
        yield port

    host_ports.register_run_checkpoint_factory(scope)


def model(seen):
    def respond(messages, info):
        seen.append(messages)
        return ModelResponse(parts=[TextPart("assistant remembers")])

    return FunctionModel(respond)


async def previous_turn():
    port = Checkpoint()
    bind_checkpoint(port)
    result = await run(
        RunRequest(task="first", run_id="prior"),
        plan=RunPlan(model=model([]), steps=("my favorite is blue",)),
    )
    assert result.status == "completed"
    return port.content


async def test_new_turn_receives_actual_prior_messages_with_fresh_usage():
    content = await previous_turn()
    current = Checkpoint()
    bind_checkpoint(current)
    reads = []

    async def history(request):
        reads.append(request.run_id)
        return {"run_id": "prior", "content": content}

    host_ports.register_run_history_provider(history)
    seen = []
    result = await run(
        RunRequest(task="second", run_id="current"),
        plan=RunPlan(model=model(seen), steps=("what is my favorite?",)),
    )
    assert result.status == "completed"
    assert result.steps == 1 and result.usage["requests"] == 1
    assert reads == ["current"]
    parts = [part for message in seen[0] for part in message.parts]
    assert [
        part.content
        for part in parts
        if getattr(part, "part_kind", None) == "user-prompt"
    ] == ["my favorite is blue", "what is my favorite?"]
    assert any(
        getattr(part, "content", None) == "assistant remembers" for part in parts
    )
    saved = json.loads(current.content)
    assert saved["run_id"] == "current" and saved["next_step"] == 1
    assert saved["usage"]["requests"] == 1


async def test_current_checkpoint_takes_precedence_over_prior_history():
    port = Checkpoint()
    bind_checkpoint(port)
    seen = []
    plan = RunPlan(model=model(seen), steps=("current task",))
    request = RunRequest(task="current", run_id="current")
    await run(request, plan=plan)

    async def history(request):
        pytest.fail("resumed current checkpoint must not read prior history")

    host_ports.register_run_history_provider(history)
    result = await run(request, plan=plan)
    assert result.status == "completed" and len(seen) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "foreign"},
        {"phase": "step_inflight"},
        {"status": "running"},
        {"messages": []},
        {"next_step": 0},
        {"messages": [{"malformed": True}]},
    ],
)
async def test_invalid_history_refuses_before_any_model_request(changes):
    raw = json.loads(await previous_turn())
    raw.update(changes)
    host_ports.register_run_checkpoint_factory(None)

    async def history(request):
        return {"run_id": "prior", "content": json.dumps(raw)}

    host_ports.register_run_history_provider(history)
    seen = []
    result = await run(
        RunRequest(task="second", run_id="current"),
        plan=RunPlan(model=model(seen), steps=("second",)),
    )
    assert result.status == "failed"
    assert seen == []
    assert "foreign" not in result.error


async def test_missing_optional_history_preserves_ordinary_run():
    seen = []
    result = await run(
        RunRequest(task="fresh"), plan=RunPlan(model=model(seen), steps=("fresh",))
    )
    assert result.status == "completed" and len(seen) == 1


async def test_prior_history_requires_current_durable_checkpoint_before_model():
    content = await previous_turn()
    host_ports.register_run_checkpoint_factory(None)

    async def history(request):
        return {"run_id": "prior", "content": content}

    host_ports.register_run_history_provider(history)
    seen = []
    result = await run(
        RunRequest(task="second", run_id="current"),
        plan=RunPlan(model=model(seen), steps=("second",)),
    )
    assert result.status == "failed"
    assert seen == []
