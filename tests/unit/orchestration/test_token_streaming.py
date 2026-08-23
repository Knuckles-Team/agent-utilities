"""Tests for token-by-token answer streaming on ``agent_runner``
(CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency, token-streaming half).

The checkpoint ``ProgressEvent`` stream (``test_progress_stream.py``) already proves the
ROUTE/tool-call/synthesis milestones stream live. This file proves the ANSWER itself now
streams on the SAME channel, as a sequence of ``stage="text_delta"`` events — never as one
final chunk — and that it interleaves with the surrounding progress events in the exact
order they were produced, because both are emitted through the ONE ``_emit`` choke point
(no second queue, so no interleaving hazard to reason about).

Never assert wall-clock timing (the shared house rule — this host is loaded and timing
assertions flake); assert ORDERING and COUNT of emitted events instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration.agent_runner import ProgressEvent, _stream_agent_run


class _Recorder:
    """An async ``progress_sink`` that records every event in arrival order."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    async def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)


class _FakeStream:
    """A minimal stand-in for pydantic-ai's ``StreamedRunResult``.

    Yields ``deltas`` one at a time from ``stream_text(delta=True)`` — exactly the
    incremental-chunk shape ``_stream_agent_run`` consumes — and exposes the
    ``get_output``/``all_messages``/``usage`` surface a completed stream carries.
    """

    def __init__(self, deltas: list[str], output: str, messages: list[Any] | None = None) -> None:
        self._deltas = deltas
        self._output = output
        self._messages = messages if messages is not None else []

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def stream_text(self, *, delta: bool = False):
        for chunk in self._deltas:
            yield chunk

    async def get_output(self) -> str:
        return self._output

    def all_messages(self, **_kwargs: Any) -> list[Any]:
        return self._messages

    def usage(self) -> Any:
        return None


class _FakeStreamingAgent:
    """A stand-in for the pydantic-ai ``Agent`` returned by ``create_agent``/
    ``create_context_agent``: ``run`` for the non-streaming call, ``run_stream`` for the
    streaming one, both fed by the same ``deltas``/``output`` fixture."""

    def __init__(self, deltas: list[str], output: str) -> None:
        self.deltas = deltas
        self.output = output
        self.run_calls = 0
        self.run_stream_calls = 0

    async def run(self, *_args: Any, **_kwargs: Any) -> Any:
        self.run_calls += 1
        return SimpleNamespace(output=self.output)

    def run_stream(self, *_args: Any, **_kwargs: Any) -> _FakeStream:
        self.run_stream_calls += 1
        return _FakeStream(list(self.deltas), self.output)


# ---------------------------------------------------------------------------
# _stream_agent_run — the primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_agent_run_with_no_sink_uses_plain_run() -> None:
    """No ``progress_sink`` -> the byte-for-byte prior ``agent.run()`` call: no streaming
    machinery touched, Native-by-default with a single code path (no opt-in flag)."""
    agent = _FakeStreamingAgent(["a", "b"], "ab")
    result = await _stream_agent_run(
        agent, "task", run_kwargs={}, progress_sink=None, run_id="run:x"
    )
    assert agent.run_calls == 1
    assert agent.run_stream_calls == 0
    assert result.output == "ab"


@pytest.mark.asyncio
async def test_stream_agent_run_emits_incremental_deltas_not_one_chunk() -> None:
    """A sink is attached -> ``run_stream`` is used and EACH delta is its own event, in
    order — never coalesced into a single final chunk."""
    recorder = _Recorder()
    agent = _FakeStreamingAgent(["The ", "answer ", "is ", "42."], "The answer is 42.")

    result = await _stream_agent_run(
        agent,
        "task",
        run_kwargs={},
        progress_sink=recorder,
        run_id="run:y",
    )

    assert agent.run_stream_calls == 1
    assert agent.run_calls == 0  # the non-streaming call path was never taken
    # Four deltas in -> four events out, not one. This is the ordering+count assertion,
    # never a wall-clock one.
    assert [e.stage for e in recorder.events] == ["text_delta"] * 4
    assert [e.detail for e in recorder.events] == ["The ", "answer ", "is ", "42."]
    assert all(e.run_id == "run:y" for e in recorder.events)
    # Concatenating the deltas in emission order reconstructs the full answer.
    assert "".join(e.detail for e in recorder.events) == "The answer is 42."
    # The normalized result still exposes the ``.output``/``.all_messages()`` contract
    # ``agent.run()``'s return value gives every existing caller.
    assert result.output == "The answer is 42."
    assert result.all_messages() == []


@pytest.mark.asyncio
async def test_stream_agent_run_skips_empty_deltas() -> None:
    """A falsy delta (some providers emit an empty string chunk) is not forwarded as a
    hollow event — every emitted ``text_delta`` carries real content."""
    recorder = _Recorder()
    agent = _FakeStreamingAgent(["hi", "", "there"], "hithere")

    await _stream_agent_run(
        agent, "task", run_kwargs={}, progress_sink=recorder, run_id="run:z"
    )

    assert [e.detail for e in recorder.events] == ["hi", "there"]


# ---------------------------------------------------------------------------
# _execute_single_server — the real entrypoint, only the model boundary faked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_single_server_streams_the_answer_incrementally() -> None:
    """Drives the REAL ``_execute_single_server`` (the seam under test) with only
    ``create_agent`` faked at the model boundary — proves the wiring from the direct
    single-server tool loop through to the progress channel, not just the primitive."""
    from agent_utilities.orchestration.agent_runner import _execute_single_server

    recorder = _Recorder()
    fake_agent = _FakeStreamingAgent(["Repo", "s: ", "3 found"], "Repos: 3 found")

    with patch(
        "agent_utilities.agent.factory.create_agent",
        return_value=(fake_agent, []),
    ):
        result = await _execute_single_server(
            config={
                "mcp_toolsets": [MagicMock()],
                "provider": "openai",
                "agent_model": "test-model",
            },
            task="list my repos",
            max_steps=5,
            agent_meta={},
            agent_name="github-mcp",
            progress_sink=recorder,
            run_id="run:" + "s" * 32,
        )

    assert fake_agent.run_stream_calls == 1
    assert result["results"]["output"] == "Repos: 3 found"
    assert [e.stage for e in recorder.events] == ["text_delta"] * 3
    assert [e.detail for e in recorder.events] == ["Repo", "s: ", "3 found"]


@pytest.mark.asyncio
async def test_execute_single_server_no_sink_is_byte_identical() -> None:
    """The ``progress_sink=None`` default keeps the exact prior non-streaming behaviour."""
    from agent_utilities.orchestration.agent_runner import _execute_single_server

    fake_agent = _FakeStreamingAgent(["x"], "answer text")

    with patch(
        "agent_utilities.agent.factory.create_agent",
        return_value=(fake_agent, []),
    ):
        result = await _execute_single_server(
            config={
                "mcp_toolsets": [MagicMock()],
                "provider": "openai",
                "agent_model": "test-model",
            },
            task="do the thing",
            max_steps=5,
            agent_meta={},
            agent_name="tester",
        )

    assert fake_agent.run_calls == 1
    assert fake_agent.run_stream_calls == 0
    assert result["results"]["output"] == "answer text"


# ---------------------------------------------------------------------------
# End-to-end through run_agent — progress events AND text deltas interleave in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_events_and_text_deltas_interleave_in_run_order() -> None:
    """Drives the REAL ``run_agent`` dispatch through the focused-tools altitude (the
    same branch ``test_progress_stream.py`` exercises), with only the fleet-URL
    resolution and the model boundary faked — everything else, including
    ``_execute_focused_tools`` -> ``_execute_single_server`` -> ``_stream_agent_run``,
    runs for real. Proves the two "channels" the task description called out are in fact
    ONE channel: a ``tool_call`` (pre-model) event, N ``text_delta`` events (the model
    call), and the terminal ``checkpoint``/``synthesis``/``done`` events (post-model) come
    out in EXACTLY that relative order — the first chunk of the run (a ``route`` progress
    event) precedes completion, and no wall-clock timing is asserted anywhere here.
    """
    shape = SimpleNamespace(
        tool_servers=("github-mcp",), resolve_agent=False, direct_complete=False
    )
    fake_agent = _FakeStreamingAgent(["agent-", "utilities is ", "a platform."], "agent-utilities is a platform.")
    fake_engine = MagicMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch(
            "agent_utilities.orchestration.execution_profile.plan_execution_shape",
            return_value=shape,
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner, "_fleet_server_url", return_value="https://github-mcp.test/mcp"
        ),
        patch.object(agent_runner, "_spawn_auth", return_value=None),
        patch(
            "agent_utilities.mcp.toolset_factory.build_http_toolset",
            return_value=MagicMock(),
        ),
        patch(
            "agent_utilities.agent.factory.create_agent",
            return_value=(fake_agent, []),
        ),
        patch.object(agent_runner, "_record_execution_trace"),
        patch.object(agent_runner, "_write_step_credit"),
        patch.object(agent_runner, "_persist_tool_calls"),
    ):
        recorder = _Recorder()
        out = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="does agent-utilities have a KG",
            run_id="run:" + "i" * 32,
            progress_sink=recorder,
        )

    assert out == "agent-utilities is a platform."
    stages = [e.stage for e in recorder.events]

    # The four text deltas appear as a contiguous run, in order, sandwiched between the
    # pre-model routing/tool_call events and the post-model checkpoint/synthesis/done
    # events -- exactly the "events AND deltas, concurrently, in order" the task asked
    # for, achieved by emitting both through the same ``_emit`` choke point rather than
    # two queues that would need separate reconciliation.
    assert stages == [
        "start",
        "route",
        "tool_call",
        "text_delta",
        "text_delta",
        "text_delta",
        "checkpoint",
        "synthesis",
        "done",
    ]
    text_delta_events = [e for e in recorder.events if e.stage == "text_delta"]
    assert [e.detail for e in text_delta_events] == [
        "agent-",
        "utilities is ",
        "a platform.",
    ]
    assert "".join(e.detail for e in text_delta_events) == out

    # The FIRST event of the run (of either kind) precedes completion -- the very first
    # emitted event is the "start" checkpoint, well before the terminal "done".
    assert stages[0] == "start"
    assert stages[-1] == "done"


@pytest.mark.asyncio
async def test_raising_sink_never_fails_a_streaming_run() -> None:
    """The existing sink-isolation guarantee (``test_progress_stream.py``) also holds for
    the new ``text_delta`` events -- a broken sink must not surface as a run failure."""
    from agent_utilities.orchestration.agent_runner import _execute_single_server

    async def _boom(_event: ProgressEvent) -> None:
        raise RuntimeError("sink is broken")

    fake_agent = _FakeStreamingAgent(["hello"], "hello")
    with patch(
        "agent_utilities.agent.factory.create_agent",
        return_value=(fake_agent, []),
    ):
        result = await _execute_single_server(
            config={
                "mcp_toolsets": [MagicMock()],
                "provider": "openai",
                "agent_model": "test-model",
            },
            task="t",
            max_steps=5,
            agent_meta={},
            agent_name="tester",
            progress_sink=_boom,
            run_id="run:" + "r" * 32,
        )
    assert result["results"]["output"] == "hello"
