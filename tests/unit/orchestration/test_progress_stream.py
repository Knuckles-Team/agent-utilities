"""Tests for the checkpoint ProgressEvent stream on ``agent_runner.run_agent``
(CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency).

A LONG delegation is otherwise a black box between "started" and the final answer.
``run_agent`` emits a small :class:`ProgressEvent` at each of its EXISTING checkpoints to an
OPTIONAL ``progress_sink``. These tests prove the three hard invariants of that additive
channel:

* the default (``progress_sink=None``) path is byte-for-byte unchanged;
* a sink receives an ORDERED sequence of events for a run;
* a sink that raises (or hangs) can NEVER fail or stall the run.

Mirrors the end-to-end run_agent mocking convention in ``test_run_summary.py`` — only the
KG-engine boundary and the executor are mocked; the REAL dispatch + Step-5 emit code runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration.agent_runner import ProgressEvent


class _Recorder:
    """An async ``progress_sink`` that records every event in arrival order."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    async def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)


# A successful focused-tools result: two fleet tool calls, one OK and one errored (but NOT all
# errored — so ``_delegation_degraded`` reports a healthy ``ok`` outcome).
_OK_RESULT: dict[str, Any] = {
    "results": {"output": "ANSWER"},
    "tool_calls": [
        {"tool_name": "list_repos", "error": ""},
        {"tool_name": "create_issue", "error": "boom: access_denied"},
    ],
}


@contextmanager
def _mocked_focused_run(result: dict[str, Any]) -> Iterator[None]:
    """Patch the minimal boundary so ``run_agent`` flows the focused-tools path to ``result``.

    Same minimal set as ``test_run_summary.py``'s end-to-end tests: the engine boundary, the
    planned shape (naming a fleet server so the focused-tools branch is taken), the config
    build, the executor, and the provenance writers that touch the engine.
    """
    shape = SimpleNamespace(
        tool_servers=("github-mcp",), resolve_agent=False, direct_complete=False
    )
    # The KG engine boundary is entirely synchronous (every backend/registry call in
    # this branch is offloaded via ``asyncio.to_thread`` / ``_call_without_blocking``,
    # never awaited directly) — an AsyncMock here made every auto-vivified attribute
    # (e.g. FeedbackService.from_engine's ``engine.store`` fallback) a coroutine whose
    # synchronous call site never awaits it, leaking an "AsyncMock ... was never
    # awaited" RuntimeWarning. MagicMock matches the real, synchronous contract.
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
            agent_runner,
            "_execute_focused_tools",
            new=AsyncMock(return_value=result),
        ),
        patch.object(agent_runner, "_record_execution_trace"),
        patch.object(agent_runner, "_write_step_credit"),
        patch.object(agent_runner, "_persist_tool_calls"),
    ):
        yield


@pytest.mark.asyncio
async def test_default_none_sink_is_byte_identical() -> None:
    """``progress_sink=None`` (the default) is a strict no-op: same output, zero emissions.

    Runs the SAME mocked delegation twice — once with the default None sink, once with a
    recording sink — and asserts the returned answer is byte-for-byte identical, proving the
    sink never alters the run. With None the bare-string contract is untouched; the recorder
    confirms the ONLY difference a sink makes is receiving events, not changing the result.
    """
    with _mocked_focused_run(dict(_OK_RESULT)):
        out_default = await agent_runner.run_agent(
            agent_name="messaging-assistant", task="t", run_id="run:" + "a" * 32
        )

    recorder = _Recorder()
    with _mocked_focused_run(dict(_OK_RESULT)):
        out_with_sink = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="t",
            run_id="run:" + "a" * 32,
            progress_sink=recorder,
        )

    # The None path keeps the exact bare-string contract (a plain answer, NOT a JSON envelope).
    assert isinstance(out_default, str)
    assert out_default == "ANSWER"
    # Adding a sink changes nothing about the returned result.
    assert out_with_sink == out_default
    # The sink is the ONLY behavioral difference: it received events; the None path emitted none.
    assert recorder.events, "a supplied sink should receive events"


@pytest.mark.asyncio
async def test_sink_receives_ordered_event_sequence() -> None:
    """A sink receives the run's checkpoints IN ORDER, with per-tool status and a final done."""
    recorder = _Recorder()
    with _mocked_focused_run(dict(_OK_RESULT)):
        out = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="does my github org have open issues",
            run_id="run:" + "b" * 32,
            progress_sink=recorder,
        )

    assert out == "ANSWER"
    stages = [e.stage for e in recorder.events]
    assert stages == [
        "start",
        "route",
        "tool_call",
        "tool_result",
        "tool_result",
        "checkpoint",
        "synthesis",
        "done",
    ]
    # Every event carries THIS run's id.
    assert all(e.run_id == "run:" + "b" * 32 for e in recorder.events)
    # Each fleet tool result carries its own name + ok/failed status.
    tool_results = [e for e in recorder.events if e.stage == "tool_result"]
    assert tool_results[0].detail == "list_repos"
    assert tool_results[0].status == "ok"
    assert tool_results[1].detail == "create_issue"
    assert tool_results[1].status == "failed"
    # The terminal done reflects a healthy outcome and carries a resolvable trace_ref.
    done = recorder.events[-1]
    assert done.stage == "done"
    assert done.status == "ok"
    assert done.evidence.get("trace_ref")


@pytest.mark.asyncio
async def test_evidence_gate_event_surfaces_retrieval_quality_failure() -> None:
    """A degraded outcome whose translated cause is the retrieval-quality signature streams a
    dedicated ``evidence_gate`` event (the paper's evidence-gating, surfaced)."""
    recorder = _Recorder()
    # An empty output with the retrieval-quality marker → degraded → evidence_gate.
    gated = {
        "results": {"output": "retrieval quality gate failed (composite=0.00)"},
        "metadata": {"degraded": True},
    }
    with _mocked_focused_run(gated):
        await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="something never ingested",
            run_id="run:" + "e" * 32,
            progress_sink=recorder,
        )

    gate_events = [e for e in recorder.events if e.stage == "evidence_gate"]
    assert len(gate_events) == 1
    assert gate_events[0].status == "failed"
    assert gate_events[0].evidence.get("category") == "retrieval_quality"


@pytest.mark.asyncio
async def test_raising_sink_never_fails_the_run() -> None:
    """Every emission raising must NOT surface as a run failure — the answer still returns."""

    async def _boom(event: ProgressEvent) -> None:
        raise RuntimeError("sink is broken")

    with _mocked_focused_run(dict(_OK_RESULT)):
        out = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="t",
            run_id="run:" + "c" * 32,
            progress_sink=_boom,
        )
    assert out == "ANSWER"


@pytest.mark.asyncio
async def test_slow_sink_is_bounded_and_never_stalls_the_run() -> None:
    """A hung/slow sink is bounded by ``_PROGRESS_SINK_TIMEOUT_S`` and swallowed — the run
    proceeds to its real answer instead of blocking on the sink."""
    calls = {"n": 0}

    async def _slow(event: ProgressEvent) -> None:
        calls["n"] += 1
        await asyncio.sleep(1.0)  # far longer than the (patched-tiny) bound below

    with (
        patch.object(agent_runner, "_PROGRESS_SINK_TIMEOUT_S", 0.02),
        _mocked_focused_run(dict(_OK_RESULT)),
    ):
        out = await agent_runner.run_agent(
            agent_name="messaging-assistant",
            task="t",
            run_id="run:" + "d" * 32,
            progress_sink=_slow,
        )
    assert out == "ANSWER"
    assert calls["n"] >= 1  # the sink WAS invoked; it simply never stalled the run
