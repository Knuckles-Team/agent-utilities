"""Tests for the iter()-based graph execution path.

Validates that :func:`run_graph_iter` correctly uses ``graph.iter()``
for step-by-step execution and yields properly structured events
including node transitions, sideband draining, and state snapshots.

CONCEPT:AU-002 Graph Orchestration
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.graph.runner import _build_state_snapshot, run_graph_iter
from agent_utilities.graph.state import GraphState


class TestBuildStateSnapshot:
    """Unit tests for the _build_state_snapshot helper."""

    def test_snapshot_captures_key_fields(self):
        state = GraphState(
            query="test",
            routed_domain="git_operations",
            mode="ask",
            topology="basic",
            node_history=["router", "executor"],
            node_transitions=5,
            session_id="sess-123",
        )
        state.results_registry = {"step_1": "ok"}
        snapshot = _build_state_snapshot(state)

        assert snapshot["routed_domain"] == "git_operations"
        assert snapshot["mode"] == "ask"
        assert snapshot["topology"] == "basic"
        assert snapshot["node_history"] == ["router", "executor"]
        assert snapshot["node_transitions"] == 5
        assert snapshot["session_id"] == "sess-123"
        assert "step_1" in snapshot["results_registry_keys"]
        assert snapshot["error"] is None

    def test_snapshot_returns_dict(self):
        state = GraphState(query="test")
        snapshot = _build_state_snapshot(state)
        assert isinstance(snapshot, dict)
        # Should be serializable
        import json

        json.dumps(snapshot)

    def test_snapshot_copies_node_history(self):
        """Snapshot should copy the list, not reference it."""
        state = GraphState(query="test", node_history=["a", "b"])
        snapshot = _build_state_snapshot(state)
        state.node_history.append("c")
        assert snapshot["node_history"] == ["a", "b"]


class _FakeEndMarker:
    """Simulates pydantic_graph.beta.graph.EndMarker for testing."""

    def __init__(self, value):
        self.value = value


class _FakeGraphTask:
    """Simulates a GraphTask for testing."""

    def __init__(self, node_id: str, task_id: str = "task:0"):
        self.node_id = node_id
        self.task_id = task_id


class _FakeGraphRun:
    """Simulates GraphRun for testing iter() behavior."""

    def __init__(self, steps: list):
        self._steps = steps
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._steps):
            raise StopAsyncIteration
        step = self._steps[self._idx]
        self._idx += 1
        return step


class _FakeGraph:
    """Simulates a Graph with iter() support for testing."""

    def __init__(self, steps: list):
        self._steps = steps

    @asynccontextmanager
    async def iter(self, *, state, deps):
        graph_run = _FakeGraphRun(self._steps)
        async with graph_run as run:
            yield run


@pytest.mark.asyncio
async def test_run_graph_iter_yields_node_transitions():
    """Verify that run_graph_iter yields node_transition events for each step."""
    tasks_step1 = [_FakeGraphTask("router", "task:0")]
    tasks_step2 = [_FakeGraphTask("executor", "task:1")]
    end_marker = _FakeEndMarker("final result")

    graph = _FakeGraph([tasks_step1, tasks_step2, end_marker])

    events = []
    with patch(
        "agent_utilities.graph.runner.load_node_agents_registry"
    ) as mock_registry, patch(
        "pydantic_graph.beta.graph.EndMarker", _FakeEndMarker
    ):
        mock_registry.return_value = MagicMock(agents=[])

        async for event in run_graph_iter(
            graph=graph,
            config={},
            query="test query",
            run_id="test-run",
        ):
            events.append(event)

    # Should get: sideband (graph_start), node_transition x2, graph_complete
    transition_events = [e for e in events if e.get("type") == "node_transition"]
    complete_events = [e for e in events if e.get("type") == "graph_complete"]

    assert len(transition_events) == 2
    assert transition_events[0]["step"] == 1
    assert transition_events[0]["active_nodes"][0]["node_id"] == "router"
    assert transition_events[1]["step"] == 2
    assert transition_events[1]["active_nodes"][0]["node_id"] == "executor"
    assert len(complete_events) == 1
    assert complete_events[0]["output"] == "final result"


@pytest.mark.asyncio
async def test_run_graph_iter_state_snapshots():
    """Verify each event includes a state_snapshot."""
    end_marker = _FakeEndMarker(None)
    graph = _FakeGraph([end_marker])

    events = []
    with patch(
        "agent_utilities.graph.runner.load_node_agents_registry"
    ) as mock_registry, patch(
        "pydantic_graph.beta.graph.EndMarker", _FakeEndMarker
    ):
        mock_registry.return_value = MagicMock(agents=[])

        async for event in run_graph_iter(
            graph=graph, config={}, query="test", run_id="s"
        ):
            events.append(event)

    for event in events:
        if event.get("type") in ("node_transition", "graph_complete"):
            assert "state_snapshot" in event
            assert isinstance(event["state_snapshot"], dict)


@pytest.mark.asyncio
async def test_run_graph_iter_handles_error():
    """Verify that exceptions during graph execution yield error events."""

    @asynccontextmanager
    async def failing_iter(*, state, deps):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    graph = MagicMock()
    graph.iter = failing_iter

    events = []
    with patch(
        "agent_utilities.graph.runner.load_node_agents_registry"
    ) as mock_registry:
        mock_registry.return_value = MagicMock(agents=[])

        async for event in run_graph_iter(
            graph=graph, config={}, query="test", run_id="err"
        ):
            events.append(event)

    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
    assert "boom" in error_events[0]["error"]


@pytest.mark.asyncio
async def test_run_graph_iter_drains_sideband():
    """Verify sideband events from event_queue are yielded."""
    end_marker = _FakeEndMarker("done")
    tasks = [_FakeGraphTask("router")]

    # Custom graph that puts events on the queue during execution
    class _SidebandGraph:
        @asynccontextmanager
        async def iter(self, *, state, deps):
            # Simulate a sideband event being emitted during execution
            if deps.event_queue:
                await deps.event_queue.put(
                    {"type": "specialist_started", "domain": "git"}
                )

            class _Run:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if not hasattr(self, "_done"):
                        self._done = True
                        return [_FakeGraphTask("router")]
                    raise StopAsyncIteration

            async with _Run() as run:
                yield run

    graph = _SidebandGraph()

    events = []
    with patch(
        "agent_utilities.graph.runner.load_node_agents_registry"
    ) as mock_registry, patch(
        "pydantic_graph.beta.graph.EndMarker", _FakeEndMarker
    ):
        mock_registry.return_value = MagicMock(agents=[])

        async for event in run_graph_iter(
            graph=graph, config={}, query="test", run_id="sb"
        ):
            events.append(event)

    sideband_events = [e for e in events if e.get("type") == "sideband"]
    assert len(sideband_events) >= 1
    # The first sideband may be from emit_graph_event("graph_start"),
    # our custom event should also appear in the sideband list.
    sideband_types = [
        e["event"].get("type") for e in sideband_events
    ]
    assert "specialist_started" in sideband_types
