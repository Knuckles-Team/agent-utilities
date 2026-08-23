"""Router-timeout + terminal-failure regression tests (D-RTR-1/2/3/4).

Covers a live production incident: a single chat turn took 264s and returned
no answer at all. Root cause, all in ``graph/_router_impl.py``:

1. The structured planning call is bounded by ``asyncio.wait_for(...,
   timeout=ctx.deps.router_timeout)`` (``router_step``), but the unstructured
   fallback that follows a planning failure was a bare, unbounded
   ``await fallback_agent.run(...)`` against the very same degraded backend
   that just stalled.
2. On total planning failure ``router_step`` used to ``return "__end__"`` — a
   value the graph discards, since ``graph/builder.py`` gives the router a
   SINGLE static outgoing edge to the dispatcher (a second edge would
   broadcast-fork pydantic-graph). Execution always proceeds to
   ``dispatcher_step`` regardless of what the router returns.
3. ``dispatcher_step``'s empty-plan branch (no results, no exploration notes)
   used to ``return None`` with no diagnosable reason. ``None`` is the ONLY
   value this function can return that reaches ``g.end_node`` without a
   crash (proven empirically: ``dispatcher_step``'s return is routed through
   ``graph/builder.py``'s ``dispatcher_route`` Decision node, an exhaustive
   Literal/type match with no ``End``-shaped branch — returning ``End(...)``
   there raises ``RuntimeError: No branch matched inputs End(...) for
   decision node dispatcher_route``), so the fix keeps returning ``None`` but
   now always stamps a concrete, actionable reason onto ``ctx.state.error``
   first.

These tests do not mock the seams they validate: ``asyncio.wait_for`` and the
real ``dispatcher_step``/``router_step`` functions run for real. Only the LLM
agent construction (``create_context_agent``) is replaced, exactly like the
existing ``tests/unit/orchestration/test_execution_budget_enforcement.py`` and
``tests/unit/graph/test_graph_steps.py`` patterns.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic_graph.step import StepContext

from agent_utilities.graph.routing import dispatcher_step, router_step
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.models.graph import GraphPlan


def _make_deps(**overrides: object) -> GraphDeps:
    return GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[], **overrides)


# ---------------------------------------------------------------------------
# D-RTR-1: the unstructured fallback must be bounded by the profile timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_unstructured_fallback_is_bounded_by_router_timeout():
    """A hanging fallback model must be cut off at ``ctx.deps.router_timeout``,
    not run unbounded — the two sequential LLM calls (structured, then
    unstructured) together must never exceed roughly 2x the profile budget."""
    state = GraphState(query="tell me about agent-utilities")
    deps = _make_deps(router_timeout=0.05)

    call_count = 0

    def _agent_factory(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        agent = MagicMock()
        if call_count == 1:
            # Structured planning call: fail immediately so we reach the
            # unstructured fallback path without waiting out its own timeout.
            agent.run_stream = MagicMock(
                side_effect=RuntimeError("router LLM backend unreachable")
            )
        else:
            # Unstructured fallback call: HANGS well past the router budget.
            # If this ever gets awaited without a timeout, the test itself
            # will hang/timeout instead of completing quickly.
            async def _hang(*_a: object, **_kw: object) -> SimpleNamespace:
                await asyncio.sleep(30)
                return SimpleNamespace(output="agent-utilities-expert")

            agent.run = _hang
        return agent

    ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)

    with patch(
        "agent_utilities.graph._router_impl.create_context_agent",
        side_effect=_agent_factory,
    ):
        start = time.monotonic()
        result = await router_step(ctx)
        elapsed = time.monotonic() - start

    # The fallback call is bounded at ~0.05s; give generous headroom for
    # scheduler jitter while still proving it did NOT run for the full 30s.
    assert elapsed < 5.0, f"unstructured fallback ran unbounded: took {elapsed:.2f}s"
    assert call_count == 2, "expected both the structured and fallback agent to run"

    # The router never terminates the graph directly (single outgoing edge to
    # the dispatcher) — it always routes onward.
    assert result == "dispatcher"

    # The timeout must be recorded as the reason, not silently dropped.
    assert state.error is not None
    assert "timed out" in state.error.lower()


# ---------------------------------------------------------------------------
# D-RTR-2/3: a total planning failure must produce a diagnosable terminal
# reason, not a silent None/empty termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_planning_failure_never_returns_dead_end_sentinel():
    """On total planning + fallback failure, ``router_step`` must route onward
    (the only edge that exists) rather than return the dead ``"__end__"``
    sentinel, and it must record why on ``ctx.state.error``."""
    state = GraphState(query="tell me about agent-utilities")
    deps = _make_deps(router_timeout=0.05)

    def _agent_factory(*_args: object, **_kwargs: object) -> MagicMock:
        agent = MagicMock()
        agent.run_stream = MagicMock(side_effect=RuntimeError("planning unreachable"))
        agent.run = MagicMock(side_effect=RuntimeError("fallback unreachable too"))
        return agent

    ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)

    with patch(
        "agent_utilities.graph._router_impl.create_context_agent",
        side_effect=_agent_factory,
    ):
        result = await router_step(ctx)

    assert result != "__end__"
    assert result == "dispatcher"
    assert state.error
    assert "planning failed" in state.error.lower()
    assert "fallback" in state.error.lower()


@pytest.mark.asyncio
async def test_dispatcher_step_empty_plan_termination_carries_a_failure_reason():
    """When the plan completes with no results and no exploration notes,
    ``dispatcher_step`` must still stamp a concrete, non-empty reason onto
    ``ctx.state.error`` before terminating — never a bare, unexplained
    ``None``. (``None`` remains the only value this function can return that
    safely reaches ``g.end_node`` through ``graph/builder.py``'s
    ``dispatcher_route`` Decision node; see the module-level docstring.)"""
    plan = GraphPlan(steps=[])
    state = GraphState(query="tell me about agent-utilities", plan=plan)
    state.step_cursor = 0
    state.results_registry = {}
    state.exploration_notes = ""
    # Simulate having already gone through router_step's planning-failure path.
    state.error = (
        "Planning failed: LLM planning timed out. Fallback also failed: "
        "the fallback attempt itself failed: Unstructured fallback planning "
        "timed out after 0.05s"
    )

    deps = _make_deps(
        # Skip the memory_selection detour so we reach the empty-plan branch
        # on the very first dispatcher entry, matching the production trace
        # (chat/webui-assistant profile: run_discovery=False).
        execution_shape=SimpleNamespace(run_discovery=False)
    )

    ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)

    result = await dispatcher_step(ctx)

    assert result is None
    # The pre-existing router failure reason must survive, not be discarded.
    assert state.error
    assert "timed out" in state.error.lower()


@pytest.mark.asyncio
async def test_dispatcher_step_empty_plan_termination_synthesizes_a_reason_if_absent():
    """Even with no prior ``ctx.state.error`` set, the empty-plan termination
    branch must synthesize a concrete, non-empty reason rather than leaving
    the failure undiagnosable."""
    plan = GraphPlan(steps=[])
    state = GraphState(query="tell me about agent-utilities", plan=plan)
    state.step_cursor = 0
    state.results_registry = {}
    state.exploration_notes = ""
    assert state.error is None or state.error == ""

    deps = _make_deps(execution_shape=SimpleNamespace(run_discovery=False))
    ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)

    result = await dispatcher_step(ctx)

    assert result is None
    assert state.error
    assert len(state.error) > 0
