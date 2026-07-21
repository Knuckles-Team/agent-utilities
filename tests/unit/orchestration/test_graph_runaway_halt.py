"""Real (non-mocked) exercise of the graph node-transition cap.

CONCEPT:AU-AHE.harness.loop-exit-conditions — exit 2 (TURN CAP). The 8-exits
work threaded ``run_agent``'s ``max_steps`` into
``AgentOrchestrationEngine.execute_graph`` -> ``GraphState.max_steps`` ->
``ExecutionBudget.max_node_transitions`` (see ``engine.py::execute_graph``),
but that threading was previously only proven with the arithmetic unit helper
(``graph_node_transition_cap``) in ``test_loop_guards.py`` — never against the
production dispatcher that actually enforces the cap.

This module drives the REAL, unmocked ``dispatcher_step`` (the same function
``agent_runner._execute_graph`` -> ``AgentOrchestrationEngine.execute_graph``
wires into the live multi-agent graph as node "dispatcher") against a
synthetic "runaway" plan far larger than the caller's requested turn budget,
using the exact ``execution_budget.max_node_transitions =
graph_node_transition_cap(max_steps)`` threading ``engine.execute_graph``
performs. It asserts the dispatcher force-terminates well before the plan is
exhausted — i.e. the cap actually HALTS a runaway multi-agent graph, not just
that the arithmetic helper returns the right number.
"""

from __future__ import annotations

import pytest
from pydantic_graph.step import StepContext

from agent_utilities.graph.routing import dispatcher_step
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.models.graph import GraphPlan
from agent_utilities.models.sdd import Task
from agent_utilities.orchestration.loop_guards import graph_node_transition_cap


def _make_deps() -> GraphDeps:
    # Minimal-but-real GraphDeps: no MCP toolsets/knowledge engine, so the
    # dispatcher's optional KG-checkpoint and plan-sync side branches are
    # inert (guarded `if ctx.deps.knowledge_engine` / `if ctx.deps.plan_sync`
    # checks in dispatcher_step), exercising only the transition-cap path.
    return GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])


@pytest.mark.asyncio
async def test_dispatcher_step_halts_runaway_graph_at_threaded_max_steps_cap():
    """A 500-step runaway plan with a tight caller max_steps=3 must be
    force-terminated by the dispatcher within the threaded cap (10 transitions),
    never anywhere close to exhausting the plan."""
    RUNAWAY_STEP_COUNT = 500
    caller_max_steps = 3
    expected_cap = graph_node_transition_cap(caller_max_steps)
    assert expected_cap == 10  # max(3*2, 10) floor

    # Every step gets a UNIQUE id so the doom-loop repeated-pattern detector
    # never trips — the ONLY exit condition that can fire here is the
    # node-transition budget under test.
    plan = GraphPlan(
        steps=[
            Task(id=f"specialist_{i}", parallel=False)
            for i in range(RUNAWAY_STEP_COUNT)
        ]
    )
    state = GraphState(query="drive the runaway graph", plan=plan)

    # This is EXACTLY what AgentOrchestrationEngine.execute_graph does when a
    # caller's max_steps is threaded (engine.py, CONCEPT:AU-AHE.harness.loop-exit-conditions):
    state.max_steps = caller_max_steps
    state.execution_budget.max_node_transitions = expected_cap

    deps = _make_deps()

    # Prime exploration_notes so the dispatcher's one-time memory_selection
    # detour (first-entry context gathering) doesn't consume a transition
    # before we start counting the runaway plan dispatch itself.
    state.exploration_notes = "primed"

    result: str | None = "unset"
    for _ in range(RUNAWAY_STEP_COUNT):
        ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)
        result = await dispatcher_step(ctx)
        if result == "error_recovery":
            break
        # A real specialist node would now run and hand control back to the
        # dispatcher; we don't need its behavior to prove the cap enforcement,
        # only that the dispatcher gets re-invoked for the next plan step.
        assert result == "parallel_batch_processor"

    # The dispatcher actually force-terminated the run ...
    assert result == "error_recovery"
    assert state.error == "Execution budget exceeded: max node transitions."

    # ... at (approximately) the threaded cap, not the ExecutionBudget's
    # untouched 50-transition default and nowhere near the runaway plan size.
    assert state.node_transitions == expected_cap + 1
    assert state.node_transitions < GraphState(query="x").MAX_NODE_TRANSITIONS
    assert state.node_transitions < RUNAWAY_STEP_COUNT

    # And the plan was genuinely cut short — most of the "runaway" steps were
    # never dispatched.
    assert state.step_cursor < RUNAWAY_STEP_COUNT
    assert state.step_cursor <= expected_cap


@pytest.mark.asyncio
async def test_dispatcher_step_without_threaded_cap_uses_untouched_default():
    """Control case: an UNTHREADED run (max_steps never set, the pre-fix
    behavior) keeps the ExecutionBudget's own default of 50 -- proving the
    halt above is actually caused by the threaded cap, not some unrelated
    guard (e.g. the doom-loop detector or the hard MAX_NODE_TRANSITIONS)."""
    plan = GraphPlan(
        steps=[Task(id=f"specialist_{i}", parallel=False) for i in range(500)]
    )
    state = GraphState(query="drive the runaway graph", plan=plan)
    state.exploration_notes = "primed"
    assert state.execution_budget.max_node_transitions == 50  # untouched default

    deps = _make_deps()

    result: str | None = "unset"
    for _ in range(60):
        ctx: StepContext = StepContext(state=state, deps=deps, inputs=None)
        result = await dispatcher_step(ctx)
        if result == "error_recovery":
            break
        assert result == "parallel_batch_processor"

    assert result == "error_recovery"
    assert state.node_transitions == 51  # default cap (50) + 1, not the tight one
