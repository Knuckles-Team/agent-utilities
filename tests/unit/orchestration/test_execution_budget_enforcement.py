"""Execution-budget enforcement (CONCEPT:AU-ORCH.execution.execution-budget-caps).

Covers the gaps closed alongside structured-output repair:

* :class:`~agent_utilities.models.usage.ExecutionBudget` ships real, finite
  defaults for every dimension (never silently unbounded), plus a new
  ``max_tool_calls`` dimension distinct from the node-transition cap.
* ``graph/_router_impl.py::dispatcher_step`` enforces the tool-call budget.
* ``graph/verification.py::error_recovery_step`` treats EVERY budget
  exhaustion as terminal — not just the node-transition variant — so a token/
  cost/duration cap can no longer be retried through the planner (spending
  more of the very budget that already tripped).
* ``orchestration/engine.py::run_graph`` classifies a budget-exhaustion
  termination explicitly (``outcome: "budget_exceeded"`` + dimension) and
  preserves partial specialist results alongside the error, instead of
  discarding everything but the error text.
* ``UsageLimits.per_request_input_tokens_limit`` (pydantic-ai-slim 2.21.0) is
  adopted at the spawn and single-server-agent sites, so a single oversized
  tool result (the real 212 KB production incident) cannot blow the run in
  one request regardless of tool behaviour.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_graph.step import StepContext

from agent_utilities.graph.routing import dispatcher_step
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.graph.verification import error_recovery_step
from agent_utilities.models.graph import GraphPlan
from agent_utilities.models.sdd import Task
from agent_utilities.models.usage import ExecutionBudget
from agent_utilities.orchestration import AgentOrchestrationEngine as runner
from agent_utilities.orchestration.loop_guards import (
    DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
)


def _make_deps() -> GraphDeps:
    return GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])


# ---------------------------------------------------------------------------
# ExecutionBudget defaults
# ---------------------------------------------------------------------------


def test_execution_budget_defaults_are_finite_not_unbounded():
    budget = ExecutionBudget()
    assert budget.max_node_transitions == 50
    assert budget.max_tool_calls == 200
    assert budget.max_total_tokens == 500_000
    assert budget.max_cost_usd == 10.0
    assert budget.max_duration_seconds == 600.0


def test_execution_budget_allows_explicit_opt_out():
    budget = ExecutionBudget(max_tool_calls=None, max_total_tokens=None)
    assert budget.max_tool_calls is None
    assert budget.max_total_tokens is None


# ---------------------------------------------------------------------------
# dispatcher_step: tool-call budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_step_halts_on_tool_call_budget():
    plan = GraphPlan(steps=[Task(id=f"specialist_{i}") for i in range(50)])
    state = GraphState(query="q", plan=plan)
    state.exploration_notes = "primed"
    state.execution_budget.max_tool_calls = 3
    state.tool_calls = [
        {"tool_name": "t", "args": "{}", "result": "r", "error": ""} for _ in range(4)
    ]

    ctx: StepContext = StepContext(state=state, deps=_make_deps(), inputs=None)
    result = await dispatcher_step(ctx)

    assert result == "error_recovery"
    assert state.error == "Execution budget exceeded: max tool calls."


@pytest.mark.asyncio
async def test_dispatcher_step_under_tool_call_budget_proceeds():
    plan = GraphPlan(steps=[Task(id="specialist_0")])
    state = GraphState(query="q", plan=plan)
    state.exploration_notes = "primed"
    state.execution_budget.max_tool_calls = 10
    state.tool_calls = [{"tool_name": "t", "args": "{}", "result": "r", "error": ""}]

    ctx: StepContext = StepContext(state=state, deps=_make_deps(), inputs=None)
    result = await dispatcher_step(ctx)

    assert result == "parallel_batch_processor"


# ---------------------------------------------------------------------------
# error_recovery_step: budget exhaustion is ALWAYS terminal
# ---------------------------------------------------------------------------


def _error_recovery_ctx(error: str, *, retry_count: int = 0) -> StepContext:
    state = GraphState(query="q")
    state.error = error
    state.retry_count = retry_count
    deps = MagicMock()
    deps.event_queue = None
    return StepContext(state=state, deps=deps, inputs=None)


@pytest.mark.parametrize(
    "error_text",
    [
        "Execution budget exceeded: max node transitions.",
        "Execution budget exceeded: max tool calls.",
        "Execution budget exceeded: max total tokens.",
        "Execution budget exceeded: max cost USD.",
        "Execution budget exceeded: max duration.",
    ],
)
@pytest.mark.asyncio
async def test_error_recovery_step_never_retries_any_budget_exhaustion(error_text):
    """Before this fix only the node-transition variant matched the terminal
    keyword list; a token/cost/duration budget was retried through the planner
    for up to 2 more rounds, each spending more of the resource that already
    tripped -- exactly the anti-pattern a hard budget must prevent."""
    ctx = _error_recovery_ctx(error_text, retry_count=0)

    result = await error_recovery_step(ctx)

    from pydantic_graph import End

    assert isinstance(result, End)
    assert result.data["error"] == error_text
    assert result.data["budget_exceeded"] is True
    # retry_count must NOT have been bumped -- no replan round was spent.
    assert ctx.state.retry_count == 0


@pytest.mark.asyncio
async def test_error_recovery_step_still_retries_a_recoverable_error():
    """Regression guard: a non-budget, non-policy-violation error is still
    retried through the planner while retries remain (unchanged behavior)."""
    ctx = _error_recovery_ctx("some transient tool error", retry_count=0)

    result = await error_recovery_step(ctx)

    assert result == "planner"
    assert ctx.state.retry_count == 1
    assert ctx.state.validation_feedback is not None


# ---------------------------------------------------------------------------
# engine.py: classified budget_exceeded outcome + partial-result preservation
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.run = AsyncMock()
    return graph


@pytest.mark.asyncio
async def test_run_graph_classifies_budget_exceeded_outcome_and_dimension(mock_graph):
    mock_graph.run.return_value = {
        "error": "Execution budget exceeded: max total tokens.",
        "results": {"specialist_0": "partial finding"},
        "budget_exceeded": True,
    }
    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None

    response = await runner().execute_graph(mock_graph, {"deps": deps}, query="hello")

    assert response["status"] == "failed"
    assert response["metadata"]["outcome"] == "budget_exceeded"
    assert response["metadata"]["budget_dimension"] == "total_tokens"
    assert response["metadata"]["degraded"] is True
    # Partial specialist output completed before the cap tripped must survive.
    assert response["results"]["partial_results"] == {"specialist_0": "partial finding"}


@pytest.mark.asyncio
async def test_run_graph_classifies_node_transitions_budget_dimension(mock_graph):
    mock_graph.run.return_value = {
        "error": "Execution budget exceeded: max node transitions.",
        "results": {},
        "budget_exceeded": True,
    }
    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None

    response = await runner().execute_graph(mock_graph, {"deps": deps}, query="hello")

    assert response["metadata"]["outcome"] == "budget_exceeded"
    assert response["metadata"]["budget_dimension"] == "node_transitions"


@pytest.mark.asyncio
async def test_run_graph_non_budget_terminal_error_keeps_generic_outcome(mock_graph):
    mock_graph.run.return_value = {
        "error": "policy violation: destructive action blocked",
        "results": {},
    }
    deps = MagicMock()
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.event_queue = None

    response = await runner().execute_graph(mock_graph, {"deps": deps}, query="hello")

    assert response["status"] == "failed"
    assert response["metadata"]["outcome"] == "graph_terminal_error"
    assert "budget_dimension" not in response["metadata"]


# ---------------------------------------------------------------------------
# per_request_input_tokens_limit -- a single oversized tool result must not
# blow the run in one request, regardless of tool behaviour
# ---------------------------------------------------------------------------


def test_spawn_usage_limits_always_bounds_per_request_input_tokens():
    from agent_utilities.graph.executor import spawn_usage_limits

    # No invoker token budget -- request-bounded only, still gets the
    # per-request input cap.
    ul = spawn_usage_limits(GraphState(query="q"))
    assert ul.per_request_input_tokens_limit == DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT

    # An invoker token budget is set too -- both caps apply together.
    ul2 = spawn_usage_limits(GraphState(query="q", invoker_budget_tokens=9000))
    assert ul2.total_tokens_limit == 9000
    assert ul2.per_request_input_tokens_limit == DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT


@pytest.mark.asyncio
async def test_execute_single_server_always_bounds_per_request_input_tokens():
    """The direct single-server tool loop -- the exact path the real ServiceNow
    212 KB-from-one-call incident ran through -- always sets
    ``per_request_input_tokens_limit``, even with no invoker budget/max_steps
    that would otherwise leave ``usage_limits`` unset entirely."""
    from agent_utilities.orchestration.agent_runner import _execute_single_server

    captured: dict = {}

    class _FakeAgent:
        async def run(self, *_args, **kwargs):
            captured.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(output="ok")

    with patch(
        "agent_utilities.agent.factory.create_agent",
        return_value=(_FakeAgent(), []),
    ):
        await _execute_single_server(
            config={
                "mcp_toolsets": [MagicMock()],
                "provider": "openai",
                "agent_model": "test-model",
            },
            task="do the thing",
            max_steps=0,
            agent_meta={},
            agent_name="tester",
        )

    limits = captured["usage_limits"]
    assert (
        limits.per_request_input_tokens_limit == DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT
    )
