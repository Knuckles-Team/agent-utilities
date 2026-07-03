"""Execution-seam robustness for the agent/workflow stack.

Three live-path defects observed through the ``graph_orchestrate`` MCP tool:

* **Symptom 1** — ``execute_workflow`` hung to the caller's 300s timeout because a
  spawned agent awaiting an unresponsive MCP tool advances zero ``max_steps`` yet
  blocks the fan-out ``asyncio.gather`` forever. Fixed by a per-spawn wall-clock
  budget (CONCEPT:ORCH-1.24).
* **Symptom 2** — a remote MCP child failure surfaced as the opaque
  "unhandled errors in a TaskGroup (1 sub-exception)" because anyio wraps it in a
  ``BaseExceptionGroup``. Fixed by flattening the group to the real leaf messages
  (CONCEPT:ORCH-1.21).
* **Symptom 3** — bound ``allowed_tools`` must reach the agent as a real callable
  toolset; a ``.filtered()`` failure or an empty toolset must fail loudly rather
  than yielding a zero-tool agent that hallucinates (CONCEPT:ORCH-1.39 / ORCH-1.21).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from agent_utilities.orchestration.agent_runner import (
    _execute_single_server,
    _flatten_exception_group,
)
from agent_utilities.orchestration.engine import (
    AGENT_WALLCLOCK_TIMEOUT_S,
    AgentOrchestrationEngine,
    _is_agent_error,
)


# --------------------------------------------------------------------------- #
# Symptom 1 — wall-clock timeout on spawned agents
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_workflow_spawned_agent_timeout():
    """A spawned agent that hangs must resolve to a structured timeout, not block.

    The whole workflow must return promptly even though every fan-out child would
    otherwise await forever — proving the gather no longer hangs to the outer
    300s budget.
    """
    engine = AgentOrchestrationEngine()

    async def _hang(*_a, **_k):
        await asyncio.Event().wait()  # never completes

    with patch(
        "agent_utilities.orchestration.agent_runner.run_agent",
        new=_hang,
    ):
        # Tiny budget so the test is fast; the fallback (no completion_state) path
        # exercises a single bounded spawn.
        result = await asyncio.wait_for(
            engine.execute_workflow(
                workflow_id="wf_timeout",
                task="do a thing",
                agent_timeout_s=0.2,
            ),
            timeout=5.0,
        )

    assert result["status"] == "executed"
    payload = json.loads(result["output"])
    assert "timed out" in payload["error"]
    assert payload["agent"] == "dynamic_worker"


@pytest.mark.asyncio
async def test_run_agent_bounded_timeout_is_excluded_from_valid_outputs():
    """A timed-out fan-out child encodes a JSON error and is NOT treated as output."""
    engine = AgentOrchestrationEngine()

    async def _hang(*_a, **_k):
        await asyncio.Event().wait()

    with patch(
        "agent_utilities.orchestration.agent_runner.run_agent",
        new=_hang,
    ):
        out = await engine._run_agent_bounded(agent_name="w0", task="t", timeout_s=0.1)

    assert _is_agent_error(out) is True
    assert json.loads(out)["agent"] == "w0"


def test_agent_wallclock_default_is_a_module_constant():
    """Configuration discipline: the budget is a module constant, not an env flag."""
    assert isinstance(AGENT_WALLCLOCK_TIMEOUT_S, float)
    assert AGENT_WALLCLOCK_TIMEOUT_S > 0


def test_is_agent_error_distinguishes_real_output():
    assert _is_agent_error('{"error": "boom", "agent": "x"}') is True
    assert _is_agent_error("the answer is 42") is False
    assert _is_agent_error('{"output": "fine"}') is False
    assert _is_agent_error("{not json") is False


# --------------------------------------------------------------------------- #
# Symptom 2 — BaseExceptionGroup unwrap
# --------------------------------------------------------------------------- #


def test_flatten_exceptiongroup_unwrap_surfaces_real_messages():
    """A nested TaskGroup error flattens to its actionable leaf messages."""
    eg = BaseExceptionGroup(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        [
            BaseExceptionGroup(
                "inner",
                [ConnectionError("remote child 'portainer-agent' refused connection")],
            )
        ],
    )
    flat = _flatten_exception_group(eg)
    assert "unhandled errors in a TaskGroup" not in flat
    assert "ConnectionError" in flat
    assert "portainer-agent" in flat


def test_flatten_exceptiongroup_unwrap_plain_exception():
    """A non-group exception is rendered unchanged (as <Type>: <msg>)."""
    flat = _flatten_exception_group(ValueError("bad input"))
    assert flat == "ValueError: bad input"


def test_flatten_exceptiongroup_unwrap_dedupes_leaves():
    eg = BaseExceptionGroup(
        "grp",
        [ConnectionError("same"), ConnectionError("same"), ValueError("other")],
    )
    flat = _flatten_exception_group(eg)
    assert flat.count("ConnectionError: same") == 1
    assert "ValueError: other" in flat


@pytest.mark.asyncio
async def test_run_agent_exceptiongroup_unwrap_end_to_end():
    """run_agent surfaces the real child error, not the opaque TaskGroup message."""
    from agent_utilities.orchestration import agent_runner

    eg = BaseExceptionGroup(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        [RuntimeError("remote child 'gitlab-mcp' streamable-http 502")],
    )

    fake_engine = AsyncMock()
    fake_engine.backend = None

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner,
            "_build_execution_config",
            return_value={"mcp_toolsets": []},
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(side_effect=eg),
        ),
        patch.object(agent_runner, "_record_execution_trace"),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        out = await agent_runner.run_agent(agent_name="some-agent", task="t")

    assert out.startswith("Agent execution failed:")
    assert "unhandled errors in a TaskGroup" not in out
    assert "gitlab-mcp" in out
    assert "502" in out


# --------------------------------------------------------------------------- #
# Symptom 3 — allowed_tools bound as a real callable toolset
# --------------------------------------------------------------------------- #


class _FakeToolset:
    """Minimal toolset stub exposing ``.filtered`` like a pydantic-ai toolset."""

    def __init__(self, name: str):
        self.name = name
        self.applied_filter = None

    def filtered(self, filter_func):
        # Return a real (distinct) filtered toolset, recording that filtering ran.
        new = _FakeToolset(self.name)
        new.applied_filter = filter_func
        return new


@pytest.mark.asyncio
async def test_allowed_tools_bound_as_real_callable_toolset():
    """The filtered toolset reaches create_agent — not just the system prompt."""
    from agent_utilities.orchestration import agent_runner

    ts = _FakeToolset("portainer")
    captured: dict = {}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return (AsyncMock(), [])

    fake_agent_run = AsyncMock()
    fake_agent_run.run.return_value.output = "ok"

    with patch("agent_utilities.agent.factory.create_agent") as mock_ca:
        mock_ca.side_effect = lambda **kw: captured.update(kw) or (fake_agent_run, [])
        await _execute_single_server(
            config={
                "mcp_toolsets": [ts],
                "invoker_allowed_tools": ["list_stacks"],
                "provider": "openai",
                "agent_model": "openai:gpt-4o-mini",
            },
            task="list the stacks",
            max_steps=5,
            agent_meta={},
            agent_name="portainer",
        )

    # A real, filtered toolset was passed through as a callable toolset.
    bound = captured["mcp_toolsets"]
    assert len(bound) == 1
    assert isinstance(bound[0], _FakeToolset)
    assert bound[0].applied_filter is not None  # filtering actually applied


@pytest.mark.asyncio
async def test_allowed_tools_bound_filter_failure_is_loud_not_silent():
    """A toolset without .filtered fails loudly instead of dropping all tools."""

    class _NoFilter:
        name = "broken"

    with pytest.raises(RuntimeError, match="does not support tool filtering"):
        await _execute_single_server(
            config={
                "mcp_toolsets": [_NoFilter()],
                "invoker_allowed_tools": ["x"],
            },
            task="t",
            max_steps=5,
            agent_meta={},
            agent_name="broken-agent",
        )


@pytest.mark.asyncio
async def test_allowed_tools_bound_empty_toolset_surfaces_clear_error():
    """A single-server agent with no toolset must error, not hallucinate."""
    with pytest.raises(RuntimeError, match="no bound\n?\\s*toolset|no bound toolset"):
        await _execute_single_server(
            config={"mcp_toolsets": []},
            task="t",
            max_steps=5,
            agent_meta={},
            agent_name="empty-agent",
        )


# --------------------------------------------------------------------------- #
# Residual (a) — a wall-clock cancellation yields a clean "timed out", not
# "Agent execution failed: CancelledError"
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_agent_reraises_cancellation_not_flattened():
    """run_agent must re-raise CancelledError so an outer wait_for can time out
    cleanly — never flatten it into an 'Agent execution failed' string."""
    import asyncio as _asyncio

    from agent_utilities.orchestration import agent_runner

    fake_engine = AsyncMock()
    fake_engine.backend = None
    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner, "_build_execution_config", return_value={"mcp_toolsets": []}
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(side_effect=_asyncio.CancelledError()),
        ),
        patch.object(agent_runner, "_record_execution_trace"),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        with pytest.raises(_asyncio.CancelledError):
            await agent_runner.run_agent(agent_name="some-agent", task="t")


@pytest.mark.asyncio
async def test_run_agent_bounded_timeout_message_is_clean():
    """End-to-end: a run_agent that swallows nothing → _run_agent_bounded returns the
    clean 'timed out' JSON (not a CancelledError string)."""
    import asyncio as _asyncio
    import json as _json

    from agent_utilities.orchestration.engine import AgentOrchestrationEngine

    async def _hang(*_a, **_k):
        await _asyncio.sleep(10)

    engine = AgentOrchestrationEngine.__new__(AgentOrchestrationEngine)
    engine.engine = None
    with patch("agent_utilities.orchestration.agent_runner.run_agent", new=_hang):
        out = await engine._run_agent_bounded("a", "t", timeout_s=0.05)
    payload = _json.loads(out)
    assert "timed out" in payload["error"] and "CancelledError" not in out


# --------------------------------------------------------------------------- #
# Residual (b) — graph-path tool scoping fails loud instead of silently
# passing a toolset through unfiltered / leaving a tool-less agent
# --------------------------------------------------------------------------- #


def test_apply_tool_scope_filters_real_toolset():
    from types import SimpleNamespace

    from agent_utilities.graph.executor import apply_tool_scope

    ts = _FakeToolset("portainer")
    state = SimpleNamespace(invoker_allowed_tools=["list_stacks"])
    _tools, toolsets = apply_tool_scope(state, [], [ts])
    assert len(toolsets) == 1 and toolsets[0].applied_filter is not None


def test_apply_tool_scope_no_allowlist_is_passthrough():
    from types import SimpleNamespace

    from agent_utilities.graph.executor import apply_tool_scope

    ts = _FakeToolset("x")
    state = SimpleNamespace(invoker_allowed_tools=None)
    tools, toolsets = apply_tool_scope(state, ["t"], [ts])
    assert tools == ["t"] and toolsets == [ts]


def test_apply_tool_scope_unfilterable_toolset_is_loud():
    from types import SimpleNamespace

    from agent_utilities.graph.executor import apply_tool_scope

    class _NoFilter:
        name = "broken"

    state = SimpleNamespace(invoker_allowed_tools=["x"])
    with pytest.raises(RuntimeError, match="does not support tool filtering"):
        apply_tool_scope(state, [], [_NoFilter()])


def test_apply_tool_scope_empty_result_is_loud():
    from types import SimpleNamespace

    from agent_utilities.graph.executor import apply_tool_scope

    def some_tool(): ...

    state = SimpleNamespace(invoker_allowed_tools=["nonexistent"])
    with pytest.raises(RuntimeError, match="eliminated every bound tool"):
        apply_tool_scope(state, [some_tool], [])
