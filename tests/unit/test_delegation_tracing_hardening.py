"""Delegation + execution-tracing hardening.

CONCEPT:AU-ORCH.execution.focused-tools-fail-closed / AU-ORCH.execution.run-trace-status-tool —
re-confirms the delegation-tracing audit's root causes against the LIVE code and pins the two
that were still real:

1. A server-name delegation (the ontology lexical gate named concrete fleet server(s) via
   ``shape.tool_servers``) whose real tools could not be reached previously fell through to
   the toolless multi-agent graph WHENEVER the top-level ``agent_name`` itself did not resolve
   as a KG ``:Server`` — the common case, since ``agent_name`` is frequently a generic or
   passthrough identity while the actual delegation target is ``shape.tool_servers``. That
   toolless graph can fabricate a plausible answer, recorded as ``status="completed"`` — the
   exact failure this program exists to catch. Fixed: the focused-tools branch now ALWAYS
   fails closed on execution failure (fail-loud, same discipline as the WorkItem
   missing-executor "unroutable" rule), never falls through to the graph.
2. ``graph_orchestrate(action="status")`` only ever read ``:Task`` nodes written by
   ``dispatch_task`` — a delegated ``execute_agent``/``execute_workflow`` run's REAL
   provenance (``:RunTrace`` + ``:ToolCall``, ORCH-1.21/KG-2.296) lives under a different id
   namespace it never queried, so ``status`` reported ``not_found`` for a run that actually
   executed. Fixed: ``status`` now routes a ``run:``/``trace:``/``wf-``/``session:``-prefixed
   ``job_id`` to the real ``RunTrace``/``ToolCall`` data via ``Orchestrator.get_run_trace`` /
   ``get_session_runs``.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _create_engine():
    """A real (in-memory) IntelligenceGraphEngine — the same fixture pattern used by
    ``tests/test_mcp_orchestrate.py`` / ``tests/test_orchestrate_mcp.py`` — so these tests
    exercise the REAL RunTrace/ToolCall write + read paths, not a mock of them."""
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    GraphComputeEngine(backend_type="rust")
    return IntelligenceGraphEngine(db_path=":memory:")


class _FakeMCP:
    """Captures the tool coroutines ``register_analysis_tools`` registers — the same minimal
    FastMCP double used in ``tests/unit/test_assurance_gate_surfaces.py`` — so we exercise the
    REAL registered ``graph_orchestrate`` coroutine (via ``kg_server._execute_tool``) without
    booting the whole MCP server."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *, name: str, description: str = "", tags=None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# 1. Focused-tools (server-name delegation) fails closed, never a silent
#    fallthrough to the toolless graph.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_focused_tools_failure_fails_closed_regardless_of_agent_name():
    """A server-name delegation whose real tools cannot be reached must FAIL —
    never silently fall through to the toolless multi-agent graph — even when the
    top-level ``agent_name`` itself is a generic/unresolved identity (the common
    case: the lexical gate names fleet servers from the TASK text, independent of
    ``agent_name`` resolution). This is the live regression test for the bug: the
    old fail-closed gate checked ``agent_meta.get("type") == "server"`` (the WRONG
    variable — that reflects ``agent_name``'s own KG resolution, not the targeted
    ``shape.tool_servers``), so this exact scenario used to fall through to
    ``_execute_graph`` silently.
    """
    from agent_utilities.orchestration.agent_runner import run_agent
    from agent_utilities.orchestration.execution_profile import ExecutionProfile

    engine = _create_engine()

    # A FOCUSED-TOOLS shape naming a concrete (unreachable/never-registered) fleet
    # server — independent of agent_name, exactly as `_lexical_capability_servers`
    # produces it in `plan_execution_shape`.
    fake_shape = ExecutionProfile(
        name="task",
        router_timeout=None,
        verifier_timeout=None,
        tool_servers=("nonexistent-fleet-server",),
    )

    with (
        patch(
            "agent_utilities.orchestration.execution_profile.plan_execution_shape",
            return_value=fake_shape,
        ),
        patch(
            "agent_utilities.orchestration.agent_runner._execute_focused_tools",
            new_callable=AsyncMock,
        ) as mock_focused,
        patch(
            "agent_utilities.orchestration.agent_runner._execute_graph",
            new_callable=AsyncMock,
        ) as mock_graph,
    ):
        mock_focused.side_effect = RuntimeError(
            "connection refused: server unreachable"
        )

        # agent_name is a generic/unresolved identity (empty KG => type stays
        # "unknown") — NOT itself a resolved KG :Server. This is the case the old
        # gate mishandled.
        result = await run_agent(
            agent_name="totally-unregistered-generic-name",
            task="do a thing on nonexistent-fleet-server",
            engine=engine,
        )

    # The toolless graph must NEVER run for a named-server delegation whose real
    # tools could not be reached — that is the confident-fabrication failure this
    # program exists to catch.
    mock_graph.assert_not_called()
    assert "could not produce a tool-grounded result" in result
    assert "connection refused" in result

    # RunTrace must be truthfully "degraded" (fed back as a negative outcome),
    # never a rubber-stamped "completed".
    trace_nodes = [
        n for n, d in engine.graph.nodes(data=True) if d.get("type") == "RunTrace"
    ]
    assert len(trace_nodes) == 1
    assert engine.graph.nodes[trace_nodes[0]]["status"] == "degraded"


@pytest.mark.asyncio
async def test_focused_tools_failure_fails_closed_even_when_agent_name_resolves_as_server():
    """Same fail-closed guarantee holds in the case the OLD code already handled
    (agent_name itself resolves as a KG :Server) — a non-regression pin."""
    from agent_utilities.orchestration.agent_runner import run_agent
    from agent_utilities.orchestration.execution_profile import ExecutionProfile

    engine = _create_engine()
    fake_shape = ExecutionProfile(
        name="task",
        router_timeout=None,
        verifier_timeout=None,
        tool_servers=("container-manager-mcp",),
    )

    with (
        patch(
            "agent_utilities.orchestration.execution_profile.plan_execution_shape",
            return_value=fake_shape,
        ),
        patch(
            "agent_utilities.orchestration.agent_runner._resolve_agent_from_kg",
            return_value={
                "type": "server",
                "server_id": "srv:container-manager-mcp",
                "tools": [],
                "capabilities": [],
                "mcp_command": "",
                "url": "https://container-manager-mcp.arpa/mcp",
                "system_prompt": "",
            },
        ),
        patch(
            "agent_utilities.orchestration.agent_runner._execute_focused_tools",
            new_callable=AsyncMock,
        ) as mock_focused,
        patch(
            "agent_utilities.orchestration.agent_runner._execute_graph",
            new_callable=AsyncMock,
        ) as mock_graph,
    ):
        mock_focused.side_effect = RuntimeError("timed out")

        result = await run_agent(
            agent_name="container-manager-mcp",
            task="list running containers",
            engine=engine,
        )

    mock_graph.assert_not_called()
    assert "could not produce a tool-grounded result" in result


# ---------------------------------------------------------------------------
# 2. graph_orchestrate(status) surfaces REAL RunTrace + ToolCall provenance.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_action_surfaces_real_run_trace_and_tool_calls(monkeypatch):
    """End-to-end (Wire-First): ``graph_orchestrate(execute_agent)`` writes a real
    ``:RunTrace`` + ``:ToolCall`` provenance into the KG; ``graph_orchestrate(status,
    job_id=<the returned run_id>)`` must surface that REAL data — not ``not_found``,
    and not an empty shell.
    """
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools

    engine = _create_engine()
    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    fake_tool_calls = [
        {
            "tool_name": "cm_docker_ps",
            "args": "{}",
            "result": "web, db, cache",
        }
    ]

    with patch(
        "agent_utilities.orchestration.agent_runner._execute_graph",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = {
            "results": {"output": "3 containers running: web, db, cache"},
            "tool_calls": fake_tool_calls,
        }
        exec_result = await kg._execute_tool(
            "graph_orchestrate",
            action="execute_agent",
            agent_name="container-manager-mcp",
            task="list running containers",
        )

    payload = json.loads(exec_result)
    run_id = payload["run_id"]
    assert run_id.startswith("run:")
    assert "3 containers running" in payload["output"]

    status_result = await kg._execute_tool(
        "graph_orchestrate", action="status", job_id=run_id
    )
    status = json.loads(status_result)

    assert status["status"] == "completed"
    assert status["run_id"] == run_id
    assert status["agent_name"] == "container-manager-mcp"
    assert status["tool_call_count"] == 1
    assert status["tool_calls"][0]["tool_name"] == "cm_docker_ps"
    assert "web, db, cache" in status["tool_calls"][0]["result_preview"]


@pytest.mark.asyncio
async def test_status_action_still_serves_legacy_dispatch_task_lookup(monkeypatch):
    """Non-regression: ``status`` for a plain ``dispatch``-created job id (the
    ``orch-<hex>`` namespace) still goes through the original ``:Task`` lookup."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools

    engine = _create_engine()
    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    dispatch_result = await kg._execute_tool(
        "graph_orchestrate", action="dispatch", task="analyze logs"
    )
    assert "Job ID:" in dispatch_result
    job_id = dispatch_result.rsplit(" ", 1)[-1]

    status_result = await kg._execute_tool(
        "graph_orchestrate", action="status", job_id=job_id
    )
    assert "pending" in status_result


@pytest.mark.asyncio
async def test_status_not_found_for_unknown_run_id(monkeypatch):
    """A run_id/trace_id that was never recorded must report not_found, not raise
    or silently return an empty-but-"completed" shell."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools

    engine = _create_engine()
    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    status_result = await kg._execute_tool(
        "graph_orchestrate", action="status", job_id="run:doesnotexist"
    )
    status = json.loads(status_result)
    assert status["status"] == "not_found"
