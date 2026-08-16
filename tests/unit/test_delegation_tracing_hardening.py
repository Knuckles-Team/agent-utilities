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
2. Delegated agent/workflow execution status reads its REAL
   provenance (``:RunTrace`` + ``:ToolCall``, ORCH-1.21/KG-2.296) lives under a different id
   namespace it never queried, so status reported ``not_found`` for a run that actually
   executed. Fixed: ``graph_jobs`` routes a ``run:``/``trace:``/``wf-``/``session:``-prefixed
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


def _ensure_standard_model_class(monkeypatch) -> None:
    """Configure a 'normal'-tier chat model so run_agent's default
    ``model_class="standard"`` resolves (_configured_model_for_class maps
    "standard" -> intelligence_level "normal"). This test environment carries
    no configured chat models at all, so without this every run_agent() call
    fails before ever reaching the mocked execution path a test is actually
    exercising. Same idiom as ``tests/unit/knowledge_graph/test_mcp_orchestrate.py``.
    """
    from agent_utilities.core.config import ChatModelConfig
    from agent_utilities.core.config import config as agent_config

    monkeypatch.setattr(
        agent_config,
        "chat_models",
        [
            ChatModelConfig(
                id="test-standard-model",
                provider="openai",
                intelligence_level="normal",
            )
        ],
    )


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
async def test_focused_tools_failure_fails_closed_regardless_of_agent_name(monkeypatch):
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

    _ensure_standard_model_class(monkeypatch)
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
        n for n, d in engine.graph.nodes(data=True) if d.get("node_type") == "RunTrace"
    ]
    assert len(trace_nodes) == 1
    assert engine.graph.nodes[trace_nodes[0]]["status"] == "degraded"


@pytest.mark.asyncio
async def test_focused_tools_failure_fails_closed_even_when_agent_name_resolves_as_server(
    monkeypatch,
):
    """Same fail-closed guarantee holds in the case the OLD code already handled
    (agent_name itself resolves as a KG :Server) — a non-regression pin."""
    from agent_utilities.orchestration.agent_runner import run_agent
    from agent_utilities.orchestration.execution_profile import ExecutionProfile

    _ensure_standard_model_class(monkeypatch)
    # _build_execution_config resolves a live toolset for a "server" agent via
    # _fleet_server_url(toolset_id), which reads FLEET_MCP_URL_TEMPLATE straight
    # from the environment (a live os.environ read, not an AgentConfig attribute)
    # rather than the mocked _resolve_agent_from_kg's "url" field -- unset, it
    # falls through to this SANDBOX's real MCP_CONFIG catalog, which is
    # environment-dependent. Same idiom as
    # tests/unit/knowledge_graph/test_orchestrate_mcp.py::test_build_execution_config.
    monkeypatch.setenv("FLEET_MCP_URL_TEMPLATE", "https://{server}.mcp.test")
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
                "url": "https://container-manager-mcp.example/mcp",
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
# 2. graph_jobs(status) surfaces REAL RunTrace + ToolCall provenance.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_action_surfaces_real_run_trace_and_tool_calls(monkeypatch):
    """End-to-end (Wire-First): ``graph_orchestrate`` writes a real
    ``:RunTrace`` + ``:ToolCall`` provenance into the KG; ``graph_jobs(status,
    job_id=<the returned run_id>)`` must surface that REAL data — not ``not_found``,
    and not an empty shell.
    """
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
    from agent_utilities.mcp.tools.job_tools import register_job_tools

    _ensure_standard_model_class(monkeypatch)
    # See test_focused_tools_failure_fails_closed_even_when_agent_name_resolves_as_server
    # for why this is required: "container-manager-mcp" resolves against this
    # sandbox's real MCP_CONFIG catalog otherwise (an environment-dependent URL).
    monkeypatch.setenv("FLEET_MCP_URL_TEMPLATE", "https://{server}.mcp.test")
    engine = _create_engine()
    register_analysis_tools(_FakeMCP())
    register_job_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    fake_tool_calls = [
        {
            "tool_name": "cm_docker_ps",
            "args": "{}",
            "result": "web, db, cache",
        }
    ]

    # "container-manager-mcp" resolves as a KG :Server (``_resolve_agent_from_kg``)
    # even against this bare in-memory engine, and the lexical ontology gate finds
    # no ``shape.tool_servers`` match for this task text, so
    # ``_is_single_server_agent`` is what routes this run -- to
    # ``_execute_single_server`` (route "resolved as a single configured MCP
    # server"), not the toolless multi-agent graph (``_execute_graph``) or the
    # FOCUSED-TOOLS path (``_execute_focused_tools``, only entered when the
    # lexical gate itself names a server). Mock the seam this run actually reaches.
    with patch(
        "agent_utilities.orchestration.agent_runner._execute_single_server",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = {
            "results": {"output": "3 containers running: web, db, cache"},
            "tool_calls": fake_tool_calls,
        }
        exec_result = await kg._execute_tool(
            "graph_orchestrate",
            agent_name="container-manager-mcp",
            task="list running containers",
        )

    payload = json.loads(exec_result)
    run_id = payload["run_id"]
    assert run_id.startswith("run:")
    assert "3 containers running" in payload["output"]

    status_result = await kg._execute_tool("graph_jobs", action="status", job_id=run_id)
    status = json.loads(status_result)

    # RunTrace deliberately never stores agent_name in plaintext -- trace_properties
    # (observability/trace_ontology.py) pseudonymizes it into attribution_ref via
    # persistence_reference("agent", agent_name, namespace="execution-trace"), the
    # SAME contract tests/unit/knowledge_graph/test_orchestrate_mcp.py's
    # test_record_execution_trace pins. There is no literal "agent_name" key.
    from agent_utilities.security.persistence_privacy import persistence_reference

    assert status["status"] == "completed"
    assert status["run_id"] == run_id
    assert status["attribution_ref"] == persistence_reference(
        "agent", "container-manager-mcp", namespace="execution-trace"
    )
    # ``tool_call_properties`` (observability/trace_ontology.py) never persists a
    # tool's raw result text either -- ``result``/``result_preview`` are always
    # written empty (only a keyed ``result_digest``, not projected by
    # ``get_run_trace``'s ToolCall query, proves what ran) -- the same privacy
    # contract ``attribution_ref`` enforces above. So "not an empty shell" is
    # proven by the un-redacted tool_name/status fields, not literal result text.
    assert status["tool_call_count"] == 1
    assert status["tool_calls"][0]["tool_name"] == "cm_docker_ps"
    assert status["tool_calls"][0]["status"] == "ok"
    assert status["tool_calls"][0]["result_preview"] == ""


def test_run_trace_status_surfaces_execution_mode() -> None:
    from unittest.mock import MagicMock

    from agent_utilities.orchestration.manager import Orchestrator

    backend = MagicMock()
    backend.execute.side_effect = [
        [
            {
                "status": "completed",
                "execution_mode": "pydantic_graph",
                "duration_ms": 12.5,
                "graph_evidence_schema_version": "graph-execution-evidence-v1",
                "graph_topology": "basic",
                "graph_topology_digest": "sha256:topology",
                "graph_version_digest": "sha256:version",
                "graph_runtime_version": "2.21.0",
                "graph_node_sequence": ["router", "__end__"],
                "graph_transition_sequence": (
                    '[{"scheduled_tasks":[{"node_id":"router",'
                    '"task_id":"task:router"}],"sequence":1}]'
                ),
                "graph_transition_count": 1,
                "graph_checkpoint_ids": ["ckpt:fixture:1"],
                "graph_resume_supported": False,
            }
        ],
        [],
        [
            {
                "status": "completed",
                "execution_mode": "pydantic_graph",
                "duration_ms": 12.5,
                "graph_transition_sequence": "[]",
            }
        ],
        [],
    ]
    engine = MagicMock()
    engine.backend = backend
    orchestrator = Orchestrator(engine)

    trace = orchestrator.get_run_trace("run:fixture")
    provenance = orchestrator._run_provenance("run:fixture")

    assert trace["execution_mode"] == "pydantic_graph"
    assert trace["graph_topology_digest"] == "sha256:topology"
    assert trace["graph_version_digest"] == "sha256:version"
    assert trace["graph_runtime_version"] == "2.21.0"
    assert trace["graph_node_sequence"] == ["router", "__end__"]
    assert trace["graph_transition_sequence"] == [
        {
            "scheduled_tasks": [{"node_id": "router", "task_id": "task:router"}],
            "sequence": 1,
        }
    ]
    assert trace["graph_checkpoint_ids"] == ["ckpt:fixture:1"]
    assert trace["graph_resume_supported"] is False
    assert provenance["execution_mode"] == "pydantic_graph"
    query = backend.execute.call_args_list[0].args[0]
    for field in (
        "graph_evidence_schema_version",
        "graph_topology",
        "graph_topology_digest",
        "graph_version_digest",
        "graph_runtime_version",
        "graph_node_sequence",
        "graph_transition_sequence",
        "graph_transition_count",
        "graph_checkpoint_ids",
        "graph_resume_supported",
    ):
        assert f"t.{field} AS {field}" in query


@pytest.mark.asyncio
async def test_graph_jobs_status_returns_durable_graph_execution_evidence(
    monkeypatch,
) -> None:
    from unittest.mock import MagicMock

    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.job_tools import register_job_tools

    backend = MagicMock()
    backend.execute.side_effect = [
        [
            {
                "status": "completed",
                "execution_mode": "pydantic_graph",
                "graph_evidence_schema_version": "graph-execution-evidence-v1",
                "graph_topology": "basic",
                "graph_topology_digest": "sha256:topology",
                "graph_version_digest": "sha256:version",
                "graph_runtime_version": "2.21.0",
                "graph_node_sequence": ["router", "dispatcher", "__end__"],
                "graph_transition_sequence": (
                    '[{"scheduled_tasks":[{"node_id":"router",'
                    '"task_id":"task:router"}],"sequence":1}]'
                ),
                "graph_transition_count": 1,
                "graph_checkpoint_ids": ["ckpt:fixture:1"],
                "graph_resume_supported": False,
            }
        ],
        [],
    ]
    engine = MagicMock()
    engine.backend = backend
    register_job_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    raw = await kg._execute_tool(
        "graph_jobs",
        action="status",
        job_id="run:durable-graph-evidence",
    )
    status = json.loads(raw)

    assert status["status"] == "completed"
    assert status["execution_mode"] == "pydantic_graph"
    assert status["graph_evidence_schema_version"] == "graph-execution-evidence-v1"
    assert status["graph_topology"] == "basic"
    assert status["graph_topology_digest"] == "sha256:topology"
    assert status["graph_version_digest"] == "sha256:version"
    assert status["graph_runtime_version"] == "2.21.0"
    assert status["graph_node_sequence"] == ["router", "dispatcher", "__end__"]
    assert status["graph_transition_sequence"] == [
        {
            "scheduled_tasks": [{"node_id": "router", "task_id": "task:router"}],
            "sequence": 1,
        }
    ]
    assert status["graph_transition_count"] == 1
    assert status["graph_checkpoint_ids"] == ["ckpt:fixture:1"]
    assert status["graph_resume_supported"] is False
    assert status["run_id"] == "run:durable-graph-evidence"
    assert status["tool_calls"] == []


@pytest.mark.asyncio
async def test_status_action_serves_dispatched_work_item_lookup(monkeypatch):
    """Status for a dispatch-created id reads its durable task record.

    A freshly dispatched WorkItem with no unresolved dependencies is created
    with status "ready" (never "pending" — see
    ``submit_orchestrator_work_item``, ``work_item.py``:
    ``status = "submitted" if dep_count else "ready"``), matching the
    established WorkItem status contract pinned throughout
    ``tests/unit/orchestration/test_work_item.py`` (e.g. line 482).
    """
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.job_tools import register_job_tools

    engine = _create_engine()
    register_job_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    dispatch_result = await kg._execute_tool(
        "graph_jobs", action="dispatch", task="analyze logs"
    )
    job_id = json.loads(dispatch_result)["job_id"]

    status_result = await kg._execute_tool("graph_jobs", action="status", job_id=job_id)
    assert "ready" in status_result


@pytest.mark.asyncio
async def test_status_not_found_for_unknown_run_id(monkeypatch):
    """A run_id/trace_id that was never recorded must report not_found, not raise
    or silently return an empty-but-"completed" shell."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.job_tools import register_job_tools

    engine = _create_engine()
    register_job_tools(_FakeMCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    status_result = await kg._execute_tool(
        "graph_jobs", action="status", job_id="run:doesnotexist"
    )
    status = json.loads(status_result)
    assert status["status"] == "not_found"
