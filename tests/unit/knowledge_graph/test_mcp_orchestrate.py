"""Tests for the graph_orchestrate MCP tool and Agent Runner.

CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — KG-to-LLM Execution Bridge tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _create_engine():
    """Create a minimal IntelligenceGraphEngine for testing.

    ``IntelligenceGraphEngine(db_path=":memory:")`` builds its own backend via
    ``create_backend()``, which constructs a bare ``EpistemicGraphBackend()``.
    That backend resolves its OWN routing graph via ``resolve_routing_graph(None)``
    *before* asking ``GraphComputeEngine`` for one -- so under the (autouse,
    test-suite-wide) ``isolate_graph_compute_engine`` fixture it lands on the
    ambient tenant's graph rather than this test's isolated graph. Binding the
    backend directly to the already-isolated ``compute`` object sidesteps that
    divergent second resolution entirely.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    compute = GraphComputeEngine(backend_type="rust")
    backend = object.__new__(EpistemicGraphBackend)
    backend._graph = compute
    backend.graph_name = compute.graph_name
    engine = IntelligenceGraphEngine(backend=backend)
    return engine


@pytest.mark.asyncio
async def test_agent_runner_resolution():
    """Test that agent_runner resolves agent capabilities from KG."""
    from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg

    engine = _create_engine()

    from unittest.mock import MagicMock

    mock_backend = MagicMock()
    engine.backend = mock_backend

    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )

    instructions = "Inspect the requested evidence and report grounded findings."

    # Mock the backend execution specifically for the CallableResource query.
    def mock_execute(query, params=None):
        if (
            "CallableResource" in query
            and params
            and params.get("name") == "test-agent"
        ):
            return [
                {
                    "rid": "skill:test-agent",
                    "rtype": "AGENT_SKILL",
                    "description": "A test skill agent",
                    "system_prompt": instructions,
                    "instruction_digest": runnable_skill_digest(instructions),
                    "source_ref": "skill://test-agent",
                    "provider_ref": "provider://xdg-local",
                }
            ]
        return []

    engine.backend.execute.side_effect = mock_execute

    meta = _resolve_agent_from_kg(engine, "test-agent")

    assert meta["type"] == "skill"


@pytest.mark.asyncio
async def test_agent_runner_binds_provider_skill_to_authenticated_server():
    """A provider skill drives its owning fleet server without retaining a path."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )
    from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg

    engine = _create_engine()

    from unittest.mock import MagicMock

    mock_backend = MagicMock()
    engine.backend = mock_backend
    instructions = "Review the named GitHub pull request using real repository tools."

    def mock_execute(query, params=None):
        params = params or {}
        if "MATCH (s:Server)" in query and params.get("name") == "github-review":
            return []
        if (
            "MATCH (r:CallableResource)" in query
            and params.get("name") == "github-review"
        ):
            return [
                {
                    "rid": "resource:skill:github-review",
                    "rtype": "AGENT_SKILL",
                    "description": "Review a GitHub pull request.",
                    "system_prompt": instructions,
                    "instruction_digest": runnable_skill_digest(instructions),
                    "source_ref": "skill://github-review",
                    "provider_ref": "provider://github-mcp",
                }
            ]
        if "MATCH (s:Server)" in query and params.get("name") == "github-mcp":
            return [
                {
                    "sid": "srv:github-mcp",
                    "name": "github-mcp",
                    "url": "https://github-mcp.example/mcp",
                    "env": "",
                }
            ]
        if "[:PROVIDES]" in query and params.get("sid") == "srv:github-mcp":
            return [
                {
                    "name": "github_pull_request_read",
                    "description": "Read pull requests.",
                }
            ]
        return []

    mock_backend.execute.side_effect = mock_execute

    meta = _resolve_agent_from_kg(engine, "github-review")

    assert meta["type"] == "server"
    assert meta["skill_of_server"] == "github-mcp"
    assert meta["tools"][0]["name"] == "github_pull_request_read"
    assert instructions in meta["system_prompt"]


def _ensure_standard_model_class(monkeypatch) -> None:
    """Configure a 'normal'-tier chat model so run_agent's default
    ``model_class="standard"`` resolves (_configured_model_for_class maps
    "standard" -> intelligence_level "normal"). This test environment carries
    no configured chat models at all, so without this every run_agent() call
    fails before ever reaching the mocked ``_execute_graph`` this test is
    actually exercising.
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


@pytest.mark.asyncio
async def test_agent_runner_execution_failure(monkeypatch):
    """Test agent runner fallback on execution failure."""
    from agent_utilities.orchestration.agent_runner import run_agent

    _ensure_standard_model_class(monkeypatch)
    engine = _create_engine()

    with patch(
        "agent_utilities.orchestration.agent_runner._execute_graph",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.side_effect = Exception("Simulated execution failure")

        result = await run_agent("non-existent-agent", "do something", engine=engine)

        assert "Simulated execution failure" in result

        # Verify trace node was added. 'type' is the retired node property;
        # the canonical name is 'node_type'.
        trace_nodes = [
            n
            for n, d in engine.graph.nodes(data=True)
            if d.get("node_type") == "RunTrace"
        ]
        assert len(trace_nodes) == 1
        assert engine.graph.nodes[trace_nodes[0]]["status"] == "failed"


@pytest.mark.asyncio
async def test_agent_runner_success(monkeypatch):
    """Test agent runner success path and provenance."""
    from agent_utilities.orchestration.agent_runner import run_agent

    _ensure_standard_model_class(monkeypatch)
    engine = _create_engine()

    with patch(
        "agent_utilities.orchestration.agent_runner._execute_graph",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = {"results": {"output": "Success response"}}

        result = await run_agent("test-agent", "do something", engine=engine)

        assert result == "Success response"

        # Verify trace node was added. 'type' is the retired node property;
        # the canonical name is 'node_type'.
        trace_nodes = [
            n
            for n, d in engine.graph.nodes(data=True)
            if d.get("node_type") == "RunTrace"
        ]
        assert len(trace_nodes) == 1
        assert engine.graph.nodes[trace_nodes[0]]["status"] == "completed"
