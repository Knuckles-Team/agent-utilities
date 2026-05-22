"""Tests for the graph_orchestrate MCP tool and Agent Runner.

CONCEPT:ORCH-1.21 — KG-to-LLM Execution Bridge tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


def _create_engine():
    """Create a minimal IntelligenceGraphEngine for testing."""
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=None)
    return engine


@pytest.mark.asyncio
async def test_agent_runner_resolution():
    """Test that agent_runner resolves agent capabilities from KG."""
    from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg

    engine = _create_engine()

    from unittest.mock import MagicMock
    mock_backend = MagicMock()
    engine.backend = mock_backend

    # Mock the backend execution specifically for the CallableResource query
    def mock_execute(query, params=None):
        if "CallableResource" in query and params and params.get("name") == "test-agent":
            return [{"rid": "skill:test-agent", "rtype": "AGENT_SKILL", "description": "A test skill agent", "skill_path": ""}]
        return []

    engine.backend.execute.side_effect = mock_execute

    meta = _resolve_agent_from_kg(engine, "test-agent")

    assert meta["type"] == "skill"


@pytest.mark.asyncio
async def test_agent_runner_execution_failure():
    """Test agent runner fallback on execution failure."""
    from agent_utilities.orchestration.agent_runner import run_agent

    engine = _create_engine()

    with patch(
        "agent_utilities.orchestration.agent_runner._execute_graph",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.side_effect = Exception("Simulated execution failure")

        result = await run_agent("non-existent-agent", "do something", engine=engine)

        assert "Simulated execution failure" in result

        # Verify trace node was added
        trace_nodes = [
            n for n, d in engine.graph.nodes(data=True) if d.get("type") == "RunTrace"
        ]
        assert len(trace_nodes) == 1
        assert engine.graph.nodes[trace_nodes[0]]["status"] == "failed"


@pytest.mark.asyncio
async def test_agent_runner_success():
    """Test agent runner success path and provenance."""
    from agent_utilities.orchestration.agent_runner import run_agent

    engine = _create_engine()

    with patch(
        "agent_utilities.orchestration.agent_runner._execute_graph",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = {"results": {"output": "Success response"}}

        result = await run_agent("test-agent", "do something", engine=engine)

        assert result == "Success response"

        # Verify trace node was added
        trace_nodes = [
            n for n, d in engine.graph.nodes(data=True) if d.get("type") == "RunTrace"
        ]
        assert len(trace_nodes) == 1
        assert engine.graph.nodes[trace_nodes[0]]["status"] == "completed"
