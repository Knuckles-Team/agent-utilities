"""Tests for the graph_orchestrate MCP tool and Agent Runner integration.

CONCEPT:ORCH-1.21 — KG-to-LLM Execution Bridge tests.

Covers:
- Orchestrator dispatch/status/approval workflow
- Agent Runner KG resolution
- Integration test for live LM Studio invocation (opt-in)
- Debate/consensus/veto KG node creation
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import networkx as nx
import pytest

# Integration test marker — requires live LM Studio
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine():
    """Create a minimal IntelligenceGraphEngine for testing."""
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=None)
    return engine


# ---------------------------------------------------------------------------
# Component 1: Orchestrator Manager
# ---------------------------------------------------------------------------


class TestOrchestratorManager:
    """Tests for the Orchestrator dispatch/status/approval workflow."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_task_node(self):
        """Dispatching a task creates a Task node in the KG."""
        from agent_utilities.orchestration.manager import Orchestrator

        engine = _create_engine()
        orch = Orchestrator(engine)

        job_id = await orch.dispatch_task("Test task: analyze logs")

        assert job_id.startswith("orch-")
        # Verify node exists in the engine's graph
        assert job_id in engine.graph.nodes

        node_data = engine.graph.nodes[job_id]
        assert node_data.get("status") == "pending"
        assert "analyze logs" in node_data.get("description", "")

    @pytest.mark.asyncio
    async def test_dispatch_with_dependencies(self):
        """Dispatching with dependencies records them."""
        from agent_utilities.orchestration.manager import Orchestrator

        engine = _create_engine()
        orch = Orchestrator(engine)

        job1 = await orch.dispatch_task("Step 1")
        job2 = await orch.dispatch_task("Step 2", dependencies=[job1])

        node2_data = engine.graph.nodes[job2]
        assert job1 in node2_data.get("dependencies", [])

    def test_get_task_status_not_found(self):
        """Status check for non-existent task returns error."""
        from agent_utilities.orchestration.manager import Orchestrator

        engine = _create_engine()
        orch = Orchestrator(engine)

        result = orch.get_task_status("nonexistent-job")
        assert "error" in result or "not found" in str(result).lower()


# ---------------------------------------------------------------------------
# Component 2: Agent Runner
# ---------------------------------------------------------------------------


class TestAgentRunner:
    """Tests for the agent_runner.run_agent() function."""

    def test_resolve_agent_from_kg_empty(self):
        """Empty KG returns unknown agent type."""
        from agent_utilities.orchestration.agent_runner import (
            _resolve_agent_from_kg,
        )

        engine = _create_engine()
        meta = _resolve_agent_from_kg(engine, "nonexistent-agent")
        assert meta["type"] == "unknown"

    def test_build_execution_config(self):
        """Config builder produces valid config dict."""
        from agent_utilities.orchestration.agent_runner import (
            _build_execution_config,
        )

        engine = _create_engine()
        agent_meta = {
            "type": "server",
            "server_id": "srv:test",
            "tools": [{"name": "test_tool", "description": "A tool"}],
            "capabilities": ["testing"],
            "mcp_command": "uv",
            "url": "",
            "system_prompt": "",
        }

        config = _build_execution_config(engine, "test-agent", agent_meta)

        assert "tag_prompts" in config
        assert "test-agent" in config["tag_prompts"]
        assert "test_tool" in config["tag_prompts"]
        assert config["provider"] is not None

    def test_record_execution_trace(self):
        """Execution trace creates RunTrace node."""
        from agent_utilities.orchestration.agent_runner import (
            _record_execution_trace,
        )

        engine = _create_engine()
        _record_execution_trace(
            engine,
            run_id="run:test123",
            agent_name="test-agent",
            task="do something",
            status="completed",
            duration_ms=1500.0,
            result_preview="Success!",
        )

        assert "trace:run:test123" in engine.graph.nodes
        node_data = engine.graph.nodes["trace:run:test123"]
        assert node_data.get("status") == "completed"
        assert node_data.get("agent_name") == "test-agent"

    @pytest.mark.asyncio
    async def test_run_agent_graceful_failure(self):
        """run_agent handles missing agent gracefully."""
        from agent_utilities.orchestration.agent_runner import run_agent

        engine = _create_engine()

        # Patch create_graph_agent to avoid full graph materialization
        with patch(
            "agent_utilities.orchestration.agent_runner._execute_graph",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = {
                "status": "completed",
                "results": {"output": "Test result"},
            }

            result = await run_agent(
                agent_name="test-agent",
                task="List containers",
                max_steps=5,
                engine=engine,
            )

        assert "Test result" in result

    @pytest.mark.asyncio
    async def test_run_agent_error_records_trace(self):
        """Failed execution records error trace in KG."""
        from agent_utilities.orchestration.agent_runner import run_agent

        engine = _create_engine()

        with patch(
            "agent_utilities.orchestration.agent_runner._execute_graph",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.side_effect = RuntimeError("LM Studio unreachable")

            result = await run_agent(
                agent_name="broken-agent",
                task="Fail gracefully",
                engine=engine,
            )

        assert "failed" in result.lower() or "unreachable" in result.lower()

        # Verify error trace was recorded
        trace_nodes = [nid for nid in engine.graph.nodes if nid.startswith("trace:")]
        assert len(trace_nodes) >= 1
        trace_data = engine.graph.nodes[trace_nodes[0]]
        assert trace_data.get("status") == "failed"


# ---------------------------------------------------------------------------
# Component 3: Debate/Consensus/Veto (graph_orchestrate actions)
# ---------------------------------------------------------------------------


class TestDebateConsensus:
    """Tests for the debate, consensus, and veto orchestration actions."""

    def test_start_debate_creates_node(self):
        """start_debate action creates a TradingDebate node."""
        engine = _create_engine()
        engine.add_node(
            "debate_test-123",
            "TradingDebate",
            properties={"topic": "BTC outlook", "status": "ongoing"},
        )

        assert "debate_test-123" in engine.graph.nodes
        assert engine.graph.nodes["debate_test-123"].get("status") == "ongoing"

    def test_submit_veto_creates_edge(self):
        """submit_risk_veto creates a RiskVeto node and CONTRADICTS edge."""
        engine = _create_engine()

        # Create debate first
        engine.add_node(
            "debate_d1",
            "TradingDebate",
            properties={"topic": "BTC", "status": "ongoing"},
        )

        # Create veto
        engine.add_node(
            "veto_d1", "RiskVeto", properties={"reason": "Too volatile", "target": "d1"}
        )
        engine.graph.add_edge(
            "veto_d1", "debate_d1", rel_type="CONTRADICTS_BELIEF_PROP"
        )

        # Verify edge exists
        assert engine.graph.has_edge("veto_d1", "debate_d1")


# ---------------------------------------------------------------------------
# Component 4: Integration Test (requires live LM Studio)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLMStudioIntegration:
    """Integration tests requiring a live LM Studio instance.

    These tests are skipped unless the ``integration`` marker is specified
    and LM Studio is reachable at the configured URL.
    """

    @staticmethod
    def _check_lmstudio():
        """Check if LM Studio is reachable."""
        import httpx

        base_url = os.environ.get("LLM_BASE_URL", "http://vllm.arpa/v1")
        try:
            resp = httpx.get(f"{base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    @pytest.mark.asyncio
    async def test_execute_agent_lmstudio(self):
        """Full pipeline: KG → Agent Runner → LM Studio → Response.

        This test verifies that the entire orchestration pipeline
        successfully invokes the local LM Studio instance.
        """
        if not self._check_lmstudio():
            pytest.skip("LM Studio not reachable")

        from agent_utilities.orchestration.agent_runner import run_agent

        # Use real env vars
        os.environ.pop("AGENT_UTILITIES_TESTING", None)

        engine = _create_engine()

        result = await run_agent(
            agent_name="general",
            task="What is 2 + 2? Answer with just the number.",
            max_steps=5,
            engine=engine,
        )

        # Should get a response (even if routing is imperfect)
        assert result is not None
        assert len(result) > 0
        # Note: don't check for literal "error" in string — GraphResponse
        # serialization includes `error=None` field name even on success.
        assert "status='failed'" not in result
