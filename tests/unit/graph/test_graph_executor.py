"""
Concepts: Graph Executor Architecture
This module contains unit tests for the graph executor, verifying the orchestration logic for
domain-specific agents, specialized steps, and MCP tool interactions.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities.graph import executor
from agent_utilities.graph.state import GraphState, GraphDeps
from agent_utilities.models.mcp import MCPAgent
from agent_utilities.models.graph import GraphPlan, ExecutionStep
from agent_utilities.models.usage import UsageStatistics

pytestmark = pytest.mark.concept("architecture")

class MockCtx:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps
        self.inputs = None
        self.step_cursor = 0

async def async_iter(items):
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_on_enter_specialist():
    deps = MagicMock()
    state = MagicMock()
    await executor.on_enter_specialist(deps, state, "test_agent")
    assert hasattr(deps, "_entry_times")

@pytest.mark.asyncio
async def test_on_exit_specialist():
    deps = MagicMock()
    deps.server_health = {}
    deps._entry_times = {"test_agent": 0}
    state = MagicMock()
    with patch("agent_utilities.graph.hsm.MCPServerHealth") as mock_health_class:
        mock_health = mock_health_class.return_value
        await executor.on_exit_specialist(deps, state, "test_agent", success=True, server_name="test_server")
        assert "test_server" in deps.server_health
        mock_health.record_success.assert_called_once()

@pytest.mark.asyncio
async def test_check_specialist_preconditions():
    agent_info = MagicMock()
    agent_info.mcp_server = "test_server"
    deps = MagicMock()
    deps.server_health = {"test_server": MagicMock(is_healthy=True)}
    result, msg = executor.check_specialist_preconditions(agent_info, deps)
    assert result is False
    assert "No MCP toolset bound" in msg

@pytest.mark.asyncio
async def test_execute_domain_logic_agent_found():
    state = GraphState(query="test")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
        provider="openai", agent_model="gpt-4o", api_key="sk-test"
    )
    ctx = MockCtx(state, deps)

    mock_agent = MagicMock()
    stream = MagicMock()
    stream.stream_messages.side_effect = lambda: async_iter([])
    stream.get_output = AsyncMock(return_value="result")
    stream.usage.return_value = UsageStatistics()

    # run_stream is an async context manager
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock()

    deps.sub_agents = {"test_domain": mock_agent}

    await executor._execute_domain_logic(ctx, "test_domain")
    assert state.results["test_domain"] == "result"

@pytest.mark.asyncio
async def test_execute_domain_logic_error():
    state = GraphState(query="test")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
    )
    ctx = MockCtx(state, deps)
    deps.sub_agents = {"test_domain": "legacy_string"}

    await executor._execute_domain_logic(ctx, "test_domain")
    assert "Legacy delegation" in state.results["test_domain"]

@pytest.mark.asyncio
async def test_execute_domain_logic_approval_required():
    from pydantic_ai import DeferredToolRequests
    state = GraphState(query="test")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
    )
    ctx = MockCtx(state, deps)

    mock_agent = AsyncMock()
    mock_res = MagicMock()
    mock_res.output = DeferredToolRequests(calls=[])
    mock_agent.run.return_value = mock_res

    with patch("agent_utilities.graph.executor.create_agent", return_value=(mock_agent, [])):
        result = await executor._execute_domain_logic(ctx, "test_domain")
        assert type(result).__name__ == "End"
        assert type(result.data).__name__ == "DeferredToolRequests"

    assert state.human_approval_required is True

@pytest.mark.asyncio
async def test_execute_specialized_step():
    state = GraphState(query="test")
    state.plan = GraphPlan(steps=[ExecutionStep(node_id="qa")])
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
        provider="openai", agent_model="gpt-4o", api_key="sk-test"
    )
    ctx = MockCtx(state, deps)

    mock_agent = MagicMock()
    stream = MagicMock()
    # delta=True → yield incremental chunks; delta=False → yield accumulated text.
    stream.stream_text.side_effect = lambda delta=True: async_iter(
        ["chunk"] if delta else ["chunk"]
    )
    stream.get_output = AsyncMock(return_value="specialized result")
    stream.usage.return_value = UsageStatistics()
    stream.all_messages = AsyncMock(return_value=[])

    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock()

    with patch("agent_utilities.graph.executor.load_specialized_prompts", return_value="prompt"), \
         patch("agent_utilities.graph.executor._get_domain_tools", new_callable=AsyncMock, return_value=([], [])), \
         patch("agent_utilities.graph.executor.Agent", return_value=mock_agent), \
         patch("agent_utilities.graph.executor.on_enter_specialist", new_callable=AsyncMock), \
         patch("agent_utilities.graph.executor.on_exit_specialist", new_callable=AsyncMock), \
         patch("agent_utilities.graph.executor.emit_graph_event"):

        result = await executor._execute_specialized_step(ctx, "qa")
        assert result == "execution_joiner"
        assert state.results["qa"] == "specialized result"

@pytest.mark.asyncio
async def test_execute_agent_package_logic_remote_a2a():
    state = GraphState(query="test")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
        approval_timeout=300, ssl_verify=True
    )
    ctx = MockCtx(state, deps)
    meta = {"type": "remote_a2a", "url": "http://peer"}

    with patch("agent_utilities.a2a.A2AClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.execute_task = AsyncMock(return_value="a2a result")

        result = await executor._execute_agent_package_logic(ctx, "peer_agent", meta)
        assert result == "execution_joiner"
        assert state.results["peer_agent"] == "a2a result"

@pytest.mark.asyncio
async def test_execute_agent_package_logic_local():
    state = GraphState(query="test")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[],
    )
    ctx = MockCtx(state, deps)
    meta = {"type": "local"}

    mock_agent = MCPAgent(name="local_agent", mcp_server="server")
    registry = MagicMock()
    registry.agents = [mock_agent]

    with patch("agent_utilities.graph.executor.load_node_agents_registry", return_value=registry), \
         patch("agent_utilities.graph.executor._execute_dynamic_mcp_agent", new_callable=AsyncMock) as mock_exec:

        mock_exec.return_value = "execution_joiner"
        result = await executor._execute_agent_package_logic(ctx, "local_agent", meta)

        assert result == "execution_joiner"
        mock_exec.assert_called_once()
