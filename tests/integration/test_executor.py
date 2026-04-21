import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from agent_utilities.graph import (
    agent_matches_node_id,
    get_discovery_registry,
)
from agent_utilities.graph.executor import (
    _get_domain_tools,
    _execute_specialized_step,
)
from agent_utilities.models import MCPAgent, ExecutionStep, GraphPlan, MCPAgentRegistryModel
from agent_utilities.graph.state import GraphState, GraphDeps

@pytest.fixture
def mock_deps():
    deps = MagicMock(spec=GraphDeps)
    deps.mcp_toolsets = []
    deps.sub_agents = {}
    deps.event_queue = asyncio.Queue()
    deps.agent_model = "test"
    deps.provider = "openai"
    deps.base_url = "http://localhost"
    deps.api_key = "test-key"
    deps.ssl_verify = True
    deps.message_history_cache = {}
    deps.server_health = {}
    deps.discovery_metadata = {}
    return deps

@pytest.mark.parametrize("agent_data, node_id, expected", [
    ({"name": "Researcher", "tag": "researcher"}, "researcher", True),
    ({"name": "Researcher", "tag": "researcher"}, "RESEARCHER", True),
    ({"name": "Researcher", "tag": "researcher"}, "research_agent", True),
    ({"name": "GitHub", "mcp_server": "github-mcp"}, "github", True),
    ({"name": "GitHub", "mcp_server": "github-mcp", "description": "Git hosting and version control"}, "git_expert", True),
    ({"name": "Random", "tag": "other"}, "researcher", False),
])
def test_agent_matches_node_id(agent_data, node_id, expected):
    agent = MCPAgent(
        name=agent_data.get("name", "Agent"),
        capabilities=[agent_data.get("tag", "")] if agent_data.get("tag") else [],
        mcp_server=agent_data.get("mcp_server", "test-server"),
        description=agent_data.get("description", "A specialist agent"),
        system_prompt=agent_data.get("system_prompt", "You are a specialist.")
    )
    result = agent_matches_node_id(agent, node_id)
    assert result == expected, f"Failed for agent {agent_data} and node_id {node_id}. Got {result}, expected {expected}"

@pytest.mark.asyncio
async def test_get_domain_tools_basic(mock_deps):
    mock_agent = MCPAgent(name="researcher", capabilities=["web-search"])
    mock_registry = MCPAgentRegistryModel(agents=[mock_agent])

    with patch("agent_utilities.graph.executor.get_discovery_registry", return_value=mock_registry):
        with patch("agent_utilities.tools.developer_tools.developer_tools", []):
            tools, skill_toolsets = await _get_domain_tools("researcher", mock_deps)
            assert isinstance(tools, list)
            assert isinstance(skill_toolsets, list)

@pytest.mark.asyncio
async def test_execute_specialized_step_subagent_target(mock_deps):
    state = GraphState(query="test query")
    state.plan = GraphPlan(steps=[ExecutionStep(node_id="specialist")])
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    mock_agent = MagicMock()
    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.stream_messages.return_value = []

    async def mock_stream_text(*args, **kwargs):
        yield "chunk"
    stream.stream_text = mock_stream_text

    stream.get_output.return_value = "Expert Result"
    stream.usage.return_value = MagicMock()
    stream.all_messages.return_value = []
    mock_agent.run_stream.return_value = stream

    # We must patch Agent in the executor module
    with patch("agent_utilities.graph.executor.Agent", return_value=mock_agent):
        with patch("agent_utilities.graph.executor.load_specialized_prompts", return_value="Prompt"):
            with patch("agent_utilities.graph.executor.on_enter_specialist", return_value={}):
                with patch("agent_utilities.graph.executor.on_exit_specialist"):
                    res = await _execute_specialized_step(ctx, "specialist")
                    assert res == "execution_joiner"
                    assert state.results["specialist"] == "Expert Result"
                    assert "specialist_0" in state.results_registry

@pytest.mark.asyncio
async def test_execute_specialized_step_error_recovery(mock_deps):
    state = GraphState(query="test query")
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = mock_deps

    # Ensuring Agent failure triggers recovery
    with patch("agent_utilities.graph.executor.Agent.run_stream", side_effect=Exception("Simulation Error")):
        with patch("agent_utilities.graph.executor.load_specialized_prompts", return_value="Prompt"):
             with patch("agent_utilities.graph.executor.on_enter_specialist", return_value={}):
                with patch("agent_utilities.graph.executor.on_exit_specialist"):
                    res = await _execute_specialized_step(ctx, "unknown")
                    assert res == "error_recovery"
