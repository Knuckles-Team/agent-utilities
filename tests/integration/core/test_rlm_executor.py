import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities.graph.executor import _execute_dynamic_mcp_agent, GraphState, GraphDeps

@pytest.mark.asyncio
async def test_executor_rlm_summary():
    # Setup state and deps
    state = GraphState(query="What was the result of the massive tool?")
    state.step_cursor = 0
    state.results_registry = {}
    state.results = {}
    state.error = None

    deps = MagicMock()
    deps.event_queue = MagicMock()
    deps.approval_manager = None
    deps.message_history_cache = {}
    deps.discovery_metadata = {}
    deps.provider = "openai"
    deps.agent_model = MagicMock()
    deps.base_url = None
    deps.api_key = "fake-key"
    deps.ssl_verify = True
    deps.mcp_toolsets = []
    deps.tag_prompts = {}
    deps.tag_env_vars = {}
    deps.sub_agents = {}

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = deps

    agent_info = MagicMock()
    agent_info.name = "test-specialist"
    agent_info.capabilities = []
    agent_info.mcp_server = "test-server"

    # Mock agent run result with massive output
    mock_res = MagicMock()
    mock_res.output = "A" * 60000
    mock_res.all_messages = MagicMock(return_value=[])
    mock_res.usage = MagicMock()

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_res)

    # Patch create_agent AND pydantic_ai.Agent
    with patch("agent_utilities.graph.executor.Agent", return_value=mock_agent):
        with patch("agent_utilities.graph.executor.create_agent", return_value=(mock_agent, [])):
            # Patch recursive_reasoner_tool
            with patch("agent_utilities.rlm.specialist.recursive_reasoner_tool", new_callable=AsyncMock) as mock_rlm_tool:
                mock_rlm_tool.return_value = "This is a summary of 60k chars."

                # Patch RLMConfig
                mock_config = MagicMock()
                mock_config.max_context_threshold = 50000

                with patch("agent_utilities.rlm.config.RLMConfig", return_value=mock_config):
                    # Patch load_node_agents_registry
                    with patch("agent_utilities.graph.executor.load_node_agents_registry", return_value=MagicMock()):
                        # Patch on_enter/exit
                        with patch("agent_utilities.graph.executor.on_enter_specialist", new_callable=AsyncMock):
                            with patch("agent_utilities.graph.executor.on_exit_specialist", new_callable=AsyncMock):
                                # Patch check_specialist_preconditions
                                with patch("agent_utilities.graph.executor.check_specialist_preconditions", return_value=(True, None)):
                                    # Patch pick_specialist_model
                                    with patch("agent_utilities.graph.executor.pick_specialist_model", return_value=MagicMock()):
                                        res = await _execute_dynamic_mcp_agent(ctx, agent_info)
                                        assert res == "execution_joiner"
                                        # Check if the result was summarized
                                        result_key = f"test-specialist_0"
                                        assert "[RLM Synthesized Summary of Massive Data]" in state.results_registry[result_key]
                                        assert "This is a summary of 60k chars." in state.results_registry[result_key]
                                        mock_rlm_tool.assert_called_once()
