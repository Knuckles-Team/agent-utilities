from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.graph.executor import GraphDeps, GraphState
from agent_utilities.graph.steps import router_step


@pytest.mark.asyncio
async def test_router_rlm_trigger():
    # Setup state and deps
    state = GraphState(query="Analyze the entire history of the project and summarize every failure.")
    state.error = None

    deps = MagicMock(spec=GraphDeps)
    deps.router_model = "test-model"
    deps.router_timeout = 60
    deps.mcp_toolsets = []
    deps.tag_prompts = {"test-specialist": "A specialist for testing"}
    deps.tag_env_vars = {}
    deps.sub_agents = {}
    deps.event_queue = MagicMock()
    deps.message_history_cache = {}
    deps.knowledge_engine = MagicMock()
    deps.plan_sync = None

    ctx = MagicMock()
    ctx.state = state
    ctx.deps = deps

    # Patch RLMEnvironment.run_full_rlm
    with patch("agent_utilities.rlm.repl.RLMEnvironment.run_full_rlm", new_callable=AsyncMock) as mock_rlm:
        mock_rlm.return_value = '{"steps": [{"node_id": "test", "input_data": {}}], "metadata": {"reasoning": "test"}}'

        # Patch fetch_unified_context to return a large string
        with patch("agent_utilities.graph.steps.fetch_unified_context", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = "A" * 60000

            # Patch RLMConfig
            mock_config = MagicMock()
            mock_config.enabled = True
            mock_config.max_context_threshold = 50000

            with patch("agent_utilities.rlm.config.RLMConfig", return_value=mock_config):
                res = await router_step(ctx)
                assert res == "dispatcher"
                mock_rlm.assert_called_once()
