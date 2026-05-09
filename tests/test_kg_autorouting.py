import pytest
from unittest.mock import MagicMock

from agent_utilities.graph.routing_policy import OntologicalFallbackChain
from agent_utilities.graph.context_filter import prune_context_by_semantic_distance


def test_ontological_fallback_chain():
    mock_engine = MagicMock()
    mock_engine.search_hybrid.return_value = [
        {"model_id": "gpt-4.1-mini", "name": "Lightweight fallback"},
        {"model_id": "claude-haiku", "name": "Anthropic fallback"},
    ]

    chain = OntologicalFallbackChain(engine=mock_engine)
    fallback = chain.get_fallback("gpt-4.1", "429")

    assert fallback == "gpt-4.1-mini"
    mock_engine.search_hybrid.assert_called_once()


def test_prune_context_by_semantic_distance():
    # Mock nodes where 'distance' represents semantic distance (lower is closer)
    nodes = [
        {"id": "n1", "distance": 0.1, "content": "A" * 40},  # ~10 tokens
        {"id": "n2", "distance": 0.9, "content": "B" * 400},  # ~100 tokens
        {"id": "n3", "distance": 0.5, "content": "C" * 200},  # ~50 tokens
    ]

    # Max 100 tokens: n1 (10) + n3 (50) = 60 <= 100. n2 (100) will overflow.
    pruned = prune_context_by_semantic_distance(nodes, "query", max_tokens=100)

    assert len(pruned) == 2
    assert pruned[0]["id"] == "n1"
    assert pruned[1]["id"] == "n3"


import pytest
import os
from unittest.mock import MagicMock
from agent_utilities.graph.routing import router_step
from pydantic_graph.beta import StepContext
from agent_utilities.graph.state import GraphState, GraphDeps


@pytest.mark.asyncio
async def test_kg_native_reasoning_escalation():
    # Mock context and dependencies
    state = GraphState(query="Calculate optimal Almgren-Chriss trajectory")
    deps = GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])
    deps.knowledge_engine = MagicMock()

    # Mock hybrid search to return a MathematicalFoundationNode matching AHE-3.25
    deps.knowledge_engine.search_hybrid.return_value = [
        {
            "name": "MathematicalFoundationNode",
            "description": "Quantitative algorithms including Almgren-Chriss.",
        }
    ]

    ctx = StepContext(state=state, deps=deps, inputs=())

    # We just want to mock the inner LLM agent run_stream so we don't actually make an LLM call.
    # Instead, we can just intercept the model selection logic.
    # Actually, we can check if the correct model was pinned to the state before the Agent runs.

    # Mock Agent.run_stream to raise an exception so we can break out early after model selection
    class MockAgent:
        def __init__(self, model, **kwargs):
            pass  # Just mock the initialization

        def run_stream(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

    import agent_utilities.graph.routing

    original_agent = agent_utilities.graph.routing.Agent
    agent_utilities.graph.routing.Agent = MockAgent

    try:
        await router_step(ctx)
    finally:
        agent_utilities.graph.routing.Agent = original_agent

    # Verify that the reasoning model was selected because of the quantitative subgraph
    assert state.pinned_model_id == os.environ.get("REASONING_MODEL", "o3-mini")


@pytest.mark.asyncio
async def test_kg_native_complex_task_escalation():
    state = GraphState(query="Execute a deep TradingPipeline")
    deps = GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])
    deps.knowledge_engine = MagicMock()

    # Mock AHE-3.24 complex topological subgraph
    def mock_hybrid(query, top_k):
        if "TradingPipeline" in query:
            return [
                {
                    "name": "TradingPipelineNode",
                    "description": "Complex financial pipeline",
                }
            ]
        return []

    deps.knowledge_engine.search_hybrid.side_effect = mock_hybrid

    ctx = StepContext(state=state, deps=deps, inputs=())

    class MockAgent:
        def __init__(self, model, **kwargs):
            pass

        def run_stream(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

    import agent_utilities.graph.routing

    original_agent = agent_utilities.graph.routing.Agent
    agent_utilities.graph.routing.Agent = MockAgent

    try:
        await router_step(ctx)
    finally:
        agent_utilities.graph.routing.Agent = original_agent

    # Verify that we did NOT fall back to lightweight model, because is_complex=True
    assert state.pinned_model_id != os.environ.get("LIGHTWEIGHT_MODEL", "gpt-4o-mini")
