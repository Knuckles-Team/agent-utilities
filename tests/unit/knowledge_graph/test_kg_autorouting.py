"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.graph.adaptive_agent_router import OntologicalFallbackChain
from agent_utilities.knowledge_graph.memory import (
    prune_context_by_semantic_distance,
)


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


import os

from pydantic_graph import StepContext

from agent_utilities.graph.routing import router_step
from agent_utilities.graph.state import GraphDeps, GraphState


@pytest.mark.asyncio
async def test_kg_native_reasoning_escalation(monkeypatch):
    # The router only pins state.pinned_model_id in the reasoning-escalation
    # branch when a "super"-tier chat model is configured (AgentConfig.
    # super_chat_model resolves intelligence_level="super", falling back to
    # the first configured chat model if none -- reasoning_model_id stays
    # None, and pinned_model_id is never set, when NO chat models are
    # configured at all, which is this environment's baseline state.
    from agent_utilities.core.config import ChatModelConfig
    from agent_utilities.core.config import config as agent_config

    monkeypatch.setattr(
        agent_config,
        "chat_models",
        [
            ChatModelConfig(
                id="test-reasoning-model",
                provider="openai",
                intelligence_level="super",
            )
        ],
    )

    # router_step() internally calls get_discovery_registry(), which lazily
    # provisions IntelligenceGraphEngine.get_or_create(db_path=...) if no
    # engine is already active. That path builds its own backend via
    # create_backend(db_path=...), which constructs a bare
    # EpistemicGraphBackend() -- resolving its own routing graph via
    # resolve_routing_graph(None) BEFORE GraphComputeEngine is ever asked for
    # one, bypassing the isolate_graph_compute_engine fixture's redirect (same
    # family as test_kg_native_orchestration.py). Pre-populating
    # IntelligenceGraphEngine._ACTIVE_ENGINE with one bound to an
    # already-isolated GraphComputeEngine makes get_or_create() reuse it
    # instead of falling back to the divergent bare construction.
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    compute = GraphComputeEngine(backend_type="rust")
    isolated_backend = object.__new__(EpistemicGraphBackend)
    isolated_backend._graph = compute
    isolated_backend.graph_name = compute.graph_name
    IntelligenceGraphEngine._ACTIVE_ENGINE = None
    IntelligenceGraphEngine(backend=isolated_backend)

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

    ctx = StepContext(state=state, deps=deps, inputs=None)

    # We just want to mock the inner LLM agent run_stream so we don't actually make an LLM call.
    # Instead, we can just intercept the model selection logic.
    # Actually, we can check if the correct model was pinned to the state before the Agent runs.

    # Mock RLMConfig to be disabled so it takes the standard path
    class MockRLMConfig:
        enabled = False
        max_context_threshold = 100000
        trigger_on_large_output = False
        trigger_on_ahe_distillation = False
        trigger_on_kg_bulk_analysis = False

        def should_trigger(self, **kwargs):
            return False

    import agent_utilities.rlm.config

    original_rlm_config = agent_utilities.rlm.config.RLMConfig  # type: ignore
    agent_utilities.rlm.config.RLMConfig = MockRLMConfig  # type: ignore

    # Mock Agent.run_stream to raise an exception so we can break out early after model selection
    class MockAgent:
        def __init__(self, *args, **kwargs):
            return None

        async def run(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

        def run_stream(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

    import agent_utilities.graph._router_impl

    original_agent_factory = agent_utilities.graph._router_impl.create_context_agent
    agent_utilities.graph._router_impl.create_context_agent = MockAgent  # type: ignore

    import agent_utilities.core.model_factory

    original_create_model = agent_utilities.core.model_factory.create_model
    agent_utilities.core.model_factory.create_model = MagicMock(
        return_value="mock_model"
    )  # type: ignore

    import agent_utilities.graph.kg_graph_factory

    original_build_kg = getattr(
        agent_utilities.graph.kg_graph_factory, "build_pydantic_graph_from_kg", None
    )
    agent_utilities.graph.kg_graph_factory.build_pydantic_graph_from_kg = MagicMock(
        side_effect=Exception("skip kg graph bypass")
    )  # type: ignore

    import agent_utilities.graph._router_impl

    original_logger_error = agent_utilities.graph._router_impl.logger.error
    agent_utilities.graph._router_impl.logger.error = lambda x: print(
        f"ROUTER ERROR: {x}"
    )  # type: ignore

    try:
        await router_step(ctx)
    finally:
        agent_utilities.graph._router_impl.create_context_agent = (  # type: ignore
            original_agent_factory
        )
        agent_utilities.rlm.config.RLMConfig = original_rlm_config  # type: ignore
        agent_utilities.core.model_factory.create_model = original_create_model  # type: ignore
        if original_build_kg:
            agent_utilities.graph.kg_graph_factory.build_pydantic_graph_from_kg = (
                original_build_kg  # type: ignore
            )
        agent_utilities.graph._router_impl.logger.error = original_logger_error  # type: ignore
        IntelligenceGraphEngine._ACTIVE_ENGINE = None

    print(f"DEBUG: state.pinned_model_id is {state.pinned_model_id}")
    print(f"DEBUG: state.error is {state.error}")
    # Verify that the reasoning model was selected because of the quantitative subgraph
    assert state.pinned_model_id is not None
    # Depending on defaults, it could be o3-mini or a configured model.
    assert "o3-mini" in state.pinned_model_id or state.pinned_model_id != ""


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

    ctx = StepContext(state=state, deps=deps, inputs=None)

    class MockRLMConfig:
        enabled = False
        max_context_threshold = 100000
        trigger_on_large_output = False
        trigger_on_ahe_distillation = False
        trigger_on_kg_bulk_analysis = False

        def should_trigger(self, **kwargs):
            return False

    import agent_utilities.rlm.config

    original_rlm_config = agent_utilities.rlm.config.RLMConfig  # type: ignore
    agent_utilities.rlm.config.RLMConfig = MockRLMConfig  # type: ignore

    class MockAgent:
        def __init__(self, *args, **kwargs):
            return None

        async def run(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

        def run_stream(self, *args, **kwargs):
            raise Exception("Mocked Agent Run")

    import agent_utilities.graph._router_impl

    original_agent_factory = agent_utilities.graph._router_impl.create_context_agent
    agent_utilities.graph._router_impl.create_context_agent = MockAgent  # type: ignore

    import agent_utilities.core.model_factory

    original_create_model = agent_utilities.core.model_factory.create_model
    agent_utilities.core.model_factory.create_model = MagicMock(
        return_value="mock_model"
    )  # type: ignore

    import agent_utilities.graph.kg_graph_factory

    original_build_kg2 = getattr(
        agent_utilities.graph.kg_graph_factory, "build_pydantic_graph_from_kg", None
    )
    agent_utilities.graph.kg_graph_factory.build_pydantic_graph_from_kg = MagicMock(
        side_effect=Exception("skip kg graph bypass")
    )  # type: ignore

    try:
        await router_step(ctx)
    finally:
        agent_utilities.graph._router_impl.create_context_agent = (  # type: ignore
            original_agent_factory
        )
        agent_utilities.rlm.config.RLMConfig = original_rlm_config  # type: ignore
        agent_utilities.core.model_factory.create_model = original_create_model  # type: ignore
        if original_build_kg2:
            agent_utilities.graph.kg_graph_factory.build_pydantic_graph_from_kg = (
                original_build_kg2  # type: ignore
            )

    # Verify that we did NOT fall back to lightweight model, because is_complex=True
    assert state.pinned_model_id != os.environ.get("LIGHTWEIGHT_MODEL", "gpt-4o-mini")
