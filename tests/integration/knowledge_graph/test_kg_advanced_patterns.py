from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.hybrid_retriever import HybridRetriever
from agent_utilities.knowledge_graph.maintainer import GraphMaintainer


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global engine and backend state before and after each test."""
    IntelligenceGraphEngine.set_active(None)
    set_active_backend(None)
    yield
    IntelligenceGraphEngine.set_active(None)
    set_active_backend(None)

@pytest.fixture
def memory_engine():
    graph = nx.MultiDiGraph()
    # Populate with some base nodes
    graph.add_node("entity:agent_a", name="Agent A", type="Agent")
    graph.add_node("entity:tool_x", name="Tool X", type="Tool")
    graph.add_node("entity:agent_b", name="Agent B", type="Agent")

    # Tool usages
    graph.add_edge("entity:agent_a", "entity:tool_x", type="USES")
    graph.add_edge("entity:agent_b", "entity:tool_x", type="USES")

    # Dependencies
    graph.add_node("entity:module_1", name="Module 1", type="Module")
    graph.add_node("entity:module_2", name="Module 2", type="Module")
    graph.add_node("entity:module_3", name="Module 3", type="Module")
    graph.add_edge("entity:module_1", "entity:module_2", type="DEPENDS_ON")
    graph.add_edge("entity:module_2", "entity:module_3", type="DEPENDS_ON")

    engine = IntelligenceGraphEngine(graph=graph)
    return engine

def test_inference_engine_fallback(memory_engine):
    """Test topological inference in NetworkX fallback mode."""
    inf_engine = memory_engine.inference_engine

    # Run inference
    new_facts = inf_engine.run_inference()

    assert new_facts > 0
    # module_1 should now depend indirectly on module_3
    assert memory_engine.graph.has_edge("entity:module_1", "entity:module_3")
    edge_data = memory_engine.graph.get_edge_data("entity:module_1", "entity:module_3")[0]
    assert edge_data["type"] == "DEPENDS_ON_INDIRECT"
    assert edge_data["inferred"] is True

@patch("agent_utilities.knowledge_graph.hybrid_retriever.create_embedding_model")
def test_hybrid_retriever_fallback(mock_create_model, memory_engine):
    """Test HybridRetriever fallback when vector backend isn't available."""
    # Setup mock to avoid hitting real APIs
    mock_model = MagicMock()
    mock_model.get_text_embedding.return_value = [0.1] * 768
    mock_create_model.return_value = mock_model

    # Initialize retriever
    retriever = HybridRetriever(memory_engine)

    # Search for Agent A
    results = retriever.retrieve_hybrid("Agent A", context_window=1, multi_hop_depth=1)

    # Should find Agent A and its neighbor Tool X
    result_ids = [r["id"] for r in results]
    assert "entity:agent_a" in result_ids
    assert "entity:tool_x" in result_ids
    assert "entity:agent_b" not in result_ids # Too far (2 hops)

@patch("agent_utilities.core.model_factory.create_model")
@patch("pydantic_ai.Agent.run_sync")
def test_consolidate_memory_llm(mock_run_sync, mock_create_model, memory_engine):
    """Test memory consolidation using the LLM judge."""
    mock_run_sync.return_value.data = "This is a consolidated summary."

    # Mock backend to simulate matching episodes
    mock_backend = MagicMock()
    mock_backend.execute.side_effect = lambda query, params=None: [
        {"id": "mem1", "description": "Episode 1"},
        {"id": "mem2", "description": "Episode 2"}
    ] if "MATCH (e:Episode)" in query else []
    memory_engine.backend = mock_backend

    maintainer = GraphMaintainer(memory_engine)

    # Consolidate
    processed = maintainer.consolidate_memory()

    assert processed == 2
    # Check if backend create was called for the summary
    create_calls = [c for c in mock_backend.execute.call_args_list if "CREATE (s:ChatSummary" in c[0][0]]
    assert len(create_calls) == 1
    assert create_calls[0][0][1]["text"] == "This is a consolidated summary."
