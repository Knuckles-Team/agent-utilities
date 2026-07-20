"""CONCEPT:AU-KG.compute.spectral-cluster-navigator"""

import pytest

from agent_utilities.knowledge_graph.core.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


@pytest.fixture
def base_graph():
    G = GraphComputeEngine(backend_type="rust", graph_name="test_analogy_base")
    if G._client:
        try:
            G._client.create_graph("test_analogy_base")
        except Exception:
            pass
        G._client.clear()

    # A small subgraph pattern (e.g. A -> B)
    node_a = RegistryNode(
        id="base_A",
        name="Base A",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[1.0, 0.0, 0.0],
    )
    node_b = RegistryNode(
        id="base_B",
        name="Base B",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 1.0, 0.0],
    )

    G.add_node("base_A", data=node_a)
    G.add_node("base_B", data=node_b)
    G.add_edge("base_A", "base_B", type="DEPENDS_ON")

    # Add a non-matching node structure
    node_c = RegistryNode(
        id="base_C",
        name="Base C",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 0.0, 1.0],
    )
    G.add_node("base_C", data=node_c)

    return G


def test_find_analogous_subgraphs(base_graph):
    engine = TopologicalAnalogyEngine(base_graph)

    # Create a target subgraph that is structurally isomorphic and semantically similar
    target_G = GraphComputeEngine(
        backend_type="rust", graph_name="test_analogy_target_1"
    )
    if target_G._client:
        try:
            target_G._client.create_graph("test_analogy_target_1")
        except Exception:
            pass
        target_G._client.clear()

    target_a = RegistryNode(
        id="target_A",
        name="Target A",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.9, 0.1, 0.0],  # Very similar to Base A
    )
    target_b = RegistryNode(
        id="target_B",
        name="Target B",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.1, 0.9, 0.0],  # Very similar to Base B
    )

    target_G.add_node("target_A", data=target_a)
    target_G.add_node("target_B", data=target_b)
    target_G.add_edge("target_A", "target_B", type="DEPENDS_ON")

    matches = engine.find_analogous_subgraphs(target_G, threshold=0.8)

    assert len(matches) == 1
    assert matches[0].name == "Analogy: Target A ≈ Base A"
    assert matches[0].matched_nodes == 2
    assert matches[0].similarity_score >= 0.89


def test_no_matches_due_to_semantic_difference(base_graph):
    engine = TopologicalAnalogyEngine(base_graph)

    # Create target subgraph that is structurally isomorphic but semantically different
    target_G = GraphComputeEngine(
        backend_type="rust", graph_name="test_analogy_target_2"
    )
    if target_G._client:
        try:
            target_G._client.create_graph("test_analogy_target_2")
        except Exception:
            pass
        target_G._client.clear()
    target_a = RegistryNode(
        id="target_A",
        name="Target A",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 0.0, 1.0],  # Dissimilar to Base A
    )
    target_b = RegistryNode(
        id="target_B",
        name="Target B",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.1, 0.9, 0.0],
    )
    target_G.add_node("target_A", data=target_a)
    target_G.add_node("target_B", data=target_b)
    target_G.add_edge("target_A", "target_B", type="DEPENDS_ON")

    matches = engine.find_analogous_subgraphs(target_G, threshold=0.8)

    assert len(matches) == 0
