"""CONCEPT:AU-KG.compute.spectral-cluster-navigator"""

from contextlib import contextmanager

import pytest

from agent_utilities.knowledge_graph.core.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


@contextmanager
def _isolated_graph(graph_name: str):
    """Own one explicit graph under matching verified test authority."""
    session = GraphSession.from_ambient().with_graph(graph_name)
    with use_session(session):
        graph = GraphComputeEngine.get_or_create(
            backend_type="rust", graph_name=graph_name
        )
        entries = graph._client.tenants.list() or []
        existing = {
            str(entry.get("name") if isinstance(entry, dict) else entry)
            for entry in entries
        }
        if graph_name not in existing:
            graph._client.tenants.create(graph_name)
        try:
            yield graph
        finally:
            if getattr(graph, "_process_root", graph) is not graph:
                graph._client.tenants.delete(graph_name)


@pytest.fixture
def base_graph(isolate_graph_compute_engine):
    graph_name = f"{isolate_graph_compute_engine}_analogy_base"
    with _isolated_graph(graph_name) as G:
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
        G.add_edge("base_A", "base_B", relationship="DEPENDS_ON")

        # Add a non-matching node structure
        node_c = RegistryNode(
            id="base_C",
            name="Base C",
            type=RegistryNodeType.TOOL_METADATA,
            embedding=[0.0, 0.0, 1.0],
        )
        G.add_node("base_C", data=node_c)

        yield G


def test_find_analogous_subgraphs(base_graph, isolate_graph_compute_engine):
    engine = TopologicalAnalogyEngine(base_graph)

    # Create a target subgraph that is structurally isomorphic and semantically similar
    target_name = f"{isolate_graph_compute_engine}_analogy_target_1"
    with _isolated_graph(target_name) as target_G:
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
        target_G.add_edge("target_A", "target_B", relationship="DEPENDS_ON")

        matches = engine.find_analogous_subgraphs(target_G, threshold=0.8)

    assert len(matches) == 1
    assert matches[0].name == "Analogy: Target A ≈ Base A"
    assert matches[0].matched_nodes == 2
    assert matches[0].similarity_score >= 0.89


def test_no_matches_due_to_semantic_difference(
    base_graph, isolate_graph_compute_engine
):
    engine = TopologicalAnalogyEngine(base_graph)

    # Create target subgraph that is structurally isomorphic but semantically different
    target_name = f"{isolate_graph_compute_engine}_analogy_target_2"
    with _isolated_graph(target_name) as target_G:
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
        target_G.add_edge("target_A", "target_B", relationship="DEPENDS_ON")

        matches = engine.find_analogous_subgraphs(target_G, threshold=0.8)

    assert len(matches) == 0
