import pytest
import networkx as nx
from agent_utilities.knowledge_graph.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


@pytest.fixture
def base_graph():
    G = nx.MultiDiGraph()
    # Add a node that will be matched
    node_data = RegistryNode(
        id="match_target",
        name="Match Target",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[1.0, 0.0, 0.0]
    )
    G.add_node("match_target", data=node_data)

    # Add some other node
    other_data = RegistryNode(
        id="other",
        name="Other",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 1.0, 0.0]
    )
    G.add_node("other", data=other_data)
    return G


def test_find_analogous_subgraphs(base_graph):
    engine = TopologicalAnalogyEngine(base_graph)

    target_G = nx.MultiDiGraph()
    target_node = RegistryNode(
        id="query_node",
        name="Query Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[1.0, 0.0, 0.0]
    )
    target_G.add_node("query_node", data=target_node)

    matches = engine.find_analogous_subgraphs(target_G, threshold=0.9)

    assert len(matches) == 1
    assert matches[0].name == "Analogy: Query Node ≈ Match Target"
    assert matches[0].similarity_score >= 0.99

def test_no_matches(base_graph):
    engine = TopologicalAnalogyEngine(base_graph)

    target_G = nx.MultiDiGraph()
    target_node = RegistryNode(
        id="query_node",
        name="Query Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 0.0, 1.0]
    )
    target_G.add_node("query_node", data=target_node)

    matches = engine.find_analogous_subgraphs(target_G, threshold=0.9)

    assert len(matches) == 0
