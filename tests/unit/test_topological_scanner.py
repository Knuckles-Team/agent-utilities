import pytest
import networkx as nx

from agent_utilities.knowledge_graph.core.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.security.topological_scanner import TopologicalScanner
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


@pytest.fixture
def analogy_engine():
    G = nx.MultiDiGraph()
    # Add a vulnerable-looking node to the main graph
    node_data = RegistryNode(
        id="exec_node",
        name="Execution Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[1.0, 0.0, 0.0],
    )
    G.add_node("exec_node", data=node_data)

    return TopologicalAnalogyEngine(G)


@pytest.fixture
def known_risk_topologies():
    risk_G = nx.MultiDiGraph()
    risk_G.graph["metadata"] = {
        "vulnerability_type": "untrusted_data_flow",
        "severity": "high",
        "mitigation_strategy": "Sanitize inputs",
    }

    risk_node = RegistryNode(
        id="risk_node",
        name="Risk Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[1.0, 0.0, 0.0],
    )
    risk_G.add_node("risk_node", data=risk_node)

    return [risk_G]


def test_scan_execution_graph(analogy_engine, known_risk_topologies):
    scanner = TopologicalScanner(analogy_engine, known_risk_topologies)

    vulnerabilities = scanner.scan_execution_graph()

    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].vulnerability_type == "untrusted_data_flow"
    assert vulnerabilities[0].severity == "high"
