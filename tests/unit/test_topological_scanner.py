"""CONCEPT:AU-KG.compute.spectral-cluster-navigator"""

import uuid

import pytest

from agent_utilities.knowledge_graph.core.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType
from agent_utilities.security.threat_defense_engine import TopologicalScanner


@pytest.fixture
def analogy_engine():
    graph_name = f"test_analogy_{uuid.uuid4().hex}"
    G = GraphComputeEngine(graph_name=graph_name, backend_type="rust")
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
    graph_name = f"test_risk_{uuid.uuid4().hex}"
    risk_G = GraphComputeEngine(graph_name=graph_name, backend_type="rust")
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
