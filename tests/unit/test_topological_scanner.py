"""CONCEPT:AU-KG.compute.spectral-cluster-navigator"""

from contextlib import contextmanager

import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.knowledge_graph.core.analogy_engine import TopologicalAnalogyEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType
from agent_utilities.security.threat_defense_engine import TopologicalScanner


@contextmanager
def _isolated_graph(graph_name: str):
    """Own one explicit graph under matching verified test authority.

    Mirrors ``tests/unit/knowledge_graph/test_topological_analogy.py``'s helper
    of the same name: a bare second ``GraphComputeEngine(graph_name=...)`` call
    within one test raises ("A process graph transport already exists"), and
    writes route by the AMBIENT ``GraphSession``'s graph, not by the
    constructor's ``graph_name`` (see ``tests/conftest.py``'s
    ``isolate_graph_compute_engine`` docstring). ``get_or_create``/``for_graph``
    is the sanctioned way to reach a second named graph, and its view
    deliberately performs no implicit tenant creation, so this helper creates
    the tenant explicitly under a session already scoped to it.
    """
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
def analogy_engine(isolate_graph_compute_engine):
    graph_name = f"{isolate_graph_compute_engine}_analogy"
    with _isolated_graph(graph_name) as G:
        # Add a vulnerable-looking node to the main graph
        node_data = RegistryNode(
            id="exec_node",
            name="Execution Node",
            type=RegistryNodeType.TOOL_METADATA,
            embedding=[1.0, 0.0, 0.0],
        )
        G.add_node("exec_node", data=node_data)

        yield TopologicalAnalogyEngine(G)


@pytest.fixture
def known_risk_topologies(isolate_graph_compute_engine):
    graph_name = f"{isolate_graph_compute_engine}_risk"
    with _isolated_graph(graph_name) as risk_G:
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

        yield [risk_G]


def test_scan_execution_graph(analogy_engine, known_risk_topologies):
    scanner = TopologicalScanner(analogy_engine, known_risk_topologies)

    vulnerabilities = scanner.scan_execution_graph()

    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].vulnerability_type == "untrusted_data_flow"
    assert vulnerabilities[0].severity == "high"
