import pytest
import networkx as nx
from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.knowledge_graph import ExperienceNode
from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (
    LatentTopologicalRAG,
)
from agent_utilities.knowledge_graph.retrieval.single_shot_sira import SingleShotSIRA
from agent_utilities.knowledge_graph.orchestration.voi_budget_controller import (
    VOIBudgetController,
)
from agent_utilities.knowledge_graph.security.cognitive_trap_defense import (
    CognitiveTrapDefense,
)
from agent_utilities.knowledge_graph.adaptation.experience_alignment import (
    ExperienceAlignmentEngine,
)


@pytest.fixture
def mock_engine():
    graph = nx.DiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
    engine.backend = MagicMock()
    return engine


def test_latent_topological_rag(mock_engine):
    rag = LatentTopologicalRAG(mock_engine)
    mock_engine.backend.execute.return_value = [
        {"id": "node1", "name": "Node 1", "score": 0.9}
    ]

    results = rag.retrieve("test query")
    assert len(results) == 1
    assert results[0]["id"] == "node1"
    mock_engine.backend.execute.assert_called_once()


def test_single_shot_sira(mock_engine):
    sira = SingleShotSIRA(mock_engine)
    # Simulate nodes from different clusters
    nodes = [
        {"id": "cluster1:a"},
        {"id": "cluster1:b"},
        {"id": "cluster1:c"},  # Should be dropped
        {"id": "cluster2:a"},
    ]

    aligned = sira.align_context(nodes, max_tokens=1000)
    assert len(aligned) == 3
    assert {"id": "cluster1:c"} not in aligned


def test_voi_budget_controller():
    controller = VOIBudgetController(engine=MagicMock(spec=IntelligenceGraphEngine), base_budget=100)

    # Should continue early on
    assert controller.should_continue_traversal(10) is True

    # Marginal utility should be low as we approach budget
    # utility = 1 - (98/100)^2 = 1 - 0.9604 = 0.0396 < 0.05
    assert controller.should_continue_traversal(98) is False

    # Exhausted budget
    assert controller.should_continue_traversal(100) is False


def test_cognitive_trap_defense(mock_engine):
    defense = CognitiveTrapDefense(mock_engine)
    # Inject a sybil cluster trap into the graph
    mock_engine.graph.add_edge("C", "F1", type="VALIDATES")
    mock_engine.graph.add_edge("C", "F2", type="VALIDATES")
    mock_engine.graph.add_edge("C", "F3", type="VALIDATES")
    mock_engine.graph.add_edge("F1", "F2", type="AGREES_WITH")
    mock_engine.graph.add_edge("F2", "F3", type="AGREES_WITH")
    mock_engine.graph.add_edge("F3", "F1", type="AGREES_WITH")

    traps = defense.scan_for_traps()
    assert len(traps) > 0

    neutralized = defense.neutralize_traps()
    assert neutralized > 0
    # Graph nodes should be removed
    assert "C" not in mock_engine.graph


def test_experience_alignment(mock_engine):
    alignment = ExperienceAlignmentEngine(mock_engine)
    exp = ExperienceNode(
        id="exp1",
        name="Test Exp",
        condition="When task fails",
        action="Retry task",
        importance_score=0.9,
    )

    # Test ingestion uses add_node
    mock_engine = MagicMock()
    mock_engine.add_node = MagicMock()
    alignment = ExperienceAlignmentEngine(mock_engine)
    alignment.ingest_experience(exp)
    mock_engine.add_node.assert_called_once_with(
        node_id=exp.id,
        node_type="Experience",
        properties=exp.model_dump()
    )

    # Test retrieval
    mock_engine.backend.execute.return_value = [
        {"id": "exp1", "name": "Test Exp", "tags": ["retry"], "score": 0.9}
    ]
    results = alignment.retrieve_aligned_experiences(["retry"])
    assert len(results) == 1
    assert results[0].id == "exp1"
