"""CONCEPT:AU-KG.compute.spectral-cluster-navigator"""

import pytest

from agent_utilities.knowledge_graph.core.semantic_subsumption import (
    SemanticSubsumptionEngine,
)
from agent_utilities.models.knowledge_graph import RegistryNode, RegistryNodeType


@pytest.fixture
def owl_classes():
    return {
        "http://example.org/ontology#Person": [1.0, 0.0, 0.0],
        "http://example.org/ontology#Organization": [0.0, 1.0, 0.0],
    }


@pytest.fixture
def owl_hierarchy():
    return {
        "http://example.org/ontology#Person": [
            "http://example.org/ontology#Agent",
            "http://example.org/ontology#Entity",
        ],
        "http://example.org/ontology#Agent": ["http://example.org/ontology#Entity"],
    }


def test_align_node_to_ontology_success(owl_classes, owl_hierarchy):
    engine = SemanticSubsumptionEngine(owl_classes, owl_hierarchy=owl_hierarchy)

    node = RegistryNode(
        id="test_node",
        name="Test Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.9, 0.1, 0.0],
    )

    alignment = engine.align_node_to_ontology(node, threshold=0.85)

    assert alignment is not None
    assert alignment.inferred_parent_class == "http://example.org/ontology#Person"
    assert alignment.inferred_lineage == [
        "http://example.org/ontology#Person",
        "http://example.org/ontology#Agent",
        "http://example.org/ontology#Entity",
    ]
    assert alignment.confidence > 0.85


def test_align_node_to_ontology_fail(owl_classes):
    engine = SemanticSubsumptionEngine(owl_classes)

    node = RegistryNode(
        id="test_node",
        name="Test Node",
        type=RegistryNodeType.TOOL_METADATA,
        embedding=[0.0, 0.0, 1.0],
    )

    alignment = engine.align_node_to_ontology(node, threshold=0.85)

    assert alignment is None
