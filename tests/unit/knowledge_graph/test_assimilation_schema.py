#!/usr/bin/python
"""Schema foundation for the graph-native assimilation engine (VU-1).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

import pytest

from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
    SDDFeatureNode,
)

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


def test_sdd_feature_node_type_registered():
    assert RegistryNodeType.SDD_FEATURE.value == "sdd_feature"


def test_assimilation_edges_registered_with_live_labels():
    # UPPER_SNAKE values must match the live Cypher labels the subsystem writes.
    assert RegistryEdgeType.ADDRESSES == "ADDRESSES"
    assert RegistryEdgeType.ADDRESSED_BY == "ADDRESSED_BY"
    assert RegistryEdgeType.RELEVANCE_SCORED == "RELEVANCE_SCORED"
    assert RegistryEdgeType.ASSIMILATED_INTO == "ASSIMILATED_INTO"
    assert RegistryEdgeType.DERIVED_FROM_RESEARCH == "DERIVED_FROM_RESEARCH"
    assert RegistryEdgeType.SATISFIED_BY == "SATISFIED_BY"


def test_reused_edges_present():
    # Dedup/synergy/supersede reuse existing edge vocabulary (no sprawl).
    assert RegistryEdgeType.SIMILAR_TO  # KG-2.3 dedup similarity
    assert RegistryEdgeType.HAS_SYNERGY_WITH  # KG-2.4 synergy
    assert RegistryEdgeType.SUPERSEDES  # supersession


def test_relevance_scored_label_matches_existing_graph():
    # Regression guard: the relevance sweep wrote "RELEVANCE_SCORED" as a literal
    # before VU-1; the registered enum value MUST equal it so edges don't split.
    assert RegistryEdgeType.RELEVANCE_SCORED.value == "RELEVANCE_SCORED"


def test_sdd_feature_node_roundtrip():
    node = SDDFeatureNode(
        id="sddf:1",
        name="executable-rag planner",
        sdd_feature_id="b2-03-planner",
        concept_ids=["AU-KG.retrieval.memory-first-retrieval"],
        research_sources=["paper:pyrag"],
        sdd_path=".specify/specs/...",
        codebase="agent-utilities",
    )
    assert node.type == RegistryNodeType.SDD_FEATURE
    assert node.status == "open"  # lifecycle default
    dumped = node.model_dump()
    assert dumped["type"] == "sdd_feature"
    assert dumped["sdd_feature_id"] == "b2-03-planner"
    assert dumped["concept_ids"] == ["AU-KG.retrieval.memory-first-retrieval"]
