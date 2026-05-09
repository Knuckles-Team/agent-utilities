#!/usr/bin/python
"""Tests for CONCEPT:KG-2.2 — Entity-Claim Extraction and MAGMA Epistemic View."""

import pytest
import networkx as nx

from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
    EntityClaimExtractor,
    ExtractionResult,
    extract_deterministic,
)
from agent_utilities.models.knowledge_graph import (
    ClaimNode,
    RegistryEdgeType,
    RegistryNodeType,
)


class TestDeterministicExtraction:
    """Test Phase 1: regex-based extraction without LLM."""

    def test_extracts_citations(self):
        """Should extract academic citation patterns."""
        content = (
            "The study (Smith et al., 2023) demonstrated that "
            "neural networks (Johnson, 2021) outperform baselines."
        )
        result = extract_deterministic(content, "doc:test")

        entities = [e for e in result.entities if e.entity_type == "paper"]
        assert len(entities) >= 1
        names = [e.name for e in entities]
        assert any("Smith" in n or "Johnson" in n for n in names)

    def test_extracts_wikilinks(self):
        """Should extract [[wikilink]] targets as concept entities."""
        content = "See [[Knowledge Graph]] for details and [[Agent OS]] overview."
        result = extract_deterministic(content, "doc:test")

        entities = [e for e in result.entities if e.entity_type == "concept"]
        names = [e.name for e in entities]
        assert "Knowledge Graph" in names
        assert "Agent OS" in names

    def test_extracts_assertions(self):
        """Should extract strong-language claims."""
        content = (
            "The system must validate all incoming data before persistence. "
            "We recommend using Pydantic models for structured validation. "
            "Therefore, the architecture should enforce typed schemas at every boundary."
        )
        result = extract_deterministic(content, "doc:test")

        assert len(result.claims) >= 1
        claim_types = [c.claim_type for c in result.claims]
        assert any(ct in ("decision", "thesis", "finding") for ct in claim_types)

    def test_generates_relationships(self):
        """Citations should produce 'cites' relationships."""
        content = "As shown by (Johnson, 2022), this is important."
        result = extract_deterministic(content, "doc:test")

        cites_rels = [r for r in result.relationships if r.relationship_type == "cites"]
        assert len(cites_rels) >= 1

    def test_empty_content(self):
        """Empty content should produce empty results."""
        result = extract_deterministic("", "doc:empty")
        assert len(result.entities) == 0
        assert len(result.claims) == 0
        assert len(result.relationships) == 0


class TestClaimNode:
    """Test ClaimNode model."""

    def test_create_claim_node(self):
        """Should create a valid ClaimNode with all fields."""
        claim = ClaimNode(
            id="claim:test01",
            name="Test claim",
            claim_text="Knowledge graphs improve agent reasoning",
            confidence=0.85,
            claim_type="thesis",
            source_ids=["doc:paper1"],
            domain="ai",
        )
        assert claim.type == RegistryNodeType.CLAIM
        assert claim.confidence == 0.85
        assert claim.claim_type == "thesis"

    def test_confidence_clamping(self):
        """Confidence should be bounded [0.0, 1.0]."""
        with pytest.raises(Exception):  # Pydantic validation
            ClaimNode(
                id="claim:bad",
                name="Bad",
                claim_text="Invalid",
                confidence=1.5,
            )


class TestEntityClaimExtractor:
    """Test the full extractor with KG persistence."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for testing."""

        class MockEngine:
            def __init__(self):
                self.graph = nx.MultiDiGraph()
                self.backend = None

            def _serialize_node(self, node, label=""):
                return node.model_dump()

            def _upsert_node(self, label, node_id, data):
                pass

            def link_nodes(self, source, target, edge_type, metadata=None):
                self.graph.add_edge(
                    source,
                    target,
                    type=edge_type.value if hasattr(edge_type, "value") else edge_type,
                    **(metadata or {}),
                )

        return MockEngine()

    def test_extract_and_persist(self, mock_engine):
        """Should extract entities and persist them to the graph."""
        # Add a source node
        mock_engine.graph.add_node("doc:test", type="article", name="Test Doc")

        extractor = EntityClaimExtractor(mock_engine)
        content = (
            "See [[Inference Engine]] for details. "
            "The system must enforce schema validation at ingestion time."
        )
        result = extractor.extract_and_persist(
            content, source_id="doc:test", domain="architecture"
        )

        # Should have extracted at least one entity
        assert len(result.entities) >= 1

        # Entities should be in the graph
        entity_nodes = [
            n
            for n, d in mock_engine.graph.nodes(data=True)
            if str(d.get("type", "")).lower() == "entity"
        ]
        assert len(entity_nodes) >= 1

    def test_claims_persisted_to_graph(self, mock_engine):
        """Claims should be persisted as ClaimNodes in the graph."""
        mock_engine.graph.add_node("doc:claims", type="article", name="Claims Doc")

        extractor = EntityClaimExtractor(mock_engine)
        content = (
            "We recommend implementing tiered validation for all graph data. "
            "Therefore, the system should auto-fix recoverable issues before flagging errors."
        )
        result = extractor.extract_and_persist(
            content, source_id="doc:claims", domain="validation"
        )

        # Claims should be in the graph
        claim_nodes = [
            n
            for n, d in mock_engine.graph.nodes(data=True)
            if str(d.get("type", "")).lower() == "claim"
        ]
        assert len(claim_nodes) >= 1


class TestEdgeTypes:
    """Test new CONCEPT:KG-2.2 edge types exist in the registry."""

    def test_builds_on_exists(self):
        assert RegistryEdgeType.BUILDS_ON == "builds_on"

    def test_exemplifies_exists(self):
        assert RegistryEdgeType.EXEMPLIFIES == "exemplifies"

    def test_authored_by_exists(self):
        assert RegistryEdgeType.AUTHORED_BY == "authored_by"

    def test_contradicts_exists(self):
        """CONTRADICTS should already exist from KG V1."""
        assert RegistryEdgeType.CONTRADICTS == "contradicts"

    def test_cites_exists(self):
        """CITES should already exist from KB edges."""
        assert RegistryEdgeType.CITES == "cites"
