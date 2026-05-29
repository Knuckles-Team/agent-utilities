from __future__ import annotations

"""Tests for CONCEPT:KG-2.6 — Risk Scoring Ontology Extension."""


import pytest

from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
    RiskAssessmentNode,
    RiskFactorNode,
    RiskMitigationNode,
)


class TestRiskAssessmentNode:
    def test_creation_defaults(self):
        node = RiskAssessmentNode(id="ra:001", name="AAPL Risk")
        assert node.type == RegistryNodeType.RISK_ASSESSMENT
        assert node.overall_risk_score == 0.0
        assert node.risk_level == "low"

    def test_full_creation(self):
        node = RiskAssessmentNode(
            id="ra:002",
            name="Portfolio Risk",
            entity_id="port:main",
            overall_risk_score=0.75,
            risk_level="high",
            assessment_type="composite",
            assessor="risk_agent",
        )
        assert node.overall_risk_score == 0.75
        assert node.risk_level == "high"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            RiskAssessmentNode(id="ra:bad", name="Bad", overall_risk_score=1.5)

    def test_serialization(self):
        node = RiskAssessmentNode(id="ra:003", name="Test", overall_risk_score=0.5)
        data = node.model_dump()
        restored = RiskAssessmentNode.model_validate(data)
        assert restored.overall_risk_score == 0.5


class TestRiskFactorNode:
    def test_creation(self):
        node = RiskFactorNode(
            id="rf:001",
            name="Market Risk",
            factor_type="market",
            severity=0.6,
            probability=0.4,
            impact=0.8,
        )
        assert node.type == RegistryNodeType.RISK_FACTOR
        assert node.factor_type == "market"
        assert node.severity == 0.6

    def test_mitigation_status(self):
        node = RiskFactorNode(
            id="rf:002",
            name="Credit Risk",
            factor_type="credit",
            mitigation_status="partial",
        )
        assert node.mitigation_status == "partial"

    def test_bounds(self):
        with pytest.raises(Exception):
            RiskFactorNode(id="rf:bad", name="Bad", severity=2.0)


class TestRiskMitigationNode:
    def test_creation(self):
        node = RiskMitigationNode(
            id="rm:001",
            name="Hedge with Puts",
            mitigation_type="hedge",
            effectiveness=0.8,
            cost=5000.0,
            status="active",
        )
        assert node.type == RegistryNodeType.RISK_MITIGATION
        assert node.mitigation_type == "hedge"
        assert node.effectiveness == 0.8
        assert node.status == "active"


class TestRiskEdgeTypes:
    def test_edge_types_exist(self):
        assert RegistryEdgeType.ASSESSED_RISK == "assessed_risk"
        assert RegistryEdgeType.HAS_RISK_FACTOR == "has_risk_factor"
        assert RegistryEdgeType.MITIGATED_BY == "mitigated_by"
        assert RegistryEdgeType.PROPAGATES_RISK_TO == "propagates_risk_to"


class TestRiskPropagationGraph:
    """Test transitive risk propagation via graph traversal."""

    def test_risk_chain(self):
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        g = GraphComputeEngine(backend_type="rust")

        # Create entities with risk assessments
        ra_a = RiskAssessmentNode(
            id="ra:assetA", name="Asset A Risk", overall_risk_score=0.8
        )
        ra_b = RiskAssessmentNode(
            id="ra:sectorB", name="Sector B Risk", overall_risk_score=0.6
        )
        rf = RiskFactorNode(
            id="rf:market", name="Market Risk", factor_type="market", severity=0.7
        )
        rm = RiskMitigationNode(
            id="rm:hedge", name="Hedge", mitigation_type="hedge", effectiveness=0.9
        )

        for n in [ra_a, ra_b, rf, rm]:
            g.add_node(n.id, **n.model_dump())

        # Build risk chain
        g.add_edge(ra_a.id, rf.id, type=RegistryEdgeType.HAS_RISK_FACTOR)
        g.add_edge(rf.id, rm.id, type=RegistryEdgeType.MITIGATED_BY)
        g.add_edge(ra_a.id, ra_b.id, type=RegistryEdgeType.PROPAGATES_RISK_TO)

        # Verify chain
        assert g.has_edge(ra_a.id, ra_b.id)
        assert g.number_of_edges() == 3

        # Verify traversal from assessment to mitigation
        path = g.get_shortest_path(ra_a.id, rm.id)
        assert path is not None
        assert len(path) == 3

    def test_multi_factor_assessment(self):
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        g = GraphComputeEngine(backend_type="rust")

        ra = RiskAssessmentNode(
            id="ra:port", name="Portfolio Risk", overall_risk_score=0.65
        )
        factors = [
            RiskFactorNode(
                id="rf:market", name="Market", factor_type="market", severity=0.8
            ),
            RiskFactorNode(
                id="rf:credit", name="Credit", factor_type="credit", severity=0.5
            ),
            RiskFactorNode(
                id="rf:liquidity",
                name="Liquidity",
                factor_type="liquidity",
                severity=0.3,
            ),
        ]

        g.add_node(ra.id, **ra.model_dump())
        for f in factors:
            g.add_node(f.id, **f.model_dump())
            g.add_edge(ra.id, f.id, type=RegistryEdgeType.HAS_RISK_FACTOR)

        # Verify all factors linked
        assert g.out_degree(ra.id) == 3

        # Verify factor severity retrieval
        severities = []
        for _, target, data in g.out_edges(ra.id, data=True):
            node_data = g.nodes[target]
            severities.append(node_data.get("severity", 0.0))
        assert max(severities) == 0.8
