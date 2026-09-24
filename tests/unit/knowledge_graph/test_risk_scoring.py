from __future__ import annotations

"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Risk Scoring Ontology Extension."""


from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
)


class TestRiskEdgeTypes:
    def test_edge_types_exist(self):
        assert RegistryEdgeType.ASSESSED_RISK == "assessed_risk"
        assert RegistryEdgeType.HAS_RISK_FACTOR == "has_risk_factor"
        assert RegistryEdgeType.MITIGATED_BY == "mitigated_by"
        assert RegistryEdgeType.PROPAGATES_RISK_TO == "propagates_risk_to"
