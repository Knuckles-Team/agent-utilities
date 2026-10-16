from __future__ import annotations

"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Risk Scoring Ontology Extension."""


from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNode,
)


def _kg_dump(node: RegistryNode) -> dict:
    """Translate a RegistryNode's Pydantic dump to GraphComputeEngine's canonical
    node property shape: the semantic ``type`` field becomes ``node_type``
    (mirrors ``agent_utilities.knowledge_graph.kb.ingestion._canonical_node_dump``
    — ``GraphComputeEngine.add_node`` retired accepting ``type`` directly)."""
    data = node.model_dump()
    node_type = data.pop("type")
    data["node_type"] = getattr(node_type, "value", node_type)
    return data


class TestRiskEdgeTypes:
    def test_edge_types_exist(self):
        assert RegistryEdgeType.ASSESSED_RISK == "assessed_risk"
        assert RegistryEdgeType.HAS_RISK_FACTOR == "has_risk_factor"
        assert RegistryEdgeType.MITIGATED_BY == "mitigated_by"
        assert RegistryEdgeType.PROPAGATES_RISK_TO == "propagates_risk_to"
