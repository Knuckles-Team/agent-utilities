from __future__ import annotations

"""Biomedical Schema Pack — clinical/healthcare domain profile.

Optimized for biomedical agents managing medical entities, procedures,
conditions, regulations, and evidence chains. Includes SNOMED-CT
OWL extension stub.
"""


from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class BiomedicalSchemaPack(SchemaPack):
    """Biomedical domain pack: medical entities, procedures, regulations."""

    name: str = "biomedical"
    description: str = (
        "Clinical/healthcare research profile. Activates medical entity, "
        "procedure, regulation, and evidence types. Suitable for clinical "
        "data management, treatment protocol tracking, and regulatory compliance."
    )
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    node_types: set[RegistryNodeType] = {
        RegistryNodeType.MEDICAL_ENTITY,
        RegistryNodeType.PROCEDURE,
        RegistryNodeType.REGULATION,
        RegistryNodeType.DOCUMENT,
        RegistryNodeType.EVIDENCE,
        RegistryNodeType.DATASET,
        RegistryNodeType.HYPOTHESIS,
        RegistryNodeType.OBSERVATION,
        RegistryNodeType.PERSON,
        RegistryNodeType.ORGANIZATION,
        RegistryNodeType.SOURCE,
        RegistryNodeType.PLACE,
    }
    edge_types: set[RegistryEdgeType] = {
        RegistryEdgeType.CITES_SOURCE,
        RegistryEdgeType.SUPPORTS_BELIEF,
        RegistryEdgeType.CONTRADICTS_BELIEF,
        RegistryEdgeType.WAS_DERIVED_FROM,
        RegistryEdgeType.WAS_ATTRIBUTED_TO,
        RegistryEdgeType.BROADER,
        RegistryEdgeType.NARROWER,
        RegistryEdgeType.EXACT_MATCH,
        RegistryEdgeType.OCCURRED_AT_PLACE,
        RegistryEdgeType.RESULTED_IN,
    }
    retrieval_boosts: dict[str, float] = {
        "cites_source": 1.4,
        "supports_belief": 1.3,
        "exact_match": 1.5,
        "broader": 1.2,
    }
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.GLOBAL
    backlink_boost_factor: float = 0.1
    owl_extensions: list[str] = []  # Future: ["snomed-ct.ttl", "mesh-headings.ttl"]
