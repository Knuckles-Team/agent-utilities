from __future__ import annotations

"""Research Schema Pack — academic/scientific domain profile.

Optimized for research agents managing papers, claims, methods,
datasets, evidence, and hypotheses. Uses CONTEXT_ONLY backlink
boost to preserve discovery of novel/low-citation papers.
"""


from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class ResearchSchemaPack(SchemaPack):
    """Research domain pack: papers, claims, methods, datasets, evidence."""

    name: str = "research-state"
    description: str = (
        "Academic/scientific research profile. Activates hypothesis, dataset, "
        "evidence, and document types with CITES/SUPPORTS retrieval boosts. "
        "Uses CONTEXT_ONLY backlink strategy to preserve novel paper discovery."
    )
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    node_types: set[RegistryNodeType] = {
        RegistryNodeType.HYPOTHESIS,
        RegistryNodeType.BELIEF,
        RegistryNodeType.DATASET,
        RegistryNodeType.DOCUMENT,
        RegistryNodeType.CREATIVE_WORK,
        RegistryNodeType.EVIDENCE,
        RegistryNodeType.SOURCE,
        RegistryNodeType.KNOWLEDGE_BASE,
        RegistryNodeType.KNOWLEDGE_BASE_TOPIC,
        RegistryNodeType.EXPERIMENT,
        RegistryNodeType.OBSERVATION,
        RegistryNodeType.PRINCIPLE,
        RegistryNodeType.SOFTWARE_PROJECT,
    }
    edge_types: set[RegistryEdgeType] = {
        RegistryEdgeType.CITES_SOURCE,
        RegistryEdgeType.SUPPORTS_BELIEF,
        RegistryEdgeType.CONTRADICTS_BELIEF,
        RegistryEdgeType.WAS_GENERATED_BY,
        RegistryEdgeType.BROADER,
        RegistryEdgeType.NARROWER,
        RegistryEdgeType.RELATED_CONCEPT,
        RegistryEdgeType.EXACT_MATCH,
        RegistryEdgeType.TESTS_HYPOTHESIS,
        RegistryEdgeType.EXPLORED_GAP,
        RegistryEdgeType.RESULTED_IN_DISCOVERY,
    }
    retrieval_boosts: dict[str, float] = {
        "cites_source": 1.5,
        "supports_belief": 1.3,
        "contradicts_belief": 1.2,
        "broader": 1.1,
        "narrower": 1.1,
        "tests_hypothesis": 1.4,
    }
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.CONTEXT_ONLY
    backlink_boost_factor: float = 0.15
