from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType

from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class EvolutionSchemaPack(SchemaPack):
    """Schema pack enabling self-improvement, continuous learning, and evolutionary mechanics.

    CONCEPT:AU-AHE.harness.harness-evolution — Agentic Harness Engineering / Evolution
    """

    name: str = "evolution"
    description: str = "Self-improvement schema including experiences, critiques, and evolutionary nodes."
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    version: str = "1.0.0"

    allowed_node_types: set[RegistryNodeType] = {
        RegistryNodeType.EXPERIENCE,
        RegistryNodeType.OUTCOME_EVALUATION,
        RegistryNodeType.CRITIQUE,
        RegistryNodeType.SELF_EVALUATION,
        RegistryNodeType.EXPERIMENT,
        RegistryNodeType.PROPOSED_SKILL,
        RegistryNodeType.REASONING_TRACE,
        RegistryNodeType.REFLECTION,
        RegistryNodeType.ROUTING_DECISION,
    }

    allowed_edge_types: set[RegistryEdgeType] = {
        RegistryEdgeType.PRODUCED_OUTCOME,
        RegistryEdgeType.SCORED_BY,
        RegistryEdgeType.GENERATED_CRITIQUE,
        RegistryEdgeType.EXPERIENCED_DURING,
        RegistryEdgeType.SUPERSEDES,
        RegistryEdgeType.LED_TO,
        RegistryEdgeType.EVALUATED_WITH,
        RegistryEdgeType.CALIBRATED_AGAINST,
    }

    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.GLOBAL
    boost_multiplier: float = 1.3
