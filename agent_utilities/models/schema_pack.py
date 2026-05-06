#!/usr/bin/python
"""Schema Pack Models — Domain-configurable KG profiles.

CONCEPT:KG-2.2 — Schema Packs

A SchemaPack defines a domain-specific subset of the Knowledge Graph ontology,
allowing agents to focus on the node/edge types relevant to their domain
(research, biomedical, finance, etc.) without loading the full 90+ type catalog.

Inspired by the gbrain schema-pack proposal (garrytan/gbrain#587) and adapted
for the agent-utilities OWL-backed architecture.

Two operating modes:
  - **ADDITIVE** (default): Pack types are merged *on top* of the core set.
  - **EXCLUSIVE**: Only the pack's types (plus a protected core) are active.

Usage::

    from agent_utilities.models.schema_pack import SchemaPack, SchemaPackMode

    pack = SchemaPack(
        name="research-state",
        mode=SchemaPackMode.ADDITIVE,
        node_types={RegistryNodeType.HYPOTHESIS, RegistryNodeType.DATASET},
        edge_types={RegistryEdgeType.CITES_SOURCE, RegistryEdgeType.SUPPORTS_BELIEF},
    )
    active = pack.get_active_node_types()  # core + research types

See docs/knowledge-graph.md §Schema Packs for the full architecture guide.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field

from .knowledge_graph import RegistryEdgeType, RegistryNodeType


class SchemaPackMode(StrEnum):
    """Operating mode for a Schema Pack.

    - ADDITIVE: Pack types are layered on top of the always-present core set.
    - EXCLUSIVE: Only the pack's declared types (plus the protected core) are active.
    """

    ADDITIVE = "additive"
    EXCLUSIVE = "exclusive"


class BacklinkBoostStrategy(StrEnum):
    """Strategy for applying backlink-density retrieval weighting (CONCEPT:KG-2.2).

    - GLOBAL: Boost is applied to all scored nodes during hybrid search.
    - CONTEXT_ONLY: Boost is applied only during context assembly (multi-hop).
    - DISABLED: No backlink boost is applied.
    """

    GLOBAL = "global"
    CONTEXT_ONLY = "context_only"
    DISABLED = "disabled"


class SchemaPack(BaseModel):
    """A domain-specific Knowledge Graph configuration profile.

    CONCEPT:KG-2.2 — Schema Packs

    Defines which node types, edge types, retrieval boosts, inference rules,
    and OWL extensions are active for a given workspace or deployment.

    The ``CORE_NODE_TYPES`` and ``CORE_EDGE_TYPES`` are *always* included
    regardless of mode, ensuring fundamental agent operations (memory,
    episodes, identity) cannot be accidentally disabled.

    Attributes:
        name: Unique identifier for the pack (e.g. ``research-state``).
        description: Human-readable summary of the pack's purpose.
        mode: ``ADDITIVE`` or ``EXCLUSIVE`` — see ``SchemaPackMode``.
        node_types: Set of ``RegistryNodeType`` members this pack activates.
        edge_types: Set of ``RegistryEdgeType`` members this pack activates.
        retrieval_boosts: Per-edge-type scoring multipliers applied during
            hybrid retrieval. Values > 1.0 increase relevance; < 1.0 decrease.
        backlink_boost_strategy: How backlink-density scoring is applied
            (CONCEPT:KG-2.2). Defaults to ``GLOBAL``.
        backlink_boost_factor: Logarithmic scaling coefficient for the
            backlink boost. Higher values amplify the effect of inbound edges.
        owl_extensions: Paths to additional ``.ttl`` files to load into the
            OWL reasoner when this pack is active.
        inference_rules: Cypher rule templates for domain-specific inference
            (e.g. transitive ``SUPPORTS`` chains).
    """

    # Protected core types — always included regardless of mode
    CORE_NODE_TYPES: ClassVar[frozenset[RegistryNodeType]] = frozenset(
        {
            RegistryNodeType.MEMORY,
            RegistryNodeType.EPISODE,
            RegistryNodeType.PERSON,
            RegistryNodeType.CONCEPT,
            RegistryNodeType.FACT,
            RegistryNodeType.ENTITY,
            RegistryNodeType.AGENT,
            RegistryNodeType.TOOL,
            RegistryNodeType.SKILL,
            RegistryNodeType.CALLABLE_RESOURCE,
            RegistryNodeType.REASONING_TRACE,
            RegistryNodeType.GOAL,
            RegistryNodeType.REFLECTION,
        }
    )

    CORE_EDGE_TYPES: ClassVar[frozenset[RegistryEdgeType]] = frozenset(
        {
            RegistryEdgeType.PROVIDES,
            RegistryEdgeType.DEPENDS_ON,
            RegistryEdgeType.RELATED_TO,
            RegistryEdgeType.MEMORY_OF,
            RegistryEdgeType.CONTAINS,
            RegistryEdgeType.CALLS,
            RegistryEdgeType.HAS_SKILL,
            RegistryEdgeType.PRODUCED_OUTCOME,
            RegistryEdgeType.WAS_DERIVED_FROM,
            RegistryEdgeType.WAS_ATTRIBUTED_TO,
        }
    )

    name: str = Field(description="Unique identifier for the pack")
    description: str = Field(
        default="", description="Human-readable summary of the pack's domain"
    )
    mode: SchemaPackMode = Field(
        default=SchemaPackMode.ADDITIVE,
        description="ADDITIVE (merge on top of core) or EXCLUSIVE (only pack + core)",
    )

    # Ontology scope
    node_types: set[RegistryNodeType] = Field(
        default_factory=set,
        description="Node types this pack activates",
    )
    edge_types: set[RegistryEdgeType] = Field(
        default_factory=set,
        description="Edge types this pack activates",
    )

    # Retrieval configuration (CONCEPT:KG-2.2)
    retrieval_boosts: dict[str, float] = Field(
        default_factory=dict,
        description="Per-edge-type scoring multipliers for hybrid retrieval",
    )
    backlink_boost_strategy: BacklinkBoostStrategy = Field(
        default=BacklinkBoostStrategy.GLOBAL,
        description="How backlink-density scoring is applied",
    )
    backlink_boost_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Logarithmic scaling coefficient for backlink boost",
    )

    # OWL extensions
    owl_extensions: list[str] = Field(
        default_factory=list,
        description="Additional .ttl files for OWL reasoner",
    )

    # Retrieval Quality (CONCEPT:KG-2.8)
    min_relevance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for retrieval results to be considered relevant. "
        "Per-domain tuning: research packs may use 0.5, finance packs 0.7.",
    )

    # Domain-specific inference
    inference_rules: list[str] = Field(
        default_factory=list,
        description="Cypher rule templates for domain-specific inference",
    )

    def get_active_node_types(self) -> frozenset[RegistryNodeType]:
        """Return the set of node types active under this pack.

        In ADDITIVE mode, returns all ``RegistryNodeType`` members.
        In EXCLUSIVE mode, returns only ``CORE_NODE_TYPES ∪ self.node_types``.
        """
        if self.mode == SchemaPackMode.EXCLUSIVE:
            return frozenset(self.CORE_NODE_TYPES | self.node_types)
        # ADDITIVE: all types are available
        return frozenset(RegistryNodeType)

    def get_active_edge_types(self) -> frozenset[RegistryEdgeType]:
        """Return the set of edge types active under this pack.

        In ADDITIVE mode, returns all ``RegistryEdgeType`` members.
        In EXCLUSIVE mode, returns only ``CORE_EDGE_TYPES ∪ self.edge_types``.
        """
        if self.mode == SchemaPackMode.EXCLUSIVE:
            return frozenset(self.CORE_EDGE_TYPES | self.edge_types)
        # ADDITIVE: all types are available
        return frozenset(RegistryEdgeType)

    def is_node_type_active(self, node_type: RegistryNodeType) -> bool:
        """Check whether a specific node type is active under this pack."""
        return node_type in self.get_active_node_types()

    def is_edge_type_active(self, edge_type: RegistryEdgeType) -> bool:
        """Check whether a specific edge type is active under this pack."""
        return edge_type in self.get_active_edge_types()

    def get_boost_for_edge(self, edge_type: str) -> float:
        """Return the retrieval boost multiplier for a given edge type.

        Returns 1.0 (neutral) if no boost is configured for this edge type.
        """
        return self.retrieval_boosts.get(edge_type, 1.0)
