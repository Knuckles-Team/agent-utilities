#!/usr/bin/python
from __future__ import annotations

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

See docs/pillars/2_epistemic_knowledge_graph.md §Schema Packs for the full architecture guide.
"""


from enum import StrEnum
from typing import ClassVar, Literal

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


class RecencyDecaySpec(BaseModel):
    """Per-type temporal recency-decay specification (CONCEPT:KG-2.22).

    Declares how a node type's retrieval score is boosted as a function of how
    recent it is, measured against its bi-temporal ``event_time`` (CONCEPT:KG-2.11).
    The boost is multiplicative and always ``>= 1.0`` so recency never *penalises*
    a result — a missing ``event_time`` yields a neutral ``1.0`` (see
    ``HybridRetriever._recency_boost``).

    Attributes:
        half_life_days: Age (in days) at which the decay term reaches 0.5.
        coefficient: Maximum additional weight a maximally-fresh node receives
            (boost ranges over ``[1.0, 1.0 + coefficient]``).
        mode: ``exponential`` (``0.5 ** (age / half_life)``) or ``hyperbolic``
            (``half_life / (half_life + age)``).
    """

    half_life_days: float = Field(gt=0.0)
    coefficient: float = Field(default=1.0, ge=0.0)
    mode: Literal["exponential", "hyperbolic"] = "exponential"


class LinkInferenceRule(BaseModel):
    """A zero-LLM regex rule for typed-edge extraction on write (CONCEPT:KG-2.33).

    On every page/document write, the active pack's rules run over the content
    deterministically (no LLM call, mirroring gbrain's ``link-inference.ts``) to
    materialise typed edges. Rule execution is ReDoS-bounded (input length cap +
    per-rule time/match budget) in ``knowledge_graph/kb/link_inference.py``.

    Attributes:
        pattern: Regular expression. A match produces one edge of ``edge_type``.
        edge_type: ``RegistryEdgeType`` *value* (e.g. ``"supports_belief"``).
        source: Where the edge's source id comes from — ``"doc"`` (the writing
            document's id) or ``"group:N"`` (the Nth regex capture group).
        target: Where the edge's target id comes from — ``"doc"`` or ``"group:N"``.
            Defaults to ``"group:1"`` (the first capture group, typically a
            ``[[wikilink]]`` or entity name).
        confidence: Provenance confidence stamped on the inferred relationship.
        flags_ignorecase: Compile the pattern case-insensitively.
    """

    pattern: str
    edge_type: str
    source: Literal["doc", "group:1", "group:2", "self"] = "doc"
    target: Literal["doc", "group:1", "group:2"] = "group:1"
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    flags_ignorecase: bool = True


class OwlObjectProperty(BaseModel):
    """A pack-declared OWL object-property characteristic.

    CONCEPT:KG-2.36 — Pack-Driven OWL Closure

    When a pack activates, these declarations are unioned into the OWL reasoning
    sets (``owl_bridge``) so the existing promote→reason→downfeed closure cycle
    infers multi-hop and inverse edges *for free* — e.g. ``supports_belief`` as a
    transitive property materialises ``A→C`` from ``A→B→C``, and ``cites_source``
    with ``inverse_of="cited_by_paper"`` materialises the back-edge automatically.
    This is a capability gbrain's flat regex edges structurally cannot provide.

    Attributes:
        edge_type: ``RegistryEdgeType`` *value* the characteristic applies to.
        transitive: Treat as ``owl:TransitiveProperty``.
        symmetric: Treat as ``owl:SymmetricProperty``.
        inverse_of: ``RegistryEdgeType`` value of the inverse edge, if any
            (``owl:inverseOf``).
    """

    edge_type: str
    transitive: bool = False
    symmetric: bool = False
    inverse_of: str | None = None


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

    # Retrieval Quality (CONCEPT:KG-2.6)
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

    # --- Schema-Pack 2.0 declarative profiles ---

    # KG-2.22 — Pack-driven temporal recency-decay (per lowercased node type/label)
    recency_decay: dict[str, RecencyDecaySpec] = Field(
        default_factory=dict,
        description="Per-node-type recency-decay specs applied during hybrid "
        "retrieval (CONCEPT:KG-2.22). Empty => no recency weighting (core default).",
    )

    # KG-2.22 — Source-trust / authority weighting (per lowercased source id/domain)
    source_trust: dict[str, float] = Field(
        default_factory=dict,
        description="Per-source multiplicative trust weights applied during hybrid "
        "retrieval (CONCEPT:KG-2.22). Missing source => neutral 1.0.",
    )

    # KG-2.22 — Autocut (score-discontinuity result trimming)
    autocut_enabled: bool = Field(
        default=False,
        description="Trim results at the largest relative score drop (CONCEPT:KG-2.22).",
    )
    autocut_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relative drop (fraction) that triggers an autocut.",
    )
    autocut_min_results: int = Field(
        default=5,
        ge=2,
        description="Never autocut a result set smaller than this (recall guard).",
    )

    # KG-2.33 — Zero-LLM pack-driven typed-edge extraction on write
    link_inference: list[LinkInferenceRule] = Field(
        default_factory=list,
        description="Regex rules for deterministic typed-edge extraction; "
        "empty means no pack-driven link inference (CONCEPT:KG-2.33).",
    )

    # KG-2.34 — Relational-intent retrieval verbs (NL phrase -> edge_type value)
    relational_verbs: dict[str, str] = Field(
        default_factory=dict,
        description="Natural-language verb phrases mapped to edge-type values for "
        "deterministic relational query parsing (CONCEPT:KG-2.34).",
    )

    # KG-2.36 — Pack-driven OWL object-property closure declarations
    owl_object_properties: list[OwlObjectProperty] = Field(
        default_factory=list,
        description="OWL object-property characteristics unioned into the reasoning "
        "closure when this pack is active (CONCEPT:KG-2.36).",
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

    def recency_spec_for(self, label: str) -> RecencyDecaySpec | None:
        """Return the recency-decay spec for a node label, or ``None`` (KG-2.22).

        Lookup is case-insensitive on the lowercased label/type so callers can
        pass a node's raw ``type``/label without normalising first.
        """
        if not self.recency_decay:
            return None
        return self.recency_decay.get(label.lower())

    def trust_for(self, source: str) -> float:
        """Return the source-trust multiplier for a source id/domain (KG-2.22).

        Returns ``1.0`` (neutral) when the source is unknown or unset.
        """
        if not self.source_trust:
            return 1.0
        return self.source_trust.get(source.lower(), 1.0)

    def get_owl_closure_sets(self) -> tuple[set[str], set[str], dict[str, str]]:
        """Return ``(transitive, symmetric, inverse_map)`` for OWL closure (KG-2.36).

        Consumed by ``owl_bridge`` to union pack-declared object-property
        characteristics into the reasoning sets used by the lightweight Python/Rust
        reasoning paths. Edge-type values are lowercased to match stored edge types.
        """
        transitive: set[str] = set()
        symmetric: set[str] = set()
        inverse_map: dict[str, str] = {}
        for prop in self.owl_object_properties:
            et = prop.edge_type.lower()
            if prop.transitive:
                transitive.add(et)
            if prop.symmetric:
                symmetric.add(et)
            if prop.inverse_of:
                inverse_map[et] = prop.inverse_of.lower()
        return transitive, symmetric, inverse_map

    def signature(self) -> str:
        """Return a stable content hash of this pack (KG-2.35).

        Used as a cache-key component so a switch of active pack cannot serve a
        previous pack's boosted/cut results (the gbrain ``knobs_hash`` analogue).
        """
        import hashlib

        return hashlib.sha256(
            self.model_dump_json(exclude_none=False).encode("utf-8")
        ).hexdigest()[:16]
