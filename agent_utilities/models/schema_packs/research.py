from __future__ import annotations

"""Research Schema Pack — academic/scientific domain profile.

CONCEPT:AU-KG.research.research-state-domain-pack — Research-State Domain Pack

Optimized for research agents managing papers, claims, methods, datasets, evidence,
and hypotheses. This is the flagship Schema-Pack 2.0 profile: it
turns the type-selection skeleton into a full domain retrieval+extraction+reasoning
profile that realises the "academic literature state" use case from the gbrain
schema-pack discussion (garrytan/gbrain#587).

It wires every Schema-Pack 2.0 capability at once:
  - zero-LLM ``link_inference`` for supports/weakens/cites/uses-dataset (CONCEPT:AU-KG.research.zero-llm-pack-link),
  - ``relational_verbs`` so "which papers support X" walks typed edges (CONCEPT:AU-KG.retrieval.relational-intent-retrieval),
  - ``recency_decay`` + ``source_trust`` + ``autocut`` (CONCEPT:EG-KG.compute.rust-native-training-loss),
  - ``owl_object_properties`` for SUPPORTS_BELIEF transitive and
    CITES_SOURCE/CITED_BY_PAPER inverse — so multi-hop support chains and citation
    back-edges are inferred for free via the OWL closure cycle (CONCEPT:AU-KG.ontology.pack-owl-closure).
Uses CONTEXT_ONLY backlink boost to preserve discovery of novel/low-citation papers.
"""


from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import (
    BacklinkBoostStrategy,
    LinkInferenceRule,
    OwlObjectProperty,
    RecencyDecaySpec,
    SchemaPack,
    SchemaPackMode,
)


class ResearchSchemaPack(SchemaPack):
    """Research domain pack: papers, claims, methods, datasets, evidence."""

    name: str = "research-state"
    description: str = (
        "Academic/scientific research profile. Activates hypothesis, dataset, "
        "evidence, and document types with CITES/SUPPORTS retrieval boosts. "
        "Uses CONTEXT_ONLY backlink strategy to preserve novel paper discovery. "
        "Schema-Pack 2.0: zero-LLM supports/weakens/cites link inference, "
        "relational-intent retrieval, recency/source-trust signals, and "
        "transitive/inverse OWL closure for support chains and citation back-edges."
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
        RegistryEdgeType.CITED_BY_PAPER,
        RegistryEdgeType.SUPPORTS_BELIEF,
        RegistryEdgeType.CONTRADICTS_BELIEF,
        RegistryEdgeType.WEAKENS,
        RegistryEdgeType.USES_DATASET,
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
        "weakens": 1.2,
        "uses_dataset": 1.1,
        "broader": 1.1,
        "narrower": 1.1,
        "tests_hypothesis": 1.4,
    }
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.CONTEXT_ONLY
    backlink_boost_factor: float = 0.15

    # KG-2.33 — zero-LLM typed-edge extraction. Patterns capture a ``[[wikilink]]``
    # target (group 1); the writing document is the edge source.
    link_inference: list[LinkInferenceRule] = [
        LinkInferenceRule(
            pattern=r"\bsupports?\s+\[\[([^\]|]+)", edge_type="supports_belief"
        ),
        LinkInferenceRule(
            pattern=r"\b(?:weakens?|undermines?|refutes?)\s+\[\[([^\]|]+)",
            edge_type="weakens",
        ),
        LinkInferenceRule(
            pattern=r"\b(?:contradicts?|disputes?)\s+\[\[([^\]|]+)",
            edge_type="contradicts_belief",
        ),
        LinkInferenceRule(
            pattern=r"\buses?\s+(?:the\s+)?dataset\s+\[\[([^\]|]+)",
            edge_type="uses_dataset",
        ),
        LinkInferenceRule(
            pattern=r"\bcites?\s+\[\[([^\]|]+)", edge_type="cites_source"
        ),
    ]

    # KG-2.34 — relational-intent verb vocabulary.
    relational_verbs: dict[str, str] = {
        "support": "supports_belief",
        "supports": "supports_belief",
        "weaken": "weakens",
        "weakens": "weakens",
        "undermine": "weakens",
        "undermines": "weakens",
        "contradict": "contradicts_belief",
        "contradicts": "contradicts_belief",
        "cite": "cites_source",
        "cites": "cites_source",
        "cited": "cites_source",
        "test": "tests_hypothesis",
        "tests": "tests_hypothesis",
    }

    # KG-2.22 — recency decay (papers stay relevant for ~1y) and source trust.
    recency_decay: dict[str, RecencyDecaySpec] = {
        "document": RecencyDecaySpec(half_life_days=365.0, coefficient=0.3),
        "creative_work": RecencyDecaySpec(half_life_days=365.0, coefficient=0.3),
        "evidence": RecencyDecaySpec(half_life_days=180.0, coefficient=0.4),
    }
    source_trust: dict[str, float] = {
        "peer_reviewed": 1.3,
        "arxiv": 1.2,
        "preprint": 0.9,
        "blog": 0.7,
    }
    autocut_enabled: bool = True
    autocut_threshold: float = 0.5
    autocut_min_results: int = 5

    # KG-2.36 — pack-driven OWL closure (the "free value-add").
    owl_object_properties: list[OwlObjectProperty] = [
        OwlObjectProperty(edge_type="supports_belief", transitive=True),
        OwlObjectProperty(edge_type="cites_source", inverse_of="cited_by_paper"),
    ]
