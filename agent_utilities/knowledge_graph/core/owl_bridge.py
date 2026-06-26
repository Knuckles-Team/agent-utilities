#!/usr/bin/python
from __future__ import annotations

"""OWL Bridge — Orchestrates LPG ↔ OWL data flow.

Handles the deterministic promote → reason → downfeed cycle that
enriches the LPG with OWL-inferred facts.
"""


import json
import logging
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Rust-native graph compute — using GraphComputeEngine

    from ..backends.base import GraphBackend
    from ..backends.owl.base import OWLBackend

logger = logging.getLogger(__name__)

# Node types registered at runtime from an external metamodel (e.g. the live
# LeanIX data model compiled by ``ontology.leanix_metamodel``). Unioned into
# every bridge's effective promotable set so generated types reach the reasoner
# without editing the static set below. (CONCEPT:KG-2.9)
DYNAMIC_PROMOTABLE_NODE_TYPES: set[str] = set()


def register_promotable_node_types(types: Iterable[str]) -> None:
    """Mark externally-generated node-type labels eligible for OWL promotion."""
    DYNAMIC_PROMOTABLE_NODE_TYPES.update(
        str(t).strip().lower() for t in types if str(t).strip()
    )


# Node types eligible for OWL promotion (must have matching class in ontology.ttl)
PROMOTABLE_NODE_TYPES: set[str] = {
    # Enrichment / code-understanding entities (CONCEPT:KG-2.8)
    "code",
    "test",
    "feature",
    "pattern",
    "route",  # HTTP route a Code handler serves (CONCEPT:KG-2.102)
    "resource",  # IaC resource (Dockerfile/K8s/Terraform) (CONCEPT:KG-2.103)
    # Harness foundry: evolvable harness as reasoned-over ontology (CONCEPT:KG-2.107)
    "harness",
    "processor",
    "harness_dimension",
    "harness_edit",
    "harness_variant",
    "harness_pathology",
    # Substitution-algebra hook contract (CONCEPT:KG-2.109)
    "harness_hook",
    "concept",
    "document",
    # Enterprise OS entities (CONCEPT:KG-2.9)
    "server",
    "hardwarenode",
    "service",
    "incident",
    "change",
    "configurationitem",
    "person",
    "application",
    "itcomponent",
    "businesscapability",
    "dataobject",
    # Technology Reference Model + risk (CONCEPT:KG-2.9)
    "technologyproduct",
    "technologystandard",
    "assetinstance",
    "technologyrisk",
    "asset",
    "warehouse",
    "calendarevent",
    "topic",
    "identityuser",
    "identitygroup",
    "identityrole",
    "employee",
    "customer",
    "salesorder",
    "item",
    "orgunit",
    "dashboard",
    "panel",
    "alert",
    "datasource",
    # A2A + data-source schema (CONCEPT:KG-2.9 / 2.10)
    "a2aagentcard",
    "table",
    "column",
    "collection",
    "graphqltype",
    "field",
    # Orchestration synthesis (CONCEPT:KG-2.10)
    "team",
    "workflow",
    "goal",
    "model",
    "agent",
    "tool",
    "skill",
    # Unified scheduling / task / loop engine (CONCEPT:OS-5.44 / KG-2.113)
    "task",
    "schedule",
    "loop",
    # Unified feed ingestion (CONCEPT:KG-2.122) — first-class RSS/Atom feed sources.
    "feedsource",
    # Computer-use (GUI) runtime entities (CONCEPT:KG-2.185 / 2.186 / ORCH-1.84).
    "computerusesession",
    "screenobservation",
    "uielement",
    "guiaction",
    "file",
    "symbol",
    "module",
    "memory",
    "event",
    "episode",
    "phase",
    "decision",
    "observation",
    "action",
    # Ontology Action System — governed verbs (CONCEPT:KG-2.25)
    "ontology_action",
    "action_invocation",
    "action_parameter",
    "belief",
    "hypothesis",
    "fact",
    "principle",
    "evidence",
    # ARA — OWL-native 4-layer research artifact (CONCEPT:KG-2.80)
    "research_artifact",
    "claim",
    "code_spec",
    "exploration_node",
    "seal_certificate",
    "trajectory",
    "deliberation",
    "reflection",
    "organization",
    "role",
    "place",
    "system",
    "reasoning_trace",
    "outcome_evaluation",
    "critique",
    "policy",
    # Standard Ontology Types (BFO, Schema.org, DC, FIBO)
    "creative_work",
    "dataset",
    "software_project",
    "medical_entity",
    "procedure",
    "regulation",
    "financial_instrument",
    "financial_transaction",
    "account",
    # AHE Types (CONCEPT:AHE-3.0)
    "change_manifest",
    "component_edit_record",
    "evidence_record",
    "constraint_state",
    # Backfill Gap Types
    "codemap",
    "pattern_template",
    "proposed_skill",
    "system_prompt",
    "prompt",
    "process_flow",
    "process_step",
    # Enterprise OS — canonical ArchiMate concepts & vendor crosswalk (KG-2.9)
    "businessprocess",
    "businesstask",
    "applicationevent",
    "erpnextissue",
    # Feedback loop — corrections → rules → eval (CONCEPT:KG-2.8)
    "correction",
    "governance_rule",
    "voice_rule",
    "source_rule",
    "preference",
    "eval_case",
    # Operating intelligence distilled from calls/docs (CONCEPT:KG-2.8)
    "insight",
    "framework",
    "playbook",
    "knowledge_base",
    "knowledge_base_topic",
    "experiment",
    # Engineering Rules Engine (CONCEPT:KG-2.2)
    "engineering_rule",
    "rule_book",
    # External Integration & SDLC Entities (CONCEPT:ORCH-1.2)
    "repository",
    "merge_request",
    "pull_request",
    "pipeline",
    "commit",
    "issue",
    "external_graph_reference",
    "external_entity",
    # Perspectival inquiry — STORM structures (CONCEPT:KG-2.127/2.128/2.129)
    "research_inquiry",
    "perspective",
    "agreement",
    "contradiction",
    "blind_spot",
    "peer_review",
    # Financial Trading Pipeline (CONCEPT:KG-2.6)
    "trading_signal",
    "order",
    "position",
    "portfolio",
    "strategy",
    # Market Data Connector Protocol (CONCEPT:ECO-4.0)
    "data_connector",
    "data_fetch_record",
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    "swarm_preset",
    "swarm_run",
    "swarm_task_record",
    # Risk Scoring Ontology (CONCEPT:KG-2.6)
    "risk_assessment",
    "risk_factor",
    "risk_mitigation",
    # Backtest Evaluation Harness (CONCEPT:AHE-3.4)
    "backtest_run",
    "backtest_metric",
    # Prompt Injection Scanner (CONCEPT:OS-5.1)
    "security_finding",
    # Tool Repetition Guard (CONCEPT:OS-5.1)
    "experience",
    # MATE Integration — Token Analytics (CONCEPT:OS-5.1)
    "token_usage_record",
    # MATE Integration — Audit Logging (CONCEPT:OS-5.1)
    "audit_log",
    # MATE Integration — Guardrail Engine (CONCEPT:OS-5.1)
    "guardrail_trigger",
    # MATE Integration — Config Versioning (CONCEPT:AHE-3.2)
    "agent_config_version",
    # MATE Integration — EvalRunner (CONCEPT:AHE-3.1)
    "eval_run",
    # Agentic-iModels (CONCEPT:AHE-3.3, AHE-3.16, KG-2.17)
    "imodel",
    "interpretability_test",
    "model_display",
    # Ecosystem Topology Map (CONCEPT:ECO-4.0)
    "ecosystem_package",
    "frontend_package",
    "kernel_package",
    "mcp_server_package",
    "skill_package",
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    "synergy_insight",
    # Formal Graph Theory Primitives (CONCEPT:KG-2.6)
    "math_foundation",
    "critical_path_result",
    # Structural Causal Reasoning (CONCEPT:KG-2.6)
    "causal_factor",
    "causal_model",
    # Optimal Execution Engine (CONCEPT:KG-2.6)
    "execution_plan",
    "market_making_quote",
    "pairs_trade_signal",
    # Context Graph Architecture (CONCEPT:KG-2.6)
    "architecture_decision",
    "archimate_element",
    # Legal Entity & Compliance domain (CONCEPT:LGC-1.0)
    "legal_trust",
    "trustee_role",
    "settlor_role",
    "beneficiary_role",
    "legal_entity",
    "company",
    "ein_application",
    "host",
    "container",
    "container_stack",
    "platform_service",
    "gpu_accelerator",
    "storage_array",
    # Hydration Core Entities
    "opportunity",
    "uptime_monitor",
    "chat_channel",
    # Capability Abstraction Layer (CONCEPT:KG-2.7)
    "service_capability",
    "vpn_purpose",
    "development_domain",
    "development_standard",
    "ea_fact_sheet",
    "process_model",
    # Ontology System — Foundry-parity type/link/function layer
    # (CONCEPT:KG-2.26 links, KG-2.38 interfaces, KG-2.41 functions)
    "relationship",
    "function",
    "function_invocation",
    "interface",
    "interface_property",
    "interface_link_constraint",
    # Object-edits durable ledger (CONCEPT:KG-2.43)
    "object_edit",
    # Object permissioning — mandatory markings (CONCEPT:KG-2.46)
    "marking",
    # Document processing pipeline — exploded chunk objects (CONCEPT:KG-2.48)
    "chunk",
    # Finance microstructure signals (CONCEPT:KG-2.81) — Kyle surveillance signal
    # reasons over grounded_in/supports transitively.
    "microstructure_signal",
    "surveillance_signal",
    # Connector → Skill synthesis proposals (CONCEPT:KG-2.90) — the distiller's
    # propose-only candidates; OWL reasoning runs transitive/inverse over their
    # automates/derived_from/composes edges (CONCEPT:KG-2.91).
    "skill_proposal",
    "skill_workflow_proposal",
    # Ops / platform connectors as typed OWL entities (CONCEPT:KG-2.155–2.161).
    # (``repository``, ``observation``, ``host``, ``person``, ``company``,
    # ``opportunity``, ``uptime_monitor`` are already promotable above.)
    "container_image",  # DockerHub image (KG-2.155)
    "trace",  # Langfuse trace (KG-2.156)
    "generation",  # Langfuse LLM-call observation (KG-2.156)
    "dns_zone",  # Technitium DNS zone (KG-2.157)
    "dns_record",  # Technitium DNS record (KG-2.157)
    "tunnel",  # tunnel-manager SSH tunnel (KG-2.158)
    "heartbeat_stat",  # Uptime Kuma heartbeat sample (KG-2.159)
    "device",  # Home Assistant device (KG-2.160)
    "entity",  # Home Assistant entity (KG-2.160)
    # Media / finance / document / genealogy connectors (CONCEPT:KG-2.163–2.166).
    # (``account``, ``document``, ``person``, ``event`` are already promotable above.)
    "library",  # Audiobookshelf library (KG-2.163)
    "book",  # Audiobookshelf book / audiobook (KG-2.163)
    "author",  # Audiobookshelf author (KG-2.163)
    "transaction",  # Firefly III transaction (KG-2.164)
    "budget",  # Firefly III budget (KG-2.164)
    "correspondent",  # Paperless-ngx correspondent (KG-2.165)
    "tag",  # Paperless-ngx tag (KG-2.165)
    "family",  # Gramps Web family (KG-2.166)
}

# Edge types eligible for OWL promotion (transitive / inferable relationships)
PROMOTABLE_EDGE_TYPES: set[str] = {
    # Perspectival inquiry edges (CONCEPT:KG-2.128/2.129)
    "asks_from",
    "agrees_with",
    "reviews",
    # Enrichment / code-understanding edges (CONCEPT:KG-2.8)
    "calls",
    "covers",
    "tests",
    "part_of_feature",
    "implements_pattern",
    "realizes",
    "inherits",
    "serves",  # Code handler → HTTP Route (CONCEPT:KG-2.102)
    "served_by",  # Route → deployed Service (CONCEPT:KG-2.102)
    "provisions",  # IaC Resource → deployed Service (CONCEPT:KG-2.103)
    # Harness foundry edges (CONCEPT:KG-2.107)
    "targets_dimension",  # HarnessEdit → HarnessDimension (formal edit scope)
    "has_variant",  # Harness → HarnessVariant (inverse variant_of)
    "variant_of",
    "applies_edit",  # HarnessVariant → HarnessEdit
    "exhibits_pathology",  # Harness/Variant → HarnessPathology
    "mitigates_pathology",  # HarnessEdit → HarnessPathology (inverse mitigated_by)
    "mitigated_by",
    "causes_regression",  # HarnessEdit → task (formal seesaw signal)
    "confirms_fix",  # HarnessEdit → predicted fix
    "at_hook",  # HarnessEdit → HarnessHook (substitution algebra, CONCEPT:KG-2.109)
    "attached_to_hook",  # Processor → HarnessHook (CONCEPT:KG-2.109)
    "implemented_by",
    "similar_to",
    "mentions",
    "relates_to",
    "corrects",
    # Enterprise OS edges (CONCEPT:KG-2.9)
    "runs_on",
    "affects",
    "assigned_to",
    "supports",
    "depends_on_it",
    "connects_via",  # Host → Tunnel (tunnel-manager, CONCEPT:KG-2.158)
    "authored_by",  # Book → Author (audiobookshelf, CONCEPT:KG-2.163)
    "tagged_with",  # Document → Tag (paperless-ngx, CONCEPT:KG-2.165)
    "placed_by",
    "member_of",
    "part_of_dashboard",
    "monitors",
    # A2A + data-source schema (CONCEPT:KG-2.9 / 2.10)
    "exposes_skill",
    "delegates_to",
    "has_table",
    "has_column",
    "foreign_key",
    "has_field",
    # Orchestration synthesis (CONCEPT:KG-2.10)
    "has_prompt",
    "uses_tool",
    "has_skill",
    "member_of_team",
    "reports_to",
    "solves",
    "orchestrates",
    "evolved_from",
    "uses_model",
    "inherits_from",
    "depends_on",
    "imports",
    "provides",
    "part_of",
    "contains",
    "triggered_by",
    "supports_belief",
    "contradicts_belief",
    "owns_system",
    "depends_on_system",
    "has_role",
    "motivated_by",
    "produced_outcome",
    "triggered_action",
    "observed_by",
    "occurred_during",
    "defined_in",
    # Standard Ontology Edges (PROV-O, SKOS, Dublin Core, FIBO)
    "was_generated_by",
    "was_derived_from",
    "was_attributed_to",
    "has_temporal_extent",
    "broader",
    "narrower",
    "related_concept",
    "exact_match",
    "close_match",
    "broad_match",
    "creator",
    "cites_source",
    "cited_by_paper",
    # Research-state domain edges (CONCEPT:KG-2.37)
    "weakens",
    "uses_dataset",
    "has_financial_instrument",
    "executed_transaction",
    "account",
    # AHE Edges (CONCEPT:AHE-3.0)
    "edited_in_round",
    "predicted_fix",
    "caused_regression",
    "confirmed_fix",
    "verified_by",
    "escalated_to",
    "applied_edit",
    "has_edit_for",
    # Engineering Rules Engine Edges (CONCEPT:KG-2.2)
    "conflicts_with",
    "corrects_bias",
    "applicable_when",
    "derived_from_book",
    "applied_in_task",
    # External Integration & SDLC Entities (CONCEPT:ORCH-1.2)
    "modified_in",
    "mentioned_in",
    "triggered",
    "targets",
    "mapped_to_external",
    # Financial Trading Pipeline (CONCEPT:KG-2.6)
    "generated_signal",
    "placed_order",
    "opened_position",
    "belongs_to_portfolio",
    "executes_strategy",
    "backtested_with",
    # BPMN process structure lift (CONCEPT:KG-2.53) — sequence flows between
    # BusinessTask elements; matches :flowsTo in the ontology.
    "flows_to",
    # Process lineage close-out (CONCEPT:ORCH-1.43) — RunTrace → the
    # BusinessProcess its workflow realizes; matches :executedProcess.
    "executed_process",
    # Market Data Connector Protocol (CONCEPT:ECO-4.0)
    "fetched_from",
    "falls_back_to",
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    "preset_of",
    "ran_preset",
    "task_depends_on",
    # Risk Scoring Ontology (CONCEPT:KG-2.6)
    "assessed_risk",
    "has_risk_factor",
    "propagates_risk_to",
    # Backtest Evaluation Harness (CONCEPT:AHE-3.4)
    "evaluated_strategy",
    "has_metric",
    "compared_to_benchmark",
    # Prompt Injection Scanner (CONCEPT:OS-5.1)
    "detected_threat",
    # Structured Retry Manager (CONCEPT:ORCH-1.3)
    "triggered_retry",
    # MATE Integration — Audit Logging (CONCEPT:OS-5.1)
    "audited_by",
    # MATE Integration — Guardrail Engine (CONCEPT:OS-5.1)
    "triggered_guardrail",
    # MATE Integration — Config Versioning (CONCEPT:AHE-3.2)
    "config_version_of",
    # MATE Integration — EvalRunner (CONCEPT:AHE-3.1)
    "evaluated_by",
    # Agentic-iModels (CONCEPT:AHE-3.3, AHE-3.16, KG-2.17)
    "evolved_model",
    "tested_interpretability",
    "display_of",
    "pareto_dominates",
    # Ecosystem Topology Map (CONCEPT:ECO-4.0)
    "provides_capability_to",
    "consumes_from_kernel",
    "visualizes",
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    "has_synergy_with",
    # Formal Graph Theory Primitives (CONCEPT:KG-2.6)
    "critical_path_of",
    "colored_with",
    # Structural Causal Reasoning (CONCEPT:KG-2.6)
    "causes",
    "causal_mechanism",
    "counterfactual_of",
    # Probabilistic Reasoning (CONCEPT:KG-2.6)
    "belief_update",
    # Optimal Execution (CONCEPT:KG-2.6)
    "executed_via",
    "pairs_with",
    "makes_market_in",
    # Context Graph Architecture — ADR edges (CONCEPT:KG-2.6)
    "impacts_concept",
    "alternatives_to",
    "decided_by",
    "supersedes",
    # Legal Entity & Compliance domain edges (CONCEPT:LGC-1.0)
    "has_trustee",
    "has_settlor",
    "has_beneficiary",
    "trust_agreement",
    "governed_by_doc",
    "filed_by",
    "has_accelerator",
    "attached_storage",
    "deployed_on",
    "belongs_to_stack",
    # Hydration Core Edges
    "blocked_by",
    "works_at",
    "related_to",
    "associated_with",
    # Ontology Action System — governed verbs (CONCEPT:KG-2.25)
    "acts_on",
    "invokes",
    "invoked_by",
    "acts_on_object",
    "may_be_invoked_by",
    # Capability Abstraction Layer (CONCEPT:KG-2.7)
    "provides_capability",
    "requires_capability",
    "swappable_with",
    "has_purpose",
    "requires_vpn_for_purpose",
    "applies_to_domain",
    "works_on_domain",
    "must_follow",
    # Universal Relationship Properties (CONCEPT:KG-2.7)
    # Lineage / Ancestry
    "has_parent",
    "has_child",
    "has_ancestor",
    "has_descendant",
    "has_sibling",
    # Participation
    "participated_in",
    "had_participant",
    "occurred_at",
    # Membership
    "has_member",
    # Ownership
    "owns",
    "owned_by",
    # Authorship
    "created_by",
    "author_of",
    # Spatial
    "located_in",
    # Succession
    "succeeds",
    "precedes",
    # Influence
    "influenced_by",
    "influenced",
    # Derivation
    "derived_from",
    "had_derivation",
    # Governance
    "governed_by",
    "approved_by",
    # Dependency
    "dependency_of",
    # Composition
    "has_part",
    # Classification
    "classified_as",
    # Alignment
    "aligned_with",
    # ARA forensic bindings (CONCEPT:KG-2.80) — verified_by/implemented_by/contains/
    # was_derived_from already promotable above; add the remaining /logic + /trace edges
    "grounded_in",
    "has_evidence",
    "pivoted_from",
    "reached_dead_end",
    "certifies",
}

# ARA forensic-edge OWL object-property characteristics, always on (CONCEPT:KG-2.80).
# Independent of any schema pack so every reasoning cycle chains a claim's grounding
# (claim -grounded_in-> evidence -grounded_in-> source ⟹ claim -grounded_in-> source)
# and materialises the evidence→claim ``supports`` inverse — letting reasoning relate
# a /logic claim to the ecosystem code/services that substantiate it.
ARA_TRANSITIVE_EDGES: frozenset[str] = frozenset({"grounded_in"})
ARA_INVERSE_EDGES: dict[str, str] = {"grounded_in": "supports"}

# Harness-foundry OWL object-property characteristics, always on (CONCEPT:KG-2.107).
# This is what makes HarnessX's "operational mirror" a formal operational *ontology*
# (CONCEPT:AHE-3.51): reasoning materialises the variant↔base and edit↔pathology
# inverses so the SHACL concentration/no-regression/pathology gate (AHE-3.53) queries
# a complete, inferred graph rather than relying on the paper's RL↔symbolic analogy.
HARNESS_INVERSE_EDGES: dict[str, str] = {
    "has_variant": "variant_of",
    "mitigates_pathology": "mitigated_by",
}


class OWLBridge:
    """Orchestrates LPG ↔ OWL data flow.

    The bridge performs a deterministic three-step cycle:

    1. **Promote** — Export stable, high-confidence nodes and edges from
       the in-memory graph compute engine to the OWL backend as individuals and
       property assertions.

    2. **Reason** — Run OWL DL reasoning (HermiT/Stardog) to discover
       new inferred facts (transitive closure, subclass inference, etc.).

    3. **Downfeed** — Write inferred facts back to the LPG as new edges
       (with ``inferred=True`` metadata) so the agent's standard queries
       immediately benefit from the reasoning.
    """

    def __init__(
        self,
        graph: Any,
        owl_backend: OWLBackend | None,
        backend: GraphBackend | None = None,
        importance_threshold: float = 0.1,
        recency_days: int = 7,
        schema_pack: Any | None = None,
    ):
        self.graph = graph
        self.backend = backend
        self.owl = owl_backend
        self.importance_threshold = importance_threshold
        self.recency_days = recency_days
        self._schema_pack = schema_pack
        self._rdf_cache: Any = None
        self._rdf_cache_hash: int = -1

        # Compute effective promotable types filtered by schema pack (CONCEPT:KG-2.2).
        # Static built-ins are unioned with any runtime-registered types (e.g. the
        # generated LeanIX metamodel) so external sources reason without hub edits.
        promotable_nodes = PROMOTABLE_NODE_TYPES | DYNAMIC_PROMOTABLE_NODE_TYPES
        if schema_pack is not None:
            active_nodes = {str(t) for t in schema_pack.get_active_node_types()}
            active_edges = {str(t) for t in schema_pack.get_active_edge_types()}
            self._effective_node_types = promotable_nodes & active_nodes
            self._effective_edge_types = PROMOTABLE_EDGE_TYPES & active_edges
        else:
            self._effective_node_types = promotable_nodes
            self._effective_edge_types = PROMOTABLE_EDGE_TYPES

        # A pack may declare its edge types as transitive/symmetric/inverse
        # object-properties; these are unioned into the lightweight reasoning sets
        # below so the existing promote→reason→downfeed cycle materialises multi-hop
        # and inverse edges for free — e.g. a research pack's supports_belief
        # transitive chains and cites_source/cited_by inverses (CONCEPT:KG-2.36).
        # Start from ARA's always-on forensic-edge characteristics (CONCEPT:KG-2.80)
        # so claim-grounding chains/inverses materialise even without a schema pack,
        # then union any pack-declared object-properties on top.
        self._pack_transitive = set(ARA_TRANSITIVE_EDGES)
        self._pack_symmetric = set()
        self._pack_inverse = {**ARA_INVERSE_EDGES, **HARNESS_INVERSE_EDGES}
        if schema_pack is not None and getattr(
            schema_pack, "owl_object_properties", None
        ):
            (
                pack_transitive,
                pack_symmetric,
                pack_inverse,
            ) = schema_pack.get_owl_closure_sets()
            self._pack_transitive |= pack_transitive
            self._pack_symmetric |= pack_symmetric
            self._pack_inverse.update(pack_inverse)

        # Ephemeral Namespaced In-Memory and Shared Cache registries
        self._namespaces: dict[str, dict[str, Any]] = {}
        self._namespace_ttls: dict[str, float] = {}

    def run_cycle(self, lightweight: bool = True) -> dict[str, Any]:
        """Full promote → reason → downfeed cycle. Returns stats.

        If lightweight=True, performs fast local RDFS+/OWL closure -- engine-native
        first (``client.rdf.owl_reason``), Python last-resort -- and needs NO owlready2
        backend (CONCEPT:KG-2.242): the engine reasons over the live graph directly, so
        owl-backend promotion is skipped. If False, executes full Description Logic
        reasoning via the (owlready2) OWL backend, which is then required.
        """
        if lightweight:
            # Engine-native (or Python-fallback) closure runs over the live graph
            # directly -- no owlready2 promotion, so it works with owl_backend=None.
            if self.owl is not None:
                self.owl.clear()
                promoted_nodes = self._promote_stable_nodes()
                promoted_edges = self._promote_stable_edges()
            else:
                promoted_nodes = 0
                promoted_edges = 0
            inferences = self._lightweight_reasoning()
        else:
            # Full DL reasoning still requires a real owlready2 backend.
            assert self.owl is not None, (
                "full-DL OWL reasoning cycle requires an owl_backend"
            )
            self.owl.clear()
            promoted_nodes = self._promote_stable_nodes()
            promoted_edges = self._promote_stable_edges()
            inferences = self.owl.reason()

        downfed = self._downfeed_inferences(inferences)

        stats = {
            "promoted_nodes": promoted_nodes,
            "promoted_edges": promoted_edges,
            "inferred": len(inferences),
            "downfed": downfed,
            "mode": "lightweight" if lightweight else "full_dl",
        }
        logger.info("OWL reasoning cycle complete: %s", stats)
        return stats

    def _lightweight_reasoning(self) -> list[dict[str, Any]]:
        """Lightweight OWL/RDFS+ reasoning -- engine-native first, Python last-resort.

        CONCEPT:KG-2.242 — OWL reasoning is demoted to the engine's native OWL 2
        (EL+/RL) reasoner (``client.rdf.owl_reason``, confidence/decay-weighted). The
        engine's RDF/OWL surface ships in every profile (the tiny/pi-tier binary
        included), so the engine path is the default everywhere. The Python
        ``_python_reasoning`` RDFS+ closure is the TRUE last-resort -- only when no
        engine is reachable at all. Both paths emit the same inference-dict shape so
        ``_downfeed_inferences`` is path-agnostic.
        """
        try:
            return self._engine_reasoning()
        except Exception as e:  # noqa: BLE001 -- degrade to the Python last-resort
            logger.debug(
                "Engine OWL reasoning unavailable (%s); falling back to Python RDFS+.",
                e,
            )
            return self._python_reasoning()

    def _engine_reasoning(self) -> list[dict[str, Any]]:
        """Execute OWL entailment via the engine's native reasoner (the memory
        inference primitive, CONCEPT:KG-2.242).

        Routes to ``self.graph.owl_reason`` (the engine ``OwlReason`` op): it
        classifies the OWL axioms in the engine's RDF projection of the live graph
        (plus any pack-declared object-property characteristics emitted as ontology
        Turtle) and returns confidence/decay-weighted entailments -- inferred subclass
        edges and class memberships -- which we convert to inference dicts. The engine
        reasoner does not emit ``owl:inverseOf`` facts, so pack-declared inverse
        closure is added on top (same as the Python path), keeping the two paths in
        agreement (CONCEPT:KG-2.36).

        Raises if the engine / ``owl`` feature is unavailable so the caller falls back
        to :meth:`_python_reasoning`.
        """
        if not hasattr(self.graph, "owl_reason"):
            raise RuntimeError("engine OWL reasoning unavailable (no owl_reason op)")

        # Seed the reasoner with the pack-declared object-property characteristics so
        # the engine classifies the active domain's transitive/symmetric edges
        # (CONCEPT:KG-2.36). An empty ontology => reason over the graph's own axioms.
        ontology = self._pack_axioms_turtle() or None
        result = self.graph.owl_reason(ontology=ontology)

        inferences: list[dict[str, Any]] = []
        # Inferred class memberships (incl. exists-restriction / role-chain reached).
        for pair in result.get("instances", []) or []:
            if not pair or len(pair) != 2:
                continue
            inst, cls = pair
            inferences.append(
                {
                    "subject": str(inst),
                    "predicate": "rdf:type",
                    "object": str(cls),
                    "inference_type": "owl_engine_instance",
                }
            )
        # Inferred subclass hierarchy (the classification result).
        for pair in result.get("subclasses", []) or []:
            if not pair or len(pair) != 2:
                continue
            sub, sup = pair
            inferences.append(
                {
                    "subject": str(sub),
                    "predicate": "rdfs:subClassOf",
                    "object": str(sup),
                    "inference_type": "owl_engine_subclass",
                }
            )

        # Pack-declared inverse closure -- the engine reasoner does not materialize
        # owl:inverseOf, so emit it here so both paths agree (CONCEPT:KG-2.36).
        inferences.extend(self._inverse_inferences())

        return inferences

    def _pack_axioms_turtle(self) -> str:
        """Render the pack-declared object-property characteristics as OWL Turtle.

        CONCEPT:KG-2.36 — the engine reasons over OWL axioms; surface the active
        pack's transitive/symmetric properties as ``owl:TransitiveProperty`` /
        ``owl:SymmetricProperty`` declarations so the native reasoner closes them.
        Returns an empty string when no pack characteristics are declared.
        """
        if not (self._pack_transitive or self._pack_symmetric):
            return ""
        lines = [
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix au: <http://agent-utilities.dev/ontology#> .",
        ]
        for prop in sorted(self._pack_transitive):
            lines.append(f"au:{prop} a owl:TransitiveProperty .")
        for prop in sorted(self._pack_symmetric):
            lines.append(f"au:{prop} a owl:SymmetricProperty .")
        return "\n".join(lines) + "\n"

    def _python_reasoning(self) -> list[dict[str, Any]]:
        """Python fallback — RDFS+ reasoning on the in-memory networkx graph.

        Performs simple transitive closure (e.g. part_of, depends_on) and
        symmetric closures (e.g. related_concept) without calling the heavy DL reasoner.
        """
        inferences = []
        transitive_props = {
            "part_of",
            "depends_on",
            "broader",
            "narrower",
            "inherits_from",
            "inherits",  # class inheritance chains (CONCEPT:KG-2.100)
        }
        symmetric_props = {
            "related_concept",
            "exact_match",
            "close_match",
            "broad_match",
            "similar_to",  # model-free code similarity (CONCEPT:KG-2.101)
        }
        # Union pack-declared object-property characteristics (CONCEPT:KG-2.36).
        transitive_props |= self._pack_transitive
        symmetric_props |= self._pack_symmetric

        # Pack-declared inverse closure: for every A -rel-> B, emit B -inverse-> A.
        inferences.extend(self._inverse_inferences())

        for u, v, data in self.graph.edges(data=True):
            rel = data.get("type")
            if not rel:
                continue

            # Symmetric closure
            if rel in symmetric_props:
                if not self.graph.has_edge(v, u) or not any(
                    e.get("type") == rel
                    for e in self.graph.get_edge_data(v, u, default={}).values()
                ):
                    inferences.append(
                        {
                            "subject": v,
                            "predicate": rel,
                            "object": u,
                            "inference_type": "symmetric_closure",
                        }
                    )

            # Transitive closure (1-hop)
            if rel in transitive_props:
                for w in self.graph.successors(v):
                    edge_data_dict = self.graph.get_edge_data(v, w, default={})
                    for w_data in edge_data_dict.values():
                        if w_data.get("type") == rel:
                            if not self.graph.has_edge(u, w) or not any(
                                e.get("type") == rel
                                for e in self.graph.get_edge_data(
                                    u, w, default={}
                                ).values()
                            ):
                                inferences.append(
                                    {
                                        "subject": u,
                                        "predicate": rel,
                                        "object": w,
                                        "inference_type": "transitive_closure",
                                    }
                                )

        return inferences

    def _inverse_inferences(self) -> list[dict[str, Any]]:
        """Emit inverse-edge facts for pack-declared ``owl:inverseOf`` properties.

        CONCEPT:KG-2.36 — for each edge ``A -rel-> B`` whose ``rel`` has a declared
        inverse ``inv``, emit ``B -inv-> A`` (deduplicated downstream by
        ``_downfeed_inferences``, so re-running the cycle is a fixpoint).
        """
        if not self._pack_inverse:
            return []
        out: list[dict[str, Any]] = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("type")
            inv = self._pack_inverse.get(rel) if rel else None
            if not inv:
                continue
            existing = self.graph.get_edge_data(v, u, default={})
            if any(e.get("type") == inv for e in existing.values()):
                continue
            out.append(
                {
                    "subject": v,
                    "predicate": inv,
                    "object": u,
                    "inference_type": "inverse_closure",
                }
            )
        return out

    def _is_eligible_node(self, node_id: str, attrs: dict[str, Any]) -> bool:
        """Check if a node meets promotion criteria."""
        node_type = attrs.get("type", "")
        if node_type not in self._effective_node_types:
            return False

        # Always promote permanent nodes
        if attrs.get("is_permanent", False):
            return True

        # Check importance threshold
        importance = float(attrs.get("importance_score", 0.0))
        if importance < self.importance_threshold:
            return False

        # Check recency
        timestamp_str = attrs.get("timestamp")
        if timestamp_str:
            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                cutoff = datetime.now(UTC) - timedelta(days=self.recency_days)
                if ts < cutoff:
                    return False
            except (ValueError, TypeError):
                pass  # If we can't parse, include it

        return True

    def _promote_stable_nodes(self) -> int:
        """Filter and promote stable nodes from the NX graph."""
        stable_nodes = []

        for node_id, attrs in self.graph.nodes(data=True):
            if self._is_eligible_node(node_id, attrs):
                node_dict = dict(attrs)
                node_dict["id"] = node_id
                stable_nodes.append(node_dict)

        if not stable_nodes:
            return 0

        assert self.owl is not None  # only reached from the backend-backed cycle
        return self.owl.promote(stable_nodes)

    def _promote_stable_edges(self) -> int:
        """Filter and promote stable edges from the NX graph."""
        stable_edges = []

        for src, tgt, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("type", "")
            if edge_type in self._effective_edge_types:
                stable_edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": edge_type,
                    }
                )

        if not stable_edges:
            return 0

        assert self.owl is not None  # only reached from the backend-backed cycle
        return self.owl.promote_edges(stable_edges)

    def _downfeed_inferences(self, inferences: list[dict[str, Any]]) -> int:
        """Write inferred facts back to the LPG as new edges and re-embed affected nodes."""
        downfed = 0
        affected_nodes = set()

        for inference in inferences:
            subject = inference.get("subject", "")
            predicate = inference.get("predicate", "")
            obj = inference.get("object", "")

            if not subject or not predicate or not obj:
                continue

            # Check if both subject and object exist in the NX graph
            if subject not in self.graph or obj not in self.graph:
                # Try to find by matching against sanitized IDs
                subject_match = self._find_node_by_owl_id(subject)
                obj_match = self._find_node_by_owl_id(obj)
                if not subject_match or not obj_match:
                    continue
                subject = subject_match
                obj = obj_match

            # Check if this exact edge already exists
            existing_edges = self.graph.get_edge_data(subject, obj)
            if existing_edges:
                already_exists = any(
                    e.get("type") == predicate for e in existing_edges.values()
                )
                if already_exists:
                    continue

            # Add inferred edge
            self.graph.add_edge(
                subject,
                obj,
                type=predicate,
                inferred=True,
                inferred_from="owl_reasoner",
                inference_type=inference.get("inference_type", "unknown"),
                timestamp=datetime.now(UTC).isoformat(),
            )
            downfed += 1
            affected_nodes.add(subject)
            affected_nodes.add(obj)

            # Entailment-aware scoping (CONCEPT:KG-2.6): a derived fact inherits
            # the most-restrictive classification of its parents so reasoning
            # can't leak a RESTRICTED node through an inferred edge.
            try:
                from .secured_reads import inherit_inferred_acl

                inherit_inferred_acl(subject, obj)
            except Exception:  # pragma: no cover - best-effort
                pass

        # Also sync to backend if available
        if downfed > 0 and self.backend:
            self._sync_inferred_to_backend(inferences, downfed)

        # Trigger Context-Aware Re-embedding (CONCEPT:KG-2.2)
        if affected_nodes:
            from .engine import IntelligenceGraphEngine

            engine = IntelligenceGraphEngine.get_active()
            if engine and hasattr(engine, "re_embed_node"):
                re_embedded = 0
                for node_id in affected_nodes:
                    if engine.re_embed_node(node_id):
                        re_embedded += 1
                logger.info(f"Re-embedded {re_embedded} nodes with new OWL context.")

        logger.info("Downfed %d inferred facts to LPG", downfed)
        return downfed

    def _find_node_by_owl_id(self, owl_id: str) -> str | None:
        """Try to match an OWL individual name back to an NX graph node ID."""
        # OWL IDs have : and / replaced with _
        for node_id in self.graph.nodes:
            safe = node_id.replace(":", "_").replace("/", "_").replace(".", "_")
            if safe == owl_id:
                return node_id
        return None

    def _sync_inferred_to_backend(
        self, inferences: list[dict[str, Any]], count: int
    ) -> None:
        """Persist inferred edges to the durable backend, synchronously and idempotently.

        Inferred relationships must survive on the durable tier (not just the
        in-memory reasoning graph) so other engines and restarts see them. The
        previous implementation queued mutations onto an asyncio queue + background
        task, which silently no-op'd whenever no event loop was running — and the
        daemon tick and pipeline phase both call this **synchronously**, so inferred
        triples never reached the backend. This writes them now via the active
        engine's ``link_nodes`` (MERGE-based; carries edge properties + provenance),
        with a direct-backend MERGE fallback. (CONCEPT:KG-2.7 — durable backfeed.)
        """
        import re as _re

        from .engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        written = 0
        for inference in inferences[:count]:
            subject = inference.get("subject", "")
            predicate = inference.get("predicate", "")
            obj = inference.get("object", "")
            if not subject or not predicate or not obj:
                continue
            src = self._find_node_by_owl_id(subject) or subject
            tgt = self._find_node_by_owl_id(obj) or obj
            props = {
                "inferred": True,
                "inferred_from": "owl_reasoner",
                "inference_type": inference.get("inference_type", "unknown"),
            }
            try:
                if (
                    engine is not None
                    and getattr(engine, "backend", None) is self.backend
                ):
                    engine.link_nodes(src, tgt, predicate, props)
                elif self.backend is not None:
                    rel = _re.sub(r"\W", "_", predicate).upper() or "INFERRED_RELATION"
                    self.backend.execute(
                        f"MATCH (s {{id: $sid}}), (t {{id: $tid}}) "
                        f"MERGE (s)-[r:{rel}]->(t) "
                        f"SET r.inferred = true, r.inferred_from = 'owl_reasoner'",
                        {"sid": src, "tid": tgt},
                    )
                written += 1
            except Exception as e:  # noqa: BLE001 — best-effort per inferred edge
                logger.debug("Inferred-edge backfeed failed (%s->%s): %s", src, tgt, e)
        if written:
            logger.info("Backfed %d inferred edges to the durable backend", written)

    def query_sparql(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query against the OWL backend or rdflib materialization.

        CONCEPT:KG-2.242 — SPARQL is demoted to the engine's native SPARQL 1.1
        surface (``client.rdf.sparql``), run over the LIVE engine graph rather than a
        materialized rdflib copy. The engine RDF/SPARQL surface ships in every profile
        (tiny/pi-tier binary included), so the engine path is the default everywhere.
        Strategies, in priority order:
        1. Engine-native SPARQL (``self.graph.sparql``) -- the default in every profile
        2. Native OWL backend SPARQL (if an OWL backend exposes ``query_sparql``)
        3. rdflib in-memory materialization -- TRUE last-resort (no engine + lib present)
        4. Basic regex-based LPG scan (last resort)

        Args:
            sparql: A SPARQL SELECT, ASK, or CONSTRUCT query string.

        Returns:
            List of result bindings as dicts. Each dict maps variable
            names to their bound values.
        """
        # 1) Engine-native SPARQL over the live graph (the demoted-to-engine path;
        #    the engine's native RDF/SPARQL surface is available in EVERY profile,
        #    tiny/pi included, so this is the default path -- not a no-engine special
        #    case). CONCEPT:KG-2.242.
        #    The engine's RDF/SPARQL surface queries the engine's RDF dataset, which
        #    only mirrors the property graph when that build projects the LPG into RDF
        #    (or it was explicitly seeded via ``add_triples``). When the engine answers
        #    with rows, it is authoritative. But an EMPTY result over a non-empty LPG
        #    means the engine's RDF projection is not populated from the live property
        #    graph -- so we fall through to the rdflib materialization (strategy 3),
        #    which materializes the LPG itself, instead of returning a false empty. A
        #    genuinely empty graph yields [] on both paths, so this never invents rows.
        if hasattr(self.graph, "sparql"):
            try:
                engine_rows = list(self.graph.sparql(sparql))
            except Exception as e:  # noqa: BLE001 -- fall through to Python last-resort
                logger.debug(
                    "Engine SPARQL unavailable (%s); falling back to Python.", e
                )
            else:
                if engine_rows:
                    return engine_rows
                logger.debug(
                    "Engine SPARQL returned no rows; its RDF surface may not mirror "
                    "the live property graph -- falling back to rdflib LPG "
                    "materialization."
                )

        # 2) A native OWL backend's own SPARQL implementation.
        if self.owl is not None and hasattr(self.owl, "query_sparql"):
            return self.owl.query_sparql(sparql)

        # 3) rdflib-based SPARQL execution -- the TRUE last-resort fallback (only when
        #    no engine is reachable AND rdflib happens to be installed).
        try:
            return self._sparql_via_rdflib(sparql)
        except ImportError:
            logger.info(
                "rdflib not installed. Install with 'pip install rdflib' "
                "for full SPARQL support. Falling back to regex scan."
            )
        except Exception as e:
            logger.warning("rdflib SPARQL execution failed: %s. Falling back.", e)

        # 4) Last-resort regex scan over the LPG.
        return self._sparql_fallback(sparql)

    def _build_rdf_graph(self) -> Any:
        """Materialize the LPG into an rdflib Graph for SPARQL queries.

        CONCEPT:KG-2.6 — RDF Materialization

        Promotes all nodes as typed OWL individuals and all edges as
        property assertions under the ``au:`` namespace. The result is
        cached and invalidated when the LPG changes.

        Returns:
            An rdflib.Graph instance populated from the current LPG state.
        """
        import rdflib

        g = rdflib.Graph()
        AU = rdflib.Namespace("http://agent-utilities.dev/ontology#")
        g.bind("au", AU)
        g.bind("rdf", rdflib.RDF)
        g.bind("rdfs", rdflib.RDFS)

        # Fast path (CONCEPT:KG-2.7): bulk-export triples from the engine in ONE
        # call (GetTriples) instead of per-node round-trips, then map to RDF. An
        # object that is itself a known subject is an edge target (URI); otherwise
        # a literal. Falls back to the per-node iteration below if unavailable.
        try:
            triples = self.graph.get_triples()
        except Exception:
            triples = None
        if triples:
            subjects = {str(t[0]) for t in triples if len(t) == 3}

            def _uri(x: str) -> Any:
                return AU[str(x).replace(" ", "_")]

            for t in triples:
                if len(t) != 3:
                    continue
                s, p, o = str(t[0]), str(t[1]), t[2]
                if p == "rdf:type":
                    g.add(
                        (
                            _uri(s),
                            rdflib.RDF.type,
                            AU[str(o).replace(" ", "_").title().replace("_", "")],
                        )
                    )
                elif str(o) in subjects:
                    g.add((_uri(s), AU[p], _uri(o)))
                else:
                    g.add((_uri(s), AU[p], rdflib.Literal(o)))
            return g

        # Promote nodes as typed individuals
        for node_id, data in self.graph.nodes(data=True):
            node_uri = AU[str(node_id).replace(" ", "_")]
            node_type = data.get("type", "Thing")
            # Type assertion
            type_class = AU[node_type.replace(" ", "_").title().replace("_", "")]
            g.add((node_uri, rdflib.RDF.type, type_class))
            # Add string properties as datatype properties
            for key, value in data.items():
                if key in ("embedding", "ewc_fisher_diag"):
                    continue  # Skip large float arrays
                if isinstance(value, str) and value:
                    g.add((node_uri, AU[key], rdflib.Literal(value)))
                elif isinstance(value, int | float):
                    g.add((node_uri, AU[key], rdflib.Literal(value)))

        # Promote edges as property assertions
        for src, tgt, data in self.graph.edges(data=True):
            src_uri = AU[str(src).replace(" ", "_")]
            tgt_uri = AU[str(tgt).replace(" ", "_")]
            edge_type = data.get("type", "relatedTo")
            prop = AU[edge_type]
            g.add((src_uri, prop, tgt_uri))

        return g

    def _sparql_via_rdflib(self, sparql: str) -> list[dict[str, Any]]:
        """Execute SPARQL via rdflib against a materialized RDF graph.

        Args:
            sparql: SPARQL query string.

        Returns:
            List of result row dicts.

        Raises:
            ImportError: If rdflib is not installed.
        """

        # Build (or use cached) RDF graph
        graph_hash = len(self.graph.nodes) + len(self.graph.edges)
        if not hasattr(self, "_rdf_cache") or self._rdf_cache_hash != graph_hash:
            self._rdf_cache = self._build_rdf_graph()
            self._rdf_cache_hash = graph_hash

        # Inject default prefix if not present
        if "PREFIX au:" not in sparql and "prefix au:" not in sparql:
            sparql = "PREFIX au: <http://agent-utilities.dev/ontology#>\n" + sparql

        qres = self._rdf_cache.query(sparql)

        # Handle different result types
        if isinstance(qres, bool):
            # ASK query
            return [{"result": qres}]

        results: list[dict[str, Any]] = []
        for row in qres:
            binding: dict[str, Any] = {}
            if hasattr(row, "labels"):
                # SELECT result
                for var in row.labels:
                    val = row[var]
                    binding[str(var)] = str(val) if val is not None else None
            elif hasattr(row, "__iter__"):
                # CONSTRUCT/DESCRIBE result (triples)
                s, p, o = row
                binding = {"subject": str(s), "predicate": str(p), "object": str(o)}
            results.append(binding)
        return results

    def _sparql_fallback(self, sparql: str) -> list[dict[str, Any]]:
        """Best-effort SPARQL fallback using the in-memory LPG.

        Handles basic ``SELECT ?s ?p ?o WHERE { ?s ?p ?o }`` patterns
        by scanning the graph compute engine. More complex queries return an
        informative error.
        """
        # Very basic pattern matching for simple triple patterns
        import re

        match = re.search(
            r"WHERE\s*\{[^}]*\?\w+\s+a\s+\w+:(\w+)", sparql, re.IGNORECASE
        )
        if match:
            target_type = match.group(1).lower()
            results = []
            for node_id, data in self.graph.nodes(data=True):
                node_type = data.get("type", "")
                if node_type == target_type or target_type in node_type:
                    results.append({"id": node_id, "type": node_type, **data})
            return results[:100]

        return [
            {
                "error": "Complex SPARQL queries require rdflib. "
                "Install with: pip install rdflib"
            }
        ]

    async def stream_api_to_graph(
        self, api_stream: Any, source_type: str, namespace: str, ttl_seconds: int = 3600
    ) -> dict[str, Any]:
        """Hydrates raw enterprise payloads from external streams into LPG nodes and edges.

        Strictly maps fields using rigid, schema-compliant R2RML mappings to prevent dynamic
        LLM hallucinations. If namespace matches 'redis://' or 'valkey://', hydrates the state
        to an external shared ephemeral cache fabric, enabling concurrent agents to share memory.

        Supports automatic TTL expiration tracking.
        """
        # R2RML mappings based on ontology.ttl classes & properties
        r2rml_mappings = {
            "servicenow:incident": {
                "class": "Incident",
                "id_field": "sys_id",
                "properties": ["short_description", "severity", "state", "description"],
                "edges": {
                    "assigned_to": ("Person", "was_attributed_to"),
                    "cmdb_ci": ("PlatformService", "monitors"),
                },
            },
            "gitlab:project": {
                "class": "Repository",
                "id_field": "id",
                "properties": ["name", "path_with_namespace", "description"],
                "edges": {"owner": ("Person", "creator")},
            },
            "gitlab:pipeline": {
                "class": "Pipeline",
                "id_field": "id",
                "properties": ["status", "ref", "sha"],
                "edges": {"project_id": ("Repository", "part_of")},
            },
        }

        mapping = r2rml_mappings.get(source_type)
        if not mapping:
            raise ValueError(
                f"No registered R2RML mapping for stream source type: {source_type}"
            )

        nodes_hydrated = 0
        edges_hydrated = 0
        hydrated_payloads = []

        # Consume api stream items (handles lists, dicts, or async iterators)
        items = []
        if isinstance(api_stream, list):
            items = api_stream
        elif isinstance(api_stream, dict):
            items = [api_stream]
        else:
            # Assume async iterator
            try:
                async for item in api_stream:
                    items.append(item)
            except Exception:
                # Fallback to standard iter
                for item in api_stream:
                    items.append(item)

        id_field = str(mapping.get("id_field", ""))
        class_name = str(mapping.get("class", ""))
        props_keys = mapping.get("properties", [])
        edges_mapping = mapping.get("edges", {})

        for item in items:
            raw_id = item.get(id_field)
            if not raw_id:
                continue

            node_id = f"{source_type}:{raw_id}"

            # Construct mapped properties
            props = {
                "type": class_name,
                "id": node_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "is_permanent": False,
            }
            if isinstance(props_keys, list):
                for p in props_keys:
                    if p in item:
                        props[p] = item[p]

            # Construct mapped edges
            edges = []
            if isinstance(edges_mapping, dict):
                for field, target_mapping in edges_mapping.items():
                    ref_val = item.get(field)
                    if ref_val:
                        tgt_class, rel_type = target_mapping
                        tgt_id = f"{tgt_class.lower()}:{ref_val}"
                        edges.append(
                            {
                                "source": node_id,
                                "target": tgt_id,
                                "type": rel_type,
                                "inferred": False,
                            }
                        )

            hydrated_payloads.append({"node": props, "edges": edges})
            nodes_hydrated += 1
            edges_hydrated += len(edges)

        # Persistence: External Valkey/Redis or local Namespaced Storage
        is_external = namespace.startswith("redis://") or namespace.startswith(
            "valkey://"
        )
        expiration_time = time.time() + ttl_seconds

        if is_external:
            try:
                import redis

                client = redis.from_url(namespace)
                # Store serialized graph nodes & edges
                key_prefix = f"company_brain:ns:{namespace.split('/')[-1]}"
                client.setex(
                    f"{key_prefix}:data", ttl_seconds, json.dumps(hydrated_payloads)
                )
                logger.info(
                    "Successfully hydrated %d nodes to external cache fabric.",
                    nodes_hydrated,
                )
            except Exception as e:
                logger.warning(
                    "Valkey/Redis hydration failed: %s. Falling back to local namespace.",
                    e,
                )
                self._save_local_namespace(
                    namespace, hydrated_payloads, expiration_time
                )
        else:
            self._save_local_namespace(namespace, hydrated_payloads, expiration_time)

        # Inject hydrated nodes and edges into active LPG context
        for hydrated in hydrated_payloads:
            n = hydrated.get("node")
            if isinstance(n, dict):
                n_id = n.get("id")
                if isinstance(n_id, str):
                    self.graph.add_node(n_id, **n)
            e_list = hydrated.get("edges")
            if isinstance(e_list, list):
                for e_item in e_list:
                    if isinstance(e_item, dict):
                        e_src = e_item.get("source")
                        e_tgt = e_item.get("target")
                        e_type = e_item.get("type")
                        e_inferred = e_item.get("inferred", False)
                        if isinstance(e_src, str) and isinstance(e_tgt, str):
                            self.graph.add_edge(
                                e_src, e_tgt, type=e_type, inferred=e_inferred
                            )

        return {
            "namespace": namespace,
            "nodes_hydrated": nodes_hydrated,
            "edges_hydrated": edges_hydrated,
            "expiration": expiration_time,
        }

    def _save_local_namespace(
        self, namespace: str, data: list[dict[str, Any]], expiration: float
    ) -> None:
        self._namespaces[namespace] = {"data": data}
        self._namespace_ttls[namespace] = expiration

    def cleanup_expired_namespaces(self) -> int:
        """Evicts expired local namespaces and clean up related cache allocations."""
        now = time.time()
        expired = [ns for ns, exp in self._namespace_ttls.items() if now >= exp]
        for ns in expired:
            # Clean up the graph nodes that belong to this namespace
            ns_data = self._namespaces.get(ns, {}).get("data", [])
            for hydrated in ns_data:
                node_id = hydrated["node"]["id"]
                if node_id in self.graph:
                    self.graph.remove_node(node_id)

            del self._namespaces[ns]
            del self._namespace_ttls[ns]
            logger.info("Evicted expired namespace cache: %s", ns)

        return len(expired)
