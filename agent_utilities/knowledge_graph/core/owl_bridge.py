#!/usr/bin/python
from __future__ import annotations

"""OWL Bridge — Orchestrates LPG ↔ OWL data flow.

Handles the deterministic promote → reason → downfeed cycle that
enriches the LPG with OWL-inferred facts.
"""


import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Rust-native graph compute — using GraphComputeEngine

    from ..backends.base import GraphBackend
    from ..backends.owl.base import OWLBackend

logger = logging.getLogger(__name__)

# Node types eligible for OWL promotion (must have matching class in ontology.ttl)
PROMOTABLE_NODE_TYPES: set[str] = {
    "agent",
    "tool",
    "skill",
    "file",
    "symbol",
    "module",
    "memory",
    "event",
    "episode",
    "phase",
    "incident",
    "decision",
    "observation",
    "action",
    "belief",
    "hypothesis",
    "fact",
    "principle",
    "concept",
    "evidence",
    "reflection",
    "organization",
    "person",
    "role",
    "place",
    "system",
    "team",
    "reasoning_trace",
    "outcome_evaluation",
    "critique",
    "goal",
    "policy",
    "server",
    "code",
    # Standard Ontology Types (BFO, Schema.org, DC, FIBO)
    "document",
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
    "task",
    "codemap",
    "pattern_template",
    "proposed_skill",
    "system_prompt",
    "prompt",
    "process_flow",
    "process_step",
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
    "alert",
    "chat_channel",
    # Capability Abstraction Layer (CONCEPT:KG-2.7)
    "service_capability",
    "vpn_purpose",
    "development_domain",
    "development_standard",
    "ea_fact_sheet",
    "process_model",
}

# Edge types eligible for OWL promotion (transitive / inferable relationships)
PROMOTABLE_EDGE_TYPES: set[str] = {
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
    "mitigated_by",
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
    "runs_on",
    "has_accelerator",
    "attached_storage",
    "deployed_on",
    "belongs_to_stack",
    # Hydration Core Edges
    "blocked_by",
    "monitors",
    "works_at",
    "related_to",
    "associated_with",
    # Capability Abstraction Layer (CONCEPT:KG-2.7)
    "provides_capability",
    "requires_capability",
    "swappable_with",
    "has_purpose",
    "requires_vpn_for_purpose",
    "applies_to_domain",
    "works_on_domain",
    "must_follow",
    # Universal Relationship Properties (CONCEPT:KG-2.8)
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
    "member_of",
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
        owl_backend: OWLBackend,
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

        # Compute effective promotable types filtered by schema pack (CONCEPT:KG-2.2)
        if schema_pack is not None:
            active_nodes = {str(t) for t in schema_pack.get_active_node_types()}
            active_edges = {str(t) for t in schema_pack.get_active_edge_types()}
            self._effective_node_types = PROMOTABLE_NODE_TYPES & active_nodes
            self._effective_edge_types = PROMOTABLE_EDGE_TYPES & active_edges
        else:
            self._effective_node_types = PROMOTABLE_NODE_TYPES
            self._effective_edge_types = PROMOTABLE_EDGE_TYPES

    def run_cycle(self, lightweight: bool = True) -> dict[str, Any]:
        """Full promote → reason → downfeed cycle. Returns stats.

        If lightweight=True, performs fast local RDFS+ closures.
        If False, executes full Description Logic reasoning via the OWL backend.
        """
        self.owl.clear()

        promoted_nodes = self._promote_stable_nodes()
        promoted_edges = self._promote_stable_edges()

        if lightweight:
            inferences = self._lightweight_reasoning()
        else:
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
        """Lightweight RDFS+ reasoning — Rust-first with Python fallback.

        CONCEPT:KG-2.23 — Rust-Accelerated Reasoning

        Attempts to use the Rust Datalog engine (epistemic-graph) for
        transitive/symmetric/domain-range/property-chain inference.
        Falls back to Python networkx implementation if the Rust engine
        is unavailable.
        """
        try:
            return self._rust_reasoning()
        except Exception:
            logger.debug("Rust reasoning unavailable, falling back to Python RDFS+")
            return self._python_reasoning()

    def _rust_reasoning(self) -> list[dict[str, Any]]:
        """Execute reasoning via the Rust Datalog engine.

        Uses epistemic-graph's compiled transitive/symmetric inference,
        domain/range rules, and property chain composition.
        """
        try:
            from epistemic_graph import EpistemicGraph
        except ImportError as exc:
            raise RuntimeError("epistemic-graph not available") from exc

        # Build a temporary Rust graph from the current in-memory networkx graph
        eg = EpistemicGraph()
        for node_id, attrs in self.graph.nodes(data=True):
            eg.add_node(str(node_id), json.dumps(attrs))
        for u, v, data in self.graph.edges(data=True):
            try:
                eg.add_edge(str(u), str(v), json.dumps(data))
            except Exception:
                pass  # Skip edges whose endpoints aren't in the graph

        # Transitive properties
        transitive_props = [
            "part_of",
            "depends_on",
            "broader",
            "narrower",
            "inherits_from",
        ]
        # Symmetric properties
        symmetric_props = [
            "related_concept",
            "exact_match",
            "close_match",
            "broad_match",
        ]

        # Run compiled Datalog reasoning
        inferred = eg.infer_transitive(transitive_props, symmetric_props)

        # Domain/range inference rules
        domain_rules = [
            ("authored_by", "Agent"),
            ("executed_by", "Agent"),
            ("created_by", "Agent"),
        ]
        range_rules = [
            ("authored_by", "Artifact"),
            ("produces", "Artifact"),
        ]
        dr_inferred = eg.infer_domain_range(domain_rules, range_rules)

        # Property chain composition
        chains = [
            ("part_of", "part_of", "part_of"),  # transitivity
            ("depends_on", "part_of", "depends_on"),  # dependency propagation
        ]
        chain_inferred = eg.infer_property_chains(chains)

        # Convert Rust results to standard inference dicts
        inferences: list[dict[str, Any]] = []
        for fact in inferred:
            inferences.append(
                {
                    "subject": fact.get("subject", ""),
                    "predicate": fact.get("predicate", ""),
                    "object": fact.get("object", ""),
                    "inference_type": fact.get("inference_type", "rust_datalog"),
                }
            )
        for fact in dr_inferred:
            inferences.append(
                {
                    "subject": fact.get("subject", ""),
                    "predicate": "rdf:type",
                    "object": fact.get("type", ""),
                    "inference_type": "domain_range_inference",
                }
            )
        for fact in chain_inferred:
            inferences.append(
                {
                    "subject": fact.get("subject", ""),
                    "predicate": fact.get("predicate", ""),
                    "object": fact.get("object", ""),
                    "inference_type": "property_chain",
                }
            )

        return inferences

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
        }
        symmetric_props = {
            "related_concept",
            "exact_match",
            "close_match",
            "broad_match",
        }

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
        """Attempt to write inferred edges to the graph backend (Cypher) asynchronously via batches."""
        try:
            # We initialize the queue and background task here to ensure they run on the correct event loop
            if not hasattr(self, "_mutation_queue"):
                self._mutation_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                self._mutation_task = asyncio.create_task(self._mutation_worker())

            for inference in inferences[:count]:
                subject = inference.get("subject", "")
                predicate = inference.get("predicate", "")
                obj = inference.get("object", "")

                if not subject or not predicate or not obj:
                    continue

                subject_match = self._find_node_by_owl_id(subject) or subject
                obj_match = self._find_node_by_owl_id(obj) or obj

                # Put onto background queue for async batching
                self._mutation_queue.put_nowait(
                    {"src": subject_match, "tgt": obj_match, "pred": predicate}
                )
        except Exception as e:
            logger.debug("Backend sync queuing failed: %s", e)

    async def _mutation_worker(self) -> None:
        """Background worker that batches mutations and flushes to the backend."""
        batch: list[dict[str, Any]] = []
        last_flush = asyncio.get_event_loop().time()

        while True:
            try:
                # Wait up to 2 seconds for a mutation
                timeout = max(0.1, 2.0 - (asyncio.get_event_loop().time() - last_flush))
                mutation = await asyncio.wait_for(
                    self._mutation_queue.get(), timeout=timeout
                )
                batch.append(mutation)
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            now = asyncio.get_event_loop().time()
            if batch and (len(batch) >= 100 or (now - last_flush) >= 2.0):
                await self._flush_batch(batch)
                batch.clear()
                last_flush = now

    async def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        """Flush a batch of mutations to the backend."""
        if not self.backend:
            return

        try:
            # For JenaFuseki, ideally we would do a single large SPARQL UPDATE,
            # but if the backend is generic, we use execute_batch.
            if hasattr(self.backend, "execute_batch"):
                query = (
                    "MATCH (a {id: $src}), (b {id: $tgt}) "
                    "MERGE (a)-[r:INFERRED_RELATION {type: $pred}]->(b) "
                    "SET r.inferred = true, r.inferred_from = 'owl_reasoner'"
                )
                # If backend has async execution, we could await it, but execute_batch is sync in SparqlAdapter
                # We offload the blocking execute_batch to a thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self.backend.execute_batch, query, batch
                )
        except Exception as e:
            logger.debug("Background batch flush failed: %s", e)

    def query_sparql(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query against the OWL backend or rdflib materialization.

        CONCEPT:KG-2.6 — SPARQL Read-Only Endpoint

        Supports three execution strategies in priority order:
        1. Native OWL backend SPARQL (if available)
        2. rdflib in-memory RDF graph materialization (preferred fallback)
        3. Basic regex-based LPG scan (last resort)

        Args:
            sparql: A SPARQL SELECT, ASK, or CONSTRUCT query string.

        Returns:
            List of result bindings as dicts. Each dict maps variable
            names to their bound values.
        """
        if hasattr(self.owl, "query_sparql"):
            return self.owl.query_sparql(sparql)

        # Try rdflib-based SPARQL execution
        try:
            return self._sparql_via_rdflib(sparql)
        except ImportError:
            logger.info(
                "rdflib not installed. Install with 'pip install rdflib' "
                "for full SPARQL support. Falling back to regex scan."
            )
        except Exception as e:
            logger.warning("rdflib SPARQL execution failed: %s. Falling back.", e)

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
