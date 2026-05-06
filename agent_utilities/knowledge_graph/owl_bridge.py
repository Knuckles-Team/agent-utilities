#!/usr/bin/python
"""OWL Bridge — Orchestrates LPG ↔ OWL data flow.

Handles the deterministic promote → reason → downfeed cycle that
enriches the LPG with OWL-inferred facts.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

    from .backends.base import GraphBackend
    from .backends.owl.base import OWLBackend

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
    # Market Data Connector Protocol (CONCEPT:ECO-4.4)
    "data_connector",
    "data_fetch_record",
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    "swarm_preset",
    "swarm_run",
    "swarm_task_record",
    # Risk Scoring Ontology (CONCEPT:KG-2.7)
    "risk_assessment",
    "risk_factor",
    "risk_mitigation",
    # Backtest Evaluation Harness (CONCEPT:AHE-3.8)
    "backtest_run",
    "backtest_metric",
    # Prompt Injection Scanner (CONCEPT:OS-5.4)
    "security_finding",
    # Tool Repetition Guard (CONCEPT:OS-5.5)
    "experience",
    # MATE Integration — Token Analytics (CONCEPT:OS-5.6)
    "token_usage_record",
    # MATE Integration — Audit Logging (CONCEPT:OS-5.7)
    "audit_log",
    # MATE Integration — Guardrail Engine (CONCEPT:OS-5.8)
    "guardrail_trigger",
    # MATE Integration — Config Versioning (CONCEPT:AHE-3.13)
    "agent_config_version",
    # MATE Integration — EvalRunner (CONCEPT:AHE-3.12)
    "eval_run",
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
    # Market Data Connector Protocol (CONCEPT:ECO-4.4)
    "fetched_from",
    "falls_back_to",
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    "preset_of",
    "ran_preset",
    "task_depends_on",
    # Risk Scoring Ontology (CONCEPT:KG-2.7)
    "assessed_risk",
    "has_risk_factor",
    "mitigated_by",
    "propagates_risk_to",
    # Backtest Evaluation Harness (CONCEPT:AHE-3.8)
    "evaluated_strategy",
    "has_metric",
    "compared_to_benchmark",
    # Prompt Injection Scanner (CONCEPT:OS-5.4)
    "detected_threat",
    # Structured Retry Manager (CONCEPT:AHE-3.11)
    "triggered_retry",
    # MATE Integration — Audit Logging (CONCEPT:OS-5.7)
    "audited_by",
    # MATE Integration — Guardrail Engine (CONCEPT:OS-5.8)
    "triggered_guardrail",
    # MATE Integration — Config Versioning (CONCEPT:AHE-3.13)
    "config_version_of",
    # MATE Integration — EvalRunner (CONCEPT:AHE-3.12)
    "evaluated_by",
}


class OWLBridge:
    """Orchestrates LPG ↔ OWL data flow.

    The bridge performs a deterministic three-step cycle:

    1. **Promote** — Export stable, high-confidence nodes and edges from
       the in-memory NetworkX graph to the OWL backend as individuals and
       property assertions.

    2. **Reason** — Run OWL DL reasoning (HermiT/Stardog) to discover
       new inferred facts (transitive closure, subclass inference, etc.).

    3. **Downfeed** — Write inferred facts back to the LPG as new edges
       (with ``inferred=True`` metadata) so the agent's standard queries
       immediately benefit from the reasoning.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
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
        """Lightweight RDFS+ reasoning on the in-memory graph.

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
                    for e in self.graph.get_edge_data(v, u, {}).values()
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
                                for e in self.graph.get_edge_data(u, w, {}).values()
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
        """Attempt to write inferred edges to the graph backend (Cypher)."""
        try:
            for inference in inferences[:count]:
                subject = inference.get("subject", "")
                predicate = inference.get("predicate", "")
                obj = inference.get("object", "")

                subject_match = self._find_node_by_owl_id(subject) or subject
                obj_match = self._find_node_by_owl_id(obj) or obj

                query = (
                    "MATCH (a {id: $src}), (b {id: $tgt}) "
                    "MERGE (a)-[r:INFERRED_RELATION {type: $pred}]->(b) "
                    "SET r.inferred = true, r.inferred_from = 'owl_reasoner'"
                )
                if self.backend:
                    self.backend.execute(
                        query,
                        {"src": subject_match, "tgt": obj_match, "pred": predicate},
                    )
        except Exception as e:
            logger.debug("Backend sync for inferred facts failed: %s", e)

    def query_sparql(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query against the OWL backend.

        CONCEPT:ORCH-1.1 — RLM × OWL Integration

        Exposes the OWL reasoner's SPARQL interface for programmatic
        queries from the RLM REPL. Supports transitive property traversal,
        SKOS hierarchy navigation, and provenance chain analysis.

        Args:
            sparql: A SPARQL SELECT query string.

        Returns:
            List of result bindings as dicts. Each dict maps variable
            names to their bound values.

        Raises:
            RuntimeError: If the OWL backend does not support SPARQL.

        Example::

            results = bridge.query_sparql('''
                PREFIX au: <http://agent-utilities.dev/ontology#>
                SELECT ?manifest ?edit WHERE {
                    ?manifest a au:ChangeManifest .
                    ?manifest au:hasEditFor ?edit .
                }
            ''')
        """
        if hasattr(self.owl, "query_sparql"):
            return self.owl.query_sparql(sparql)

        # Fallback: convert to in-memory graph traversal
        logger.warning(
            "OWL backend does not support SPARQL directly. "
            "Running a filtered graph scan instead."
        )
        return self._sparql_fallback(sparql)

    def _sparql_fallback(self, sparql: str) -> list[dict[str, Any]]:
        """Best-effort SPARQL fallback using the in-memory LPG.

        Handles basic ``SELECT ?s ?p ?o WHERE { ?s ?p ?o }`` patterns
        by scanning the NetworkX graph. More complex queries return an
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
                "error": "Complex SPARQL queries require a backend with native SPARQL support."
            }
        ]
