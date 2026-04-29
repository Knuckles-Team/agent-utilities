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
    ):
        self.graph = graph
        self.backend = backend
        self.owl = owl_backend
        self.importance_threshold = importance_threshold
        self.recency_days = recency_days

    def run_cycle(self) -> dict[str, int]:
        """Full promote → reason → downfeed cycle. Returns stats."""
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
        }
        logger.info("OWL reasoning cycle complete: %s", stats)
        return stats

    def _is_eligible_node(self, node_id: str, attrs: dict[str, Any]) -> bool:
        """Check if a node meets promotion criteria."""
        node_type = attrs.get("type", "")
        if node_type not in PROMOTABLE_NODE_TYPES:
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
            if edge_type in PROMOTABLE_EDGE_TYPES:
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
        """Write inferred facts back to the LPG as new edges."""
        downfed = 0

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

        # Also sync to backend if available
        if downfed > 0 and self.backend:
            self._sync_inferred_to_backend(inferences, downfed)

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
